"""Meaningful protocol/regression checks; synthetic data never represents findings."""

import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import torch
from IPython.core.interactiveshell import InteractiveShell
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from src.config import Config
from src.dataset import (balanced_subset, get_or_create_split, make_loader,
                         make_transforms, prepare_data, validate_split)
from src.evaluate import evaluate_clean_test
from src.model import build_model
from src.train import run_directory, run_epoch, run_experiment
from src.utils import (capture_rng, load_checkpoint, restore_rng, save_checkpoint,
                       seed_everything)

ROOT = Path(__file__).resolve().parents[1]


class StochasticFixture(Dataset):
    """Exercise torch, Python and NumPy augmentation RNG plus loader shuffle."""

    def __init__(self, count, training):
        generator = torch.Generator().manual_seed(101 + count)
        self.images = torch.rand(count, 3, 32, 32, generator=generator)
        self.targets = torch.arange(count) % 100
        self.training = training

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.images[index].clone()
        if self.training:
            image = image * (0.8 + 0.1 * random.random() + 0.1 * np.random.random())
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
        return image, self.targets[index]


def fixture_data(config, device):
    generators = {name: torch.Generator().manual_seed(config.seed + offset)
                  for name, offset in (("train", 0), ("val", 1))}
    train = make_loader(StochasticFixture(8, True), config, device,
                        generators["train"], shuffle=True)
    val = make_loader(StochasticFixture(4, False), config, device, generators["val"])
    manifest = {"split_path": str(Path(config.artifact_root) / "fixture.json"),
                "split_sha256": "SYNTHETIC_REGRESSION_FIXTURE",
                "train_indices": list(range(8)), "val_indices": list(range(8, 12)),
                "train_count": 8, "val_count": 4, "classes": list(map(str, range(100)))}
    return train, val, generators, manifest, {"fixture": True}


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42, cpu_threads=2)

    def test_notebook_schema_syntax_and_source_sync(self):
        notebook = nbformat.read(ROOT / "notebooks/01_baseline.ipynb", as_version=4)
        nbformat.validate(notebook)
        shell = InteractiveShell.instance()
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(shell.input_transformer_manager.transform_cell(cell.source),
                        f"notebook:{cell.id}", "exec")
                self.assertEqual(cell.outputs, [])
                if cell.source.startswith("%%writefile"):
                    directive, embedded = cell.source.split("\n", 1)
                    name = directive.rsplit("/", 1)[-1]
                    folder = "src" if cell.id.startswith("module-") else "third_party"
                    self.assertEqual(embedded, (ROOT / folder / name).read_text())
        for path in ROOT.glob("src/*.py"):
            compile(path.read_text(), str(path), "exec")
        settings_cell = next(cell for cell in notebook.cells if cell.id == "configuration")
        namespace = {}
        exec(settings_cell.source, namespace)
        configured = Config(**namespace["SETTINGS"])
        self.assertEqual(configured.to_dict(), Config().to_dict())

    def test_split_balance_persistence_and_tamper_detection(self):
        targets = np.tile(np.arange(100), 500).tolist()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            split = get_or_create_split(targets, path, 42)
            self.assertEqual(len(split["train_indices"]), 45000)
            self.assertEqual(len(split["val_indices"]), 5000)
            self.assertFalse(set(split["train_indices"]) & set(split["val_indices"]))
            self.assertEqual(get_or_create_split(targets, path, 42), split)
            self.assertEqual(get_or_create_split(targets, Path(temporary) / "new.json", 42), split)
            for key, per_class in (("train_indices", 2), ("val_indices", 1)):
                subset = balanced_subset(split[key], targets, per_class, 43)
                self.assertTrue(set(subset) <= set(split[key]))
                np.testing.assert_array_equal(np.bincount(np.asarray(targets)[subset]),
                                              np.full(100, per_class))
            with self.assertRaises(ValueError):
                validate_split(split, targets, 43)
            split["val_indices"][0] = split["train_indices"][0]
            with self.assertRaises(ValueError):
                validate_split(split, targets, 42)

    def test_architecture_forward_backward_optimizer_and_portable_checkpoint(self):
        model = build_model()
        self.assertEqual([len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)],
                         [2, 2, 2, 2])
        self.assertIsInstance(model.maxpool, nn.Identity)
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        before = model.fc.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        images, targets = torch.randn(2, 3, 32, 32), torch.tensor([0, 99])
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 100))
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertGreater(model.conv1.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fc.weight))
        model.eval()
        with torch.inference_mode():
            expected = model(images)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "portable.pt"
            save_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "rng": capture_rng()})
            saved = load_checkpoint(path)
            restored = build_model().eval()
            restored.load_state_dict(saved["model"])
            with torch.inference_mode():
                self.assertTrue(torch.equal(restored(images), expected))

    def test_all_rng_states_restore(self):
        seed_everything(77, cpu_threads=2)
        state = capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_rng(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_training_only_loading_and_transform_isolation(self):
        class Raw:
            targets = np.tile(np.arange(100), 500).tolist()
            classes = list(map(str, range(100)))

            def __getitem__(self, index):
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[:, :16, :] = index % 255
                return Image.fromarray(image), self.targets[index]

        with tempfile.TemporaryDirectory() as temporary, patch(
                "src.dataset.datasets.CIFAR100", return_value=Raw()) as dataset:
            config = Config(artifact_root=temporary, num_workers=0)
            train, val, _, manifest, _ = prepare_data(config, torch.device("cpu"))
            dataset.assert_called_once_with(config.data_root, train=True, download=True)
            self.assertEqual((len(train.dataset), len(val.dataset)), (200, 100))
            self.assertFalse(set(manifest["train_indices"]) & set(manifest["val_indices"]))
            image1, target1 = val.dataset[0]
            image2, target2 = val.dataset[0]
            self.assertTrue(torch.equal(image1, image2))
            self.assertEqual(target1, target2)
            train_transform, eval_transform = make_transforms(config)
            self.assertEqual([type(t).__name__ for t in train_transform.transforms],
                             ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"])
            self.assertEqual([type(t).__name__ for t in eval_transform.transforms],
                             ["ToTensor", "Normalize"])

    def test_sample_weighted_metrics_with_partial_batch(self):
        logits = torch.tensor([[8., 0.], [0., 1.], [0., 4.], [1., 0.], [2., 0.]])
        targets = torch.tensor([0, 1, 0, 1, 0])
        loader = torch.utils.data.DataLoader(TensorDataset(logits, targets), batch_size=2)
        metrics = run_epoch(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertAlmostEqual(metrics["loss"], nn.CrossEntropyLoss()(logits, targets).item(), places=6)
        self.assertEqual(metrics["accuracy_pct"], 60.0)
        self.assertEqual(metrics["count"], 5)

    def test_test_set_is_guarded_for_smoke_and_incomplete_runs(self):
        with patch("src.evaluate.CIFAR100") as dataset:
            with self.assertRaises(ValueError):
                evaluate_clean_test(Config())
            with patch("src.evaluate.load_checkpoint", return_value={"epoch": 1, "history": [{}]}):
                with self.assertRaises(ValueError):
                    evaluate_clean_test(Config(smoke_test=False))
            dataset.assert_not_called()

    def test_schedule_matches_mode_and_invalid_settings_fail(self):
        self.assertEqual(Config().epochs, 2)
        self.assertEqual(Config(smoke_test=False).epochs, 100)
        with self.assertRaises(ValueError):
            Config(run_id="../escape").validate()

    def test_interrupted_resume_matches_uninterrupted_including_workers(self):
        # Real ResNet18 optimization on synthetic fixtures, never claimed as CIFAR results.
        # Both the main-process and multiworker augmentation paths must replay exactly.
        for workers in (0, 2):
            with self.subTest(num_workers=workers), tempfile.TemporaryDirectory() as temporary:
                config = Config(artifact_root=str(Path(temporary) / "continuous"),
                                num_workers=workers, batch_size=4, cpu_threads=2, use_amp=False)
                interrupted = replace(config, artifact_root=str(Path(temporary) / "resumed"))
                with patch("src.train.prepare_data", side_effect=fixture_data), patch(
                        "src.train.select_device", return_value=torch.device("cpu")):
                    continuous_dir, _ = run_experiment(config)
                    resumed_dir, _ = run_experiment(interrupted, stop_after_epoch=1)
                    # Simulate a crash during best-model publication and damaged metric export.
                    (resumed_dir / "checkpoints/best_model.pt").unlink()
                    (resumed_dir / "metrics.json").write_text("interrupted export")
                    with self.assertRaises(FileExistsError):
                        run_experiment(interrupted)
                    with self.assertRaises(ValueError):
                        run_experiment(replace(interrupted, learning_rate=0.2), resume=True)
                    with patch("src.train.source_digest", return_value="changed_source"):
                        with self.assertRaises(ValueError):
                            run_experiment(interrupted, resume=True)
                    run_experiment(interrupted, resume=True)
                continuous = load_checkpoint(continuous_dir / "checkpoints/last_checkpoint.pt")
                resumed = load_checkpoint(resumed_dir / "checkpoints/last_checkpoint.pt")
                for key, value in continuous["model_state_dict"].items():
                    self.assertTrue(torch.equal(value, resumed["model_state_dict"][key]), key)
                self.assertEqual(continuous["scheduler_state_dict"], resumed["scheduler_state_dict"])
                self.assertEqual(continuous["scaler_state_dict"], resumed["scaler_state_dict"])
                self.assertEqual(continuous["optimizer_state_dict"]["param_groups"],
                                 resumed["optimizer_state_dict"]["param_groups"])
                for parameter, state in continuous["optimizer_state_dict"]["state"].items():
                    for name, value in state.items():
                        self.assertTrue(torch.equal(value, resumed["optimizer_state_dict"]["state"][parameter][name]))
                self.assertEqual(continuous["best_epoch"], resumed["best_epoch"])
                self.assertEqual(continuous["best_validation_accuracy_pct"],
                                 resumed["best_validation_accuracy_pct"])
                for name, state in continuous["loader_rng_state"].items():
                    self.assertTrue(torch.equal(state, resumed["loader_rng_state"][name]))
                for name in ("python", "numpy"):
                    self.assertEqual(continuous["rng_state"][name], resumed["rng_state"][name])
                self.assertTrue(torch.equal(continuous["rng_state"]["torch"], resumed["rng_state"]["torch"]))
                for left, right in zip(continuous["history"], resumed["history"]):
                    for key in ("epoch", "learning_rate", "train_loss", "val_loss",
                                "train_accuracy_pct", "val_accuracy_pct"):
                        self.assertEqual(left[key], right[key], key)
                self.assertEqual([row["learning_rate"] for row in resumed["history"]], [0.1, 0.05])
                self.assertEqual(len(json.loads((resumed_dir / "metrics.json").read_text())), 2)
                for name in ("accuracy", "loss", "learning_rate"):
                    self.assertTrue((resumed_dir / "plots" / f"{name}.png").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
