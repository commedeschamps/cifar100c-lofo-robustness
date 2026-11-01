"""LOFO protocol, operator-reference, leakage and resume regressions."""
import hashlib
import json
import random
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import torch
from PIL import Image
from IPython.core.interactiveshell import InteractiveShell
from torch.utils.data import Dataset

from src.config import Config, LOFOConfig, load_config
from src.corruptions import LOFOTrainingTransform, OPERATORS, apply_corruption
from src.dataset import make_loader, make_transforms, prepare_data, split_fingerprint
from src.lofo import (CORRUPTION_NAMES, FAMILIES, ORIGINAL_SPLIT_SHA256,
                      PLANNED_SEEDS, allowed_for, protocol_metadata)
from src.train import run_directory, run_experiment
from src.utils import load_checkpoint, select_device
from third_party import cifar_c_reference as reference

ROOT = Path(__file__).resolve().parents[1]


def condition(family="noise", **kwargs):
    return LOFOConfig(experiment_name="lofo_" + family, held_out_family=family,
                      allowed_corruptions=allowed_for(family), **kwargs)


class ImagesFixture(Dataset):
    def __init__(self, count, transform):
        self.pixels = np.random.RandomState(7).randint(0, 256, (count, 32, 32, 3), dtype=np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        transformed = self.transform(Image.fromarray(self.pixels[index]))
        if isinstance(transformed, tuple):
            tensor, code, severity = transformed
            return tensor, index % 100, code, severity
        return transformed, index % 100


def lofo_fixture(config, device):
    generators = {name: torch.Generator().manual_seed(config.seed + offset)
                  for name, offset in (("train", 0), ("val", 1))}
    _, clean = make_transforms(config)
    train = make_loader(ImagesFixture(24, LOFOTrainingTransform(config)), config, device,
                        generators["train"], shuffle=True)
    val = make_loader(ImagesFixture(12, clean), config, device, generators["val"])
    manifest = {"split_path": "SYNTHETIC_ONLY", "train_count": 24, "val_count": 12,
                "split_sha256": "synthetic_resume_regression"}
    return train, val, generators, manifest, {"fixture": "synthetic_only"}


class LOFOTests(unittest.TestCase):
    def test_reference_equivalence_all_operators_severities_and_edges(self):
        # Reference bodies are pinned/extracted from upstream, not our implementation.
        pixels = np.random.RandomState(1).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        images = [Image.fromarray(pixels), Image.new("RGB", (32, 32), 0), Image.new("RGB", (32, 32), "white")]
        for name in OPERATORS:
            for severity in range(1, 6):
                for image in images:
                    with self.subTest(name=name, severity=severity, pixel=image.getpixel((0, 0))):
                        np.random.seed(2026)
                        expected = np.uint8(getattr(reference, name)(image, severity))
                        actual = apply_corruption(image, name, severity, np.random.RandomState(2026))
                        self.assertEqual(actual.size, (32, 32))
                        self.assertEqual(actual.mode, "RGB")
                        values = np.asarray(actual)
                        self.assertEqual(values.dtype, np.uint8)
                        self.assertTrue(np.isfinite(values).all())
                        self.assertGreaterEqual(int(values.min()), 0)
                        self.assertLessEqual(int(values.max()), 255)
                        np.testing.assert_array_equal(values, expected)

    def test_invalid_operator_inputs(self):
        image = Image.new("RGB", (32, 32))
        for severity in (0, 6, 1.5, True):
            with self.assertRaises(ValueError):
                apply_corruption(image, "fog", severity)
        with self.assertRaises(ValueError):
            apply_corruption(image, "shot_noise", 1)
        for invalid in (Image.new("L", (32, 32)), Image.new("RGB", (64, 64))):
            with self.assertRaises(ValueError):
                apply_corruption(invalid, "fog", 1)

    def test_sampling_balance_and_entire_family_exclusion(self):
        for family in FAMILIES:
            transform = LOFOTrainingTransform(condition(family))
            rng = np.random.RandomState(42)
            counts = {}
            size = 30000
            for _ in range(size):
                name, severity = transform.sample(rng)
                counts[(name, severity)] = counts.get((name, severity), 0) + 1
                self.assertNotIn(name, FAMILIES[family])
            clean = counts[("clean", 0)]
            self.assertLess(abs(clean / size - 0.5), 0.015)
            for name in transform.allowed:
                for severity in range(1, 6):
                    self.assertLess(abs(counts[(name, severity)] / size - 1/30), 0.006)
            for probability, expected_clean in ((0., True), (1., False), (0.25, None)):
                sampler = LOFOTrainingTransform(replace(condition(family), corruption_probability=probability))
                draws = [sampler.sample(rng)[0] == "clean" for _ in range(3000)]
                if expected_clean is not None:
                    self.assertTrue(all(value == expected_clean for value in draws))
                else:
                    self.assertLess(abs(sum(draws) / len(draws) - 0.75), 0.04)

    def test_configuration_rejects_invalid_or_unfair_conditions(self):
        invalids = [dict(allowed_corruptions=("gaussian_noise", "fog", "jpeg_compression")),
                    dict(allowed_corruptions=("glass_blur", "fog", "jpeg_compression")),
                    dict(held_out_family="unknown"), dict(experiment_name="lofo_blur"),
                    dict(corruption_probability=-0.1), dict(corruption_probability=float("nan")),
                    dict(corruption_selection="weighted"), dict(allowed_severities=(1, 2)),
                    dict(split_seed=123), dict(split_sha256="new_split"), dict(seed=9),
                    dict(seed=123), dict(learning_rate=0.2), dict(full_epochs=99), dict(batch_size=64)]
        for overrides in invalids:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(condition(), **overrides).validate()

    def test_all_configs_seeds_isolation_and_comparable_settings(self):
        directories = set()
        baseline = Config()
        for family in FAMILIES:
            config = load_config(ROOT / "configs" / f"lofo_{family}.json")
            for key in ("architecture", "full_epochs", "batch_size", "learning_rate", "momentum",
                        "weight_decay", "eta_min", "optimizer", "scheduler", "loss", "mean", "std"):
                self.assertEqual(config.to_dict()[key], baseline.to_dict()[key])
            for seed in PLANNED_SEEDS:
                for smoke in (False, True):
                    selected = replace(config, seed=seed, run_id=f"seed{seed}", smoke_test=smoke)
                    selected.validate()
                    directories.add(str(run_directory(selected)))
        self.assertEqual(len(directories), 24)

    def test_spatial_then_single_corruption_then_tensor(self):
        transform = LOFOTrainingTransform(condition(corruption_probability=1))
        image = Image.new("RGB", (32, 32), "red")
        spatial_image = Image.new("RGB", (32, 32), "blue")
        with patch.object(transform, "spatial", return_value=spatial_image), patch(
                "src.corruptions.apply_corruption", return_value=spatial_image) as corruption:
            tensor, code, severity = transform(image)
            self.assertEqual(tuple(tensor.shape), (3, 32, 32))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertIn(CORRUPTION_NAMES[code], transform.allowed)
            self.assertEqual(corruption.call_count, 1)
            self.assertEqual(corruption.call_args.args, (spatial_image, CORRUPTION_NAMES[code], severity))
            self.assertIsInstance(corruption.call_args.kwargs["rng"], np.random.RandomState)

    def test_operator_random_draws_do_not_change_future_sampling(self):
        traces, torch_states = [], []
        image = Image.fromarray(np.random.RandomState(1).randint(0, 256, (32, 32, 3), dtype=np.uint8))
        for family in FAMILIES:
            sampler = LOFOTrainingTransform(condition(family))
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            trace = []
            for _ in range(50):
                _, code, severity = sampler(image)
                name = CORRUPTION_NAMES[code]
                trace.append((name == "clean", severity,
                              None if name == "clean" else sampler.allowed.index(name)))
            traces.append(trace)
            torch_states.append(torch.get_rng_state())
        self.assertTrue(all(trace == traces[0] for trace in traces))
        self.assertTrue(all(torch.equal(state, torch_states[0]) for state in torch_states))

    def test_original_split_reused_without_writes_and_clean_validation(self):
        path = ROOT / "splits/cifar100_seed42.json"
        original = path.read_bytes()
        self.assertEqual(split_fingerprint(json.loads(original)), ORIGINAL_SPLIT_SHA256)
        config = condition(split_file=str(path), num_workers=0, download=False)
        with tempfile.TemporaryDirectory() as temporary, patch("src.dataset.get_or_create_split") as generate:
            config = replace(config, artifact_root=temporary)
            train, val, _, manifest, split = prepare_data(config, torch.device("cpu"))
            generate.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(manifest["split_sha256"], ORIGINAL_SPLIT_SHA256)
            self.assertFalse(set(manifest["train_indices"]) & set(manifest["val_indices"]))
            self.assertTrue(set(manifest["train_indices"]) <= set(split["train_indices"]))
            self.assertTrue(set(manifest["val_indices"]) <= set(split["val_indices"]))
            with patch("src.corruptions.apply_corruption", side_effect=AssertionError("validation leaked")):
                left, label = val.dataset[0]
                right, label2 = val.dataset[0]
            self.assertTrue(torch.equal(left, right))
            self.assertEqual(label, label2)
            self.assertEqual(len(next(iter(val))), 2)
            self.assertEqual(len(next(iter(train))), 4)

    def test_missing_changed_split_fails_before_dataset_access(self):
        with tempfile.TemporaryDirectory() as temporary, patch("src.dataset.datasets.CIFAR100") as data:
            path = Path(temporary) / "split.json"
            config = condition(split_file=str(path))
            with self.assertRaises(FileNotFoundError):
                prepare_data(config, torch.device("cpu"))
            split = json.loads((ROOT / "splits/cifar100_seed42.json").read_text())
            split["train_indices"][0], split["train_indices"][1] = split["train_indices"][1], split["train_indices"][0]
            path.write_text(json.dumps(split))
            with self.assertRaises(ValueError):
                prepare_data(config, torch.device("cpu"))
            data.assert_not_called()

    def test_future_evaluation_metadata_covers_all_15_corruptions(self):
        for family in FAMILIES:
            config = condition(family)
            categories = protocol_metadata(config)["evaluation_categories"]
            self.assertEqual(len(categories), 15)
            for name, entry in categories.items():
                expected = ("entirely_held_out_family" if name in FAMILIES[family]
                            else "exact_seen" if name in config.allowed_corruptions
                            else "unseen_within_represented_family")
                self.assertEqual(entry["exposure_category"], expected)

    def test_device_selection_order_and_explicit_errors(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
            with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
                self.assertEqual(select_device().type, expected)
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                select_device("mps")

    def test_launcher_schema_syntax_no_duplicate_training_and_bundle(self):
        notebook = nbformat.read(ROOT / "notebooks/02_lofo_training.ipynb", as_version=4)
        nbformat.validate(notebook)
        shell = InteractiveShell.instance()
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(shell.input_transformer_manager.transform_cell(cell.source), cell.id, "exec")
                self.assertNotIn("def run_epoch", cell.source)
                self.assertNotIn("def run_experiment", cell.source)
        with zipfile.ZipFile(ROOT / "dist/lofo_colab.zip") as archive:
            self.assertEqual(archive.read("splits/cifar100_seed42.json"), (ROOT / "splits/cifar100_seed42.json").read_bytes())
            for name in archive.namelist():
                self.assertFalse(name.startswith(("data/", "results/", ".venv/")))
                if name.startswith("src/"):
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes())

    def test_lofo_resume_replays_parameters_metrics_and_consumed_exposures(self):
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as temporary:
                continuous = condition(artifact_root=str(Path(temporary) / "continuous"),
                                       num_workers=workers, device="cpu", cpu_threads=2)
                interrupted = replace(continuous, artifact_root=str(Path(temporary) / "resumed"))
                with patch("src.train.prepare_data", side_effect=lofo_fixture):
                    left_dir, _ = run_experiment(continuous)
                    right_dir, _ = run_experiment(interrupted, stop_after_epoch=1)
                    (right_dir / "corruption_counts.json").write_text("damaged export")
                    # Copying another experiment/seed's checkpoint into this run must not work.
                    for wrong in (condition("blur", artifact_root=interrupted.artifact_root, device="cpu", cpu_threads=2, num_workers=workers),
                                  replace(interrupted, seed=123, run_id="seed123")):
                        wrong_path = run_directory(wrong) / "checkpoints"
                        wrong_path.mkdir(parents=True)
                        import shutil
                        shutil.copy(right_dir / "checkpoints/last_checkpoint.pt", wrong_path / "last_checkpoint.pt")
                        with self.assertRaises(ValueError):
                            run_experiment(wrong, resume=True)
                    run_experiment(interrupted, resume=True)
                left = load_checkpoint(left_dir / "checkpoints/last_checkpoint.pt")
                right = load_checkpoint(right_dir / "checkpoints/last_checkpoint.pt")
                for name, tensor in left["model_state_dict"].items():
                    self.assertTrue(torch.equal(tensor, right["model_state_dict"][name]), name)
                self.assertEqual(left["corruption_history"], right["corruption_history"])
                self.assertEqual(left["scheduler_state_dict"], right["scheduler_state_dict"])
                for a, b in zip(left["history"], right["history"]):
                    for key in ("train_loss", "val_loss", "train_accuracy_pct", "val_accuracy_pct", "learning_rate"):
                        self.assertEqual(a[key], b[key])
                audit = json.loads((right_dir / "corruption_counts.json").read_text())
                self.assertEqual(audit["total_images"], 48)
                self.assertEqual(sum(audit["totals"].values()), 48)
                for row in audit["epochs"]:
                    self.assertEqual(sum(row["counts"].values()), 24)
                self.assertTrue((right_dir / "training.log").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
