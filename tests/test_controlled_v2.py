"""Controlled bridge contracts. No forward/backward passes or fitted parameter updates."""
import copy
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.config import LOFOConfig
from src.controlled_v2.config import (ControlledConfig, EVALUATION_MODELS, TRAINING_CONDITIONS,
                                      VERSION, configs)
from src.controlled_v2.data import ControlledTransform, prepare_data
from src.controlled_v2.evaluate import expected_keys, validate_complete, aggregate
from src.controlled_v2.provenance import HISTORICAL_INITIAL, check_stack, load_initial
from src.controlled_v2.train import (_train, run_manifest, selected_row, validate_checkpoint,
                                    verify_best)
from src.dataset import split_fingerprint
from src.lofo import FAMILIES, REPRESENTATIVES, ORIGINAL_SPLIT_SHA256, allowed_for, exposure_bins, validate_exposures
from src.model import build_model
from src.utils import capture_rng, load_checkpoint, model_digest, save_checkpoint, seed_everything


class ControlledTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)

    def test_registry_freezes_both_conditions_and_legacy_remains_strict(self):
        from src.benchmark.checkpoints import MODELS
        self.assertEqual(len(MODELS), 5)
        self.assertEqual(len(EVALUATION_MODELS), 2)
        for config in configs().values():
            config.validate()
            self.assertEqual(json.loads(Path(f"configs/controlled_v2/{config.condition}.json").read_text()),
                             config.to_dict())
            self.assertTrue(str(config.output).startswith("results/controlled_v2/"))
            for overrides in ({"seed": 123}, {"smoke_test": True}, {"device": "cpu"},
                              {"batch_size": 64}, {"full_epochs": 99}, {"use_amp": True},
                              {"num_workers": 0}, {"mean": (0, 0, 0)}, {"download": True},
                              {"protocol_version": "v3"}, {"condition": "lofo_noise"}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    replace(config, **overrides).validate()
        for family in FAMILIES:
            config = LOFOConfig(experiment_name="lofo_"+family, held_out_family=family,
                                allowed_corruptions=allowed_for(family))
            config.validate()
            with self.assertRaises(ValueError):
                replace(config, allowed_corruptions=tuple(TRAINING_CONDITIONS["all_four_representatives"][1])).validate()
            forbidden = exposure_bins()
            forbidden[REPRESENTATIVES[family]+":severity1"] = 1
            with self.assertRaises(ValueError):
                validate_exposures(config, forbidden, 1)

    def test_sampling_probabilities_and_clean_exclusion(self):
        config = configs()["all_four_representatives"]
        transform = ControlledTransform(config)
        counts = exposure_bins()
        for _ in range(40000):
            name, severity, _ = transform.sample()
            counts["clean" if name == "clean" else f"{name}:severity{severity}"] += 1
        validate_exposures(config, counts, 40000)
        self.assertLess(abs(counts["clean"]/40000 - .5), .015)
        for name in config.allowed_corruptions:
            for severity in range(1, 6):
                self.assertLess(abs(counts[f"{name}:severity{severity}"]/40000 - 1/40), .004)
        clean = configs()["clean_baseline"]
        sampler = ControlledTransform(clean)
        self.assertTrue(all(sampler.sample()[:2] == ("clean", 0) for _ in range(100)))
        with self.assertRaises(ValueError):
            validate_exposures(clean, counts, 40000)

    def test_spatial_streams_pair_and_operators_cannot_perturb_sampling(self):
        image = Image.new("RGB", (32, 32), "red")
        states, traces = [], []
        for name, config in configs().items():
            seed_everything(42)
            transform = ControlledTransform(config)
            trace = []
            for _ in range(40):
                tensor, code, severity = transform(image)
                self.assertEqual(tensor.shape, (3, 32, 32))
                trace.append((code, severity))
            states.append(torch.get_rng_state())
            traces.append(trace)
        self.assertTrue(torch.equal(*states))
        self.assertTrue(any(code for code, _ in traces[0]))
        self.assertTrue(all(code == severity == 0 for code, severity in traces[1]))
        transform = ControlledTransform(configs()["all_four_representatives"])
        # A corruption receives only its independent RNG, after spatial augmentation.
        with patch("src.controlled_v2.data.apply_corruption", return_value=image) as operator:
            for _ in range(20):
                transform(image)
            self.assertTrue(operator.called)
            self.assertIsInstance(operator.call_args.args[3], np.random.RandomState)

    def test_exact_split_clean_validation_and_no_test_or_split_writes(self):
        original = Path("splits/cifar100_seed42.json").read_bytes()
        for config in configs().values():
            with patch("src.controlled_v2.data.CIFAR100", wraps=__import__(
                    "torchvision").datasets.CIFAR100) as dataset:
                train, val, generators, manifest, split = prepare_data(config, torch.device("cpu"))
                self.assertEqual(dataset.call_args.kwargs, {"train": True, "download": False})
            self.assertEqual((len(train.dataset), len(val.dataset)), (45000, 5000))
            self.assertEqual((len(train), len(val)), (352, 40))
            self.assertEqual(split_fingerprint(split), ORIGINAL_SPLIT_SHA256)
            self.assertEqual(manifest["train_indices"], split["train_indices"])
            self.assertEqual(manifest["val_indices"], split["val_indices"])
            with patch("src.controlled_v2.data.apply_corruption", side_effect=AssertionError("validation corruption")):
                self.assertTrue(torch.equal(val.dataset[0][0], val.dataset[0][0]))
        self.assertEqual(original, Path("splits/cifar100_seed42.json").read_bytes())
        with tempfile.TemporaryDirectory() as tmp, patch("src.controlled_v2.data.CIFAR100") as dataset:
            split = json.loads(original)
            split["train_indices"][0] = split["val_indices"][0]
            path = Path(tmp)/"bad.json"
            path.write_text(json.dumps(split))
            with self.assertRaises(ValueError):
                prepare_data(replace(ControlledConfig(), split_file=str(path)), torch.device("cpu"))
            dataset.assert_not_called()

    def test_initial_digest_and_roundtrip_no_forward(self):
        model = build_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11220132)
        self.assertEqual(model_digest(model), HISTORICAL_INITIAL)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"initial.pt"
            save_checkpoint(path, {"model_state_dict": model.state_dict()})
            other = build_model()
            other.load_state_dict(load_checkpoint(path)["model_state_dict"], strict=True)
            self.assertEqual(model_digest(other), HISTORICAL_INITIAL)

    def test_initial_tamper_and_stack_drift_fail(self):
        from src.controlled_v2.provenance import historical_stack
        stack = historical_stack()
        with patch("src.controlled_v2.provenance.current_stack", return_value={**stack, "mps_available": False}):
            check_stack(require_mps=False)
            with self.assertRaises(RuntimeError):
                check_stack(require_mps=True)
        with patch("src.controlled_v2.provenance.current_stack", return_value={**stack, "torch": "wrong"}):
            with self.assertRaises(ValueError):
                check_stack(require_mps=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"initial.pt"
            path.write_bytes(b"tampered")
            with patch("src.controlled_v2.provenance.load_freeze", return_value={"initial_file_sha256": "wrong"}):
                with self.assertRaises(ValueError):
                    load_initial(replace(ControlledConfig(), initial_state=str(path)))

    def test_manifests_ties_checkpoint_selection_and_evaluation_grid(self):
        self.assertEqual(selected_row([{"epoch": 1, "val_accuracy_pct": 70},
                                      {"epoch": 2, "val_accuracy_pct": 70}])["epoch"], 1)
        for history in ([], [{"epoch": 2, "val_accuracy_pct": 70}],
                        [{"epoch": 1, "val_accuracy_pct": float("nan")}]):
            with self.assertRaises(ValueError):
                selected_row(history)
        cfg = ControlledConfig()
        freeze = {"configs": {"clean_baseline": cfg.to_dict()}, "freeze_sha256": "test",
                  "initial_file_sha256": "file", "initial_model_sha256": "model"}
        manifest = run_manifest(cfg, freeze, {"train_count": 45000})
        self.assertFalse(manifest["test_set_used_for_selection"])
        with self.assertRaises(ValueError):
            run_manifest(replace(cfg, data_root="other"), freeze, {})
        self.assertEqual(len(expected_keys()), 152)
        from src.benchmark.metrics import condition_metrics
        targets = np.zeros(10000, dtype=np.int64)
        rows = []
        for m, c, s in expected_keys():
            predictions = np.ones(10000, dtype=np.int64)
            predictions[:5000 if m == "clean_baseline" else 5100] = 0
            rows.append(condition_metrics(m, c, s, predictions, targets, 0., 1.))
        validate_complete(rows)
        aggregates = aggregate(rows)
        self.assertEqual(len(aggregates), 12)
        for row in aggregates:
            self.assertTrue(math.isfinite(row["matched_clean_difference_pp"]))
            self.assertEqual(row["matched_clean_difference_pp"],
                             0.0 if row["model"] == "clean_baseline" else 1.0)
        with self.assertRaises(ValueError):
            validate_complete(rows[:-1])
        with self.assertRaises(ValueError):
            validate_complete(rows+[rows[0]])

    def test_resume_and_best_recovery_with_mock_epochs_no_fitting(self):
        # Exercise real serialization, scheduler, RNG restore and optimizer hooks, with no
        # model forward/backward, no gradients and no parameter changes.
        class NoForward(nn.Linear):
            def to(self, *args, **kwargs):
                return self
            def forward(self, *args):
                raise AssertionError("Tests must never run model inference/training")

        class Loader(list):
            def __init__(self, generator, count):
                super().__init__(range(count))
                self.generator = generator

        def fake_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
            value = random.random() + float(np.random.random()) + torch.rand(()).item()
            value += torch.rand((), generator=loader.generator).item()
            if optimizer is not None:
                for _ in range(352):
                    optimizer.step()  # No gradients exist: parameters remain unchanged.
                counts = exposure_bins()
                counts["clean"] = 45000
                return {"count": 45000, "loss": value, "accuracy_pct": 50., "corruption_counts": counts}
            return {"count": 5000, "loss": value, "accuracy_pct": 50.}

        for condition in TRAINING_CONDITIONS:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                saved = []
                for name, pauses in (("continuous", (3,)), ("resumed", (1, 3))):
                    root = Path(tmp)/name
                    cfg = ControlledConfig(condition=condition, artifact_root=str(root),
                                           initial_state=str(root/"results/controlled_v2/shared/initial_seed42.pt"))
                    cfg.output.mkdir(parents=True)
                    seed_everything(42)
                    model = NoForward(1, 1)
                    digest = model_digest(model)
                    before = copy.deepcopy(model.state_dict())
                    freeze = {"configs": {condition: cfg.to_dict()}, "freeze_sha256": "fixture",
                              "initial_model_sha256": digest, "initial_file_sha256": "fixture"}
                    data = {"train_count": 45000, "val_count": 5000}
                    for i, pause in enumerate(pauses):
                        seed_everything(42)
                        generators = {"train": torch.Generator().manual_seed(42), "val": torch.Generator().manual_seed(43)}
                        with patch("src.controlled_v2.train.prepare_data", return_value=(Loader(generators["train"], 352), Loader(generators["val"], 40), generators, data, {})), patch(
                                "src.controlled_v2.train.load_initial", return_value=model), patch(
                                "src.controlled_v2.train.run_epoch", side_effect=fake_epoch), patch(
                                "src.controlled_v2.train.synchronize"):
                            _train(cfg, freeze, {"fixture": True}, resume=i > 0, stop_after_epoch=pause)
                        if name == "resumed" and i == 0:
                            (cfg.output/"checkpoints/best_model.pt").unlink()  # recover last→best crash
                    last = load_checkpoint(cfg.output/"checkpoints/last_checkpoint.pt")
                    best = load_checkpoint(cfg.output/"checkpoints/best_model.pt")
                    manifest = run_manifest(cfg, freeze, data)
                    verify_best(best, last, manifest, complete=False)
                    self.assertEqual(best["epoch"], 1)  # all validation ties
                    for key in before:
                        self.assertTrue(torch.equal(before[key], last["model_state_dict"][key]))
                    for mutation in ("identity", "selection", "scheduler", "optimizer", "exposures"):
                        bad = copy.deepcopy(last)
                        if mutation == "identity":
                            bad["run_manifest"]["freeze_sha256"] = "wrong"
                        elif mutation == "selection":
                            bad["best_epoch"] = 3
                        elif mutation == "scheduler":
                            bad["scheduler_state_dict"]["last_epoch"] = 0
                        elif mutation == "optimizer":
                            bad["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.9
                        else:
                            bad["corruption_history"][0]["counts"]["clean"] = 0
                        with self.assertRaises(ValueError):
                            validate_checkpoint(bad, manifest)
                    with self.assertRaises(ValueError):
                        verify_best(best, last, manifest, complete=True)
                    complete = copy.deepcopy(last)
                    complete["history"] = [{**last["history"][0], "epoch": e,
                          "learning_rate": .1*(1+math.cos(math.pi*(e-1)/100))/2} for e in range(1, 101)]
                    complete["corruption_history"] = [{"epoch": e, "counts": copy.deepcopy(
                          last["corruption_history"][0]["counts"])} for e in range(1, 101)]
                    complete["epoch"] = 100
                    complete["scheduler_state_dict"]["last_epoch"] = 100
                    complete["scheduler_state_dict"]["_last_lr"] = [0.0]
                    complete["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.0
                    verify_best(best, complete, manifest, complete=True)
                    saved.append(last)
                left, right = saved
                for a, b in zip(left["history"], right["history"]):
                    for key in ("learning_rate", "train_loss", "val_loss", "val_accuracy_pct"):
                        self.assertEqual(a[key], b[key])
                self.assertTrue(torch.equal(left["rng_state"]["torch"], right["rng_state"]["torch"]))
                self.assertEqual(left["rng_state"]["python"], right["rng_state"]["python"])
                self.assertEqual(left["rng_state"]["numpy"], right["rng_state"]["numpy"])
                for name in ("train", "val"):
                    self.assertTrue(torch.equal(left["loader_rng_state"][name], right["loader_rng_state"][name]))
                self.assertEqual(left["scheduler_state_dict"], right["scheduler_state_dict"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
