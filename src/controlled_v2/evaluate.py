"""Two-model v2 evaluation registry, caches and completion rules; no legacy gate reuse."""
import json
from pathlib import Path

import numpy as np
import torch

from ..benchmark.data import (CORRUPTIONS, file_hash, fingerprint, inspect_benchmark,
                              load_clean, severity_slice)
from ..benchmark.runner import infer, save_condition, load_condition, csv_write
from ..benchmark.metrics import condition_metrics
from ..model import build_model
from ..utils import load_checkpoint, save_json, seed_everything
from .config import EVALUATION_MODELS, EVALUATION_ROOT, ROOT, VERSION, configs
from .provenance import check_stack, load_freeze
from .train import exclusive_run, run_manifest, verify_best


def expected_keys():
    return {(m, c, s) for m in EVALUATION_MODELS for c in CORRUPTIONS for s in range(1, 6)} | {
        (m, "clean", 0) for m in EVALUATION_MODELS}


def load_run(config, freeze, classes):
    run = config.output
    best_path, last_path = (run / "checkpoints" / name for name in ("best_model.pt", "last_checkpoint.pt"))
    best, last = load_checkpoint(best_path), load_checkpoint(last_path)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest != run_manifest(config, freeze, manifest["data_manifest"]):
        raise ValueError("Evaluation run manifest differs from frozen protocol.")
    from ..dataset import split_fingerprint, validate_split
    from torchvision.datasets import CIFAR100
    split = json.loads((run / "split.json").read_text())
    raw = CIFAR100(config.data_root, train=True, download=False)
    validate_split(split, raw.targets, 42)
    data = manifest["data_manifest"]
    if (split_fingerprint(split) != config.split_sha256 or data["classes"] != list(classes)
            or data["train_indices"] != split["train_indices"] or data["val_indices"] != split["val_indices"]
            or data["train_count"] != 45000 or data["val_count"] != 5000):
        raise ValueError("Evaluation split/classes mismatch.")
    verify_best(best, last, manifest, complete=True)
    if best["environment"] != last["environment"]:
        raise ValueError("Best/last training environments differ.")
    if {k: best["environment"][k] for k in freeze["historical_stack"]} != freeze["historical_stack"]:
        raise ValueError("Checkpoint training stack mismatch.")
    model = build_model()
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    return model, {"best_sha256": file_hash(best_path), "last_sha256": file_hash(last_path),
                   "selected_epoch": best["epoch"], "run_manifest": manifest}


def validate_complete(rows):
    keys = [(r["model"], r["corruption"], r["severity"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys():
        raise ValueError("V2 requires exactly 150 corrupted conditions and two clean checks.")
    if any(r["n_samples"] != 10000 for r in rows):
        raise ValueError("Complete evaluation requires all 10,000 test identities.")
    from ..benchmark.data import FAMILY_OF
    for row in rows:
        if (type(row["n_correct"]) is not int or not 0 <= row["n_correct"] <= 10000
                or row["seed"] != 42
                or row["family"] != ("clean" if row["corruption"] == "clean" else FAMILY_OF[row["corruption"]])
                or not np.isclose(row["accuracy_pct"], row["n_correct"]/100, atol=1e-10, rtol=0)
                or not np.isclose(row["error_pct"], 100-row["accuracy_pct"], atol=1e-10, rtol=0)):
            raise ValueError("Invalid complete-evaluation metrics.")


def aggregate(rows):
    validate_complete(rows)
    from ..lofo import FAMILIES
    output = []
    for model in EVALUATION_MODELS:
        for scope, names in [("overall_15_type", CORRUPTIONS), *FAMILIES.items()]:
            selected = [r["accuracy_pct"] for r in rows if r["model"] == model and r["corruption"] in names]
            base = [r["accuracy_pct"] for r in rows if r["model"] == "clean_baseline" and r["corruption"] in names]
            output.append({"model": model, "scope": scope, "n_conditions": len(selected),
                           "accuracy_pct": float(np.mean(selected)),
                           "error_pct": 100-float(np.mean(selected)),
                           "matched_clean_difference_pp": float(np.mean(selected)-np.mean(base))})
        families = [r["accuracy_pct"] for r in output if r["model"] == model and r["scope"] in FAMILIES]
        baseline_families = [float(np.mean([r["accuracy_pct"] for r in rows
                            if r["model"] == "clean_baseline" and r["corruption"] in names]))
                            for names in FAMILIES.values()]
        output.append({"model": model, "scope": "equal_four_family", "n_conditions": 75,
                       "accuracy_pct": float(np.mean(families)),
                       "error_pct": 100-float(np.mean(families)),
                       "matched_clean_difference_pp": float(np.mean(families)) - float(np.mean(baseline_families))})
    return output


def evaluation_preflight():
    freeze = load_freeze()
    report = {"protocol_version": VERSION, "freeze_sha256": freeze["freeze_sha256"],
              "models": list(EVALUATION_MODELS), "expected_corruption_conditions": 150,
              "expected_clean_conditions": 2, "forward_passes": 0, "test_metrics_used": False,
              "status": "waiting_for_authorized_training", "runs": {}}
    for name, config in configs().items():
        paths = [config.output / "checkpoints" / p for p in ("best_model.pt", "last_checkpoint.pt")]
        report["runs"][name] = {"best": str(paths[0]), "last": str(paths[1]),
                               "checkpoints_exist": all(p.is_file() for p in paths)}
    # Independent preparation artifact, never creates benchmark metrics/caches or loads test images.
    save_json(EVALUATION_ROOT / "preflight/registry.json", report)
    return report


def evaluate(resume=False):
    seed_everything(42)
    freeze = load_freeze()
    stack = check_stack(require_mps=True)
    device = torch.device("mps")
    # Structural gate: both completed selected checkpoints must validate before any inference.
    clean, clean_identity = load_clean("data")
    models, identities = {}, {}
    for name, config in configs().items():
        models[name], identities[name] = load_run(config, freeze, clean.classes)
    corruption_root, corruption_identity = inspect_benchmark("data", clean.targets)
    identity = {"protocol_version": VERSION, "freeze_sha256": freeze["freeze_sha256"],
                "checkpoints": identities, "clean_data": clean_identity, "corruptions": corruption_identity,
                "stack": stack, "batch_size": 128, "precision": "FP32", "mode": "full"}
    identity_digest = fingerprint(identity)
    output = EVALUATION_ROOT / "seed42"
    with exclusive_run(output):
        path = output / "manifest.json"
        if path.exists():
            if not resume or json.loads(path.read_text()) != identity:
                raise ValueError("Existing evaluation identity mismatch or --resume missing.")
        elif resume:
            raise FileNotFoundError("No evaluation manifest to resume.")
        elif any(p.name != ".run.lock" for p in output.iterdir()):
            raise FileExistsError("Unrecognized evaluation artifacts; refusing overwrite.")
        save_json(path, identity)
        save_json(output / "status.json", {"status": "in_progress", "scientific_result": False})
        rows = []
        targets = np.asarray(clean.targets, dtype=np.int64)
        for name, model in models.items():
            model.to(device)
            conditions = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in range(1, 6)]
            for corruption, severity in conditions:
                cache_path = output / f"cache/{name}/{corruption}__{severity}.npz"
                cache_identity = {"run": identity_digest, "model": name, "corruption": corruption,
                                  "severity": severity}
                if cache_path.exists():
                    row, _ = load_condition(cache_path, cache_identity, name, corruption, severity, targets)
                else:
                    images = clean.data if corruption == "clean" else np.load(
                        corruption_root / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)[severity_slice(severity)]
                    predictions, loss, seconds = infer(model, images, targets, device, 128)
                    row = condition_metrics(name, corruption, severity, predictions, targets, seconds, loss)
                    save_condition(cache_path, cache_identity, row, predictions)
                rows.append(row)
                csv_write(output / "metrics/per_condition.csv", rows)
            model.cpu()
        summaries = aggregate(rows)
        csv_write(output / "metrics/aggregates.csv", summaries)
        save_json(output / "metrics/evaluation.json", {"conditions": rows, "aggregates": summaries,
                  "protocol_version": VERSION, "single_seed_descriptive_bridge_only": True})
        save_json(output / "status.json", {"status": "complete", "scientific_result": True,
                  "completed_corruption_conditions": 150, "completed_clean_checks": 2,
                  "test_metrics_used_for_selection": False})
    return output
