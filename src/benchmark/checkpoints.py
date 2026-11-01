"""Explicit adapter for audited Part 1/2 checkpoints; no partial loading."""
import hashlib
import json
from pathlib import Path

import torch

from ..config import Config, LOFOConfig
from ..dataset import split_fingerprint
from ..lofo import FAMILIES, ORIGINAL_SPLIT_SHA256, allowed_for, protocol_metadata, validate_exposures
from ..model import build_model
from ..utils import load_checkpoint
from .data import file_hash

MODELS = ("clean_baseline", *(f"lofo_{f}" for f in FAMILIES))


def default_runs(root):
    root = Path(root)
    return {m: root / m / ("clean_seed42" if m == "clean_baseline" else "seed42") for m in MODELS}


def snapshot_digest(folder):
    h = hashlib.sha256()
    for path in sorted(folder.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def adapt_checkpoint(best, last, model_id, classes, split):
    """Both inspected schemas use model_state_dict; reject undocumented alternatives."""
    try:
        if best.get("format_version") != 1 or last.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint schema format_version; expected 1.")
        cfg = best["config"]
        kind = Config if model_id == "clean_baseline" else LOFOConfig
        config = kind(**cfg)
        config.validate()
        ref = Config()
        shared = ("dataset", "architecture", "num_classes", "seed", "split_seed", "full_epochs",
                  "batch_size", "learning_rate", "momentum", "weight_decay", "eta_min", "mean", "std",
                  "selection", "train_per_class", "val_per_class", "optimizer", "scheduler", "loss")
        for key in shared:
            value, expected = cfg[key], getattr(ref, key)
            if key in ("mean", "std"):
                value, expected = tuple(value), tuple(expected)
            if value != expected:
                raise ValueError(f"Incompatible {model_id} {key}: {value!r}; expected {expected!r}.")
        if cfg["smoke_test"] or best["seed"] != 42 or last["seed"] != 42:
            raise ValueError("Only the completed, non-smoke seed-42 models belong to this protocol.")
        if last["config"] != cfg or last["epoch"] != 100 or len(last["history"]) != 100:
            raise ValueError("Training completion/configuration does not match the selected checkpoint.")
        history = last["history"]
        selected = max(history, key=lambda row: row["val_accuracy_pct"])
        if (best["epoch"] != selected["epoch"] or best["epoch"] != last["best_epoch"]
                or best["best_epoch"] != best["epoch"]
                or best["best_validation_accuracy_pct"] != selected["val_accuracy_pct"]
                or best["best_validation_accuracy_pct"] != last["best_validation_accuracy_pct"]):
            raise ValueError("Checkpoint is not the first highest-clean-validation epoch.")
        for checkpoint in (best, last):
            manifest = checkpoint["data_manifest"]
            if (manifest["classes"] != list(classes)
                    or manifest["split_sha256"] != ORIGINAL_SPLIT_SHA256
                    or manifest["train_count"] != 45000 or manifest["val_count"] != 5000
                    or manifest["train_indices"] != split["train_indices"]
                    or manifest["val_indices"] != split["val_indices"]):
                raise ValueError("Checkpoint class ordering or train/validation split is incompatible.")
        if best["source_sha256"] != last["source_sha256"]:
            raise ValueError("Best and last checkpoints identify different training sources.")
        if model_id != "clean_baseline":
            family = model_id.removeprefix("lofo_")
            if cfg["held_out_family"] != family or tuple(cfg["allowed_corruptions"]) != allowed_for(family):
                raise ValueError("LOFO model identity/exclusion mismatch.")
            if best["lofo_protocol"] != protocol_metadata(config):
                # JSON checkpoints encode tuples as lists inside config, but metadata families are tuples.
                if json.dumps(best["lofo_protocol"], sort_keys=True) != json.dumps(protocol_metadata(config), sort_keys=True):
                    raise ValueError("Checkpoint LOFO exposure protocol mismatch.")
            if len(last["corruption_history"]) != 100:
                raise ValueError("Missing completed LOFO training exposure audit.")
            for epoch, audit in enumerate(last["corruption_history"], 1):
                if audit["epoch"] != epoch:
                    raise ValueError("LOFO exposure audit epoch mismatch.")
                validate_exposures(config, audit["counts"], 45000)
        state = best["model_state_dict"]
        if any(t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all())
               for t in state.values()):
            raise ValueError("Weights must be finite FP32 tensors.")
        model = build_model()
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 3, 32, 32))
        if logits.shape != (2, 100) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite [N,100] logits from the CIFAR ResNet18.")
        return model
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(f"Incompatible {model_id} checkpoint schema/architecture/metadata: {exc}") from exc


def load_run(run, model_id, classes):
    run = Path(run).resolve()
    best_path, last_path = (run / "checkpoints" / n for n in ("best_model.pt", "last_checkpoint.pt"))
    best, last = load_checkpoint(best_path), load_checkpoint(last_path)
    split = json.loads((run / "split.json").read_text())
    if split_fingerprint(split) != ORIGINAL_SPLIT_SHA256:
        raise ValueError(f"{run}: saved split differs from the original Part 1 split.")
    if (snapshot_digest(run / "source") != best["source_sha256"]
            or (run / "source/model.py").read_bytes() != Path(__file__).parents[1].joinpath("model.py").read_bytes()):
        raise ValueError(f"{run}: training source snapshot or architecture does not match checkpoint.")
    protocol = json.loads((run / "protocol.json").read_text())
    if protocol["eval_transforms"] != ["ToTensor", "Normalize(config.mean,config.std)"]:
        raise ValueError(f"{run}: unsupported evaluation preprocessing.")
    if protocol["effective_amp"] != best["effective_amp"]:
        raise ValueError("Training precision provenance mismatch.")
    model = adapt_checkpoint(best, last, model_id, classes, split)
    info = {"path": str(best_path), "sha256": file_hash(best_path),
            "last_checkpoint_sha256": file_hash(last_path), "state_key": "model_state_dict",
            "selected_epoch": best["epoch"], "best_clean_validation_accuracy_pct": best["best_validation_accuracy_pct"],
            "seed": best["seed"], "training_environment": best["environment"],
            "training_effective_amp": best["effective_amp"], "training_source_sha256": best["source_sha256"],
            "split_sha256": ORIGINAL_SPLIT_SHA256,
            "protocol": protocol, "config": best["config"]}
    return model, info
