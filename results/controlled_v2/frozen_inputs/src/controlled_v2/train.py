"""Explicitly invoked v2 training and checkpoint contracts. Never called by preflight."""
import fcntl
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from ..lofo import validate_exposures
from ..benchmark.data import file_hash
from ..train import run_epoch, write_corruption_audit
from ..utils import (capture_rng, load_checkpoint, model_digest, restore_rng, save_checkpoint,
                     save_json, seed_everything, synchronize, utc_now, write_metrics)
from .config import VERSION
from .data import prepare_data
from .provenance import check_stack, load_freeze, load_initial


@contextmanager
def exclusive_run(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {output}.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_manifest(config, freeze, data):
    config.validate()
    if config.to_dict() != freeze["configs"][config.condition]:
        raise ValueError("Configuration differs from the frozen run grid.")
    return {"protocol_version": VERSION, "condition": config.condition, "seed": 42,
            "freeze_sha256": freeze["freeze_sha256"], "config": config.to_dict(),
            "initial_model_sha256": freeze["initial_model_sha256"],
            "initial_file_sha256": freeze["initial_file_sha256"],
            "data_manifest": data, "device": "mps", "precision": "FP32", "effective_amp": False,
            "allowed_corruptions": list(config.allowed_corruptions),
            "corruption_probability": config.corruption_probability,
            "selection": config.selection, "test_set_used_for_selection": False,
            "scheduler_T_max": 100, "attempted_updates_per_epoch": 352}


def selected_row(history):
    if not history or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("History must contain consecutive completed epochs.")
    if any(not math.isfinite(r["val_accuracy_pct"]) or not 0 <= r["val_accuracy_pct"] <= 100 for r in history):
        raise ValueError("Nonfinite validation history.")
    return max(history, key=lambda r: r["val_accuracy_pct"])  # First maximum wins ties.


def validate_checkpoint(checkpoint, manifest, complete=False):
    if checkpoint.get("format_version") != VERSION or checkpoint.get("run_manifest") != manifest:
        raise ValueError("Checkpoint identity/configuration/provenance mismatch.")
    history = checkpoint["history"]
    selected = selected_row(history)
    epoch = checkpoint["epoch"]
    if (epoch != len(history) or not 1 <= epoch <= 100
            or (complete and epoch != 100)
            or checkpoint["best_epoch"] != selected["epoch"]
            or checkpoint["best_validation_accuracy_pct"] != selected["val_accuracy_pct"]):
        raise ValueError("Checkpoint completion/clean-validation selection mismatch.")
    audits = checkpoint["corruption_history"]
    if [r["epoch"] for r in audits] != list(range(1, epoch + 1)):
        raise ValueError("Incomplete consumed-image audit.")
    from .config import ControlledConfig
    config = ControlledConfig(**manifest["config"])
    for row in audits:
        validate_exposures(config, row["counts"], 45000)
    for row in history:
        e = row["epoch"]
        expected_lr = 0.1 * (1 + math.cos(math.pi * (e-1) / 100)) / 2
        if (row["attempted_optimizer_updates"] != 352 or row["successful_optimizer_updates"] != 352
                or not math.isclose(row["learning_rate"], expected_lr, abs_tol=1e-12)):
            raise ValueError("Optimizer update count or schedule mismatch.")
    if checkpoint["scheduler_state_dict"]["last_epoch"] != epoch:
        raise ValueError("Scheduler is not positioned at the next epoch.")
    scheduler = checkpoint["scheduler_state_dict"]
    next_lr = 0.1 * (1 + math.cos(math.pi * epoch / 100)) / 2
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    if (scheduler["T_max"] != 100 or scheduler["eta_min"] != 0
            or scheduler["base_lrs"] != [0.1] or len(groups) != 1
            or not math.isclose(groups[0]["lr"], next_lr, abs_tol=1e-12)
            or not math.isclose(scheduler["_last_lr"][0], next_lr, abs_tol=1e-12)):
        raise ValueError("Resume optimizer/cosine schedule mismatch.")
    for key, expected in {"momentum": 0.9, "weight_decay": 5e-4, "dampening": 0,
                          "nesterov": False, "maximize": False}.items():
        if groups[0][key] != expected:
            raise ValueError(f"Resume optimizer {key} mismatch.")
    for tensor in checkpoint["model_state_dict"].values():
        if tensor.is_floating_point() and (tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
            raise ValueError("Model tensors must be finite FP32.")


def verify_best(best, last, manifest, complete=True):
    validate_checkpoint(last, manifest, complete=complete)
    validate_checkpoint(best, manifest)
    selected = selected_row(last["history"])
    if (best["epoch"] != selected["epoch"] or best["best_epoch"] != best["epoch"]
            or best["history"] != last["history"][:best["epoch"]]
            or best["corruption_history"] != last["corruption_history"][:best["epoch"]]):
        raise ValueError("Best checkpoint is not the first maximum of clean validation accuracy.")


def train(config, resume=False, stop_after_epoch=None):
    config.validate()
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= 100:
        raise ValueError("Pause epoch must be within 1–100; the cosine schedule stays 100.")
    seed_everything(42)
    freeze = load_freeze()
    environment = check_stack(require_mps=True)
    with exclusive_run(config.output):
        return _train(config, freeze, environment, resume, stop_after_epoch)


def _train(config, freeze, environment, resume, stop_after_epoch):
    output = config.output
    last_path = output / "checkpoints/last_checkpoint.pt"
    best_path = output / "checkpoints/best_model.pt"
    if not resume and any(p.name != ".run.lock" for p in output.iterdir()):
        raise FileExistsError("Existing run artifacts: use --resume; never overwrite a run.")
    if resume and not last_path.is_file():
        raise FileNotFoundError("No completed epoch checkpoint to resume.")
    device = torch.device("mps")
    train_loader, val_loader, generators, data, split = prepare_data(config, device)
    manifest = run_manifest(config, freeze, data)
    model = load_initial(config).to(device)
    if model_digest(model) != manifest["initial_model_sha256"]:
        raise ValueError("Initial state changed on transfer to MPS.")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history, audits, start = [], [], 1
    best_accuracy, best_epoch = -math.inf, 0
    if resume:
        last = load_checkpoint(last_path)
        validate_checkpoint(last, manifest)
        if last["environment"] != environment:
            raise ValueError("Resume requires the same runtime environment.")
        if json.loads((output / "run_manifest.json").read_text()) != manifest:
            raise ValueError("Run manifest changed.")
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        if not best_path.exists() or load_checkpoint(best_path)["epoch"] != last["best_epoch"]:
            if last["epoch"] != last["best_epoch"]:
                raise ValueError("Earlier best checkpoint missing; restore its backup.")
            save_checkpoint(best_path, last)  # Recover interrupted last→best publication.
        verify_best(load_checkpoint(best_path), last, manifest, complete=False)
        history, audits, start = last["history"], last["corruption_history"], last["epoch"] + 1
        best_accuracy, best_epoch = last["best_validation_accuracy_pct"], last["best_epoch"]
        restore_rng(last["rng_state"])
        for name, generator in generators.items():
            generator.set_state(last["loader_rng_state"][name])
    else:
        save_json(output / "run_manifest.json", manifest)
        save_json(output / "split.json", split)
        save_json(output / "environment.json", environment)
        save_json(output / "config.json", config.to_dict())
    session = {"started_utc": utc_now(), "resume": resume, "start_epoch": start,
               "pause_after_epoch": stop_after_epoch, "environment": environment}
    with (output / "sessions.jsonl").open("a") as stream:
        stream.write(json.dumps(session) + "\n")
    updates = [0]
    hook = optimizer.register_step_post_hook(lambda *args, **kwargs: updates.__setitem__(0, updates[0]+1))
    try:
        for epoch in range(start, (stop_after_epoch or 100) + 1):
            started = time.perf_counter()
            updates[0] = 0
            lr = optimizer.param_groups[0]["lr"]
            fitted = run_epoch(model, train_loader, nn.CrossEntropyLoss(), device, optimizer, scaler, False)
            validation = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device, amp=False)
            synchronize(device)
            counts = fitted["corruption_counts"]
            validate_exposures(config, counts, 45000)
            if fitted["count"] != 45000 or validation["count"] != 5000 or updates[0] != 352:
                raise ValueError("Incomplete epoch/image/update accounting.")
            improved = validation["accuracy_pct"] > best_accuracy
            if improved:
                best_accuracy, best_epoch = validation["accuracy_pct"], epoch
            scheduler.step()
            history.append({"epoch": epoch, "learning_rate": lr,
                            "train_loss": fitted["loss"], "train_accuracy_pct": fitted["accuracy_pct"],
                            "val_loss": validation["loss"], "val_accuracy_pct": validation["accuracy_pct"],
                            "attempted_optimizer_updates": len(train_loader),
                            "successful_optimizer_updates": updates[0], "seconds": time.perf_counter()-started})
            audits.append({"epoch": epoch, "counts": counts})
            checkpoint = {"format_version": VERSION, "run_manifest": manifest,
                          "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
                          "history": history, "corruption_history": audits,
                          "best_epoch": best_epoch, "best_validation_accuracy_pct": best_accuracy,
                          "environment": environment, "rng_state": capture_rng(),
                          "loader_rng_state": {n: g.get_state() for n, g in generators.items()}}
            validate_checkpoint(checkpoint, manifest)
            save_checkpoint(last_path, checkpoint)
            if improved:
                save_checkpoint(best_path, checkpoint)
            write_metrics(output, history)
            write_corruption_audit(output, audits)
            message = f"{config.condition} | epoch {epoch}/100 | val {validation['accuracy_pct']:.2f}%"
            print(message, flush=True)
            with (output / "training.log").open("a") as stream:
                stream.write(message + "\n")
    finally:
        hook.remove()
    # Recover exports when resuming a completed run; no optimizer work occurs.
    write_metrics(output, history)
    write_corruption_audit(output, audits)
    save_json(output / "summary.json", {"protocol_version": VERSION, "completed_epochs": len(history),
              "completed": len(history) == 100, "best_epoch": best_epoch,
              "best_validation_accuracy_pct": best_accuracy, "test_evaluated": False,
              "initial_model_sha256": manifest["initial_model_sha256"],
              "initial_file_sha256": manifest["initial_file_sha256"],
              "best_file_sha256": file_hash(best_path), "last_file_sha256": file_hash(last_path)})
    return output
