"""Clean baseline and LOFO training; official test images are never loaded here."""

import argparse
import json
import math
import shutil
import time
import warnings
from pathlib import Path
from dataclasses import replace
from contextlib import nullcontext

import torch
from torch import nn

from .config import LOFOConfig, load_config
from .lofo import CORRUPTION_NAMES, exposure_bins, protocol_metadata, validate_exposures
from .dataset import prepare_data
from .model import build_model
from .utils import (capture_rng, environment_info, load_checkpoint, restore_rng,
                    save_checkpoint, save_json, seed_everything, select_device,
                    source_digest, utc_now, write_metrics, synchronize, model_digest)


def run_directory(config):
    if isinstance(config, LOFOConfig) and config.smoke_test:
        return Path(config.artifact_root) / "results" / "smoke_test" / config.experiment_name / config.run_id
    return Path(config.artifact_root) / "results" / config.mode / config.run_id


def compatible_config(saved, current):
    # Relocating Drive/cache directories does not change the experiment.
    ignored = {"data_root", "artifact_root", "download", "split_file"}
    return {k: v for k, v in saved.items() if k not in ignored} == {
        k: v for k, v in current.items() if k not in ignored}


def write_corruption_audit(output_dir, history):
    totals = exposure_bins()
    for row in history:
        for key, value in row["counts"].items():
            totals[key] += value
    save_json(Path(output_dir) / "corruption_counts.json", {
        "unit": "actually_consumed_training_images", "epochs": history, "totals": totals,
        "total_images": sum(totals.values()), "clean_images": totals["clean"],
        "corrupted_images": sum(totals.values()) - totals["clean"],
    })


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    # Metal does not support float64. CPU/CUDA retain Part 1's float64 accumulation.
    accumulator_dtype = torch.float32 if device.type == "mps" else torch.float64
    total_loss = torch.zeros((), device=device, dtype=accumulator_dtype)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    count = 0
    exposures = None
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            images, targets = batch[:2]
            if len(batch) == 4:
                if not training:
                    raise ValueError("Corruption metadata is forbidden in clean validation/test batches.")
                if exposures is None:
                    exposures = exposure_bins()
                codes, severities = batch[2:]
                for code, severity in zip(codes.tolist(), severities.tolist()):
                    if not 0 <= code < len(CORRUPTION_NAMES) or not (severity == 0 if code == 0 else 1 <= severity <= 5):
                        raise ValueError("Invalid corruption audit code/severity.")
                    name = CORRUPTION_NAMES[code]
                    exposures["clean" if code == 0 else f"{name}:severity{severity}"] += 1
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type) if amp else nullcontext():
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect configuration and inputs.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            size = targets.size(0)
            count += size
            total_loss += loss.detach().to(accumulator_dtype) * size
            total_correct += (logits.detach().argmax(1) == targets).sum()
    if not count:
        raise ValueError("An empty loader cannot be trained or evaluated.")
    result = {"loss": total_loss.item() / count,
              "accuracy_pct": 100.0 * total_correct.item() / count, "count": count}
    if exposures is not None:
        result["corruption_counts"] = exposures
    return result


def plot_history(history, output_dir, smoke_test, show=False, experiment_name="Clean CIFAR-100 baseline"):
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("No epoch metrics to plot.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    label = f"SMOKE TEST — {experiment_name}" if smoke_test else experiment_name
    specs = [
        ("accuracy", "Accuracy (%)", [("train_accuracy_pct", "Training", "#0072B2"),
                                       ("val_accuracy_pct", "Validation", "#D55E00")]),
        ("loss", "Cross-entropy loss", [("train_loss", "Training", "#0072B2"),
                                        ("val_loss", "Validation", "#D55E00")]),
        ("learning_rate", "Learning rate", [("learning_rate", "SGD", "#0072B2")]),
    ]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300}):
        for filename, ylabel, series in specs:
            fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
            for key, legend, color in series:
                ax.plot(epochs, [row[key] for row in history], label=legend,
                        color=color, linewidth=1.8, marker="o" if len(epochs) < 10 else None)
            ax.set(xlabel="Epoch", ylabel=ylabel, title=label)
            if filename == "accuracy":
                ax.set_ylim(0, 100)
            else:
                maximum = max(row[key] for key, _, _ in series for row in history)
                ax.set_ylim(0, maximum * 1.08 if maximum > 0 else 1)
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.22)
            ax.legend(frameon=False)
            fig.savefig(output_dir / f"{filename}.png", dpi=300)
            if show:
                plt.show()
            plt.close(fig)


def run_experiment(config, resume=False, stop_after_epoch=None):
    """Checkpoint at epoch boundaries; a partial epoch is replayed after interruption.

    stop_after_epoch allows a deliberate pause without changing the planned cosine
    schedule. It is also used by resume regression tests. No test-set evaluation.
    """
    config.validate()
    lofo = isinstance(config, LOFOConfig)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config.epochs:
        raise ValueError("stop_after_epoch must be within the planned training schedule.")
    output_dir = run_directory(config)
    checkpoint_dir = output_dir / "checkpoints"
    last_path, best_path = (checkpoint_dir / "last_checkpoint.pt",
                            checkpoint_dir / "best_model.pt")
    if resume and not last_path.exists():
        raise FileNotFoundError(f"Cannot resume: {last_path} does not exist.")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already contains artifacts. Use resume=True "
                              "or choose a new run_id to preserve the previous run.")

    seed_everything(config.seed, config.deterministic, config.cpu_threads)
    device = select_device() if config.device == "auto" else select_device(config.device)
    amp = config.use_amp and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
    environment = environment_info(device)
    if lofo:
        from .corruptions import implementation_metadata
        environment["corruptions"] = implementation_metadata()
    print(json.dumps({**environment, "effective_amp": amp}, indent=2))
    if device.type == "cpu" and not config.smoke_test:
        warnings.warn("Full training on CPU is slow. Enable a Colab GPU before starting.")
    train_loader, val_loader, generators, data_manifest, split = prepare_data(config, device)
    model = build_model().to(device)
    initial_model_sha256 = model_digest(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    history, best_accuracy, best_epoch, start_epoch, elapsed = [], -math.inf, 0, 1, 0.0
    corruption_history = []
    digest = source_digest()
    session = {"started_utc": utc_now(), "environment": environment, "effective_amp": amp}

    if resume:
        checkpoint = load_checkpoint(last_path)
        if not compatible_config(checkpoint["config"], config.to_dict()):
            raise ValueError("Resume configuration differs from the saved experiment; "
                             "keep the original schedule/settings or use a new run_id.")
        if checkpoint["source_sha256"] != digest:
            raise ValueError("Implementation changed since this checkpoint; restore its "
                             "saved source snapshot before resuming.")
        if checkpoint["data_manifest"] != data_manifest:
            # Ignore the location of the split file when Drive/cache is relocated.
            old = {k: v for k, v in checkpoint["data_manifest"].items() if k != "split_path"}
            new = {k: v for k, v in data_manifest.items() if k != "split_path"}
            if old != new:
                raise ValueError("Resume data indices/classes differ from the saved run.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if amp and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint["environment"] != environment or checkpoint["effective_amp"] != amp:
            warnings.warn("Runtime or precision changed. Loading is portable, but bitwise "
                          "replay is only expected on the same hardware/software stack.")
        history = checkpoint["history"]
        if lofo:
            corruption_history = checkpoint["corruption_history"]
            if len(corruption_history) != len(history):
                raise ValueError("Resume corruption history is incomplete.")
            for row in corruption_history:
                validate_exposures(config, row["counts"], data_manifest["train_count"])
        best_accuracy, best_epoch = checkpoint["best_validation_accuracy_pct"], checkpoint["best_epoch"]
        start_epoch, elapsed = checkpoint["epoch"] + 1, checkpoint["total_training_seconds"]
        # Recover a crash between committing 'last' and publishing a new 'best'.
        best = load_checkpoint(best_path) if best_path.exists() else None
        if best is None or best["epoch"] != best_epoch:
            if checkpoint["epoch"] == best_epoch:
                save_checkpoint(best_path, checkpoint)
            else:
                raise RuntimeError("Best checkpoint is missing/inconsistent. Restore it "
                                   "from backup; the last model cannot replace an earlier best.")
        restore_rng(checkpoint["rng_state"])
        for name, generator in generators.items():
            generator.set_state(checkpoint["loader_rng_state"][name])
        sessions = json.loads((output_dir / "sessions.json").read_text())
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "config.json", config.to_dict())
        save_json(output_dir / "split.json", split)
        save_json(output_dir / "data_manifest.json", data_manifest)
        save_json(output_dir / "environment.json", environment)
        (output_dir / "model.txt").write_text(str(model) + "\n", encoding="utf-8")
        shutil.copytree(Path(__file__).parent, output_dir / "source",
                        ignore=shutil.ignore_patterns("__pycache__"))
        notices = Path(__file__).parent.parent / "third_party"
        if notices.exists():
            shutil.copytree(notices, output_dir / "third_party",
                            ignore=shutil.ignore_patterns("__pycache__"))
        save_json(output_dir / "protocol.json", {
            "mode": config.mode, "pipeline_check_only": config.smoke_test,
            "source_sha256": digest, "scheduler_T_max": config.epochs,
            "effective_amp": amp, "parameter_count": sum(p.numel() for p in model.parameters()),
            "initial_model_sha256": initial_model_sha256,
            "selection": config.selection, "test_set_used_for_selection": False,
            "train_transforms": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip(p=0.5)"]
                                + (["LOFO:zero_or_one_corruption_before_normalization"] if lofo else [])
                                + ["ToTensor", "Normalize(config.mean,config.std)"],
            "eval_transforms": ["ToTensor", "Normalize(config.mean,config.std)"],
            **(protocol_metadata(config) if lofo else {}),
        })
        sessions = []
    sessions.append(session)
    save_json(output_dir / "sessions.json", sessions)
    with (output_dir / "training.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps(session) + "\n")
    # Checkpoint history is authoritative if a disconnect interrupted metric exports.
    write_metrics(output_dir, history)
    if lofo:
        write_corruption_audit(output_dir, corruption_history)
    final_epoch = min(config.epochs, stop_after_epoch or config.epochs)
    for epoch in range(start_epoch, final_epoch + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]  # LR actually used this epoch.
        synchronize(device)
        train_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        synchronize(device)
        train_seconds = time.perf_counter() - train_start
        val_metrics = run_epoch(model, val_loader, criterion, device, amp=amp)
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed += epoch_seconds
        improved = val_metrics["accuracy_pct"] > best_accuracy  # First epoch wins ties.
        if improved:
            best_accuracy, best_epoch = val_metrics["accuracy_pct"], epoch
        scheduler.step()  # Saved state is ready for the next epoch, including on resume.
        history.append({
            "mode": config.mode, "epoch": epoch, "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"], "train_accuracy_pct": train_metrics["accuracy_pct"],
            "val_loss": val_metrics["loss"], "val_accuracy_pct": val_metrics["accuracy_pct"],
            "train_seconds": train_seconds, "epoch_seconds": epoch_seconds,
            "total_training_seconds": elapsed,
        })
        if lofo:
            counts = train_metrics.get("corruption_counts")
            if counts is None:
                raise ValueError("LOFO training requires consumed-batch corruption audit metadata.")
            validate_exposures(config, counts, train_metrics["count"])
            if train_metrics["count"] != data_manifest["train_count"]:
                raise ValueError("Training did not consume every selected training image exactly once.")
            corruption_history.append({"epoch": epoch, "counts": counts})
        checkpoint = {
            "format_version": 1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "best_validation_accuracy_pct": best_accuracy, "best_epoch": best_epoch,
            "config": config.to_dict(), "seed": config.seed, "source_sha256": digest,
            "environment": environment, "effective_amp": amp, "data_manifest": data_manifest,
            "rng_state": capture_rng(),
            "loader_rng_state": {name: gen.get_state() for name, gen in generators.items()},
            "history": history, "total_training_seconds": elapsed,
            **({"corruption_history": corruption_history,
                "lofo_protocol": protocol_metadata(config)} if lofo else {}),
        }
        save_checkpoint(last_path, checkpoint)
        if improved:
            save_checkpoint(best_path, checkpoint)
        write_metrics(output_dir, history)
        if lofo:
            write_corruption_audit(output_dir, corruption_history)
        message = (f"{config.experiment_name if lofo else config.mode} | epoch {epoch}/{config.epochs} | lr {learning_rate:.6f} | "
              f"train {train_metrics['accuracy_pct']:.2f}% | val {val_metrics['accuracy_pct']:.2f}% | "
              f"{epoch_seconds:.1f}s")
        print(message, flush=True)
        with (output_dir / "training.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")

    completed = len(history) == config.epochs
    save_json(output_dir / "summary.json", {
        "mode": config.mode, "scientific_result": not config.smoke_test and completed,
        **({"experiment_name": config.experiment_name} if lofo else {}),
        "completed": completed, "completed_epochs": len(history), "planned_epochs": config.epochs,
        "best_epoch": best_epoch, "best_validation_accuracy_pct": best_accuracy,
        "total_training_seconds": elapsed,
        "timing_definition": "Sum of train+validation epoch durations; excludes setup, "
                             "checkpoint/metric writes, plots, and downtime between sessions.",
        "test_evaluated": (output_dir / "clean_test.json").exists(),
    })
    if history:
        plot_history(history, output_dir / "plots", config.smoke_test,
                     experiment_name=config.experiment_name if lofo else "Clean CIFAR-100 baseline")
    return output_dir, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full", action="store_true", help="Explicitly run the predefined 100-epoch schedule")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--data-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--split-file", help="LOFO only: existing Part 1 split; never regenerated")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Pause at this epoch without changing the schedule")
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {key: getattr(args, key) for key in
                 ("seed", "run_id", "device", "data_root", "artifact_root", "num_workers")
                 if getattr(args, key) is not None}
    if args.seed is not None and args.run_id is None:
        overrides["run_id"] = f"seed{args.seed}"
    if args.full or args.smoke_test:
        overrides["smoke_test"] = not args.full
    if args.no_download:
        overrides["download"] = False
    if args.split_file:
        if not isinstance(config, LOFOConfig):
            parser.error("--split-file applies only to LOFO configurations.")
        overrides["split_file"] = args.split_file
    config = replace(config, **overrides)
    config.validate()
    run_experiment(config, resume=args.resume, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
