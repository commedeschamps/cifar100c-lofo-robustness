"""Opt-in official clean test evaluation, after the full validation protocol ends."""

import json
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import CIFAR100

from .dataset import make_loader, make_transforms
from .model import build_model
from .train import compatible_config, run_directory, run_epoch
from .utils import (environment_info, load_checkpoint, save_json, seed_everything,
                    select_device, source_digest, utc_now)


def evaluate_clean_test(config):
    config.validate()
    if config.smoke_test:
        raise ValueError("Official test evaluation is disabled for smoke tests.")
    output_dir = run_directory(config)
    result_path = output_dir / "clean_test.json"
    last = load_checkpoint(output_dir / "checkpoints" / "last_checkpoint.pt")
    best_path = output_dir / "checkpoints" / "best_model.pt"
    best = load_checkpoint(best_path)
    if last["epoch"] != config.epochs or len(last["history"]) != config.epochs:
        raise ValueError("Finish the full planned schedule before accessing the test set.")
    if (not compatible_config(last["config"], config.to_dict())
            or not compatible_config(best["config"], config.to_dict())
            or best["epoch"] != last["best_epoch"]
            or best["source_sha256"] != source_digest()):
        raise ValueError("Final checkpoint/configuration mismatch; use the original frozen run.")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result["source_sha256"] != best["source_sha256"]
                or result["selected_epoch"] != best["epoch"]
                or result["split_sha256"] != best["data_manifest"]["split_sha256"]):
            raise ValueError("Saved test result does not match the selected checkpoint.")
        print("Returning the saved final test result; no repeated test-set access.")
        return result
    seed_everything(config.seed, config.deterministic, config.cpu_threads)
    device = select_device()
    model = build_model().to(device)
    model.load_state_dict(best["model_state_dict"])
    _, transform = make_transforms(config)
    # This is the ONLY train=False dataset construction in the implementation.
    try:
        test_set = CIFAR100(config.data_root, train=False, transform=transform,
                            download=config.download)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Cannot load official CIFAR-100 test data. Check data_root, "
                           "internet access, and free disk space.") from exc
    if len(test_set) != 10000:
        raise ValueError("Expected all 10,000 official clean test images.")
    loader = make_loader(test_set, config, device,
                         torch.Generator().manual_seed(config.seed + 2))
    # Full precision for the final clean test; selection still uses validation only.
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device, amp=False)
    result = {
        "dataset": "CIFAR100_official_clean_test", "evaluated_utc": utc_now(),
        "test_accuracy_pct": metrics["accuracy_pct"], "test_loss": metrics["loss"],
        "test_count": metrics["count"], "selected_epoch": best["epoch"],
        "selected_validation_accuracy_pct": best["best_validation_accuracy_pct"],
        "selection": config.selection, "checkpoint": str(best_path),
        "source_sha256": best["source_sha256"], "split_sha256": best["data_manifest"]["split_sha256"],
        "evaluation_precision": "float32", "environment": environment_info(device),
        "configuration_frozen": True, "used_for_tuning": False,
    }
    save_json(result_path, result)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["test_evaluated"] = True
        save_json(summary_path, summary)
    print(f"Final official clean test: accuracy {metrics['accuracy_pct']:.2f}%, "
          f"loss {metrics['loss']:.4f}. Do not tune the configuration using this result.")
    return result
