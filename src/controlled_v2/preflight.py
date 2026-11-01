"""Bounded data/initial-state checks, without model inference or training."""
import json
from pathlib import Path

import torch

from ..benchmark.data import fingerprint
from ..lofo import CORRUPTION_NAMES, exposure_bins, validate_exposures
from ..utils import model_digest, save_json, seed_everything, utc_now
from .config import ROOT, VERSION, configs
from .data import prepare_data
from .evaluate import evaluation_preflight
from .provenance import check_stack, load_freeze, load_initial
from .train import run_manifest


def preflight(require_mps=False):
    freeze = load_freeze()
    seed_everything(42)
    stack = check_stack(require_mps=require_mps)
    reports = {}
    first_ids = []
    for name, config in configs().items():
        seed_everything(42)
        # CPU loaders exercise the same crop/flip/corruption and two worker processes.
        train, val, _, data, _ = prepare_data(config, torch.device("cpu"))
        model = load_initial(config)
        digest = model_digest(model)
        mps_digest = None
        if stack["mps_available"]:
            model.to("mps")
            mps_digest = model_digest(model)
            if mps_digest != digest:
                raise ValueError("Initial-state digest changed on MPS transfer.")
            model.cpu()
        counts = exposure_bins()
        iterator = iter(train)
        batches = [next(iterator), next(iterator)]  # 256 training-file examples, no forward pass.
        del iterator
        labels = []
        for images, targets, codes, severities in batches:
            if images.shape != (128, 3, 32, 32) or not torch.isfinite(images).all():
                raise ValueError("Invalid preflight training batch.")
            labels += targets.tolist()
            for code, severity in zip(codes.tolist(), severities.tolist()):
                corruption = CORRUPTION_NAMES[code]
                counts["clean" if code == 0 else f"{corruption}:severity{severity}"] += 1
        validate_exposures(config, counts, 256)
        first_ids.append(labels)
        a, label = val.dataset[0]
        b, other_label = val.dataset[0]
        if not torch.equal(a, b) or label != other_label or len(val.dataset[0]) != 2:
            raise ValueError("Validation must be clean and deterministic.")
        manifest = run_manifest(config, freeze, data)
        target = ROOT / "preflight" / name
        save_json(target / "run_manifest.json", manifest)
        reports[name] = {"config_valid": True, "train_count": len(train.dataset), "val_count": len(val.dataset),
                         "train_batches_per_epoch": len(train), "val_batches_per_epoch": len(val),
                         "initial_model_sha256": digest, "mps_initial_model_sha256": mps_digest,
                         "manifest_sha256": fingerprint(manifest), "sampled_training_images": 256,
                         "sample_exposure_counts": counts, "clean_validation_verified": True,
                         "run_output": str(config.output), "production_output_exists": config.output.exists()}
    if first_ids[0] != first_ids[1]:
        raise ValueError("Paired conditions have different loader label orders.")
    report = {"protocol_version": VERSION, "checked_utc": utc_now(), "freeze_sha256": freeze["freeze_sha256"],
              "software_stack_matches_historical_lofo": True, "mps_available": stack["mps_available"],
              "device_readiness": "ready" if stack["mps_available"] else "requires_MPS_visible_process",
              "runs": reports, "shared_initial_state_verified": True,
              "historical_initial_digest_match": freeze["historical_digest_match"],
              "forward_passes": 0, "backward_passes": 0, "optimizer_updates": 0,
              "cifar100c_test_metrics_read": False, "evaluation": evaluation_preflight()}
    save_json(ROOT / "preflight/report.json", report)
    return report
