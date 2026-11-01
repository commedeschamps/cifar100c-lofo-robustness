"""Interruption-safe frozen FP32 evaluation, with mandatory clean-baseline gate."""
import csv
import fcntl
import json
import logging
import os
import resource
import subprocess
import platform
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..lofo import FAMILIES, REPRESENTATIVES
from ..utils import (atomic_write, environment_info, save_json, seed_everything,
                     select_device, synchronize, utc_now)
from .checkpoints import MODELS, default_runs, load_run
from .data import (CORRUPTIONS, MEAN, STD, file_hash, fingerprint, inspect_benchmark,
                   load_clean, normalize_rgb, severity_slice, discover)
from .metrics import (DEFINITIONS, aggregate, condition_metrics, expected_keys,
                      paired_bootstrap, validate_complete)

BASELINE_ACCURACY = 76.29
BASELINE_TOLERANCE_PP = 0.05  # fixed before corruption inference; five test images
TITLE = "How Far Does Corruption Robustness Transfer? A Leave-One-Family-Out Study on CIFAR-100-C."



def hardware_info():
    info = {"machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        for key, query in (("cpu_brand", "machdep.cpu.brand_string"), ("physical_memory_bytes", "hw.memsize")):
            try:
                info[key] = subprocess.check_output(["/usr/sbin/sysctl", "-n", query], text=True,
                                                     stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                info[key] = None
    return info


def memory_info(device):
    # ru_maxrss is bytes on macOS, KiB on Linux. This includes checkpoint audit/load peaks.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    info = {"process_peak_rss_bytes": rss if sys.platform == "darwin" else rss*1024}
    if device.type == "mps":
        info["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        info["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return info

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"utc": utc_now(), "level": record.levelname, "message": record.getMessage()})


def logger_for(output):
    logger = logging.getLogger("part3")
    for previous_handler in logger.handlers:
        previous_handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "logs/evaluation.jsonl")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    return logger


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluation holds the output lock: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def cache_path(output, model, corruption, severity):
    return Path(output) / "cache" / model / f"{corruption}__{severity}.npz"


def save_condition(path, identity, row, predictions):
    metadata = {"identity": identity, "row": row}
    def writer(temporary):
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, predictions=predictions.astype(np.uint8), metadata=json.dumps(metadata))
    atomic_write(path, writer)


def load_condition(path, identity, model, corruption, severity, targets):
    try:
        with np.load(path, allow_pickle=False) as cache:
            predictions = cache["predictions"]
            metadata = json.loads(str(cache["metadata"]))
        if (metadata["identity"] != identity or predictions.dtype != np.uint8
                or predictions.shape != np.asarray(targets).shape or np.any(predictions >= 100)):
            raise ValueError("Cache identity, shape or class range mismatch.")
        saved = metadata["row"]
        row = condition_metrics(model, corruption, severity, predictions, targets, saved["seconds"], saved["loss"])
        if saved != row or not np.isfinite(saved["loss"]) or saved["seconds"] < 0:
            raise ValueError("Cached metrics do not match per-image predictions.")
        return row, predictions
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"Invalid cached condition {path}; investigate/remove this cache file to recompute: {exc}") from exc


def infer(model, images, targets, device, batch_size, progress=True):
    if len(images) != len(targets) or not len(images):
        raise ValueError("Inference images/targets must match and be nonempty.")
    model.eval()
    predictions = np.empty(len(targets), dtype=np.uint8)
    loss = 0.0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            stop = min(start+batch_size, len(targets))
            x = normalize_rgb(images[start:stop], device)
            y = torch.as_tensor(np.asarray(targets[start:stop], dtype=np.int64), device=device)
            logits = model(x)
            if logits.shape != (stop-start, 100) or not torch.isfinite(logits).all():
                raise ValueError("Model returned invalid or nonfinite [batch,100] logits.")
            predictions[start:stop] = logits.argmax(1).cpu().numpy().astype(np.uint8)
            loss += float(nn.functional.cross_entropy(logits, y, reduction="sum").item())
            if progress and sys.stderr.isatty():
                filled = int(24*stop/len(targets))
                print(f"\r[{'='*filled}{' '*(24-filled)}] {stop}/{len(targets)}", end="", file=sys.stderr, flush=True)
    synchronize(device)
    if progress and sys.stderr.isatty():
        print(file=sys.stderr)
    return predictions, loss/len(targets), time.perf_counter()-started


def csv_write(path, rows, fields=None):
    if not rows and fields is None:
        return
    def writer(temporary):
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            out = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            out.writeheader()
            out.writerows(rows)
    atomic_write(path, writer)


def export(output, rows, manifest, status, aggregates=None):
    complete = status == "complete"
    if complete:
        validate_complete(rows)
        if manifest["mode"] != "full" or not (manifest.get("baseline_gate") or {}).get("passed"):
            raise ValueError("Only a full, gated benchmark can be marked complete.")
    csv_write(output / "metrics/per_condition.csv", [r for r in rows if r["corruption"] != "clean"],
              ["model", "seed", "family", "corruption", "severity", "n_samples", "n_correct", "accuracy_pct",
               "error_pct", "exposure_category", "seconds", "loss"])
    clean = [dict(r, best_clean_validation_accuracy_pct=manifest["checkpoints"][r["model"]]["best_clean_validation_accuracy_pct"])
             for r in rows if r["corruption"] == "clean"]
    csv_write(output / "metrics/clean_accuracy.csv", clean)
    summary = {"title": TITLE, "status": status, "benchmark_complete": complete,
               "scientific_result": complete, "mode": manifest["mode"],
               "pipeline_check_only": manifest["mode"] == "smoke",
               "completed_corruption_conditions": sum(r["corruption"] != "clean" for r in rows),
               "expected_corruption_conditions": 375, "completed_clean_checks": len(clean),
               "corruption_predictions": sum(r["n_samples"] for r in rows if r["corruption"] != "clean"),
               "baseline_gate": manifest.get("baseline_gate"), "definitions": DEFINITIONS,
               "official_corruption_dataset_available": manifest["dataset"]["corruptions"] is not None,
               "full_evaluation_ready": bool(manifest["dataset"]["corruptions"] is not None
                                             and (manifest.get("baseline_gate") or {}).get("passed"))}
    if aggregates is not None:
        summary["overall"] = aggregates["overall"]
        for name, values in aggregates.items():
            if isinstance(values, list):
                csv_write(output / "metrics" / f"{name}.csv", values)
    # A summary is published last; consumers must require benchmark_complete=true.
    save_json(output / "manifests/provenance.json", manifest)
    save_json(output / "metrics/evaluation.json", {"summary": summary, "conditions": rows, "aggregates": aggregates})
    report = [f"# {TITLE}", "", f"Status: **{status}**. Complete scientific benchmark: **{complete}**.", "",
              "Smoke outputs are software checks only. Partial outputs must not support publication aggregates.", "",
              "## Clean test checks", "", "| Model | Clean test accuracy (%) | Best clean validation (%) |",
              "|---|---:|---:|"]
    report += [f"| {r['model']} | {r['accuracy_pct']:.2f} | {r['best_clean_validation_accuracy_pct']:.2f} |" for r in clean]
    gate = manifest.get("baseline_gate")
    if gate:
        report += ["", f"Baseline gate passed: **{gate['passed']}**; measured minus recorded accuracy: "
                   f"**{gate['difference_pp']:+.4f} pp** (fixed tolerance {gate['tolerance_pp']:.2f} pp)."]
    report += ["", f"Completed corruption conditions: {summary['completed_corruption_conditions']}/375. "
               f"Official corruption dataset available: {summary['official_corruption_dataset_available']}. "
               f"Full evaluation ready: {summary['full_evaluation_ready']}."]
    if aggregates:
        report += ["", "## Held-out family transfer", "", "| Family | LOFO (%) | Baseline (%) | Difference (pp) |", "|---|---:|---:|---:|"]
        report += [f"| {r['family']} | {r['held_out_accuracy_pct']:.2f} | {r['baseline_accuracy_pct']:.2f} | {r['difference_pp']:+.2f} |"
                   for r in aggregates["held_out_transfer"] if r["scope"] == "family"]
    if aggregates and "bootstrap_held_out" in aggregates:
        report += ["", "## Paired test-image uncertainty (conditional on seed 42)", "",
                   "Marginal 95% percentile intervals; no training-seed variability or multiplicity correction.", "",
                   "| Held-out family | Difference (pp) | 95% interval (pp) |", "|---|---:|---:|"]
        report += [f"| {r['family']} | {r['difference_pp']:+.2f} | [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] |"
                   for r in aggregates["bootstrap_held_out"]]
    report += ["", "## Interpretation and limitations", "",
               "The baseline trained on CUDA with AMP; LOFO trained on MPS with FP32. All Part 3 inference uses FP32. "
               "Hardware, training precision and library differences limit the comparison.", "",
               "Positive held-out gains suggest transfer relative to clean-only training; they do not establish universal or causal generalization. "
               "The three exposed LOFO models have different training mixtures. Represented operators are not pixel-identical benchmark images.", "",
               "One training seed cannot estimate training-seed variability. Optional paired bootstrap intervals quantify test-image uncertainty "
               "conditional on these trained models, preserving repeated-image clusters. No significance claims are made.", "",
               "Overall accuracy weights each of the 15 corruptions equally. The separate four-family average weights families equally. "
               "Noise has three types; the other families have four. Mean corruption error here is unnormalized.", "",
               "See manifests/provenance.json and metrics/evaluation.json for content identities, configuration, gate and completion status."]
    atomic_write(output / "report.md", lambda p: p.write_text("\n".join(report)+"\n"))
    save_json(output / "metrics/summary.json", summary)


def evaluation(args):
    output = Path(args.output or ("results/evaluation/seed42" if args.mode == "full"
                                 else f"results/evaluation/{args.mode}_seed42")).resolve()
    if args.mode == "smoke" and "smoke" not in output.name.lower():
        raise ValueError("Smoke output directory name must contain 'smoke'.")
    if args.mode == "full" and any("smoke" in p.lower() for p in output.parts):
        raise ValueError("Full scientific outputs cannot use a smoke directory.")
    with run_lock(output):
        return _evaluation(args, output)


def _evaluation(args, output):
    for directory in ("metrics", "plots", "manifests", "logs", "cache"):
        (output / directory).mkdir(exist_ok=True)
    log = logger_for(output)
    manifest_path = output / "manifests/provenance.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"Existing evaluation at {output}; use --resume or a fresh --output.")
    if args.batch_size < 1 or args.smoke_images not in range(1, 10001) or args.benchmark_images not in range(1, 10001):
        raise ValueError("Positive batch size and image counts <=10000 required.")
    started = time.perf_counter()
    device = select_device(args.device)
    seed_everything(42, deterministic=True, cpu_threads=args.cpu_threads)
    clean, clean_identity = load_clean(args.data_root)
    targets = np.asarray(clean.targets, dtype=np.int64)
    models, infos = {}, {}
    for model_id, run in default_runs(args.results_root).items():
        models[model_id], infos[model_id] = load_run(run, model_id, clean.classes)
        log.info("Checkpoint verified: %s (best clean validation %.2f%%)", model_id,
                 infos[model_id]["best_clean_validation_accuracy_pct"])
    corruption_root, corruption_identity = None, None
    try:
        corruption_root = discover(args.corruption_root or args.data_root)
    except FileNotFoundError:
        if args.mode == "full":
            raise
        log.warning("Official CIFAR-100-C missing; this run will check clean data only. "
                    "Download explicitly: python -m src.benchmark download --data-root data")
    if corruption_root is not None:
        corruption_root, corruption_identity = inspect_benchmark(corruption_root, targets)
    sources = {str(p.relative_to(Path(__file__).parents[2])): file_hash(p)
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    for name in ("model.py", "utils.py", "lofo.py", "config.py", "dataset.py"):
        p = Path(__file__).parents[1] / name
        sources[f"src/{name}"] = file_hash(p)
    environment = environment_info(device)
    hardware = hardware_info()
    identity = {"mode": args.mode, "checkpoint_sha256": {m: i["sha256"] for m, i in infos.items()},
                "clean": clean_identity, "corruptions": corruption_identity, "sources": sources,
                "environment": environment, "hardware": hardware, "batch_size": args.batch_size, "precision": "float32",
                "mean": MEAN, "std": STD, "smoke_images": args.smoke_images if args.mode == "smoke" else None,
                "baseline_reference_pct": BASELINE_ACCURACY, "baseline_tolerance_pp": BASELINE_TOLERANCE_PP}
    run_id = fingerprint(identity)
    manifest = {"title": TITLE, "created_utc": utc_now(), "mode": args.mode, "identity": run_id,
                "configuration": identity, "environment": environment,
                "hardware": hardware,
                "checkpoints": infos, "dataset": {"clean": clean_identity, "corruptions": corruption_identity},
                "preprocessing": "uint8 RGB 32x32 -> FP32 /255 -> channel normalization; no resize/TTA/adaptation",
                "precision": "float32, autocast disabled", "batch_size": args.batch_size,
                "families": FAMILIES, "representatives": REPRESENTATIVES, "aggregation_definitions": DEFINITIONS,
                "baseline_gate": None, "commands": [shlex.join([sys.executable, "-m", "src.benchmark", *sys.argv[1:]])],
                "training_environment_limitation": "Baseline CUDA/AMP; LOFO MPS/FP32; training library versions differ."}
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError("No evaluation manifest to resume.")
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != run_id:
            raise ValueError("Resume identity changed (data, checkpoints, source, preprocessing, device, stack or batch size). Use a new directory.")
        manifest["created_utc"] = previous["created_utc"]
        manifest["commands"] = previous["commands"] + manifest["commands"]
        manifest["sessions"] = previous.get("sessions", [])
    manifest.setdefault("sessions", []).append({"started_utc": utc_now(), "command": manifest["commands"][-1]})
    for relative, digest in sources.items():
        target = output / "manifests/source" / relative
        if target.exists():
            if file_hash(target) != digest:
                raise ValueError(f"Evaluation source snapshot integrity mismatch: {target}")
        else:
            content = (Path(__file__).parents[2] / relative).read_bytes()
            atomic_write(target, lambda p, content=content: p.write_bytes(content))
    rows = []
    export(output, rows, manifest, "running")

    def condition(model_id, corruption, severity, images, labels):
        path = cache_path(output, model_id, corruption, severity)
        condition_id = fingerprint({"run": run_id, "model": model_id, "corruption": corruption,
                                    "severity": severity, "n": len(labels)})
        if path.exists():
            row, _ = load_condition(path, condition_id, model_id, corruption, severity, labels)
            log.info("Cached: %s %s severity=%s", model_id, corruption, severity)
        else:
            predictions, loss, seconds = infer(models[model_id], images, labels, device, args.batch_size)
            row = condition_metrics(model_id, corruption, severity, predictions, labels, seconds, loss)
            save_condition(path, condition_id, row, predictions)
            log.info("Completed: %s %s severity=%s n=%s accuracy=%.2f%% seconds=%.2f", model_id,
                     corruption, severity, len(labels), row["accuracy_pct"], seconds)
        rows.append(row)
        export(output, rows, manifest, "running")
        return row

    try:
        for model_id in MODELS:
            models[model_id].to(device)
            n = args.smoke_images if args.mode == "smoke" else 10000
            row = condition(model_id, "clean", 0, clean.data[:n], targets[:n])
            if model_id == "clean_baseline" and args.mode != "smoke":
                delta = row["accuracy_pct"]-BASELINE_ACCURACY
                manifest["baseline_gate"] = {"expected_accuracy_pct": BASELINE_ACCURACY,
                    "measured_accuracy_pct": row["accuracy_pct"], "measured_loss": row["loss"],
                    "difference_pp": delta, "tolerance_pp": BASELINE_TOLERANCE_PP,
                    "passed": abs(delta) <= BASELINE_TOLERANCE_PP+1e-10}
                if not manifest["baseline_gate"]["passed"]:
                    raise ValueError("BASELINE REPRODUCIBILITY GATE FAILED. Stop: inspect architecture, normalization, class order, "
                                     "checkpoint selection, saved baseline source and backend arithmetic. Do not tune to the test set.")
            # Warm up, then time a fixed real clean-data throughput probe. No training or generated inputs.
            if model_id == "clean_baseline":
                probe_n = args.benchmark_images
                infer(models[model_id], clean.data[:min(args.batch_size, probe_n)], targets[:min(args.batch_size, probe_n)],
                      device, args.batch_size, progress=False)
                _, _, seconds = infer(models[model_id], clean.data[:probe_n], targets[:probe_n], device,
                                      args.batch_size, progress=False)
                manifest["performance_probe"] = {"data": "official clean test images; throughput diagnostic only",
                    "n_images": probe_n, "seconds": seconds, "images_per_second": probe_n/seconds,
                    "estimated_3750000_prediction_seconds": 3750000*seconds/probe_n,
                    "limitations": "clean-array throughput extrapolation; excludes corruption I/O, hashes, checkpoints, plots; not MPS timing unless device=mps"}
                log.info("Measured throughput: %.1f images/s (%s)", probe_n/seconds, device)
            models[model_id].to("cpu")
        if corruption_root is not None and args.mode != "preflight":
            for model_id in MODELS:
                models[model_id].to(device)
                for corruption in CORRUPTIONS:
                    images = np.load(corruption_root / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)
                    for severity in range(1, 6):
                        block = images[severity_slice(severity)]
                        n = args.smoke_images if args.mode == "smoke" else 10000
                        condition(model_id, corruption, severity, block[:n], targets[:n])
                    del block, images
                models[model_id].to("cpu")
        aggregates = aggregate(rows) if args.mode == "full" else None
        if aggregates and args.bootstrap:
            intervals = []
            def correctness(model_id, corruption, severity):
                with np.load(cache_path(output, model_id, corruption, severity), allow_pickle=False) as cached:
                    return cached["predictions"] == targets
            for family, names in FAMILIES.items():
                values = []
                for model_id in (f"lofo_{family}", "clean_baseline"):
                    values.append(np.stack([correctness(model_id, c, s)
                                            for c in names for s in range(1, 6)]))
                intervals.append({"family": family, **paired_bootstrap(*values, repeats=args.bootstrap)})
            aggregates["bootstrap_held_out"] = intervals
        status = "complete" if args.mode == "full" else "smoke_complete" if args.mode == "smoke" else "preflight_passed" if corruption_root is not None else "preflight_clean_only"
        manifest["memory"] = memory_info(device)
        manifest["sessions"][-1].update(ended_utc=utc_now(), runtime_seconds=time.perf_counter()-started, status=status)
        export(output, rows, manifest, status, aggregates)
        if aggregates:
            from .plots import plot_saved
            plot_saved(output)
        log.info("%s. Outputs: %s", status, output)
        return manifest
    except BaseException as exc:
        manifest["sessions"][-1].update(ended_utc=utc_now(), runtime_seconds=time.perf_counter()-started,
                                       error=f"{type(exc).__name__}: {exc}")
        export(output, rows, manifest, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        log.exception("Evaluation stopped; completed conditions are cached safely.")
        raise
