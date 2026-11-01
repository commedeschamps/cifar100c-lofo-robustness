"""Reproducibility, portable checkpoints, and atomic artifact writes."""

import os
# Must be set before CUDA context creation (including notebook diagnostics).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, payload):
    atomic_write(path, lambda p: p.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def save_checkpoint(path, payload):
    atomic_write(path, lambda p: torch.save(payload, p))


def load_checkpoint(path):
    # Store only tensors and primitive containers; no arbitrary object unpickling.
    return torch.load(path, map_location="cpu", weights_only=True)


def seed_everything(seed, deterministic=True, cpu_threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fail explicitly rather than quietly use a nondeterministic operation.
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            warnings.warn("CUDA device count changed; exact replay is not guaranteed.")


def select_device(preference="auto"):
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if preference == "auto":
        return torch.device(next(name for name in ("cuda", "mps", "cpu") if available[name]))
    if preference not in available or not available[preference]:
        raise RuntimeError(f"Requested device {preference!r} is unavailable. "
                           "Use --device cpu for smoke checks, or enable the requested accelerator.")
    return torch.device(preference)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def environment_info(device):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    gpu = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "gpu_model": gpu.name if gpu else "Apple Metal (MPS)" if device.type == "mps" else None,
        "mps_available": torch.backends.mps.is_available(),
        "gpu_memory_gib": gpu.total_memory / 2**30 if gpu else None,
        "git_commit": commit, "git_dirty": dirty,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cpu_threads": torch.get_num_threads(),
    }


def source_digest():
    sources = sorted(Path(__file__).parent.glob("*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_metrics(output_dir, history):
    output_dir = Path(output_dir)
    save_json(output_dir / "metrics.json", history)
    if history:
        def writer(path):
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                csv_writer.writeheader()
                csv_writer.writerows(history)
        atomic_write(output_dir / "metrics.csv", writer)
