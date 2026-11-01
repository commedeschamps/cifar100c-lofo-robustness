"""Frozen inputs, stack enforcement and shared initialization. No model forward pass."""
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch

from ..benchmark.data import file_hash, fingerprint
from ..corruptions import implementation_metadata
from ..model import build_model
from ..utils import (environment_info, load_checkpoint, model_digest, save_checkpoint,
                     save_json, seed_everything, utc_now)
from .config import ROOT, VERSION, configs

HISTORICAL_INITIAL = "fe7a5fa407bd0a5669d4cca9949a6cda1f9af4c7145e368d956d1c12b414a50b"
STACK_KEYS = ("python", "platform", "torch", "torchvision", "numpy", "cuda", "cudnn",
              "device", "gpu_model", "cpu_threads", "deterministic_algorithms",
              "cublas_workspace_config", "corruptions")


def current_stack():
    result = environment_info(torch.device("mps"))
    result["corruptions"] = implementation_metadata()
    return result


def historical_stack():
    stacks = [json.loads(Path(f"results/lofo_{f}/seed42/environment.json").read_text())
              for f in ("noise", "blur", "weather", "digital")]
    expected = {k: stacks[0][k] for k in STACK_KEYS}
    if any({k: stack[k] for k in STACK_KEYS} != expected for stack in stacks):
        raise ValueError("Historical LOFO stacks disagree.")
    return expected


def check_stack(require_mps=True):
    current = current_stack()
    differences = {key: {"expected": value, "actual": current[key]}
                   for key, value in historical_stack().items() if current[key] != value}
    if differences:
        raise ValueError(f"Exact historical software stack required: {differences}")
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in importlib.metadata.distributions()}
    for line in Path("requirements-local-lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        if installed.get(name.lower().replace("_", "-")) != version:
            raise ValueError(f"Historical environment lock mismatch: {line}.")
    if require_mps and not current["mps_available"]:
        raise RuntimeError("MPS is unavailable in this process. No CPU fallback is permitted for training/evaluation.")
    return current


def frozen_files():
    return sorted([*Path("src").rglob("*.py"), *Path("tests").glob("test_controlled_v2.py"),
                   *Path("configs/controlled_v2").glob("*.json"),
                   Path("results/controlled_v2/PROTOCOL_v2.0.0.md"),
                   Path("results/controlled_v2/RUN_PLAN.md"),
                   Path("requirements-local-lock.txt"), Path("requirements-lofo.txt")])


def load_freeze():
    freeze = json.loads((ROOT / "freeze.json").read_text())
    if freeze["protocol_version"] != VERSION:
        raise ValueError("Protocol version mismatch.")
    for path, expected in freeze["files"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen input changed: {path}. Create a new protocol version; do not refreeze existing runs.")
    if fingerprint(freeze["files"]) != freeze["freeze_sha256"]:
        raise ValueError("Freeze identity mismatch.")
    if set(map(str, frozen_files())) != set(freeze["frozen_source_files"]):
        raise ValueError("Frozen source/config inventory changed.")
    return freeze


def load_initial(config):
    freeze = load_freeze()
    path = Path(config.initial_state)
    if file_hash(path) != freeze["initial_file_sha256"]:
        raise ValueError("Shared initial-state file changed.")
    payload = load_checkpoint(path)
    if payload["protocol_version"] != VERSION or payload["seed"] != 42:
        raise ValueError("Initial-state identity mismatch.")
    # Preserve the caller's initialization RNG stream, including model-factory draws.
    with torch.random.fork_rng(devices=[]):
        model = build_model()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model_digest(model) != freeze["initial_model_sha256"]:
        raise ValueError("Initial tensor digest mismatch.")
    if any(t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all())
           for t in model.state_dict().values()):
        raise ValueError("Initial tensors must be finite FP32.")
    return model


def freeze_protocol():
    if (ROOT / "freeze.json").exists():
        raise FileExistsError("Protocol already frozen. Use preflight to verify; never overwrite the freeze.")
    for config in configs().values():
        config.validate()
    seed_everything(42)
    stack = check_stack(require_mps=False)
    initial_path = ROOT / "shared/initial_seed42.pt"
    if initial_path.exists():
        raise FileExistsError(f"Refusing to overwrite {initial_path}.")
    model = build_model()  # CPU construction matches the historical factory and RNG order.
    tensor_digest = model_digest(model)
    historical_protocols = {str(p): json.loads(p.read_text())["initial_model_sha256"]
                            for p in sorted(Path("results").glob("lofo_*/seed42/protocol.json"))}
    if len(historical_protocols) != 4 or set(historical_protocols.values()) != {HISTORICAL_INITIAL}:
        raise ValueError("Historical initial-state evidence is incomplete/inconsistent.")
    historical_sources = []
    for folder in sorted(Path("results").glob("lofo_*/seed42/source")):
        for name in ("model.py", "corruptions.py"):
            path = folder / name
            if path.read_bytes() != (Path("src") / name).read_bytes():
                raise ValueError(f"Historical architecture/operator source differs: {path}.")
            historical_sources.append(path)
    save_checkpoint(initial_path, {"protocol_version": VERSION, "seed": 42,
                                 "model_state_dict": model.state_dict()})
    reloaded = build_model()
    reloaded.load_state_dict(load_checkpoint(initial_path)["model_state_dict"], strict=True)
    if model_digest(reloaded) != tensor_digest:
        raise ValueError("Initial-state round-trip failed.")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    save_json(ROOT / "shared/packages.json", packages)
    freeze_paths = frozen_files() + [initial_path, ROOT / "shared/packages.json",
                                    Path("splits/cifar100_seed42.json"),
                                    Path("data/cifar-100-python/train"), Path("data/cifar-100-python/meta"),
                                    Path("results/research_audit/RESEARCH_AUDIT.md")]
    freeze_paths += [Path(p) for p in historical_protocols]
    freeze_paths += sorted(Path("results").glob("lofo_*/seed42/environment.json"))
    freeze_paths += historical_sources
    files = {str(p): file_hash(p) for p in freeze_paths}
    snapshot = ROOT / "frozen_inputs"
    for path in frozen_files():
        destination = snapshot / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    freeze = {"protocol_version": VERSION, "frozen_utc": utc_now(),
              "client_date": "2026-10-29", "client_timezone": "Asia/Almaty",
              "freeze_sha256": fingerprint(files), "files": files,
              "frozen_source_files": list(map(str, frozen_files())),
              "stack": stack, "historical_stack": historical_stack(),
              "initial_file_sha256": file_hash(initial_path), "initial_model_sha256": tensor_digest,
              "historical_initial_model_sha256": HISTORICAL_INITIAL,
              "historical_protocol_evidence": historical_protocols,
              "historical_digest_match": tensor_digest == HISTORICAL_INITIAL,
              "initialization_limit": "Digest equivalence under the historical model_digest algorithm; historical initial tensors were not saved, so direct tensor-by-tensor proof and complete trajectory replay are unavailable.",
              "training_started": False, "full_inference_started": False,
              "run_order": list(configs()), "configs": {k: c.to_dict() for k, c in configs().items()}}
    save_json(ROOT / "freeze.json", freeze)
    return freeze
