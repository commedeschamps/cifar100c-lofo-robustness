"""Official arrays, fixed RGB preprocessing, and content identities."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR100

from ..config import Config
from ..lofo import FAMILIES

CORRUPTIONS = tuple(c for names in FAMILIES.values() for c in names)
FAMILY_OF = {c: f for f, names in FAMILIES.items() for c in names}
MEAN, STD = Config().mean, Config().std
SOURCE = "https://zenodo.org/records/3555552"
URL = SOURCE + "/files/CIFAR-100-C.tar?download=1"
ARCHIVE_MD5 = "11f0ed0f1191edbf9fa23466ae6021d3"


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def severity_slice(severity, block_size=10000):
    if type(severity) is not int or severity not in range(1, 6):
        raise ValueError("Severity must be an integer from 1 through 5.")
    return slice((severity - 1) * block_size, severity * block_size)


def validate_labels(labels, official_targets):
    targets = np.asarray(official_targets)
    n = len(targets)
    if (labels.ndim != 1 or labels.shape not in ((n,), (5*n,))
            or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() >= 100):
        raise ValueError("labels.npy must contain 10,000 labels or five repeated blocks, integer [0,99].")
    for block in labels.reshape(-1, n):
        if not np.array_equal(block, targets):
            raise ValueError("Benchmark labels do not align with the official fine-label test ordering.")
    return targets.astype(np.int64)


def validate_images(images, block_size=10000):
    if images.shape != (5*block_size, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(f"Expected uint8 NHWC RGB [{5*block_size},32,32,3], got {images.shape}/{images.dtype}.")
    # uint8 enforces finite [0,255]. No resize, channel swap, or image generation.
    return {"shape": list(images.shape), "dtype": str(images.dtype),
            "value_range": [int(images.min()), int(images.max())],
            "severity_boundaries": [[s*block_size, (s+1)*block_size] for s in range(5)]}


def normalize_rgb(images, device):
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("Inference requires unresized uint8 NHWC RGB images.")
    # Copy only this batch: mmap slices are read-only. Match ToTensor / Normalize FP32 order.
    x = torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    x = x.div(255.0)
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    return x.sub(mean).div(std).contiguous()


def load_clean(root):
    dataset = CIFAR100(str(root), train=False, download=False)
    if dataset.data.shape != (10000, 32, 32, 3) or len(dataset.classes) != 100:
        raise ValueError("Expected the complete official CIFAR-100 test set and 100 fine classes.")
    base = Path(root) / dataset.base_folder
    identity = {"root": str(Path(root).resolve()), "test_sha256": file_hash(base / "test"),
                "meta_sha256": file_hash(base / "meta"), "classes": dataset.classes,
                "targets_sha256": fingerprint(dataset.targets),
                "integrity": "torchvision official extracted-file MD5 checks"}
    return dataset, identity


def discover(root):
    root = Path(root).expanduser().resolve()
    candidates = [p for p in (root, root / "CIFAR-100-C") if (p / "labels.npy").is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot uniquely locate CIFAR-100-C under {root}. "
                                "Run: python -m src.benchmark download --data-root data")
    return candidates[0]


def inspect_benchmark(root, targets):
    root = discover(root)
    receipt_path = root / "official_provenance.json"
    if not receipt_path.is_file():
        raise ValueError("Existing arrays need official archive verification: python -m src.benchmark "
                         "verify --archive data/CIFAR-100-C.tar --corruption-root " + str(root))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("archive_md5") != ARCHIVE_MD5 or receipt.get("source") != SOURCE:
        raise ValueError("Benchmark provenance does not identify the pinned official Zenodo archive.")
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    validate_labels(labels, targets)
    records = {}
    for name in ("labels", *CORRUPTIONS):
        path = root / f"{name}.npy"
        digest = file_hash(path)
        if receipt.get("files", {}).get(path.name) != digest:
            raise ValueError(f"Dataset integrity mismatch: {path}. Re-verify against the official archive.")
        if name != "labels":
            images = np.load(path, mmap_mode="r", allow_pickle=False)
            details = validate_images(images)
            del images
        else:
            details = {"shape": list(labels.shape), "dtype": str(labels.dtype)}
        records[path.name] = {**details, "sha256": digest, "bytes": path.stat().st_size}
    return root, {"root": str(root), "source": SOURCE, "archive_md5": ARCHIVE_MD5,
                  "files": records, "rgb_evidence": "verified archive identity and published RGB serialization",
                  "excluded_extra_types": sorted(p.stem for p in root.glob("*.npy")
                                                 if p.stem not in (*CORRUPTIONS, "labels"))}
