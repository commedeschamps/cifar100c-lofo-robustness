"""Windows-friendly, zero-copy memory-mapped CIFAR-100 training cache.

Workers receive only paths instead of pickled 150 MB CIFAR-100 image buffers.
Files are derived exclusively from the official TRAIN file, never the test set.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from .core import read_json,sha_file,atomic_json,now

CACHE=Path("data/replication_v3")
IMAGES=CACHE/"cifar100_train_images.npy"
LABELS=CACHE/"cifar100_train_labels.npy"
RECEIPT=CACHE/"cache_manifest.json"


def _save_npy_atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            np.save(stream,data,allow_pickle=False)
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def ensure_cache(verify_hash=True):
    """Build cache if absent, validate identity if present. Does not access TEST."""
    if not IMAGES.exists() or not LABELS.exists() or not RECEIPT.exists():
        if any(p.exists() for p in (IMAGES,LABELS,RECEIPT)):
            raise RuntimeError("Incomplete CIFAR cache; preserve and inspect before rebuilding")
        from torchvision.datasets import CIFAR100
        raw=CIFAR100("data",train=True,download=False)
        if raw.data.shape!=(50000,32,32,3) or len(raw.targets)!=50000:
            raise ValueError("Unexpected CIFAR-100 training array")
        images=np.asarray(raw.data,dtype=np.uint8)
        labels=np.asarray(raw.targets,dtype=np.int64)
        _save_npy_atomic(IMAGES,images)
        _save_npy_atomic(LABELS,labels)
        atomic_json(RECEIPT,{"made_utc":now(),"origin":"official_train_only",
            "original_train_file_sha256":sha_file(Path("data/cifar-100-python/train")),
            "images_sha256":sha_file(IMAGES),"labels_sha256":sha_file(LABELS),
            "shapes":[list(images.shape),list(labels.shape)]})
    receipt=read_json(RECEIPT)
    if receipt.get("origin")!="official_train_only":raise ValueError("Unexpected cache origin")
    if receipt["original_train_file_sha256"]!=sha_file(Path("data/cifar-100-python/train")):
        raise ValueError("CIFAR cache derived from different official train file")
    if verify_hash:
        for path,key in ((IMAGES,"images_sha256"),(LABELS,"labels_sha256")):
            if sha_file(path)!=receipt[key]:raise ValueError(f"Corrupt cache file: {path}")
    return ArrayStore(IMAGES,LABELS)


class ArrayStore:
    """Lazily opens immutable .npy arrays inside each spawned DataLoader worker."""
    def __init__(self,images=IMAGES,labels=LABELS):
        self.images_path=str(images)
        self.labels_path=str(labels)
        self._data=None
        self._targets=None

    def __getstate__(self):
        # Worker spawn must NOT serialize numpy arrays/memory maps.
        return {"images_path":self.images_path,"labels_path":self.labels_path,
                "_data":None,"_targets":None}

    @property
    def data(self):
        if self._data is None:
            self._data=np.load(self.images_path,mmap_mode="r",allow_pickle=False)
            if self._data.shape!=(50000,32,32,3) or self._data.dtype!=np.uint8:
                raise ValueError("Invalid CIFAR training image cache")
        return self._data

    @property
    def targets(self):
        if self._targets is None:
            self._targets=np.load(self.labels_path,mmap_mode="r",allow_pickle=False)
            if self._targets.shape!=(50000,) or self._targets.dtype!=np.int64:
                raise ValueError("Invalid CIFAR training label cache")
        return self._targets
