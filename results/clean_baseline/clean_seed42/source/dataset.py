"""Official training-only split, independent transforms, and seeded loaders."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .utils import save_json, seed_worker


def split_fingerprint(split):
    encoded = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_split(split, targets, split_seed):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (50000,) or not np.array_equal(np.bincount(targets, minlength=100),
                                                      np.full(100, 500)):
        raise ValueError("Expected official CIFAR-100 training targets: 500 per class.")
    label_hash = hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest()
    if (split["split_seed"] != split_seed or split["targets_sha256"] != label_hash
            or split["dataset"] != "CIFAR100_official_train" or split["version"] != 1):
        raise ValueError("Saved split does not match the dataset or split seed.")
    train = np.asarray(split["train_indices"])
    val = np.asarray(split["val_indices"])
    if (train.shape != (45000,) or val.shape != (5000,)
            or not np.issubdtype(train.dtype, np.integer)
            or not np.issubdtype(val.dtype, np.integer)):
        raise ValueError("Saved split must contain 45,000/5,000 integer indices.")
    all_indices = np.concatenate([train, val])
    if not np.array_equal(np.sort(all_indices), np.arange(50000)):
        raise ValueError("Split contains overlaps, omissions, or invalid indices.")
    for indices, expected in ((train, 450), (val, 50)):
        if not np.array_equal(np.bincount(targets[indices], minlength=100), np.full(100, expected)):
            raise ValueError("Split is not exactly stratified (450 train / 50 val per class).")


def get_or_create_split(targets, path, split_seed=42):
    path = Path(path)
    targets = np.asarray(targets, dtype=np.int64)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
    else:
        if targets.shape != (50000,) or not np.array_equal(
                np.bincount(targets, minlength=100), np.full(100, 500)):
            raise ValueError("Only the 50,000 official training images may define the split.")
        rng = np.random.default_rng(split_seed)
        train, val = [], []
        for label in range(100):
            indices = rng.permutation(np.flatnonzero(targets == label))
            val.extend(indices[:50].tolist())
            train.extend(indices[50:].tolist())
        split = {
            "version": 1, "dataset": "CIFAR100_official_train", "split_seed": split_seed,
            "targets_sha256": hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest(),
            "train_indices": sorted(train), "val_indices": sorted(val),
        }
        validate_split(split, targets, split_seed)
        save_json(path, split)
    validate_split(split, targets, split_seed)
    return split


def balanced_subset(indices, targets, per_class, seed):
    indices = np.asarray(indices)
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(100):
        candidates = indices[targets[indices] == label]
        if len(candidates) < per_class:
            raise ValueError(f"Insufficient examples for class {label}.")
        chosen.extend(rng.permutation(candidates)[:per_class].tolist())
    return sorted(chosen)


class TransformSubset(Dataset):
    """Share raw images, but apply distinct training and validation transforms."""

    def __init__(self, raw, indices, transform):
        self.raw, self.indices, self.transform = raw, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.raw[self.indices[index]]
        return self.transform(image), label


def make_transforms(config):
    normalize = transforms.Normalize(config.mean, config.std)
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])
    return train, evaluation


def make_loader(dataset, config, device, generator, shuffle=False):
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=generator,
        # Recreate workers each epoch so restored generator state reproduces worker seeds.
        persistent_workers=False, drop_last=False,
    )


def prepare_data(config, device):
    config.validate()
    try:
        raw = datasets.CIFAR100(config.data_root, train=True, download=config.download)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not load official CIFAR-100 from {config.data_root}. "
            "Enable internet for the first download (~170 MB archive), check free disk "
            "space, or supply an existing torchvision CIFAR-100 cache."
        ) from exc
    split_path = Path(config.artifact_root) / "splits" / f"cifar100_seed{config.split_seed}.json"
    split = get_or_create_split(raw.targets, split_path, config.split_seed)
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    if config.smoke_test:
        train_indices = balanced_subset(train_indices, raw.targets,
                                        config.smoke_train_per_class, config.split_seed + 1)
        val_indices = balanced_subset(val_indices, raw.targets,
                                      config.smoke_val_per_class, config.split_seed + 2)
    train_transform, eval_transform = make_transforms(config)
    generators = {name: torch.Generator().manual_seed(config.seed + offset)
                  for name, offset in (("train", 0), ("val", 1))}
    train_loader = make_loader(TransformSubset(raw, train_indices, train_transform),
                               config, device, generators["train"], shuffle=True)
    val_loader = make_loader(TransformSubset(raw, val_indices, eval_transform),
                             config, device, generators["val"])
    manifest = {
        "split_path": str(split_path), "split_sha256": split_fingerprint(split),
        "train_indices": train_indices, "val_indices": val_indices,
        "train_count": len(train_indices), "val_count": len(val_indices),
        "classes": raw.classes,
    }
    print(f"{config.mode}: {len(train_indices):,} training / {len(val_indices):,} validation; "
          "official test set is not loaded.")
    return train_loader, val_loader, generators, manifest, split
