"""Training-file-only data preparation and independent v2 augmentation streams."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

from ..corruptions import apply_corruption, require_opencv
from ..dataset import TransformSubset, make_loader, make_transforms, split_fingerprint, validate_split
from ..lofo import CORRUPTION_NAMES


class ControlledTransform:
    def __init__(self, config):
        config.validate()
        self.allowed = config.allowed_corruptions
        self.probability = config.corruption_probability
        self.severities = config.allowed_severities
        self.spatial = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=0.5)])
        self.tensor = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])
        self.streams = None
        if "defocus_blur" in self.allowed:
            require_opencv()

    def sample(self):
        # Nonpersistent workers restart with saved loader-generator seeds each epoch.
        # Fixed draws even for the clean control preserve independent spatial/branch streams.
        if self.streams is None:
            seed = torch.initial_seed() % 2**32
            self.streams = [np.random.RandomState((seed + offset) % 2**32)
                            for offset in (0, 100003, 200003)]
        branch, operator, severity = self.streams
        corrupt = branch.random() < self.probability
        slot = operator.randint(4)
        level = int(severity.choice(self.severities))
        operator_seed = random.getrandbits(32)
        if not corrupt:
            return "clean", 0, operator_seed
        return self.allowed[slot], level, operator_seed

    def __call__(self, image):
        image = self.spatial(image)
        name, severity, operator_seed = self.sample()
        if name != "clean":
            image = apply_corruption(image, name, severity, np.random.RandomState(operator_seed))
        return self.tensor(image), CORRUPTION_NAMES.index(name), severity


def prepare_data(config, device):
    config.validate()
    split_path = Path(config.split_file)
    split = json.loads(split_path.read_text())
    if split_fingerprint(split) != config.split_sha256:
        raise ValueError("Controlled v2 requires the immutable historical split.")
    raw = CIFAR100(config.data_root, train=True, download=False)
    validate_split(split, raw.targets, 42)
    _, evaluation = make_transforms(config)
    generators = {name: torch.Generator().manual_seed(config.seed + offset)
                  for name, offset in (("train", 0), ("val", 1))}
    train = make_loader(TransformSubset(raw, split["train_indices"], ControlledTransform(config)),
                        config, device, generators["train"], shuffle=True)
    val = make_loader(TransformSubset(raw, split["val_indices"], evaluation),
                      config, device, generators["val"])
    manifest = {"split_path": str(split_path), "split_sha256": split_fingerprint(split),
                "train_indices": split["train_indices"], "val_indices": split["val_indices"],
                "train_count": len(train.dataset), "val_count": len(val.dataset),
                "classes": raw.classes}
    return train, val, generators, manifest, split
