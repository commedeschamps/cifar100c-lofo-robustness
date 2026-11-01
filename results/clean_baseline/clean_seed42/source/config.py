"""One serializable configuration for both notebook and command-line execution."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Config:
    smoke_test: bool = True
    seed: int = 42
    split_seed: int = 42  # Keep fixed across seeds and future experimental conditions.
    full_epochs: int = 100
    smoke_epochs: int = 2
    smoke_train_per_class: int = 2
    smoke_val_per_class: int = 1
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 2
    use_amp: bool = True
    deterministic: bool = True
    cpu_threads: int = 4
    data_root: str = "data"
    artifact_root: str = "."
    run_id: str = "seed42"
    download: bool = True
    # Fixed protocol, serialized explicitly for future comparisons.
    dataset: str = "CIFAR100"
    num_classes: int = 100
    train_per_class: int = 450
    val_per_class: int = 50
    architecture: str = "cifar_resnet18_scratch"
    mean: tuple = (0.5071, 0.4867, 0.4408)
    std: tuple = (0.2675, 0.2565, 0.2761)
    optimizer: str = "SGD"
    scheduler: str = "CosineAnnealingLR"
    eta_min: float = 0.0
    loss: str = "CrossEntropyLoss"
    selection: str = "highest_validation_accuracy_first_tie"

    @property
    def epochs(self):
        return self.smoke_epochs if self.smoke_test else self.full_epochs

    @property
    def mode(self):
        return "smoke_test" if self.smoke_test else "clean_baseline"

    def to_dict(self):
        # JSON round-trip normalizes tuples and other config representations.
        return json.loads(json.dumps(asdict(self)))

    def validate(self):
        reference = Config()
        for key in ("dataset", "num_classes", "train_per_class", "val_per_class",
                    "architecture", "optimizer", "scheduler", "loss", "selection"):
            if getattr(self, key) != getattr(reference, key):
                raise ValueError(f"{key} is fixed by the clean-baseline protocol.")
        if self.epochs < 1 or self.batch_size < 2 or self.num_workers < 0:
            raise ValueError("Use positive epochs, batch_size >= 2 and num_workers >= 0.")
        if self.cpu_threads < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid CPU thread count, learning rate or weight decay.")
        if not 0 <= self.momentum < 1 or not 0 <= self.eta_min <= self.learning_rate:
            raise ValueError("Invalid momentum or cosine minimum learning rate.")
        if not 1 <= self.smoke_train_per_class <= 450 or not 1 <= self.smoke_val_per_class <= 50:
            raise ValueError("Smoke subsets must fit within the fixed split.")
        if len(self.mean) != 3 or len(self.std) != 3 or any(s <= 0 for s in self.std):
            raise ValueError("Provide three channel means and positive standard deviations.")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("run_id must be a single directory name.")


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = Config(**json.load(handle))
    config.validate()
    return config
