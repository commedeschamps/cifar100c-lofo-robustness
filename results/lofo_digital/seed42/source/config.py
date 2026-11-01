"""One serializable configuration for both notebook and command-line execution."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from .lofo import ORIGINAL_SPLIT_SHA256


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
    device: str = "auto"  # auto selects CUDA, then Apple MPS, then CPU.
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
        if self.device not in ("auto", "cuda", "mps", "cpu"):
            raise ValueError("device must be auto, cuda, mps, or cpu.")
        if not 0 <= self.seed < 2**32 or not 0 <= self.split_seed < 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32).")


@dataclass(frozen=True)
class LOFOConfig(Config):
    experiment_name: str = "lofo_noise"
    held_out_family: str = "noise"
    allowed_corruptions: tuple = ("defocus_blur", "fog", "jpeg_compression")
    corruption_probability: float = 0.5
    corruption_selection: str = "uniform"
    severity_selection: str = "uniform"
    allowed_severities: tuple = (1, 2, 3, 4, 5)
    split_file: str = "splits/cifar100_seed42.json"
    split_sha256: str = ORIGINAL_SPLIT_SHA256
    corruption_implementation: str = "hendrycks_cifar_c_8190fe3_v1"

    @property
    def mode(self):
        return "smoke_test" if self.smoke_test else self.experiment_name

    def validate(self):
        from .lofo import FAMILIES, ORIGINAL_SPLIT_SHA256, PLANNED_SEEDS, allowed_for
        super().validate()
        expected = allowed_for(self.held_out_family)
        if set(self.allowed_corruptions) & set(FAMILIES[self.held_out_family]):
            raise ValueError("Allowed corruptions include the entirely held-out family.")
        if self.experiment_name != f"lofo_{self.held_out_family}":
            raise ValueError("Experiment name must match its held-out family.")
        if tuple(self.allowed_corruptions) != expected:
            raise ValueError(f"Use exactly the predefined representatives in order: {expected}.")
        if not 0 <= self.corruption_probability <= 1:
            raise ValueError("corruption_probability must be in [0,1].")
        if (self.corruption_selection != "uniform" or self.severity_selection != "uniform"
                or tuple(self.allowed_severities) != (1, 2, 3, 4, 5)):
            raise ValueError("Use uniform selection over the three representatives and severities 1–5.")
        if self.seed not in PLANNED_SEEDS:
            raise ValueError(f"Planned LOFO training seeds are {PLANNED_SEEDS}.")
        if self.run_id != f"seed{self.seed}" and not self.run_id.endswith(f"_seed{self.seed}"):
            raise ValueError("LOFO run_id must be seedN or end with _seedN for this seed.")
        if (self.split_seed != 42 or self.split_sha256 != ORIGINAL_SPLIT_SHA256
                or not self.split_file):
            raise ValueError("LOFO requires the original persisted Part 1 seed-42 split and fingerprint.")
        if self.corruption_implementation != LOFOConfig().corruption_implementation:
            raise ValueError("Unknown corruption implementation/version.")
        baseline = Config()
        for key in ("full_epochs", "batch_size", "learning_rate", "momentum", "weight_decay",
                    "eta_min", "mean", "std"):
            value, reference = getattr(self, key), getattr(baseline, key)
            if key in ("mean", "std"):
                value, reference = list(value), list(reference)
            if value != reference:
                raise ValueError(f"{key} must match the predefined clean baseline for fair comparisons.")


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
        config = (LOFOConfig if "experiment_name" in data else Config)(**data)
    config.validate()
    return config
