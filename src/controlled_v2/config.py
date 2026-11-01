"""Versioned training/evaluation registry, deliberately separate from legacy LOFO."""
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..lofo import ORIGINAL_SPLIT_SHA256, REPRESENTATIVES

VERSION = "controlled_v2.0.0"
ROOT = Path("results/controlled_v2")
EVALUATION_ROOT = Path("results/evaluation_controlled_v2")
# Order is frozen before execution; no wall-time comparisons are planned.
TRAINING_CONDITIONS = {
    "all_four_representatives": (0.5, tuple(REPRESENTATIVES.values())),
    "clean_baseline": (0.0, ()),
}
EVALUATION_MODELS = tuple(TRAINING_CONDITIONS)


@dataclass(frozen=True)
class ControlledConfig(Config):
    protocol_version: str = VERSION
    condition: str = "clean_baseline"
    smoke_test: bool = False
    device: str = "mps"
    use_amp: bool = False
    download: bool = False
    split_file: str = "splits/cifar100_seed42.json"
    split_sha256: str = ORIGINAL_SPLIT_SHA256
    initial_state: str = "results/controlled_v2/shared/initial_seed42.pt"
    corruption_selection: str = "uniform"
    severity_selection: str = "uniform"
    allowed_severities: tuple = (1, 2, 3, 4, 5)
    corruption_implementation: str = "hendrycks_cifar_c_8190fe3_v1"

    @property
    def mode(self):
        return self.condition

    @property
    def corruption_probability(self):
        return TRAINING_CONDITIONS[self.condition][0]

    @property
    def allowed_corruptions(self):
        return TRAINING_CONDITIONS[self.condition][1]

    @property
    def output(self):
        return Path(self.artifact_root) / ROOT / self.condition / self.run_id

    def validate(self):
        super().validate()
        if self.condition not in TRAINING_CONDITIONS:
            raise ValueError("Unknown controlled-v2 condition.")
        reference = ControlledConfig()
        # This bridge is exactly two seed-42 runs. New conditions/seeds require a new freeze.
        for key in self.to_dict():
            if key in {"condition", "data_root", "artifact_root", "split_file", "initial_state"}:
                continue
            value, expected = getattr(self, key), getattr(reference, key)
            if key in {"mean", "std", "allowed_severities"}:
                value, expected = tuple(value), tuple(expected)
            if value != expected:
                raise ValueError(f"{key} is fixed by {VERSION}.")
        expected_initial = Path(self.artifact_root) / ROOT / "shared/initial_seed42.pt"
        if Path(self.initial_state).resolve() != expected_initial.resolve():
            raise ValueError("Use the shared v2 initial state inside the controlled output root.")


def configs():
    return {name: ControlledConfig(condition=name) for name in TRAINING_CONDITIONS}
