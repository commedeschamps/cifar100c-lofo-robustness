"""Regenerate the standalone notebook and default JSON from canonical src files.

Uses only the Python standard library. Embedded %%writefile cells make the single
notebook sufficient in a fresh Colab session, without downloading repository code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config


def build():
    cells = []

    def add(kind, source, name):
        cell = {"cell_type": kind, "id": name, "metadata": {},
                "source": source.lstrip("\n").splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(source, name):
        add("markdown", source, name)

    def code(source, name):
        add("code", source, name)

    md("""
# 01 · Clean ResNet18 baseline on CIFAR-100

Research: **How Far Does Corruption Robustness Transfer? A Leave-One-Family-Out
Study on CIFAR-100-C**. This notebook implements only the clean control condition.

**Default: a two-epoch smoke test.** Its small-subset results check software and
must never appear as scientific findings. Full training requires changing
`SMOKE_TEST` below. The official clean test has a separate, opt-in final cell.
No CIFAR-100-C data is downloaded or evaluated.

In Colab choose **Runtime → Change runtime type → GPU** (a T4 is sufficient), then
run the cells in order. This notebook is self-contained: source-writing cells
below materialize the exact implementation into a temporary runtime package.
No repository clone or automatic package installation is required.
""", "intro")
    code("""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Before importing torch/CUDA.
try:
    import torch
    import torchvision
    import numpy as np
    import matplotlib
except (ImportError, OSError, RuntimeError) as exc:
    raise RuntimeError(
        "A compatible torch/torchvision pair, NumPy, and Matplotlib are required. "
        "Use a fresh Colab runtime, or install requirements.txt locally. "
        "For binary operator errors reinstall torch and torchvision as a matched pair "
        "using https://pytorch.org/get-started/locally/ and restart the runtime."
    ) from exc
if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
    raise RuntimeError("This notebook requires PyTorch >= 2.3. Update the runtime and restart.")
print("Python / packages imported successfully.")
print("PyTorch:", torch.__version__, "Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available(), "CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}; VRAM: {gpu.total_memory / 2**30:.1f} GiB")
elif torch.backends.mps.is_available():
    print("Apple MPS available. Automatic device selection prefers MPS over CPU.")
else:
    print("CPU mode: suitable for a smoke test. Enable a GPU before full training.")
""", "dependencies")
    md("""
## Experiment settings — edit here

The full schedule is predefined: SGD (LR 0.1, momentum 0.9, weight decay 0.0005),
batch 128, 100 epochs, cosine decay to zero, ordinary cross entropy, seed 42.
These are starting settings, with no claim of optimality. The smoke schedule is
two epochs over **200 training / 100 validation** images, balanced over all classes.
The cosine `T_max` always equals the actual planned epochs for the selected mode.

Use the **same `SPLIT_SEED` and saved split** across later experimental conditions.
Keep `RUN_ID` unchanged to resume; set `RESUME=True`. A fresh run refuses to
overwrite an existing run directory. Use a new run ID for a different experiment.
Mount Drive to preserve artifacts across Colab disconnections; the dataset cache
may remain on the faster ephemeral local disk.
""", "configuration-notes")
    code("""
SMOKE_TEST = True             # False explicitly starts the full 100-epoch schedule.
USE_GOOGLE_DRIVE = False      # True mounts Drive and saves persistent artifacts there.
RESUME = False               # True resumes this mode / RUN_ID from last_checkpoint.pt.
RUN_ID = "seed42"
SPLIT_SEED = 42
SEED = 42

from pathlib import Path
IN_COLAB = Path("/content").exists()
DATA_ROOT = Path("/content/data") if IN_COLAB else Path.cwd() / "data"
LOCAL_ARTIFACT_ROOT = Path("/content/clean_baseline_artifacts") if IN_COLAB else Path.cwd()
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/corruption-robustness-research")

SETTINGS = dict(
    full_epochs=100, smoke_epochs=2,
    smoke_train_per_class=2, smoke_val_per_class=1,
    batch_size=128, learning_rate=0.1, momentum=0.9, weight_decay=5e-4,
    eta_min=0.0, num_workers=2, use_amp=True, deterministic=True, cpu_threads=4, device="auto",
    download=True,
    mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
)
""", "configuration")
    code("""
if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Drive mounting is available in Colab. Set USE_GOOGLE_DRIVE=False locally.") from exc
    drive.mount("/content/drive")
    ARTIFACT_ROOT = DRIVE_ARTIFACT_ROOT
else:
    ARTIFACT_ROOT = LOCAL_ARTIFACT_ROOT
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
import shutil
print("Artifacts:", ARTIFACT_ROOT)
print(f"Free space: {shutil.disk_usage(ARTIFACT_ROOT).free / 2**30:.1f} GiB")
if not USE_GOOGLE_DRIVE and IN_COLAB:
    print("Local Colab storage is temporary. Enable Drive for a persistent full run.")
""", "storage")
    md("""
## Reusable implementation

The following cells contain the maintained Python modules. Run them unchanged.
They write to a temporary package, without overwriting your repository files.
For development, edit `src/*.py` and run `python tools/build_notebook.py`; this
regenerates the notebook from the same sources to avoid diverging implementations.

The CIFAR stem uses a 3×3 convolution at stride 1 with no max pooling. This retains
spatial detail in 32×32 inputs. Standard ResNet18 residual blocks and stage strides
are preserved; the 100-output classifier and all other parameters start from
scratch (`weights=None`).

Training uses only RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, and
fixed conventional CIFAR-100 normalization. Validation/test use only ToTensor
and the same normalization. The quoted constants are fixed, rounded reference
statistics; this experiment never estimates preprocessing values from held-out
images. They are not statistics recomputed on the 45,000-image subset.
""", "implementation-notes")
    code("""
import sys
import tempfile
RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="clean_baseline_"))
RUNTIME_PACKAGE = str(RUNTIME_ROOT / "src")
Path(RUNTIME_PACKAGE).mkdir()
RUNTIME_ASSETS = str(RUNTIME_ROOT / "third_party")
Path(RUNTIME_ASSETS).mkdir()
sys.path.insert(0, str(RUNTIME_ROOT))
# Running all cells again should import the newly materialized sources.
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]
print("Runtime package:", RUNTIME_PACKAGE)
""", "runtime-package")
    for name in ("__init__", "lofo", "config", "utils", "model", "corruptions", "dataset", "train", "evaluate"):
        source = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8")
        code(f"%%writefile $RUNTIME_PACKAGE/{name}.py\n{source}", f"module-{name.strip('_')}")
    for name in ("NOTICE.md", "robustness-LICENSE.txt"):
        source = (ROOT / "third_party" / name).read_text(encoding="utf-8")
        code(f"%%writefile $RUNTIME_ASSETS/{name}\n{source}", "notice-" + name.replace(".", "-"))
    md("""
## Inspect and freeze the effective configuration

The saved `config.json`, environment metadata, split indices, source snapshot,
model description, and checkpoint states make the run auditable. Python, NumPy,
PyTorch, CUDA, and loader workers are seeded. Deterministic algorithms are required;
cuDNN benchmarking and TF32 are disabled. CUDA AMP is enabled on supported GPUs
(compute capability ≥ 7); MPS/CPU use float32. Device order is CUDA → MPS → CPU.

Exact replay is expected only within an unchanged software/hardware environment.
PyTorch does not guarantee identical results across releases, CPU/GPU, or GPU models.
Changing precision or runtime on resume emits a warning and records a new session.
Epoch timing includes loading, train and validation work, and excludes persistence,
setup, plotting, and downtime. All accuracy fields use **percent** (0–100).
""", "reproducibility-notes")
    code("""
from src.config import Config
from src.utils import seed_everything, select_device
from src.model import build_model
from src.train import run_experiment, plot_history, run_directory
import json

config = Config(
    smoke_test=SMOKE_TEST, seed=SEED, split_seed=SPLIT_SEED,
    data_root=str(DATA_ROOT), artifact_root=str(ARTIFACT_ROOT), run_id=RUN_ID, **SETTINGS,
)
config.validate()
print(json.dumps({**config.to_dict(), "effective_epochs": config.epochs,
                  "output_directory": str(run_directory(config))}, indent=2))
# Lightweight architecture check. Training reseeds before constructing its own model.
seed_everything(config.seed, config.deterministic, config.cpu_threads)
check_model = build_model().eval()
with torch.inference_mode():
    check_logits = check_model(torch.zeros(2, 3, 32, 32))
assert tuple(check_logits.shape) == (2, 100), check_logits.shape
assert check_model.conv1.kernel_size == (3, 3) and check_model.conv1.stride == (1, 1)
assert isinstance(check_model.maxpool, torch.nn.Identity)
print("Architecture check passed: [2, 3, 32, 32] → [2, 100]")
del check_model, check_logits
""", "effective-config")
    md("""
## Train and select using validation only

This cell downloads **official CIFAR-100** via torchvision if needed, creates or
validates the saved 45,000/5,000 split (450/50 per class), and trains the selected
mode. Smoke subsets are drawn only from their respective sides of that split.
The official archive includes the clean test file, but this cell never constructs
or evaluates a test dataset. No corruption benchmark is accessed.

The first epoch with the highest validation accuracy becomes `best_model.pt`.
`last_checkpoint.pt` includes optimizer, cosine scheduler, AMP scaler, RNG and
loader states, configuration, and history. Interruptions replay the incomplete
epoch. Resume never resets or shortens the original learning-rate schedule.
Keep the same configuration and run ID, set `RESUME=True`, and rerun all cells.
""", "training-notes")
    code("""
output_dir, history = run_experiment(config, resume=RESUME)
print("Saved artifacts:", output_dir)
print("SMOKE TEST ONLY — not a scientific result" if SMOKE_TEST else "Full schedule completed; validation selection is fixed.")
""", "training")
    code("""
# Readable 300-DPI PNGs are already saved; this also displays the plots inline.
# Later figures can be regenerated from metrics.json without retraining.
saved_history = json.loads((output_dir / "metrics.json").read_text())
plot_history(saved_history, output_dir / "plots", config.smoke_test, show=True)
print(json.dumps(json.loads((output_dir / "summary.json").read_text()), indent=2))
""", "figures")
    md("""
## Final official clean test — explicit opt-in, after full training only

Leave this off during development. After the full schedule finishes and model
selection/settings are fixed, set `RUN_FINAL_TEST=True` and run **this cell**.
The function rejects smoke or incomplete runs and loads the validation-selected
best checkpoint. It evaluates all 10,000 official clean test images in float32 and
saves `clean_test.json`; subsequent calls return the stored result. Never use the
test result to alter the selected configuration or select among seeds.

This task does not run this cell against the test set, and never downloads or
evaluates CIFAR-100-C. Future corruption training must augment only training
images. Future LOFO/AugMix conditions must reuse this architecture, saved split,
normalization, selection rule, and comparable optimization budgets.
""", "final-test-notes")
    code("""
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    from src.evaluate import evaluate_clean_test
    clean_test_result = evaluate_clean_test(config)
    print(json.dumps(clean_test_result, indent=2))
else:
    print("Official test evaluation not requested. No test images were evaluated.")
""", "final-test")
    md("""
## Artifacts and troubleshooting

`ARTIFACT_ROOT/splits/cifar100_seed42.json` is shared across conditions.
Each run lives at `ARTIFACT_ROOT/results/{smoke_test|clean_baseline}/RUN_ID/`:
`config.json`, `environment.json`, `sessions.json`, `protocol.json`, `model.txt`,
`split.json`, `data_manifest.json`, `source/`, `metrics.csv`, `metrics.json`,
`summary.json`, `plots/*.png`, and `checkpoints/{best_model.pt,last_checkpoint.pt}`.
All smoke outputs live beneath `smoke_test` and plot titles/metric rows label them.

If download fails, check internet/disk space or set `DATA_ROOT` to a prepared
torchvision cache and `download=False`. If workers fail in a local interactive
environment, use `num_workers=0` **before starting a fresh run**. If GPU memory is
insufficient, restart with a smaller batch size and a new run ID; record this as a
changed configuration. Enable Drive before long training. Allow roughly 1 GB of
free space for dataset extraction and checkpoints. Keep environment versions fixed
for later conditions; do not silently resume with changed code or settings.

References: [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html),
[torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
""", "artifacts")
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": "01_baseline.ipynb", "provenance": []}, "accelerator": "GPU",
    }, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "notebooks" / "01_baseline.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "configs" / "baseline.json").write_text(
        json.dumps(Config().to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
