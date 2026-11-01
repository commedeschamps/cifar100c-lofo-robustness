"""Generate four configs, a lightweight Colab launcher, and a source-only ZIP.

No datasets, results, checkpoints, or baseline logic are duplicated in the launcher.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import LOFOConfig
from src.lofo import FAMILIES, ORIGINAL_SPLIT_SHA256, allowed_for


def build():
    for family in FAMILIES:
        config = LOFOConfig(experiment_name="lofo_" + family, held_out_family=family,
                            allowed_corruptions=allowed_for(family))
        config.validate()
        (ROOT / "configs" / f"lofo_{family}.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    cells = []

    def add(kind, name, source):
        cell = {"cell_type": kind, "id": name, "metadata": {},
                "source": source.lstrip("\n").splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
# Part 2 · Leave-One-Family-Out training

Launcher only: imports the existing Python implementation. It contains no training
logic. In Colab select **Runtime → Change runtime type → GPU**. Upload the generated
`dist/lofo_colab.zip` when prompted (build locally with `python tools/build_lofo_launcher.py`).
The ZIP includes source, configurations, licenses and the **original Part 1 split**;
it excludes all image data and results. A cloned repository can be used instead.

Default mode is smoke. `RUN_TRAINING=False` prevents accidental starts. First run a
smoke check, then explicitly switch to full training. No test or CIFAR-100-C
evaluation is offered in this notebook.
""")
    add("code", "configuration", """
from pathlib import Path
EXPERIMENT = "lofo_noise"  # lofo_noise, lofo_blur, lofo_weather, lofo_digital
SEED = 42                 # Planned seeds: 42, 123, 2026
SMOKE_TEST = True
RUN_TRAINING = False       # Set True explicitly to launch the selected run.
RESUME = False
USE_GOOGLE_DRIVE = True
DEVICE = "auto"            # CUDA → MPS → CPU. Use cpu for local checks.
NUM_WORKERS = 2
IN_COLAB = Path("/content").exists()
PROJECT_ROOT = Path("/content/research_cv") if IN_COLAB else Path.cwd()
DATA_ROOT = Path("/content/data") if IN_COLAB else PROJECT_ROOT / "data"
LOCAL_ARTIFACT_ROOT = PROJECT_ROOT
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/corruption-robustness-research")
""")
    add("code", "project", """
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if not (PROJECT_ROOT / "src/train.py").is_file():
    if not IN_COLAB:
        raise FileNotFoundError("Set PROJECT_ROOT to the research_cv repository.")
    from google.colab import files
    import io, zipfile
    uploaded = files.upload()
    if len(uploaded) != 1 or not next(iter(uploaded)).endswith(".zip"):
        raise ValueError("Upload exactly the generated lofo_colab.zip source bundle.")
    with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe path in ZIP archive.")
        archive.extractall(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
import sys
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import torch, torchvision, numpy, matplotlib, cv2
except (ImportError, OSError, RuntimeError) as exc:
    raise RuntimeError("Use compatible torch/torchvision and install missing requirements-lofo.txt dependencies. "
                       "Colab normally supplies them; do not install multiple OpenCV variants.") from exc
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__, "OpenCV:", cv2.__version__)
""")
    add("code", "storage", """
if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Set USE_GOOGLE_DRIVE=False for local execution.") from exc
    drive.mount("/content/drive")
    ARTIFACT_ROOT = DRIVE_ARTIFACT_ROOT
else:
    ARTIFACT_ROOT = LOCAL_ARTIFACT_ROOT
    if IN_COLAB:
        print("Local Colab artifacts are temporary. Enable Drive before a full run.")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
print("Persistent artifacts:", ARTIFACT_ROOT)
print("Dataset cache:", DATA_ROOT)
""")
    add("markdown", "protocol-notes", """
Each condition excludes one entire family. Allowed representatives are fixed:
Noise→gaussian_noise, Blur→defocus_blur, Weather→fog, Digital→jpeg_compression.
Each training image receives the baseline crop/flip, then with probability 0.5 one
uniformly selected allowed corruption at uniformly selected severity 1–5, then
tensor conversion and normalization. Validation remains clean. Actual consumed
exposure counts are checkpointed and exported to `corruption_counts.json`.

Every model starts from scratch. Full settings match Part 1: SGD 0.1 / momentum
0.9 / weight decay 0.0005, batch 128, 100 epochs, cosine decay, cross entropy and
best clean-validation accuracy. Do not tune separately by held-out benchmark results.
The original split is mandatory and hash-checked. Missing/changed split files fail;
this launcher never generates a replacement.

To resume, restore this exact experiment/seed/settings/artifact location and set
RESUME=True. The incomplete epoch is replayed with restored RNG/worker states.
Cross-hardware/software/codec replay is not guaranteed. Strict deterministic MPS
support depends on installed PyTorch; if unsupported, start a documented CPU run.
""")
    add("code", "effective-config", """
from dataclasses import replace
import json
from src.config import load_config
from src.train import run_directory, run_experiment, plot_history
from src.utils import select_device, environment_info
if EXPERIMENT not in ("lofo_noise", "lofo_blur", "lofo_weather", "lofo_digital"):
    raise ValueError("Choose one of the four LOFO conditions.")
config = replace(load_config(PROJECT_ROOT / "configs" / f"{EXPERIMENT}.json"),
                 seed=SEED, run_id=f"seed{SEED}", smoke_test=SMOKE_TEST,
                 artifact_root=str(ARTIFACT_ROOT), data_root=str(DATA_ROOT),
                 split_file=str(PROJECT_ROOT / "splits/cifar100_seed42.json"),
                 num_workers=NUM_WORKERS, device=DEVICE)
config.validate()
print(json.dumps(config.to_dict(), indent=2))
print(json.dumps(environment_info(select_device(DEVICE)), indent=2))
print("Output:", run_directory(config))
""")
    add("code", "training", """
if RUN_TRAINING:
    output_dir, history = run_experiment(config, resume=RESUME)
    print("SMOKE CHECK ONLY — no scientific findings" if SMOKE_TEST else "Selected full schedule completed.")
else:
    print("Training not started. Set RUN_TRAINING=True explicitly after reviewing the configuration.")
""")
    add("code", "figures", """
if RUN_TRAINING:
    plot_history(history, output_dir / "plots", config.smoke_test, show=True,
                 experiment_name=config.experiment_name)
    print(json.dumps(json.loads((output_dir / "summary.json").read_text()), indent=2))
    print(json.dumps(json.loads((output_dir / "corruption_counts.json").read_text())["totals"], indent=2))
""")
    add("markdown", "artifacts", """
Full runs: `ARTIFACT_ROOT/results/lofo_FAMILY/seedN/`.
Smoke runs: `ARTIFACT_ROOT/results/smoke_test/lofo_FAMILY/seedN/`.
Artifacts retain the baseline layout plus `training.log`, `corruption_counts.json`
and protocol exposure categories for all 15 future evaluation corruptions.
Training seconds include on-the-fly augmentation/loading; compare runtime only
under matched hardware, worker counts and software. This notebook never accesses
the test split or a corruption benchmark, and does not implement AugMix.
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "02_lofo_training.ipynb", "provenance": []},
        "accelerator": "GPU"}}
    (ROOT / "notebooks/02_lofo_training.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    split = json.loads((ROOT / "splits/cifar100_seed42.json").read_text())
    digest = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != ORIGINAL_SPLIT_SHA256:
        raise ValueError("Refusing to package a changed Part 1 split.")
    (ROOT / "dist").mkdir(exist_ok=True)
    paths = [*ROOT.glob("src/*.py"), *ROOT.glob("configs/*.json"), *ROOT.glob("third_party/*.*"),
             ROOT / "splits/cifar100_seed42.json", ROOT / "requirements.txt", ROOT / "requirements-lofo.txt"]
    with zipfile.ZipFile(ROOT / "dist/lofo_colab.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(ROOT)))


if __name__ == "__main__":
    build()
