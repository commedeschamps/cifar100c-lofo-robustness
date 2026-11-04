# MacBook Pro M5 (16 GB) - setup and overnight use

Run from your existing repo root, NOT the patch extraction directory.

## One-time prerequisite

Install v3 patch with `install_patch.py`, and use a local SSD, AC power and a Python virtual environment. Install a compatible PyTorch/torchvision pair with MPS support according to official PyTorch instructions. Then install existing project requirements and v3 requirements:

```bash
cd /Users/rakhman/Desktop/research_cv
python3 -m venv .venv-v3
source .venv-v3/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-lofo.txt -r requirements-replication-v3.txt
python -m pytest -q tests/test_replication_v3*.py
python -m src.replication_v3 plan
```

Check that `torch.backends.mps.is_available()` is true and that CIFAR-100 is already installed in `data/cifar-100-python/`. The official CIFAR-100 test file may also be installed, but the *trainer never loads it*.

## Prepare the same experimental data for both machines

```bash
python -m src.replication_v3 prepare-cache
python -m src.replication_v3 prepare-split
python -m src.replication_v3 prepare-inits
python -m src.replication_v3 make-golden
```

This creates one exact-content-disjoint 45k/5k split, 5 CPU initialization tensors and 20 golden corruption fingerprints. CPU initial tensors may total hundreds of MB and must be copied as files to Windows, not regenerated separately.

**Important:** Before freeze, review the scientific design document. Commit your reviewed v3 source changes *locally* (without pushing) so `git status --porcelain --untracked-files=no` is clean. Then:

```bash
python -m src.replication_v3 freeze --confirm-design
python -m src.replication_v3 export-kit --to "$HOME/Desktop/lofo_v3_frozen_kit.zip"
```

Copy the entire ZIP safely to your teammate. Both machines must use the **same committed Git source**, same frozen ZIP and the official CIFAR-100 dataset. After freeze, do not edit scientific code; any edit requires a new version/freeze.

## Validate on real MPS BEFORE production

```bash
python -m src.replication_v3 doctor --machine mac_m5 --device mps
python -m src.replication_v3 benchmark --machine mac_m5 --device mps --workers 2
python -m src.replication_v3 status --machine mac_m5
```

The benchmark is a small synthetic-duration *protocol smoke run* over real training images, 2 epochs with only a few batches each. It is not a seed-level scientific run, and its output is isolated under `results/replication_v3_smoke/`.

Test file integrity, manual pause, and retry behavior in a smoke environment. Confirm the benchmark does not hit MPS unsupported ops. The Windows machine must pass an analogous smoke test.

## Overnight training (ONLY AFTER BOTH MACHINES PASS)

```bash
caffeinate -dimsu python -m src.replication_v3 start \
  --machine mac_m5 --device mps --workers 2 --confirm-production
```

Leave AC power connected. Do not close the lid unless the specific supported display/power configuration allows operation. If macOS sleeps or the machine is switched off, training stops; on restart use the same command with `resume` replacing `start`. The queue verifies earlier runs and resumes from epoch checkpoints.

The local status page is `results/replication_v3/mac_m5/dashboard.html`; it updates while the queue is running. `python -m src.replication_v3 status --machine mac_m5` shows a text snapshot.

Pause safely from a *second Terminal window*:

```bash
python -m src.replication_v3 pause --machine mac_m5
```

To resume after the worker has exited:

```bash
caffeinate -dimsu python -m src.replication_v3 resume \
  --machine mac_m5 --device mps --workers 2 --confirm-production
```

To back up only verified runs (optional):

```bash
python -m src.replication_v3 backup --machine mac_m5 --to /Volumes/YourExternalDrive/LOFO_v3_backup
```

After 30/30 complete, perform official test evaluation separately:

```bash
python -m src.replication_v3 verify --machine mac_m5
python -m src.replication_v3 evaluate --machine mac_m5 --device mps
```

Full evaluation has 76 conditions for every run and may take significant additional time.
