# LOFO CIFAR-100-C Replication v3 - practical implementation package

**Status: engineering prototype, not production-certified on Mac M5 or RTX 3050 Ti.**

This is a new, isolated module for the existing `research_cv` project. It does **not** contain the historical repository, training dataset, trained weights or private checkpoints. The old seed-42 results are never overwritten.

## Quick orientation

- 5 distinct seeds: **52, 67, 999, 555, 1666**.
- 6 conditions per seed, both Mac MPS and Windows CUDA: **30 runs per machine, 60 total**.
- 100 epochs / run, CIFAR-adapted ResNet-18, SGD, FP32, batch size 128 unless explicitly revised *before* freeze.
- One run per GPU at a time, start next automatically, checkpoints every epoch, automatically skip verified completions.
- Crash recovery from last completed epoch; previous checkpoint retained for corruption fallback.
- Run-level SHA-256 proofs and platform-aware per-seed analysis.
- Memory-mapped CIFAR-100 training cache and persistent Windows workers avoid repeatedly pickling 150 MB of images every epoch.
- No test data seen by the trainer. CIFAR-100-C evaluation separately gated until a machine completes all 30 runs.
- No GitHub push or cloud storage actions are performed automatically.

## What was actually tested here

- Python compilation of all v3 modules.
- 12 tests passed using this standalone patch. One original-repository integration test is skipped because this workspace lacks the original source tree.
- Tested a real CPU training restart: interrupted a synthetic 2-epoch run at an epoch boundary, resumed, and compared final weights to an uninterrupted synthetic run. The weights matched.
- Did **not** train on real CIFAR-100, run an MPS/CUDA epoch, verify Windows behavior, or evaluate 2.7 GB CIFAR-100-C release arrays in this environment.

**Treat this as candidate code for Codex/local engineering review, not as a proven 60-run autopilot.**

## Installation and setup

Copy this patch into your EXISTING research repository, using `install_patch.py` from the extracted patch folder:

```bash
python install_patch.py --project /Users/rakhman/Desktop/research_cv --apply
```

On Windows the target is the local Git clone path, not a Mac path. The installer refuses to overwrite pre-existing code by default. It never installs dependencies, touches experiment history or runs training.

Run the commands below **from the root of your existing research_cv repository**, never from the extracted patch folder:

```bash
python -m src.replication_v3 plan
python -m pytest -q tests/test_replication_v3*.py
```

Detailed instructions:

- [Scientific protocol and non-causal limitation](docs/REPLICATION_V3_PROTOCOL_REVIEW.md)
- [Mac M5 setup and operation](docs/REPLICATION_V3_MAC.md)
- [Windows RTX 3050 Ti setup and operation](docs/REPLICATION_V3_WINDOWS.md)
- [Unattended operation and recovery](docs/REPLICATION_V3_OPERATIONS.md)

## Deliberate research-design trade-off

Shared operators are matched exactly at probability **1/8 per image** across all conditions that include them. The excluded operator's slot becomes an identity/no-op. This makes LOFO use **37.5% corrupted images** versus **50% in all-four**. The shared-operator exposure is matched, but total augmentation exposure and clean-image frequency are NOT matched. This must be accepted as a limited comparison or expanded with additional control arms. **Freezing requires an explicit flag acknowledging this limitation.**

## Data layout and queue

Training outputs are isolated under:

```
results/replication_v3/mac_m5/seed_52/lofo_noise/
results/replication_v3/win_rtx3050ti/seed_52/lofo_noise/
```

Each production run saves `last_checkpoint.pt`, `previous_checkpoint.pt`, `best_model.pt`, `history.csv`, `corruption_counts.json`, `progress.json`, `completion_manifest.json`, and `train.log`.

Queue state is SQLite on a **local disk**. No active files are written into Google Drive or a network-synced folder. Optional verified-run backups can go to a separate destination.

## Practical constraints

This package depends on existing `src/model.py`, `src/corruptions.py`, `src/benchmark/data.py`, `src/utils.py`, original CIFAR-100 and the verified CIFAR-100-C release. The two devices may have different supported PyTorch versions, so the short native hardware tests and the 20 golden-corruption image hashes are compulsory. If the cross-platform corruption pixels differ, training is intentionally blocked.

A Windows laptop with only 4 GB VRAM may OOM at batch size 128. The runner **will not silently reduce batch size or enable AMP**. Measure before freezing; any revised batch size must be the same for both devices, all 6 conditions and 5 seeds.

Do not edit the frozen protocol, seeds, learning-rate schedule, augmentation slots, split, checkpoints or source during production. Store any new experiment with a new protocol version rather than changing v3 in place.
