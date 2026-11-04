# Operations, recovery and final analysis

## Expected status files

- `results/replication_v3/{machine}/queue.sqlite3`: local persistent run state. Never sync an active SQLite database to Drive.
- `results/replication_v3/{machine}/dashboard.html`: refreshable local status snapshot.
- `results/replication_v3/{machine}/seed_{seed}/{condition}/run_manifest.json`: immutable identity for checkpoint resume.
- `checkpoints/last_checkpoint.pt`: state after the latest completed epoch.
- `checkpoints/previous_checkpoint.pt`: independent previous-epoch backup.
- `checkpoints/best_model.pt`: weights from first highest clean validation epoch.
- `progress.json`: latest epoch and heartbeat timestamp.
- `train.log`: per-attempt stdout and error tracebacks, appended.
- `history.csv`, `corruption_counts.json`: complete epoch logs.
- `completion_manifest.json`: artifact SHA-256 verification and 100-epoch completion proof.

## Startup and interruption

`start` and `resume` both validate the frozen protocol, confirm the selected GPU is available, lock the queue, and then run its predefined six-condition grid for each of five seeds. Each run executes in a fresh process. Verified completed runs are skipped. A partially completed run reloads model/optimizer/scheduler/RNG and continues from the next epoch.

`pause` creates a file read by the worker at epoch boundaries. The worker writes the completed epoch checkpoint first. When a machine is hard-powered off, only the currently executing epoch may be lost. The program does **not** continue while a machine is off, and OS automatic resume on reboot is not installed by this patch.

A worker that exits successfully but lacks a verified completion manifest is treated as failed, not completed. Fatal failures such as OOM, frozen configuration mismatch, disk-full, corrupted irrecoverable artifacts, missing dataset or non-finite loss stop the queue instead of silently modifying scientific parameters. Unexpected process failures are retried once by default using a verified checkpoint.

The most recently completed checkpoint is backed up as `previous_checkpoint.pt` before every new checkpoint publication. If the last checkpoint is unreadable, the previous one may be used, with a recovery record. If both are bad, the program stops.

## Safe nightly checklist

Before leaving either machine unattended:

1. The *same* code and frozen kit are installed and `doctor` passes.
2. The actual MPS/CUDA `benchmark` succeeds with 2 workers and batch 128.
3. Check 20 golden corruption hashes and same initial state file hashes.
4. AC power, clear ventilation, no scheduled OS update/reboot, sleep disabled while plugged in.
5. At least 12 GB free disk before queue start, and ideally >=20 GB free beyond dataset storage.
6. A complete stop/resume smoke drill is performed before full training.
7. Confirm the queue shows the expected 5 seeds and 6 conditions, and no old v2 output paths.

## Backup and sync

`backup --to <separate directory>` copies only verified completed run directories. Files are copied into staging, individually hash-checked, and then atomically promoted. The optional queue argument `--backup-to <path>` attempts backup after each run; a failed backup does not block the next GPU training run. Do not use a network-mounted cloud-sync location as the **active** checkpoint root.

Large checkpoints/prediction caches belong in local or Drive storage (and preferably redundant backup), not ordinary Git. Git tracks code, split, frozen protocol, compact per-seed summaries and citation-ready tables. No automatic pushing is implemented.

## Finishing training and evaluation

After 30 verified training runs per machine, perform evaluation as an explicitly separate stage; it reads official CIFAR-100 test and 75 CIFAR-100-C corruption/severity arrays per model, saving compact per-image prediction caches. It supports interruption/resume at the individual condition level and verifies array identity and correct counts.

Do not compare test results mid-training to pick epochs, seeds, operators, doses, or other protocol settings. The protocol and number of runs are predetermined.

Once Windows results are copied under `results/replication_v3/win_rtx3050ti/` on Mac, execute:

```bash
python -m src.replication_v3 aggregate --machines mac_m5 win_rtx3050ti
```

Outputs:

- `seed_level_model_results.csv` - accuracy per machine/seed/condition.
- `paired_contrasts.csv` - held-out gains and transfer ratios per machine/seed/family.
- `platform_model_summary.csv` - mean, sample SD, min/max, n=5, per machine and condition.
- `platform_paired_summary.csv` - paired held-out differences with n=5 per platform, plus sample SD and approximate t intervals (strong n=5 assumptions).
- `convergence_diagnostics.csv` - best epochs, final validation and loss changes during the last ten epochs; descriptive, not proof of convergence.
- `matched_seed_platform_differences.csv` - within-seed Mac-vs-Windows contrasts.
- `analysis_manifest.json` - explicit counts and limitations.

The aggregator checks that the realized operator/severity exposure counts are exactly matched among conditions sharing that operator. If not, it stops and reports the first mismatch.

A bootstrap over test images, when used, is *secondary* to variability across the five independent training seeds. Do not call two hardware repetitions '10 independent seeds'.

## Readiness limitations

This patch was checked with Python compilation, lightweight unit tests and a tiny CPU interruption-resume simulation. It has **not yet passed a real CIFAR-100 training smoke on MPS or RTX 3050 Ti, a full CIFAR-100-C evaluation, or the final cross-platform bundle exchange**. Verify those before depending on the unattended queue for the deadline.
