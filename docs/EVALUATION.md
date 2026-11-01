# Verification and evaluation reproduction

The seven completed runs and their two evaluations are immutable research records. Commands below distinguish checking existing evidence from generating a separate evaluation.

## Verify saved artifacts

For the Git-distributed source and saved results, use Python 3.10 or later:

```sh
python3 -B tools/verify_repository.py
```

This standard-library check verifies all seven metric histories, first-maximum validation selection, complete saved evaluation grids, summary arithmetic, table provenance, split indices and hashes of available frozen inputs. It does not require datasets or model tensors. Recorded artifact hashes and portable parameter records are in [reproducibility/](reproducibility/models.json).

For the complete local research repository, including the omitted artifacts and hash-bound planning records, use its recorded environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/verify_saved_results.py
```

This reads all seven best/last checkpoint pairs, histories, configuration and split identities, first-maximum validation selection, finite FP32 tensors, and complete evaluation registries. It performs no model forward pass or parameter update. It does not reconstruct cross-entropy from predicted classes. Prediction caches are scientific evidence and remain preserved.

The Git distribution deliberately omits dataset images, checkpoint tensors, prediction caches, internal planning files and machine-specific execution manifests. They must be restored from the complete research archive before full-artifact verification or inference; no public download endpoint for the model artifacts is supplied. Canonical sources and manuscript files are not duplicated through the local submission directory.

## Reproduce evaluation into a new directory

Full inference requires the complete research repository, original datasets, fourteen best/last checkpoint files, frozen metadata and the recorded MPS/FP32 environment. The compact submission alone is insufficient. The historical clean model remains a CUDA/AMP-trained model even when evaluated on MPS.

The following opt-in command runs the unchanged five-model and two-model evaluators sequentially in a new workspace. Its output must not exist and must be outside the research repository:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/reproduce_evaluation.py \
  --research-root . --output /tmp/research_cv_evaluation_new --execute
```

The wrapper links existing inputs, preserves the frozen relative layout, and creates new evaluation outputs only. It refuses an existing destination; it does not train, download data, replace caches, or resume a completed evaluation. Stdout and stderr are saved in the new directory. Both evaluators require MPS; no backend fallback is introduced. The historical registry retains its 76.29% baseline gate (tolerance 0.05 percentage points); v2 retains its structural checks without that score gate. A rerun is a numerical reproduction, not a new training replication. Exact numerical equality across environments is not guaranteed.

Results are written to `historical/` and `workspace/results/evaluation_controlled_v2/seed42/` below the new destination. Original `results/evaluation/seed42/` and `results/evaluation_controlled_v2/seed42/` are never used as outputs. Do not call the frozen v2 evaluator directly from the completed working tree: its output location is fixed.

## Data and metrics

The official CIFAR-100 training file is split into 45,000 fitted and 5,000 validation images; the official 10,000-image test file is separate. Canonical split SHA-256: `ab04334a218db1266d508e8c9adaaf91d45d81e0f4410739243ef4991fdc3ee9`. CIFAR-100-C archive MD5: `11f0ed0f1191edbf9fa23466ae6021d3`; original receipts retain per-file SHA-256 values.

The 15 primary types have five consecutive severity blocks of 10,000 images with aligned test labels. Four additional released types (gaussian_blur, saturate, spatter, speckle_noise) are excluded. Inference uses FP32, evaluation/inference mode, RGB /255 and recorded normalization, with no resizing, TTA, calibration or adaptation.

Completeness requires 375 corrupted conditions plus 5 clean checks for the historical cohort, and 150 plus 2 for v2; every condition contains 10,000 identities. Overall accuracy weights the 15 types and five severities equally. Family means weight their members and severities equally; the separate four-family mean gives each family equal weight. Error is 100 minus accuracy, not reference-normalized mCE.

## Historical training reference

The original entrypoint is `src.train`; legacy configurations default to smoke mode and require `--full` for the 100-epoch schedule. Per-run source snapshots and environments define the completed experiments. Current default Mac settings do not reproduce the historical CUDA/AMP baseline exactly.

Controlled-v2 is defined by its [immutable protocol](../results/controlled_v2/PROTOCOL_v2.0.0.md), manifests and shared initializer. Its completed directories reject a fresh launch. New training requires a separate experimental version and artifact root; changing frozen code or configuration invalidates provenance.

The full legacy test suite and legacy evaluator preflight include fitting or inference and are not lightweight verification. The frozen controlled-v2 split integration test also requires the local CIFAR-100 training file; it is not a dataset-free clone test. Run only explicitly selected non-training tests when checking a source-only checkout. Scientific limitations and cohort distinctions are summarized in the [main README](../README.md); exact manuscript table inputs are in [paper/tables/](../paper/tables/seed42_results.csv).
