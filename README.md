# How Far Does Corruption Robustness Transfer?

**A Leave-One-Family-Out Study on CIFAR-100-C**

Does training with representative corruptions from three families improve robustness to a fourth, excluded family? This repository contains seven completed seed-42 models, their training records, and two separately identified evaluations.

## Data and method

The official CIFAR-100 training set contains 50,000 images. A fixed stratified [split](splits/cifar100_seed42.json) assigns 45,000 to fitting and 5,000 to validation (450/50 per class). The separate official test set contains 10,000 images, 100 per class.

Evaluation uses 15 primary CIFAR-100-C types, each at five severity levels and 10,000 test identities per condition:

| Family | Evaluated types | Training representative |
|---|---|---|
| Noise | gaussian_noise, shot_noise, impulse_noise | gaussian_noise |
| Blur | defocus_blur, glass_blur, motion_blur, zoom_blur | defocus_blur |
| Weather | snow, frost, fog, brightness | fog |
| Digital | contrast, elastic_transform, pixelate, jpeg_compression | jpeg_compression |

The release also contains gaussian_blur, saturate, spatter and speckle_noise; these four additional types are excluded from primary metrics. Dataset fingerprints are preserved with the original data and evaluation manifests.

Each LOFO model excludes one family's representative and trains with the other three. This is representative-operator training, not training on every type in the retained families. After random crop (32 pixels, padding 4) and horizontal flip, corruption is applied with probability 0.5; one allowed operator and severity 1–5 are selected uniformly. Validation remains clean. The all-four control uses all four representatives at the same total probability; clean controls use none. All models are evaluated on all 15 types; the excluded family's mean is the LOFO comparison of interest. See [operator details](docs/CORRUPTIONS.md).

All models use a scratch ResNet-18 with a 3×3, stride-1 input convolution, no initial max-pooling, and 100 outputs (11,220,132 parameters). Training uses 100 epochs, batch 128, SGD (learning rate 0.1, momentum 0.9, weight decay 0.0005), cosine annealing to zero, and cross-entropy. RGB tensors are divided by 255 and normalized with mean (0.5071, 0.4867, 0.4408), standard deviation (0.2675, 0.2565, 0.2761). The selected checkpoint is the first maximum of clean-validation accuracy. The recorded selection rule does not use corruption-test metrics.

## Results

Accuracy is in percent. Overall corruption accuracy is the equally weighted mean of 75 type/severity conditions. These are descriptive results from one training seed.

| Cohort | Model | Best epoch | Validation | Clean test | Overall corruption |
|---|---|---:|---:|---:|---:|
| Historical | Clean baseline (CUDA/AMP) | 90 | 76.82 | 76.29 | 47.70 |
| Historical | LOFO-Noise (MPS/FP32) | 100 | 73.52 | 73.95 | 60.15 |
| Historical | LOFO-Blur (MPS/FP32) | 100 | 72.70 | 72.28 | 63.48 |
| Historical | LOFO-Weather (MPS/FP32) | 91 | 72.40 | 71.21 | 62.40 |
| Historical | LOFO-Digital (MPS/FP32) | 95 | 73.96 | 73.87 | 64.99 |
| Controlled-v2 | Clean baseline (MPS/FP32) | 94 | 77.20 | 76.68 | 47.83 |
| Controlled-v2 | All-four representatives (MPS/FP32) | 98 | 73.34 | 72.68 | 65.20 |

[Full-precision seven-model table](paper/tables/seed42_results.csv), [historical evaluation](results/evaluation/seed42/metrics/evaluation.json), and [controlled-v2 evaluation](results/evaluation_controlled_v2/seed42/metrics/evaluation.json) retain family and condition results. Historical runs are under `results/clean_baseline/clean_seed42/` and `results/lofo_{noise,blur,weather,digital}/seed42/`; the two controls are under `results/controlled_v2/{clean_baseline,all_four_representatives}/seed42/`.

## Reproducibility

`src/`, `configs/`, `splits/` and per-run source/environment records define the implementation. The [frozen v2 protocol](results/controlled_v2/PROTOCOL_v2.0.0.md) specifies the bridge controls. Exact reproduction requires the recorded hardware, precision, package versions and data, not merely compatible current dependencies.

From a Git clone, verify saved metric arithmetic, manuscript table provenance, the split and available frozen inputs using Python 3.10 or later (standard library only):

```sh
python3 -B tools/verify_repository.py
```

The repository includes scientific metadata, histories, source snapshots and figures. Dataset images, checkpoint tensors, prediction caches, machine-specific execution manifests and internal planning records are not distributed through Git. [Artifact identities](docs/reproducibility/models.json), [recorded training parameters](docs/reproducibility/recorded_parameters.json) and [dataset fingerprints](docs/reproducibility/data_identities.json) document the retained full research records. No checkpoint download endpoint is supplied.

In the complete local research archive and recorded environment, verify checkpoint tensors, histories and evaluation completeness without training or inference:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/verify_saved_results.py
```

The full-archive check requires the omitted data, checkpoints and original hash-bound planning files; it is not a standalone Git-clone check. See [evaluation reproduction](docs/EVALUATION.md) for separate-output evaluation and [manuscript instructions](paper/README.md) for IEEE compilation. Scientific figures and tables remain in the canonical project tree; the local `submission/` copy is excluded from Git.

## Limitations

- Only training seed 42 is completed; there is no estimate of training-seed variance or established seed-level significance. Repeated corruptions of 10,000 test identities are not independent training replications.
- The historical baseline used CUDA/AMP; LOFO used MPS/FP32. The v2 controls match the LOFO software stack but use a separately versioned RNG schedule. The four LOFO models were not retrained as part of v2; the seven models are not a fully matched cohort.
- The v2 initializer matches the historical LOFO digest, but historical initial tensors were not retained. Digest equivalence does not prove direct tensor identity or trajectory replay; no comparable initializer digest exists for the historical clean baseline.
- At total corruption probability 0.5, a retained operator has probability 1/6 in LOFO and 1/8 in all-four. Exclusion and dose redistribution are intertwined.
- Transfer is heterogeneous. Historical held-out-weather fog and brightness differences are −8.072 and −5.600 percentage points against the historical clean baseline, despite a positive weather-family mean. Families are taxonomic groups, not physically disjoint effects.
- Official data contain 4 shared exact pixel contents across fitted/validation roles, 8 across fitted/test and 2 across validation/test, including conflicting labels. Indices are disjoint; no deduplication or relabeling was applied. Archived sensitivity analysis changes held-out gains by less than 0.008 percentage points after excluding affected test identities.
- Pilot test outcomes were known before the prospective v2 bridge. The recorded checkpoint rule uses clean validation; artifact inspection cannot reconstruct all unlogged historical human decisions. Conclusions are limited to this architecture and benchmark.
