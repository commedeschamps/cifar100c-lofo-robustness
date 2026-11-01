# How Far Does Corruption Robustness Transfer?

**A Leave-One-Family-Out Study on CIFAR-100-C**

This repository contains the code, saved experimental results and manuscript for an unpublished study of corruption-robustness transfer. Seven scratch-trained CIFAR-adapted ResNet-18 runs compare representative-operator leave-one-family-out (LOFO) augmentation with clean-only training and an all-four-representative control. Results are reported by corruption family, type and severity, including negative transfer and the cost in clean accuracy.

## Research question

Does training with representatives from three corruption families improve classification on a fourth family excluded from explicit augmentation?

## Data and method

[CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) contains 50,000 training and 10,000 test images in 100 classes. The persisted seed-42 [split](splits/cifar100_seed42.json) assigns 45,000 training-file images to fitting and 5,000 to clean validation, exactly 450/50 per class. Evaluation uses the released [CIFAR-100-C version 1](https://zenodo.org/records/3555552) arrays: 15 primary corruption types at five severities, each applied to the same 10,000 test identities.

| Family | Evaluated types | Training representative |
|---|---|---|
| Noise | Gaussian noise, shot noise, impulse noise | Gaussian noise |
| Blur | Defocus, glass, motion, zoom blur | Defocus blur |
| Weather | Snow, frost, fog, brightness | Fog |
| Digital | Contrast, elastic transform, pixelate, JPEG | JPEG compression |

Four additional released types—Gaussian blur, saturate, spatter and speckle noise—are excluded from primary metrics.

Each LOFO model omits one family's representative and trains with the remaining three. After random crop and horizontal flip, one allowed operator is applied with probability 0.5, with operator and severity sampled uniformly. The all-four control uses all four representatives at the same total probability; clean controls use no corruption augmentation. See [operator definitions](docs/CORRUPTIONS.md).

Models have a 3×3, stride-1 stem, no initial max-pooling, 100 outputs and 11,220,132 parameters. All runs use 100 epochs, batch size 128, SGD with learning rate 0.1, momentum 0.9 and weight decay 0.0005, cosine annealing and cross-entropy. Checkpoints are selected by the first maximum of clean-validation accuracy.

## Results

The **primary comparator is the controlled-v2 MPS/FP32 clean baseline**. Accuracy is in percent; overall corrupted accuracy averages all 75 type–severity conditions equally.

| Model | Clean | Corrupted |
|---|---:|---:|
| Primary clean baseline, MPS/FP32 | 76.68 | 47.83 |
| LOFO-Noise | 73.95 | 60.15 |
| LOFO-Blur | 72.28 | 63.48 |
| LOFO-Weather | 71.21 | 62.40 |
| LOFO-Digital | 73.87 | 64.99 |
| All-four representatives | 72.68 | 65.20 |
| Historical clean reference, CUDA/AMP | 76.29 | 47.70 |

Held-out-family results use the corresponding LOFO network and the primary baseline:

| Held-out family | Primary baseline | LOFO | Gain (pp) |
|---|---:|---:|---:|
| Noise | 25.91 | 44.15 | +18.23 |
| Blur | 46.33 | 58.58 | +12.24 |
| Weather | 59.38 | 60.33 | +0.95 |
| Digital | 54.23 | 63.32 | +9.09 |

All-four training gains 17.36 pp in overall corrupted accuracy and loses 4.00 pp in clean accuracy. Transfer is uneven: glass blur gains 31.51 pp, but fog loses 8.23 pp and brightness loses 5.88 pp. Fourteen of the 75 held-out type–severity differences are negative.

[Full-precision model summary](paper/tables/seed42_results.csv), [held-out comparisons](paper/tables/held_out_transfer.csv) and [per-type comparisons](paper/tables/per_corruption_transfer.csv) retain the underlying values. Historical evaluation files and plots under `results/evaluation/` use the historical clean comparator; current manuscript comparisons use the primary baseline.

## Repository structure

```text
configs/        Training configurations
src/            Training, augmentation and evaluation implementations
splits/         Persisted fitting/validation indices
results/        Saved metrics, histories, protocols and source snapshots
notebooks/      Baseline and LOFO training notebooks
paper/          ELCVIA manuscript, figures, tables and asset generator
docs/           Operator details, artifact identities and reproduction instructions
tools/          Saved-result verification and reproduction utilities
tests/          Protocol and output-isolation tests
third_party/    Corruption reference code and license notices
```

## Setup and reproducibility

Saved-result checks require Python 3.10 or later and the standard library only:

```sh
python3 -B tools/verify_repository.py
python3 -B tools/verify_manuscript.py
python3 -B -m unittest discover -s tests -p test_submission.py
```

These commands check existing records without fitting models or running inference. The full test suite also contains model-execution tests; it is not the lightweight verification command above.

To regenerate manuscript figures and tables from saved correct counts:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
.venv/bin/python paper/generate_assets.py
```

Research-code dependencies are listed in [requirements-lofo.txt](requirements-lofo.txt). Per-run environment records and [requirements-local-lock.txt](requirements-local-lock.txt) preserve the completed environment; a compatible installation alone does not reproduce a training trajectory. [Evaluation instructions](docs/EVALUATION.md) describe the original artifact requirements and isolated-output reproduction command.

## Manuscript and citation

The complete [manuscript source](paper/main.tex) follows the official **ELCVIA** LaTeX article template. [Compilation instructions](paper/README.md), vector figures, tables and twelve references are included. The article is unpublished; no acceptance or publication is claimed. Citation metadata are provided in [CITATION.cff](CITATION.cff).

## Limitations

- One training seed (42) per condition; repeated corruptions do not estimate training-seed variability.
- The primary baseline matches MPS/FP32, but legacy LOFO and controlled-v2 RNG schedules differ. Pilot outcomes were known before the v2 protocol was frozen.
- A retained operator has probability 1/6 in LOFO versus 1/8 in all-four, so the control is not dose matched. No augmentation-probability ablation was run. LOFO-Noise and LOFO-Blur select epoch 100 at the budget boundary.
- Official data contain exact-content overlaps across disjoint index partitions. Excluding ten affected test identities changes primary held-out gains by less than 0.006 pp; validation-selection effects and near-duplicates remain unresolved.
- Four representatives, one architecture and one synthetic benchmark limit generalization. The study establishes descriptive transfer patterns, not a universal robustness mechanism or state-of-the-art result.

## Code and data availability

The repository includes source code, configurations, the split, training histories, frozen source snapshots, saved evaluation counts, manuscript assets and artifact fingerprints. Dataset images, checkpoint tensors and prediction caches remain in the full local research record and are not distributed here. No public model-download endpoint is supplied. Dataset sources are linked above; [artifact identities](docs/reproducibility/models.json) and [data fingerprints](docs/reproducibility/data_identities.json) identify the retained records.

Third-party notices are preserved in [third_party/](third_party/NOTICE.md) and [paper/THIRD_PARTY.md](paper/THIRD_PARTY.md). A license for the original project code has not been selected.
