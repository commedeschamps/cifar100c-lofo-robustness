# How Far Does Corruption Robustness Transfer? A Leave-One-Family-Out Study on CIFAR-100-C.

Status: **complete**. Complete scientific benchmark: **True**.

Smoke outputs are software checks only. Partial outputs must not support publication aggregates.

## Clean test checks

| Model | Clean test accuracy (%) | Best clean validation (%) |
|---|---:|---:|
| clean_baseline | 76.29 | 76.82 |
| lofo_noise | 73.95 | 73.52 |
| lofo_blur | 72.28 | 72.70 |
| lofo_weather | 71.21 | 72.40 |
| lofo_digital | 73.87 | 73.96 |

Baseline gate passed: **True**; measured minus recorded accuracy: **+0.0000 pp** (fixed tolerance 0.05 pp).

Completed corruption conditions: 375/375. Official corruption dataset available: True. Full evaluation ready: True.

## Held-out family transfer

| Family | LOFO (%) | Baseline (%) | Difference (pp) |
|---|---:|---:|---:|
| noise | 44.15 | 25.51 | +18.63 |
| blur | 58.58 | 46.42 | +12.16 |
| weather | 60.33 | 58.88 | +1.45 |
| digital | 63.32 | 54.45 | +8.87 |

## Interpretation and limitations

The baseline trained on CUDA with AMP; LOFO trained on MPS with FP32. All Part 3 inference uses FP32. Hardware, training precision and library differences limit the comparison.

Positive held-out gains suggest transfer relative to clean-only training; they do not establish universal or causal generalization. The three exposed LOFO models have different training mixtures. Represented operators are not pixel-identical benchmark images.

One training seed cannot estimate training-seed variability. Optional paired bootstrap intervals quantify test-image uncertainty conditional on these trained models, preserving repeated-image clusters. No significance claims are made.

Overall accuracy weights each of the 15 corruptions equally. The separate four-family average weights families equally. Noise has three types; the other families have four. Mean corruption error here is unnormalized.

See manifests/provenance.json and metrics/evaluation.json for content identities, configuration, gate and completion status.
