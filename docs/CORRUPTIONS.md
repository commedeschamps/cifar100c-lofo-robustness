# Training corruption implementation

The operators adapt the authors' [CIFAR reference source](https://github.com/hendrycks/robustness/blob/8190fe329f5f072a06e3a2aea02bb5dda69aed9f/ImageNet-C/create_c/make_cifar_c.py),
pinned at commit `8190fe329f5f072a06e3a2aea02bb5dda69aed9f`. Its CIFAR severity
settings differ from ImageNet-C defaults. No benchmark images or original
top-level dataset-generation loop are imported.

| Severity | Gaussian std on [0,1] | Defocus (radius, alias sigma) | Fog (strength, wibble decay) | JPEG quality |
| --- | --- | --- | --- | --- |
| 1 | 0.04 | (0.3, 0.4) | (0.2, 3) | 80 |
| 2 | 0.06 | (0.4, 0.5) | (0.5, 3) | 65 |
| 3 | 0.08 | (0.5, 0.6) | (0.75, 2.5) | 58 |
| 4 | 0.09 | (1, 0.2) | (1, 2) | 50 |
| 5 | 0.10 | (1.5, 0.1) | (1.5, 1.75) | 40 |

- Gaussian adds independent normal samples to each RGB value, then clips.
- Defocus builds the reference 17×17 normalized binary disk, antialiases it
  with OpenCV's 3×3 Gaussian kernel and applies per-channel `filter2D` with the
  default reflect-101 boundary. This is the disk operator, not a replacement
  Gaussian image filter. Cached kernels and one OpenCV thread avoid unnecessary
  worker overhead.
- Fog uses the 32×32 diamond-square plasma map, including the reference wibble
  expression and maximum-value rescaling. The black-input behavior is retained.
- JPEG uses Pillow encode/decode at the listed qualities, preserving default
  encoder/subsampling behavior. No external files are written.

Inputs must be 32×32 RGB PIL images. Outputs are clipped, truncated to uint8 as
in benchmark serialization, and returned as 32×32 RGB PIL images before tensor
conversion/normalization. The input is not mutated. Invalid operators, severities,
channels or dimensions raise errors. Operators execute in CPU data-loader workers;
CUDA/MPS train on the resulting tensors.

The port changes obsolete `np.float_` to its equivalent `np.float64`, injects an
RNG, validates inputs, quantizes outputs and returns exposure codes. Tests compare
all 20 operator/severity combinations pixel for pixel with the pinned reference
on random RGB, black and white inputs (**60 comparisons**) under identical
RandomState draws and installed libraries. NumPy, OpenCV, Pillow and JPEG-codec
versions are recorded. This verifies the operators on the tested stack, not
released CIFAR-100-C image identity. Draws, preceding spatial augmentation and
historical library/codec versions can differ. Evaluation uses the released benchmark arrays.

In the original LOFO protocol, NumPy's seeded worker RNG controls clean/type/severity sampling; Python's seeded
worker RNG seeds an independent RandomState per corrupted image; Torch controls
crop/flip. Operator-internal draw counts therefore cannot change future sampling
choices. With the same seed/loader settings, conditions share clean masks,
severity draws and representative-position draws without coupling their operators.
Nonpersistent workers, seeded loader generators and checkpoint RNG restoration
support replay within an unchanged stack. Counts are aggregated from consumed
batches in the main process, so abandoned prefetch does not inflate exposure.
Resume restores the checkpoint's count history and repairs interrupted exports.

Apache-2.0 source attribution, full upstream license, modification notices and
the diamond-square helper credit are retained in
[`third_party/`](../third_party/NOTICE.md). OpenCV also carries an
[Apache-2.0 license](https://github.com/opencv/opencv/blob/4.x/LICENSE); runtime
wheels retain their third-party notices. The retained portable source bundle carries the attribution files. Pillow/libjpeg remain normal runtime dependencies.

Controlled-v2 uses separate worker-local streams for branch, operator and severity draws. Its frozen protocol and source
record this versioned RNG change; it does not establish replay of the original LOFO augmentation trajectory.
