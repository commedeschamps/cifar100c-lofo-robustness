# CIFAR corruption reference

The selected corruption/helper functions in `cifar_c_reference.py` are extracted
from Dan Hendrycks and Thomas Dietterich's `hendrycks/robustness` repository,
`ImageNet-C/create_c/make_cifar_c.py`, commit
`8190fe329f5f072a06e3a2aea02bb5dda69aed9f` (Apache-2.0; full license alongside).
Source: https://github.com/hendrycks/robustness/blob/8190fe329f5f072a06e3a2aea02bb5dda69aed9f/ImageNet-C/create_c/make_cifar_c.py

Modifications to the extracted reference: unrelated imports/functions and all
dataset-loading/generation code removed; the removed NumPy alias `np.float_`
replaced by its equivalent `np.float64` for NumPy 2 compatibility. No benchmark
images are included. The reference is used only by operator equivalence tests.

`src/corruptions.py` adapts these operators with explicit RNG injection, checked
32×32 RGB PIL inputs/outputs and uint8 serialization. Gaussian/defocus/fog/JPEG
parameters and formulas follow this CIFAR-specific reference. The original
diamond-square helper credits https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py.
Training-time on-the-fly samples are not claimed bit-identical to the released
benchmark: random draws, spatial augmentation, versions and JPEG codecs differ.
