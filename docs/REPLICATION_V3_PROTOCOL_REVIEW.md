# Scientific design review - required before approving Replication v3

## Research question

In specifically configured scratch ResNet-18 runs, how do three representative corruption operators affect performance on the deliberately excluded CIFAR-100-C family relative to a clean baseline and an all-four representative control? This is a **controlled replication after exploratory results**, not an untouched preregistration; the original seed-42 exploratory data were already known before design.

## Fixed grid

5 preselected seeds `52, 67, 999, 555, 1666` × 6 conditions × 2 platforms = 60 train runs, 5 unique training seeds repeated on both platforms. All 6 conditions use identical initial CPU state tensors per seed (copied from Mac in a SHA-256-checked bundle) and the same shuffled image identity order per epoch, keyed by seed/epoch.

Per seed: clean_baseline, lofo_noise, lofo_blur, lofo_weather, lofo_digital, all_four.

## Exposure design and its unavoidable trade-off

A common *8-slot sampling variable* is used for each image presentation, keyed to `(seed, epoch, image_id)`. Slots `0,1,2,3` correspond to Gaussian noise, defocus blur, fog, JPEG, each with probability 1/8; slots 4–7 are clean (4/8).

- All-Four: all four operators receive 1/8 exposure, and the clean branch has 4/8.
- Any LOFO: three retained operators receive 1/8 exposure each, the excluded slot becomes an identity image, and the clean branch has **5/8**.
- Clean baseline: all eight slots become identity images, no corruption branch.

Within the same seed, epoch and image identity, common operators, severities and random operator parameters are sampled from the same deterministic keys. Thus, their **realized operator+severity exposures** are exactly matched across relevant conditions, not only matched in expectation. An audit checks this after all runs.

**Remaining confound:** LOFO has 37.5% corrupted images while all-four has 50%. Holding total augmentation rate fixed would require a replacement operation in the excluded slot; that operation would introduce another treatment. This six-arm design does **not** isolate a pure effect of family omission from the change in overall corruption dose. Transfer ratios remain descriptive. If full dose matching is a publication requirement, add a separate, prospectively defined matched-regularizer control and start a new protocol, not a silent v3 change.

## What the experiment can and cannot establish

- Supports reproducibility of observed contrasts across five initializations on each platform.
- Supports paired within-seed contrasts and cross-platform agreement checks.
- Does not establish causality of a specific perceptual or frequency mechanism.
- Does not automatically isolate generic augmentation regularization from true cross-family transfer.
- Does not establish invariance to all possible representatives or corruption distributions.
- Does not yield 10 unique independent training seeds just because the same 5 are repeated on two devices.
- Does not prove loss of generality to natural shifts, other networks, other datasets or all possible severity levels.

## Training and model selection

Official CIFAR-100 fitting file only. New deterministic, stratified 45k fit / 5k clean validation split: exact pixel contents seen more than once in the 50k train file are kept on the fitting side, and validation is chosen from globally unique-content candidates. Training does not inspect official *test* images to choose the split. This removes exact **fit/validation** overlap, but does not claim removal of all train/test or near-duplicate overlap.

CIFAR-specific ResNet-18, 100 epochs, batch 128, SGD lr 0.1 momentum 0.9 weight decay 0.0005, cosine annealing, no AMP or label smoothing, best checkpoint = first strict maximum clean-validation accuracy. Standard crop pad4, horizontal flip p=.5, corruption occurs after crop+flip. Same normalization as historical work. CPU input transformations derive per-image/per-epoch random streams via BLAKE2b; DataLoader ordering cannot affect individual image augmentation decisions.

Official test evaluation is a **separate explicit phase** after all 30 train runs on a machine pass completion verification. Full CIFAR-100-C: 15 types × 5 severities plus clean. Historical 7 seed-42 models remain untouched and are not used as new replication seeds.

## Cross-platform caveats

MPS and CUDA are distinct compute environments. Identical initial weights, input pixels, split and code do not imply bit-for-bit identical training trajectories. Golden test outputs verify the corruption generation on both hosts; identical PyTorch source algorithm and recorded versions do not remove backend numerical differences. Report individual-platform seed statistics, not a pooled n=10 seed claim.

## Approval gate

Review and accept the 37.5%-versus-50% exposure trade-off, the new train/validation split, exact proposed epochs, batch size, test-only evaluation barrier and limited nature of causal claims. Only after review may the Mac operator run `freeze --confirm-design`. That frozen lock and CPU initialization files are then transferred unmodified to Windows.
