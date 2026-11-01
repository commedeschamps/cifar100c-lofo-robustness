# Controlled bridge protocol v2.0.0

Prospective freeze date: 29 October 2026, Asia/Almaty. Protocol ID: `controlled_v2.0.0`.
Stage: two seed-42 bridge controls following `results/research_audit/RESEARCH_AUDIT.md`, sections 8–9.
No training or test inference is authorized by preparation of this protocol.
The exact freeze timestamp and SHA-256 identities are committed to `freeze.json` before training.

## Question, scope and run grid

The exploratory pilot has already revealed CIFAR-100-C outcomes. These audit-motivated controls address the historical
baseline's CUDA/AMP environment confound and the missing all-representative comparison. They are prospective controls
after an observed pilot, not an untouched confirmatory study. No improvement, significance or power is promised.

The entire executable v2.0.0 grid is two runs, training seed 42, fixed data-split seed 42. Frozen execution order:

| Order | Condition | P(corrupt) | Operators conditional on corruption |
|---|---|---:|---|
| 1 | all_four_representatives | 0.5 | gaussian_noise, defocus_blur, fog, jpeg_compression, each 1/4 |
| 2 | clean_baseline | 0 | none |

No LOFO retraining, new seed, dose-matched condition, hyperparameter search or clean-training inference is part of this stage.
The audit's future six-condition/five-seed grid remains future work requiring another freeze and authorization.
This bridge alone does not establish training-seed uncertainty or make the legacy cohort fully controlled.
Run order is fixed before execution. No timing comparison is planned; no random-order or thermal-control claim is made.
Record interruptions and execution sessions. Hardware/software execution remains subject to MPS reproducibility limits.

## Fixed training contract

- Official CIFAR-100 **training file only**. Reuse `splits/cifar100_seed42.json` without regeneration or changes.
  Exactly 45,000 fitted / 5,000 clean validation indices, 450/50 per fine class, complete disjoint index partition.
  Canonical split SHA-256: `ab04334a218db1266d508e8c9adaaf91d45d81e0f4410739243ef4991fdc3ee9`.
  Persisted split file SHA-256: `f97c88d2ad9d118c98a9f0cc9229a3294b4181f6764b83794d577fddae15a617`.
- Both conditions: RandomCrop(32,padding=4), RandomHorizontalFlip(p=0.5), then zero or one corruption,
  then RGB ToTensor FP32 /255 and Normalize(mean=(0.5071,0.4867,0.4408), std=(0.2675,0.2565,0.2761)).
  The clean branch still receives crop/flip. Validation receives only ToTensor and the same normalization.
- All-four: independent Bernoulli 0.5, uniform one-of-four operator, uniform severity 1–5. No compositions.
  P(clean)=1/2; P(each operator)=1/8; P(each operator,severity)=1/40. No severity/type/test tuning.
  The clean condition has no generated corruption. Both conditions export consumed-batch exposure audits.
- Reuse the unchanged, pinned CIFAR-reference corruption bodies, quantization, OpenCV and Pillow JPEG defaults.
  Reference commit `8190fe329f5f072a06e3a2aea02bb5dda69aed9f`; original notices retained.
  “All-family” means **four representatives**, not all fifteen benchmark types.
- torchvision ResNet18(weights=None,num_classes=100), 3x3 stride-1 no-bias stem, Identity maxpool,
  11,220,132 parameters. No pretrained weights or architecture change.
- SGD(lr=0.1,momentum=0.9,weight_decay=0.0005,dampening=0,nesterov=False,maximize=False).
  CrossEntropyLoss with default mean reduction and no label smoothing.
  CosineAnnealingLR(T_max=100,eta_min=0), stepped after each completed epoch. Exactly 100 epochs.
  Batch 128, two nonpersistent seeded workers, shuffle fitted data, no drop_last (352 updates/epoch), CPU threads 4.
- **MPS/FP32**, effective AMP disabled. No auto-device selection or CPU/CUDA fallback. Deterministic algorithms enabled,
  TF32 disabled. Match all four LOFO environment manifests: Python 3.14.5 (same build), macOS 26.5.2 arm64,
  torch 2.14.1, torchvision 0.29.1, NumPy 2.5.4, OpenCV 5.0.0 (distribution 5.0.0.93), Pillow 13.0.0,
  JPEG codec 6.2. Full installed distributions and existing lock are saved/hashed; no package installation or upgrade.
  `check_stack` rejects changes to recorded software/platform/operator metadata. Device visibility is checked separately.

## Initialization and random streams

Create one CPU state using seed 42 and the unchanged historical model factory; save `shared/initial_seed42.pt` before training.
Verify serialization round-trip and canonical model-state digest, then load this file into **both** conditions.
Verify again after transfer to MPS; do not train a freshly recreated replacement. Hash model tensors and the serialized file
separately. The historical LOFO digest is
`fe7a5fa407bd0a5669d4cca9949a6cda1f9af4c7145e368d956d1c12b414a50b`.
The freeze records the measured comparison against all four saved LOFO protocols. If it differs, do not claim matching.
If it matches, claim digest equivalence under the historical `model_digest` algorithm only: historical initial tensors were
not saved, preventing direct tensor comparison. The historical CUDA/AMP clean baseline has no comparable initial digest.
A matching initial digest does not establish historical trajectory replay or exchangeability.

V2 deliberately versions its RNG schedule rather than silently modifying legacy LOFO semantics. Initialization is CPU-seeded;
loader generators use training seed and seed+1; Torch controls crop/flip. Each worker uses separate NumPy RandomState streams
for Bernoulli, operator slot and severity, seeded from worker Torch initial seed plus offsets 0, 100003 and 200003 modulo 2^32.
Every image draws each stream even on the clean branch; Python draws an independent operator seed every image. Operator
internals receive a fresh RandomState. Thus internal corruption draws cannot perturb later selection/spatial randomness.
Both conditions share loader order and spatial RNG draws. Workers restart each epoch, so loader-generator and main RNG states
are sufficient for epoch-boundary replay. Do not claim identical v2 augmentation trajectories to historical LOFO sampling.

## Selection, interruption and provenance

Select the **first** epoch attaining maximum clean-validation accuracy (strict `>`). No early stopping or validation-loss tiebreak.
CIFAR-100-C and clean test outcomes never enter training, scheduler, checkpoint selection, run retention or retry decisions.
Keep every completed run, including poor outcomes. Save both best and last atomically at each completed epoch, with the last
checkpoint authoritative after interrupted exports. Count actual optimizer step calls; with FP32 and no scaler all 352 must
succeed. Retain all 100 histories and per-image-operator/severity counts, initial/best/last hashes, manifests and RNG states.

Resume only an unchanged frozen protocol, exact config/data/classes/initial identity and the same runtime. Restore model,
optimizer, 100-epoch scheduler, all main RNG and loader-generator states. A partial epoch is replayed from the previous boundary;
its discarded computation is not part of the committed history. Recover a crash between last and best writes only when last
itself is the selected best. A missing earlier best requires backup restoration; never substitute the last model.
Retry technical failures only, recording the cause in `results/controlled_v2/retries.jsonl` and retaining session/failure logs.
No reruns because of accuracy. A failure before the first checkpoint requires preserving its directory under
`results/controlled_v2/failed_attempts/` before a same-config restart; later failures use --resume.
Any config/code change after freeze requires a **new version/output namespace**; never rewrite this freeze to permit resume.

The freeze hashes source recursively, the v2 test file/configs, protocol, lock, packages, official train/meta files, split,
audit report and historical environment/protocol/operator/model evidence. Source/config snapshots are copied under
`frozen_inputs/`. Runtime loading verifies the inventory and hashes before work. New outputs are confined to
`results/controlled_v2` and `results/evaluation_controlled_v2`; legacy checkpoints, metrics, caches and audit files are read-only.

## Evaluation and analysis, only after separate authorization

The versioned evaluator admits exactly these two completed v2 models, not the five-model legacy registry.
Require 100 epochs, matching split/class/source/initial/stack manifests, finite FP32 weights and the first-best clean-validation
checkpoint before any model inference. There is no requirement that the new baseline reproduce the historical 76.29% score:
that was a different training stack. A test-accuracy gate must not reject a valid new baseline for its performance.

Frozen eval-mode models and inference_mode, MPS/FP32, batch 128, RGB/no resizing/TTA/adaptation. Evaluate the full official
10,000 clean test images and 15 official primary corruption types x five severities x 10,000 identities per model.
Exclude gaussian_blur, saturate, spatter and speckle_noise. Authenticate arrays/labels with the official existing provenance
receipt and file SHA-256 before inference. Identity-bound per-condition caches include checkpoint/data/protocol hashes;
reject changed/incomplete cache identity and never use legacy caches. Completeness requires 150 corrupted conditions plus
two clean checks; partial outputs carry no complete/scientific-result status.

Report both clean accuracies; all type/severity cells (including negative differences); macro accuracy/error over the 75
conditions, separate four-family equal-weight mean, family means, and paired all-four minus matched-clean differences.
These are descriptive seed-42 bridge outcomes. No training-seed SD/CI, hypothesis test or superiority claim is planned.
Do not combine a bridge family mean with different historical LOFO models and label it as one deployed model.
Comparison to the historical LOFO runs must disclose mixed age and the different v2 RNG schedule; a complete paired
multi-seed cohort remains necessary. Inclusion changes retained-operator dose: 1/8 each here vs 1/6 in historical LOFO.
This is not a pure target-inclusion intervention. No benchmark-derived parameter changes or checkpoint reselection.

Keep the official primary split and scores. Disclose audit findings of 4/8/2 shared exact pixel contents across fitted/validation,
fitted/test and validation/test roles, including conflicting labels. Do not silently deduplicate or relabel in this bridge.
No additional sensitivity inference, bootstrap or clean-training diagnostic is included in this computational budget.

## Computational scope and preparation evidence

Per run: 4,500,000 fitted image presentations; 500,000 validation presentations; 35,200 optimizer updates.
Two runs: 9,000,000 fitted and 1,000,000 validation presentations; 70,400 updates. Full later evaluation: 1,520,000 predictions.
No runtime estimate is promised before execution; run sequentially on the existing machine and retain actual timing.
Preparation uses no model forward/backward pass, optimizer parameter update or test metric inspection.
Focused tests simulate resume with mocked epoch outputs and no gradients; they do not fit a model.
Preflight loads the official training file/split, transforms 256 fitted images per condition and one validation image twice,
verifies shared initial tensors (including MPS transfer when available), checks manifests, and reports pending checkpoints.
The evaluation preflight reads no test arrays and runs no inference. All preparation is clearly distinct from scientific results.
