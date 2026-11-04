# Codex integration / independent engineering review prompt

You are reviewing the attached Replication v3 patch for `/Users/rakhman/Desktop/research_cv`. This is ACTUAL proposed code, not a feature specification. Before any full training:

1. Read `README_REPLICATION_V3.md`, `docs/REPLICATION_V3_PROTOCOL_REVIEW.md`, and `docs/REPLICATION_V3_OPERATIONS.md`.
2. Audit `src/replication_v3/` against actual existing project modules (`src/model.py`, `src/corruptions.py`, `src/utils.py`, `src/benchmark/data.py`). Verify imports, torch serialization, CIFAR inputs, source hashing, cache identity, sampling and saved manifests.
3. Check queue correctness: cross-platform locks, SQLite recovery, subprocess exit codes, stop/pause, completion invariants, backup, retry classification, power interruption.
4. Run available tests. Add short synthetic integration tests on the real Mac. Perform a real MPS 2-epoch smoke using `benchmark`, without running 100 epochs. On the teammate Windows machine, separately verify CUDA and batch-size-128 VRAM with real data.
5. Verify the research-design limitation: matched shared-operator exposure is paired *exactly*, but total corrupted-image fraction is 37.5% in LOFO and 50% in All-Four. This is still a confound. Obtain author approval for this six-arm design or propose a separate matched-regularizer control; do NOT silently change probabilities.
6. Verify identical artifact kit SHA-256, initial weights, split, and all 20 per-operator golden hashes on Mac and Windows.
7. Verify no production training reads CIFAR-100-C test data, checkpoints resume without optimizer/scheduler reset, best checkpoint comes from first maximum clean validation accuracy.
8. Make minimal fixes to any bugs discovered; after edits, recreate frozen source identity/lock BEFORE the production queue.
9. Provide copy-paste-ready macOS and PowerShell commands, report exactly which checks passed and which remain unverified on Windows.
10. Never modify historical seed-42 artifacts, overwrite the existing paper, create a GitHub push, or start the 60 full runs without direct user approval.

Critical: The bundle has standalone unit tests and a synthetic CPU stop/resume test, but its MPS/CUDA operation has not been verified here. DO NOT claim it is production-certified before machine-specific smoke checks succeed.
