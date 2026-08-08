# Frozen Model Artifacts

- `v2-clean-rc1/`: V2 baseline policy, warm start, normalization, and training
  provenance.
- `v3-clean-rc1/`: final V3 policy, normalization, manifests, training logs,
  and paired Clean evaluation evidence.
- `chunk-ablation-v3/`: chunk 1, 5, and 10 policies plus their training
  provenance. Chunk 20 is the final V3 policy.

Inference-only policy files are retained. Large optimizer-state resume
checkpoints are intentionally excluded after final verification.
