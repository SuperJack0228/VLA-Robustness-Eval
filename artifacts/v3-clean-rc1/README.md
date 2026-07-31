# MiniVLA V3 Clean RC1 Artifacts

This directory contains the frozen policy and the evidence required to audit
the V3 Clean release candidate.

## Contents

- `mini_vla_v3_policy.pth`: selected Epoch 22 policy checkpoint, stored with
  Git LFS.
- `dataset_stats_v3.npz`: action and state normalization statistics.
- `language_augmentations_v3.json`: language catalog copied from the training
  run.
- `training_log_v3.csv`: per-epoch training and validation metrics.
- `training_metadata_v3.json`: model, optimizer, data, and checkpoint metadata.
- `preflight_report_v3.json`: dataset and warm-start gate from the HPC run.
- `postflight_clean_v3.*`: six-episode training-job smoke evaluation.
- `minivla-v3-1349625.out`: complete Slurm training console log.
- `clean_comparison/`: two paired seeds for V2 and V3, including CSV, JSON,
  and console logs.
- `SHA256SUMS`: checksums for release integrity verification.

## Retrieve LFS Objects

```bash
git lfs install
git lfs pull
```

## Verify Integrity

```bash
cd artifacts/v3-clean-rc1
shasum -a 256 -c SHA256SUMS
```

## Evaluate

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n mujoco310 \
  python scripts/evaluate_policy_v2.py \
    --policy artifacts/v3-clean-rc1/mini_vla_v3_policy.pth \
    --num-episodes 120 \
    --max-steps 200 \
    --seed 20260714 \
    --replan-interval 1 \
    --ensemble-mode temporal \
    --temporal-profile robust \
    --visual-perturbation clean \
    --local-files-only \
    --output-prefix results/v3_release_check
```
