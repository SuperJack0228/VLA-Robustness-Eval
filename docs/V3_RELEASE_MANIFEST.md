# MiniVLA V3 Clean RC1 Release Manifest

## Release Scope

MiniVLA V3 Clean RC1 is the frozen candidate policy for the six Clean Panda
manipulation tasks. It is not yet a final robustness release: formal V3
perturbation curves and closed-loop held-out-language tests remain outstanding.

## Model Configuration

- Architecture: `MiniVLAV2`, architecture version 4
- State: 17 dimensions, 5-step causal history
- Observations: 112 x 112 agent and wrist RGB images
- Control: 7D `OSC_POSE` delta action
- ACT chunk size: 20
- Hidden dimension: 512
- Transformer: 2 encoder layers, 3 decoder layers, 8 heads
- Language model: frozen `distilbert-base-uncased`
- Total parameters: 101,784,921
- Trainable parameters: 33,977,113

## Frozen Artifacts

Generated datasets and checkpoints are intentionally excluded from Git.

| Artifact | Repository-relative location | SHA256 |
| --- | --- | --- |
| V3 policy | `results/hpc/v3_hpc_1349625/mini_vla_v3_policy.pth` | `9fc20e1587472dbe9b78bb635f747c9edfb69d98bc4d03f8185ebadc85ff6429` |
| V3 normalization | `results/hpc/v3_hpc_1349625/dataset_stats_v3.npz` | `429663756f8a0921e3b9ff9c9d4516155f85b43c19dcd221b2af103baa3809dd` |
| Language catalog | `configs/language_augmentations_v3.json` | `90e45afedc008511389b27337a001e441b5e0af3380ea5977ad2673f29d57ca2` |

The frozen DistilBERT weights are not embedded in the policy checkpoint. The
exact pretrained model must be available in the Hugging Face cache when using
`--local-files-only`.

## Data Contract

- V2 Clean: 1,200 accepted episodes, 200 per task
- V3 Recovery: 600 accepted episodes, 100 per task
- Recovery types: 150 forced pick misses, 150 forced push misses, and 300
  target-displacement episodes
- Language: 60 training and 30 disjoint evaluation expressions per task

## Verified Clean Result

Two paired seeds produced 240 episodes per policy:

- V2 Clean: 222 / 240 = 92.50%
- V3: 235 / 240 = 97.92%
- Paired McNemar test: `p = 0.0072`
- V3 task totals: Pick A/B/C 40/40 each, Push A 40/40, Push B 37/40,
  Push C 38/40

The source results are stored locally under
`results/v3_tests/clean_comparison/` and are excluded from Git.

## Verification

Run the lightweight regression suite on macOS with Linux-only MuJoCo variables
removed:

```bash
env -u MUJOCO_GL \
  conda run --no-capture-output -n mujoco310 \
  python -m pytest -q
```

Run the complete V3 data and warm-start gate:

```bash
env -u MUJOCO_GL \
  conda run --no-capture-output -n mujoco310 \
  python scripts/preflight_v3.py \
    --base-data-dir results/dataset_v2_clean \
    --recovery-data-dir results/dataset_v3_recovery \
    --expected-base-episodes 1200 \
    --expected-recovery-episodes 600 \
    --init-policy results/v2_clean/mini_vla_v2_clean_policy.pth \
    --batch-size 32 \
    --num-workers 0 \
    --local-files-only \
    --output results/v3/preflight_report_v3.json
```

Use multiple workers on an unrestricted local shell or HPC compute node.
