# MiniVLA V3 Chunk-Size Ablation Policies

The three independently trained ablation policies are stored below. The
canonical chunk-20 policy is the final V3 policy in `../v3-clean-rc1/`.

- `chunk_1/mini_vla_v3_policy.pth`
- `chunk_5/mini_vla_v3_policy.pth`
- `chunk_10/mini_vla_v3_policy.pth`
- chunk 20: `../v3-clean-rc1/mini_vla_v3_policy.pth`

Each ablation directory retains its normalization statistics, language catalog,
training log, metadata, preflight report, and postflight report. Optimizer-heavy
`best` and `last` resume checkpoints were removed after policy verification.

The final figures and statistical report are in
`final_report/03_chunk_size_ablation/`. Raw paired evaluations are in
`results/benchmarks/chunk_size_ablation/`.
