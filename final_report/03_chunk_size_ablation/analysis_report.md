# MiniVLA V3 ACT Chunk-Size Ablation

## Protocol Audit

- Status: PASS.
- Raw episode rows: 1920.
- Paired scene conditions: 480.
- Two evaluation seeds; 120 paired episodes per chunk/mode/level.
- Clean and 4 cm target displacement; injection compliance at 4 cm: 100%.
- No privileged state was used to assist policy execution.
- Chunk 20 is the frozen V3 full-warm-start reference; chunks 1/5/10 use reinitialized action queries.

## Success Rates

| Chunk | Temporal Clean | Temporal 4 cm | Latest Clean | Latest 4 cm |
|---:|---:|---:|---:|---:|
| 1 | 96.67% | 85.00% | 96.67% | 85.00% |
| 5 | 97.50% | 89.17% | 95.83% | 88.33% |
| 10 | 98.33% | 95.00% | 99.17% | 92.50% |
| 20 | 99.17% | 90.83% | 97.50% | 94.17% |

## Statistical Tests

- temporal at 0 cm: Cochran Q=3.750, df=3, p=0.2898.
- temporal at 4 cm: Cochran Q=11.100, df=3, p=0.0112.
- latest-only at 0 cm: Cochran Q=5.000, df=3, p=0.1718.
- latest-only at 4 cm: Cochran Q=13.059, df=3, p=0.004511.

Pairwise exact McNemar comparisons against chunk 20:

| Mode | Level | Candidate | Difference vs 20 | Holm p |
|---|---:|---:|---:|---:|
| temporal | 0 cm | 1 | -2.50 pp | 0.75 |
| temporal | 0 cm | 5 | -1.67 pp | 1 |
| temporal | 0 cm | 10 | -0.83 pp | 1 |
| temporal | 4 cm | 1 | -5.83 pp | 0.5391 |
| temporal | 4 cm | 5 | -1.67 pp | 0.7744 |
| temporal | 4 cm | 10 | +4.17 pp | 0.5391 |
| latest-only | 0 cm | 1 | -0.83 pp | 1 |
| latest-only | 0 cm | 5 | -1.67 pp | 1 |
| latest-only | 0 cm | 10 | +1.67 pp | 1 |
| latest-only | 4 cm | 1 | -9.17 pp | 0.01025 |
| latest-only | 4 cm | 5 | -5.83 pp | 0.1309 |
| latest-only | 4 cm | 10 | -1.67 pp | 0.625 |

## Conclusions

1. Clean success is high for every chunk (95.83%-99.17%); the overall chunk effect is not significant in either execution mode.
2. At 4 cm, chunk size has a significant overall effect: temporal p=0.0112 and latest-only p=0.00451.
3. The prespecified simple hypothesis 'shorter chunks are more dynamically robust' is not supported. Chunk 1 is the weakest at 4 cm (85.0%), while chunk 10 is best under classic temporal ensembling (95.0%).
4. Chunk 20 is best under latest-only (94.17%) but falls to 90.83% under classic temporal ensembling. The -3.33 pp temporal difference is directionally consistent with historical inertia but is not statistically significant (McNemar p=0.388).
5. Chunk 10 provides the best measured Clean/4 cm temporal balance. Its 95.0% 4 cm rate exceeds chunk 20 by 4.17 pp, but the paired difference is not individually significant after correction.
6. Temporal contributor count rises strongly with chunk size, confirming that longer chunks integrate older predictions; performance nevertheless follows a non-monotonic optimum rather than a simple linear degradation.
7. Because chunk 20 inherited V2 action queries while chunks 1/5/10 reinitialized them, the study is an operational ablation with a documented initialization confound.

## Figure Guide

- `01_chunk_success_clean_vs_4cm`: primary success result.
- `02_robustness_retention_and_drop`: normalized displacement cost.
- `03_clean_robustness_pareto`: operating-point comparison.
- `04_temporal_ensembling_effect`: temporal minus latest-only effect.
- `05_per_task_4cm_heatmaps`: task-specific behavior.
- `06_failure_taxonomy_4cm`: failure mechanisms.
- `07_reacquisition_and_contact`: visual recovery.
- `08_execution_cost_and_safety`: time, prediction integration, clipping, wrong contact.
- `09_seed_consistency_4cm`: replication stability.
- `10_training_endpoint_comparison`: optimization diagnostics.
