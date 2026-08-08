# V2 vs V3 Dynamic Target Displacement Analysis

## Protocol

- Primary estimand: Temporal ensemble, paired scenes, intention-to-treat.
- Two seed schedules, 120 paired episodes per displacement level and model.
- All non-zero Temporal episodes received the requested displacement.
- No privileged execution assistance was enabled.
- Latest-only is diagnostic because V2 missed 8 scheduled injections in one seed run.

## Primary Result

| Shift | V2 success (95% Wilson CI) | V3 success (95% Wilson CI) | Paired gain | Holm p |
|---:|---:|---:|---:|---:|
| 0 cm | 90.8% [84.3, 94.8] | 97.5% [92.9, 99.1] | +6.7 pp | 0.00781 |
| 1 cm | 90.8% [84.3, 94.8] | 98.3% [94.1, 99.5] | +7.5 pp | 0.00781 |
| 2 cm | 86.7% [79.4, 91.6] | 97.5% [92.9, 99.1] | +10.8 pp | 0.00293 |
| 3 cm | 57.5% [48.6, 66.0] | 94.2% [88.4, 97.1] | +36.7 pp | 3.35e-11 |
| 4 cm | 40.0% [31.7, 48.9] | 90.8% [84.3, 94.8] | +50.8 pp | 3.4e-15 |
| 5 cm | 27.5% [20.3, 36.1] | 87.5% [80.4, 92.3] | +60.0 pp | 6.2e-19 |
| 6 cm | 21.7% [15.2, 29.9] | 83.3% [75.7, 88.9] | +61.7 pp | 1.84e-19 |
| 7 cm | 20.0% [13.8, 28.0] | 80.0% [72.0, 86.2] | +60.0 pp | 3.67e-18 |
| 8 cm | 16.7% [11.1, 24.3] | 75.0% [66.6, 81.9] | +58.3 pp | 2.55e-16 |

## Main Findings

- V2 has a robustness cliff between 2 cm and 3 cm; V3 remains above 90% through 4 cm.
- The point-estimate 80% boundary moves from 2 cm (V2) to 7 cm (V3).
- V3 largely removes post-contact grasp collapse and wrong-object contact.
- At large shifts, V3's remaining bottleneck is push completion distance rather than target selection.
- V2 often reacquires and contacts the displaced target at 3-4 cm but still fails task completion, so its failure is not purely visual grounding.

## Aggregate Temporal Failure Counts

| Failure category | V2 | V3 |
|---|---:|---:|
| grasp_failed_after_contact | 173 | 1 |
| gripper_never_closed | 2 | 10 |
| insufficient_lift | 2 | 6 |
| insufficient_push_distance | 160 | 52 |
| lateral_push_error | 0 | 2 |
| missed_grasp | 49 | 0 |
| object_dropped | 9 | 0 |
| object_launched | 6 | 0 |
| recovery_limit | 10 | 28 |
| target_not_contacted | 28 | 12 |
| target_not_reached | 1 | 0 |
| target_toppled_before_grasp | 17 | 0 |
| workspace_stall | 6 | 0 |
| wrong_object_contact | 75 | 4 |

## Output Guide

- `01_v2_v3_temporal_success_decay`: main paper robustness curve.
- `03_v3_paired_gain_and_significance`: paired improvement and significance.
- `06_per_task_v3_gain_heatmap`: task-level gains.
- `07_failure_taxonomy_v2_v3`: failure mechanism transition.
- `08_reacquisition_rate_and_latency`: visual recovery evidence.
- `09_recovery_cost_and_wrong_contact`: recovery efficiency and safety.

## Audit Note

- Paired model outcomes: 2160.
- Latest-only missed injections: 8.
