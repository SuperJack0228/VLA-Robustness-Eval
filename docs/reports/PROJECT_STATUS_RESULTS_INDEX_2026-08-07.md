# MiniVLA Project Status and Results Index

Date: 2026-08-07

## 1. Current frozen baseline

- Final candidate policy: MiniVLA V3, ACT chunk size 20.
- Inputs: language instruction, agent-view image, wrist image, and five-step 17D robot-state history.
- Outputs: a 20-step chunk of 7D OSC_POSE delta actions.
- Training data: 1,200 balanced V2 Clean episodes plus 600 balanced V3 recovery episodes.
- Language split: 60 train and 30 held-out evaluation expressions per task, six tasks total.
- Training: 30 epochs on one NVIDIA L40S; V3 preflight and postflight passed.

## 2. Completed evidence

### 2.1 Paired Clean benchmark

- V2: 222/240 = 92.50%.
- V3: 235/240 = 97.92%.
- Exact paired McNemar p = 0.0072.
- V3 rescued 17 V2 failures and introduced 4 regressions.

Source files:

- `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/clean_comparison/v2_20260714.json`
- `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/clean_comparison/v2_20261017.json`
- `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/clean_comparison/v3_20260714.json`
- `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/clean_comparison/v3_20261017.json`

### 2.2 Paired dynamic target-displacement benchmark

- Levels: 0, 1, 2, 3, 4, 5, 6, 7, and 8 cm.
- Two matched seed schedules.
- Temporal and latest-only execution modes.
- 4,320 raw episode records; 2,160 paired V2/V3 outcomes.
- Primary Temporal analysis: 120 paired episodes per displacement level.
- V2 normalized robustness AUC: 0.4974.
- V3 normalized robustness AUC: 0.8974.
- Point-estimate 80% boundary: V2 2 cm, V3 7 cm.
- Conservative lower-Wilson-CI 80% boundary: V2 1 cm, V3 5 cm.

Primary report and tables:

- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/analysis_report.md`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/comparison_summary.json`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/paired_v2_v3_comparison.csv`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/model_success_summary.csv`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/task_success_summary.csv`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/failure_taxonomy.csv`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/paired_episode_outcomes.csv`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/protocol_audit.json`

Raw benchmark directories:

- `/Users/superjack/VLA-Robustness-Eval/results/benchmarks/dynamic_displacement/v2_seed_20260801`
- `/Users/superjack/VLA-Robustness-Eval/results/benchmarks/dynamic_displacement/v2_seed_20260811`
- `/Users/superjack/VLA-Robustness-Eval/results/benchmarks/dynamic_displacement/v3_seed_20260801`
- `/Users/superjack/VLA-Robustness-Eval/results/benchmarks/dynamic_displacement/v3_seed_20260811`

## 3. Visualizations

Every comparison figure below is available as both PNG and vector PDF.

1. Main robustness decay curve:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/01_v2_v3_temporal_success_decay.png`
2. Temporal versus latest-only diagnostic:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/02_all_models_and_execution_modes.png`
3. Paired V3 gain and statistical significance:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/03_v3_paired_gain_and_significance.png`
4. Success retention normalized to clean performance:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/04_normalized_success_retention.png`
5. Pick-versus-push family degradation:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/05_pick_push_family_comparison.png`
6. Per-task V3 improvement heatmap:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/06_per_task_v3_gain_heatmap.png`
7. Failure Taxonomy:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/07_failure_taxonomy_v2_v3.png`
8. Target reacquisition rate and latency:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/08_reacquisition_rate_and_latency.png`
9. Recovery cost and wrong-object contact:
   `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/09_recovery_cost_and_wrong_contact.png`

Supervisor-summary figures generated from the frozen artifacts:

10. Paired Clean aggregate and six-task comparison:
    `/Users/superjack/VLA-Robustness-Eval/final_report/01_clean_baseline/paired_clean_v2_vs_v3.png`
11. V3 training diagnostics:
    `/Users/superjack/VLA-Robustness-Eval/final_report/04_training_diagnostics/v3_training_diagnostics.png`
12. Metrics used by figures 10 and 11:
    `/Users/superjack/VLA-Robustness-Eval/final_report/01_clean_baseline/clean_summary_metrics.json`

Older V3-only figures, useful as appendix material:

- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/appendix_v3_only/success_decay_curve.png`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/appendix_v3_only/task_family_decay_curve.png`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/appendix_v3_only/failure_taxonomy_temporal.png`
- `/Users/superjack/VLA-Robustness-Eval/final_report/02_dynamic_displacement/appendix_v3_only/reacquisition_metrics.png`

Existing presentation:

- `/Users/superjack/VLA-Robustness-Eval/docs/presentations/2026-07-28-LiuYining-MiniVLA-V3-Update.pptx`

The presentation predates the final two-seed displacement comparison and should not be treated as the final paper deck.

## 4. Training and reproducibility artifacts

- V3 release bundle: `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1`
- V3 training log: `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/training_log_v3.csv`
- V3 training metadata: `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/training_metadata_v3.json`
- V3 preflight: `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/preflight_report_v3.json`
- V3 postflight: `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/postflight_clean_v3.json`
- Language catalog: `/Users/superjack/VLA-Robustness-Eval/artifacts/v3-clean-rc1/language_augmentations_v3.json`
- V2 dataset manifest: `/Users/superjack/VLA-Robustness-Eval/data/dataset_v2_clean/dataset_manifest.csv`
- V3 recovery manifest: `/Users/superjack/VLA-Robustness-Eval/data/dataset_v3_recovery/dataset_manifest_v3_recovery.csv`
- Supervisor-figure generator: `/Users/superjack/VLA-Robustness-Eval/scripts/build_supervisor_summary_figures.py`

## 5. Completed ACT chunk-size ablation

ACT chunk-size training and paired Mac evaluation are complete for chunk sizes 1, 5, 10, and 20. The final evaluation contains 1,920 episodes across two seeds, Clean/4 cm conditions, and temporal/latest-only modes. Protocol audit passed.

- Submission script: `/Users/superjack/VLA-Robustness-Eval/hpc/train_v3_chunk_ablation.sbatch`
- Analysis report: `/Users/superjack/VLA-Robustness-Eval/final_report/03_chunk_size_ablation/analysis_report.md`
- Figures and tables: `/Users/superjack/VLA-Robustness-Eval/final_report/03_chunk_size_ablation`

Classic temporal 4 cm success is 85.00%, 89.17%, 95.00%, and 90.83% for chunks 1, 5, 10, and 20. The overall paired chunk effect is significant (`p=0.0112`), but the result is non-monotonic and does not support the simple claim that shorter chunks are always more dynamically robust.

## 6. Not yet completed

1. Formal visual degradation curves: Gaussian noise, blur, and brightness reduction with multiple controlled levels and matched seeds.
2. Physics drift curves: object mass, object friction, and table friction.
3. OOD distractor benchmark: unseen geometry, controlled count/placement, collision-integrity validation.
4. Camera-extrinsic benchmark: controlled translation and rotation of agentview and wrist cameras.
5. Held-out language benchmark: evaluate the 180 reserved expressions and semantic paraphrases without explicit task-label leakage.
6. Efficiency table: parameter count, GPU/Mac inference latency, control rate, peak memory, and checkpoint size.
7. Final paper consolidation: methods diagram, experiment table, statistical wording, limitations, and appendix protocol.

## 7. Recommended supervisor-meeting sequence

1. State the system: language + dual-view vision + five-step 17D history -> 20-step 7D ACT policy.
2. Show paired Clean result: V2 92.50% -> V3 97.92%.
3. Show figure 01: V3 moves the point-estimate 80% displacement boundary from 2 cm to 7 cm.
4. Show figure 06: improvement is largest on displaced pick tasks; push tasks remain the bottleneck.
5. Show figure 07: failure mechanism changes from grasp collapse/wrong target to insufficient push distance.
6. Show the completed chunk-size ablation: chunk 10 is best under classic temporal 4 cm testing, while the relation is non-monotonic.

## 8. Claim boundaries

- The completed evidence is MuJoCo/robosuite simulation evidence, not real-robot or sim-to-real validation.
- The 7 cm result is an 80% point estimate; the conservative 95% confidence lower-bound threshold is 5 cm.
- Temporal versus latest-only is not uniformly ordered. No within-chunk temporal/latest-only difference is statistically significant, so V3's gain should not be attributed solely to temporal ensembling.
- The current data strongly support better closed-loop recovery under dynamic displacement, not universal robustness to every perturbation type.
