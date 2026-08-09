# Script Index

Scripts remain in one import-stable directory because collectors, training
jobs, tests, and HPC commands import each other by their existing module paths.

## Data Collection and Validation

- `collect_data_v2.py`: 1,200-episode balanced clean expert collector.
- `collect_data_v3.py`: 600-episode displacement-recovery collector.
- `preflight_v2.py`, `preflight_v3.py`: dataset and initialization gates.
- `postflight_v2.py`: full-trajectory policy gate.

## Training

- `train_v2.py`: V2 Clean training.
- `train_v3.py`: V3 recovery fine-tuning and chunk-size ablation training.

## Evaluation

- `evaluate_policy_v2.py`: raw-policy closed-loop evaluator.
- `benchmark_robustness_v2.py`: paired perturbation benchmark driver.
- `benchmark_visual_robustness_v3.py`: paired dual-camera Gaussian-noise,
  Gaussian-blur, and brightness robustness curves for the frozen V3 policy;
  its default grid spans the mild regime through deliberately severe visual
  corruption to identify sub-80% and sub-50% failure boundaries.
- `benchmark_physics_robustness_v3.py`: paired target mass/inertia and contact-
  friction drift curves with per-episode parameter restoration audits.
- `benchmark_camera_extrinsics_v3.py`: paired MuJoCo `agentview` position and
  quaternion drift with deterministic per-scene directions and pose restoration.
- `benchmark_ood_distractors_v3.py`: paired 0-3 unseen static distractors with
  safe-path, target-visibility, collision, and target-selection diagnostics.
- `benchmark_language_generalization_v3.py`: 180 paired canonical/held-out
  language scenes (360 rollouts) with raw-text passthrough and no ground-truth
  task routing in policy execution.
- `benchmark_oracle_v2.py`: scripted expert baseline.
- `benchmark_blue_push_heights_v2.py`: targeted blue-ball diagnostic.

## Paper Analysis

- `analyze_v2_v3_displacement_comparison.py`: final paired V2/V3 curves.
- `analyze_dynamic_displacement_v3.py`: V3-only appendix analysis.
- `analyze_chunk_ablation_v3.py`: chunk-size ablation analysis.
- `analyze_visual_robustness_v3.py`: protocol audit, paired statistics, and
  paper figures for the three visual-degradation curves.
- `analyze_physics_robustness_v3.py`: paired physics-drift statistics, task-
  family sensitivity, failure taxonomy, and paper figures.
- `analyze_camera_extrinsics_v3.py`: camera-pose protocol audit, paired
  statistics, task heatmap, diagnostics, and failure taxonomy figures.
- `analyze_ood_distractors_v3.py`: OOD layout audit, paired significance tests,
  collision-aware success, target-selection failures, and paper figures.
- `analyze_language_generalization_v3.py`: language protocol audit, paired
  success statistics, six-task confusion matrices, and paper figures.
- `build_supervisor_summary_figures.py`: paired Clean and training figures.
- `run_final_dynamic_v3_mac.sh`: reproducible Mac displacement launcher.
- `run_chunk_ablation_evaluation_mac.sh`: reproducible Mac ablation launcher.
- `run_ood_distractor_v3_mac.sh`: resumable two-seed, 480-episode OOD launcher
  followed by protocol audit and paper-figure generation.

## Interactive Demo and Smoke Tests

- `launch_demo_v3.py`: PySide6 desktop application.
- `hpc_smoke_test.py`: CUDA, EGL, MuJoCo, and checkpoint smoke test.
- `test_mujoco.py`, `test_robosuite.py`: minimal local runtime checks.
