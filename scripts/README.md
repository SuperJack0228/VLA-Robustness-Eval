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
- `benchmark_oracle_v2.py`: scripted expert baseline.
- `benchmark_blue_push_heights_v2.py`: targeted blue-ball diagnostic.

## Paper Analysis

- `analyze_v2_v3_displacement_comparison.py`: final paired V2/V3 curves.
- `analyze_dynamic_displacement_v3.py`: V3-only appendix analysis.
- `analyze_chunk_ablation_v3.py`: chunk-size ablation analysis.
- `build_supervisor_summary_figures.py`: paired Clean and training figures.
- `run_final_dynamic_v3_mac.sh`: reproducible Mac displacement launcher.
- `run_chunk_ablation_evaluation_mac.sh`: reproducible Mac ablation launcher.

## Interactive Demo and Smoke Tests

- `launch_demo_v3.py`: PySide6 desktop application.
- `hpc_smoke_test.py`: CUDA, EGL, MuJoCo, and checkpoint smoke test.
- `test_mujoco.py`, `test_robosuite.py`: minimal local runtime checks.
