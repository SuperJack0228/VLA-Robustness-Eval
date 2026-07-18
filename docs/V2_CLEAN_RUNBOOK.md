# MiniVLA V2 Clean Runbook

Run every command from the repository root with the `mujoco310` conda
environment installed. Datasets, checkpoints, and logs under `results/` are
intentionally excluded from Git.

## 1. Collect 1200 Balanced Demonstrations

The collector writes 200 clean trajectories for each of the six task buckets.
Collection is resumable and saves only trajectories that pass the schema and
collision-integrity checks.

```bash
mkdir -p results/v2_clean
set -o pipefail

caffeinate -dimsu env \
  PYTHONUNBUFFERED=1 \
  NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache \
  conda run --no-capture-output -n mujoco310 \
  python scripts/collect_data_v2.py \
    --num-episodes 1200 \
    --data-dir results/dataset_v2_clean \
    --seed 20260817 \
  2>&1 | tee results/v2_clean/collection_console_v2_clean.log
```

## 2. Preflight Gate

Do not train unless the command ends with `V2 CLEAN PREFLIGHT: PASS`.

```bash
conda run --no-capture-output -n mujoco310 \
python scripts/preflight_v2.py \
  --data-dir results/dataset_v2_clean \
  --expected-episodes 1200 \
  --batch-size 32 \
  --num-workers 4 \
  --device mps
```

## 3. Train

Only the V2.2 perception stack is warm-started. Interaction and action heads
are trained for the clean schema.

```bash
mkdir -p results/v2_clean
set -o pipefail

caffeinate -dimsu env \
  PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n mujoco310 \
  python scripts/train_v2.py \
    --data-dir results/dataset_v2_clean \
    --output-dir results/v2_clean \
    --epochs 40 \
    --batch-size 32 \
    --num-workers 8 \
    --init-policy results/v2_2/mini_vla_v2_2_policy.pth \
    --init-scope perception \
    --local-files-only \
  2>&1 | tee results/v2_clean/training_console_v2_clean.log
```

Resume an interrupted run without `--init-policy`:

```bash
set -o pipefail

caffeinate -dimsu env \
  PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n mujoco310 \
  python scripts/train_v2.py \
    --data-dir results/dataset_v2_clean \
    --output-dir results/v2_clean \
    --epochs 40 \
    --batch-size 32 \
    --num-workers 8 \
    --resume results/v2_clean/mini_vla_v2_clean_last.pth \
    --local-files-only \
  2>&1 | tee -a results/v2_clean/training_console_v2_clean.log
```

## 4. Full-Trajectory Postflight

```bash
conda run --no-capture-output -n mujoco310 \
python scripts/postflight_v2.py \
  --policy results/v2_clean/mini_vla_v2_clean_policy.pth \
  --data-dir results/dataset_v2_clean \
  --output results/v2_clean/postflight_v2_clean.json \
  --batch-size 32 \
  --num-workers 4 \
  --local-files-only \
  --enforce
```

## 5. Clean 80 Percent Gate

The 120 episodes are balanced across all six instructions. Evaluation executes
raw model actions with only generic robot-workspace safety limits.

```bash
set -o pipefail

NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache \
conda run --no-capture-output -n mujoco310 \
python scripts/evaluate_policy_v2.py \
  --policy results/v2_clean/mini_vla_v2_clean_policy.pth \
  --num-episodes 120 \
  --max-steps 200 \
  --seed 20260817 \
  --replan-interval 1 \
  --visual-perturbation clean \
  --log-every 20 \
  --output-prefix results/v2_clean/evaluation_clean_v2_clean \
  --local-files-only \
  --enforce-80 \
  2>&1 | tee results/v2_clean/evaluation_clean_v2_clean.log
```

Add `--render` for an OpenCV live view. Robustness perturbations are `bright`,
`dark`, `gaussian_noise`, `camera_shift`, and `center_occlusion`; use a unique
output prefix for every run.

## 6. Dynamic-Displacement Robustness Sweep

The robustness runner evaluates identical scene seeds at every displacement
level and runs both temporal ensembling and latest-prediction-only execution.
Distances are specified in meters. External target displacement is removed
from Push scoring, so teleportation cannot count as policy progress.

```bash
set -o pipefail

caffeinate -dimsu env \
  PYTHONUNBUFFERED=1 \
  NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache \
  conda run --no-capture-output -n mujoco310 \
  python scripts/benchmark_robustness_v2.py \
    --policy results/v2_clean/mini_vla_v2_clean_policy.pth \
    --perturbation target-displacement \
    --levels 0 0.02 0.04 0.06 0.08 \
    --episodes-per-level 120 \
    --max-steps 200 \
    --seed 20261101 \
    --ensemble-modes temporal latest-only \
    --replan-interval 1 \
    --log-every 20 \
    --local-files-only \
    --output-dir results/robustness/target_displacement \
  2>&1 | tee results/robustness/target_displacement_console.log
```

Primary outputs:

- `benchmark_episodes.csv`: per-episode perturbation, recovery, and failure data.
- `benchmark_summary.json`: success-decay curves and per-task failure counts.
- `runs/*.csv` and `runs/*.json`: independently inspectable level/mode runs.
