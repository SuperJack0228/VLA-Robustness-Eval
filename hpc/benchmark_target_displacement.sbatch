#!/usr/bin/env bash
#SBATCH --job-name=vla-robust
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/users/3153782y/slurm-logs/%x-%j.out

set -euo pipefail

ENV_NAME="${ENV_NAME:-vla310}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/25.3.0-3/none-none/a-4lhn6xy}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/VLA-Robustness-Eval}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/scratch/users/${USER}/VLA-Robustness-Eval}"
POLICY="${POLICY:-${SHARED_ROOT}/checkpoints/v2_clean/mini_vla_v2_clean_policy.pth}"
RUN_NAME="${RUN_NAME:-target_displacement_v2_${SLURM_JOB_ID}}"
OUTPUT_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
LOCAL_ROOT="/tmp/users/${USER}/${SLURM_JOB_ID}"
LEVELS="${LEVELS:-0 0.01 0.02 0.03 0.04}"
ENSEMBLE_MODES="${ENSEMBLE_MODES:-temporal latest-only}"

module purge
module load "${MINIFORGE_MODULE}"

mkdir -p "${OUTPUT_DIR}" "${LOCAL_ROOT}/numba"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="${LOCAL_ROOT}/numba"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

read -r -a LEVEL_ARGS <<< "${LEVELS}"
read -r -a MODE_ARGS <<< "${ENSEMBLE_MODES}"

cd "${PROJECT_ROOT}"
echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Levels: ${LEVELS}"
echo "Ensemble modes: ${ENSEMBLE_MODES}"
nvidia-smi

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/benchmark_robustness_v2.py \
    --policy "${POLICY}" \
    --perturbation target-displacement \
    --levels "${LEVEL_ARGS[@]}" \
    --episodes-per-level "${EPISODES_PER_LEVEL:-30}" \
    --max-steps "${MAX_STEPS:-200}" \
    --seed "${SEED:-20261101}" \
    --perturbation-seed "${PERTURBATION_SEED:-22261104}" \
    --ensemble-modes "${MODE_ARGS[@]}" \
    --replan-interval 1 \
    --temporal-profile robust \
    --ensemble-decay "${ENSEMBLE_DECAY:-0.75}" \
    --max-prediction-age "${MAX_PREDICTION_AGE:-3}" \
    --grounding-reset-threshold "${GROUNDING_RESET_THRESHOLD:-0.01}" \
    --log-every "${LOG_EVERY:-20}" \
    --local-files-only \
    --output-dir "${OUTPUT_DIR}"

echo "Robustness outputs: ${OUTPUT_DIR}"
