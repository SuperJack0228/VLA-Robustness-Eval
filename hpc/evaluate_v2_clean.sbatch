#!/usr/bin/env bash
#SBATCH --job-name=minivla-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/3153782y/slurm-logs/%x-%j.out

set -euo pipefail

ENV_NAME="${ENV_NAME:-vla310}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/25.3.0-3/none-none/a-4lhn6xy}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/VLA-Robustness-Eval}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/scratch/users/${USER}/VLA-Robustness-Eval}"
POLICY="${POLICY:-${SHARED_ROOT}/checkpoints/v2_clean/mini_vla_v2_clean_policy.pth}"
RUN_NAME="${RUN_NAME:-evaluation_clean_${SLURM_JOB_ID}}"
OUTPUT_PREFIX="${SHARED_ROOT}/runs/${RUN_NAME}/evaluation"
LOCAL_ROOT="/tmp/users/${USER}/${SLURM_JOB_ID}"

module purge
module load "${MINIFORGE_MODULE}"

mkdir -p "$(dirname "${OUTPUT_PREFIX}")" "${LOCAL_ROOT}/numba"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="${LOCAL_ROOT}/numba"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_ROOT}"
nvidia-smi

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/evaluate_policy_v2.py \
    --policy "${POLICY}" \
    --num-episodes "${NUM_EPISODES:-120}" \
    --max-steps 200 \
    --seed "${SEED:-20261017}" \
    --replan-interval 1 \
    --ensemble-mode temporal \
    --temporal-profile "${TEMPORAL_PROFILE:-legacy}" \
    --visual-perturbation clean \
    --log-every 20 \
    --local-files-only \
    --output-prefix "${OUTPUT_PREFIX}"
