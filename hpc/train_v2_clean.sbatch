#!/usr/bin/env bash
#SBATCH --job-name=minivla-train
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/3153782y/slurm-logs/%x-%j.out

set -euo pipefail

ENV_NAME="${ENV_NAME:-vla310}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/25.3.0-3/none-none/a-4lhn6xy}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/VLA-Robustness-Eval}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/scratch/users/${USER}/VLA-Robustness-Eval}"
SHARED_DATASET="${SHARED_DATASET:-${SHARED_ROOT}/datasets/dataset_v2_clean}"
RUN_NAME="${RUN_NAME:-v2_clean_hpc_${SLURM_JOB_ID}}"
SHARED_OUTPUT="${SHARED_OUTPUT:-${SHARED_ROOT}/runs/${RUN_NAME}}"
LOCAL_ROOT="/tmp/users/${USER}/${SLURM_JOB_ID}"
LOCAL_DATASET="${LOCAL_ROOT}/dataset_v2_clean"
LOCAL_OUTPUT="${LOCAL_ROOT}/output"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-40}"

module purge
module load "${MINIFORGE_MODULE}"

mkdir -p "${LOCAL_DATASET}" "${LOCAL_OUTPUT}" "${LOCAL_ROOT}/numba" "${SHARED_OUTPUT}"

sync_back() {
  rsync -a "${LOCAL_OUTPUT}/" "${SHARED_OUTPUT}/"
}
trap sync_back EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="${LOCAL_ROOT}/numba"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Staging dataset from Shared Scratch to Local Scratch..."
rsync -a "${SHARED_DATASET}/" "${LOCAL_DATASET}/"

cd "${PROJECT_ROOT}"
nvidia-smi

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/preflight_v2.py \
    --data-dir "${LOCAL_DATASET}" \
    --expected-episodes 1200 \
    --batch-size 32 \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --output "${LOCAL_OUTPUT}/preflight_v2_clean.json"

train_args=(
  python scripts/train_v2.py
  --data-dir "${LOCAL_DATASET}"
  --output-dir "${LOCAL_OUTPUT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --local-files-only
)

if [[ -n "${INIT_POLICY:-}" ]]; then
  train_args+=(--init-policy "${INIT_POLICY}" --init-scope "${INIT_SCOPE:-all}")
fi
if [[ -n "${RESUME:-}" ]]; then
  train_args+=(--resume "${RESUME}")
fi

srun conda run --no-capture-output -n "${ENV_NAME}" "${train_args[@]}"

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/postflight_v2.py \
    --policy "${LOCAL_OUTPUT}/mini_vla_v2_clean_policy.pth" \
    --data-dir "${LOCAL_DATASET}" \
    --output "${LOCAL_OUTPUT}/postflight_v2_clean.json" \
    --batch-size 32 \
    --num-workers "${NUM_WORKERS}" \
    --local-files-only

echo "Training artifacts will be synchronized to ${SHARED_OUTPUT}"
