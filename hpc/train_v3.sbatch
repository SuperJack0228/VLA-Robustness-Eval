#!/usr/bin/env bash
#SBATCH --job-name=minivla-v3
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/3153782y/slurm-logs/%x-%j.out

set -euo pipefail

ENV_NAME="${ENV_NAME:-vla310}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/25.3.0-3/none-none/a-4lhn6xy}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/VLA-Robustness-Eval}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/scratch/users/${USER}/VLA-Robustness-Eval}"
SHARED_BASE_DATASET="${SHARED_BASE_DATASET:-${SHARED_ROOT}/datasets/dataset_v2_clean}"
SHARED_RECOVERY_DATASET="${SHARED_RECOVERY_DATASET:-${SHARED_ROOT}/datasets/dataset_v3_recovery}"
SHARED_INIT_POLICY="${SHARED_INIT_POLICY:-${SHARED_ROOT}/checkpoints/v2_clean/mini_vla_v2_clean_policy.pth}"
RUN_NAME="${RUN_NAME:-v3_hpc_${SLURM_JOB_ID}}"
SHARED_OUTPUT="${SHARED_OUTPUT:-${SHARED_ROOT}/runs/${RUN_NAME}}"
LOCAL_ROOT="/tmp/users/${USER}/${SLURM_JOB_ID}"
LOCAL_BASE_DATASET="${LOCAL_ROOT}/dataset_v2_clean"
LOCAL_RECOVERY_DATASET="${LOCAL_ROOT}/dataset_v3_recovery"
LOCAL_INIT_POLICY="${LOCAL_ROOT}/mini_vla_v2_clean_policy.pth"
LOCAL_OUTPUT="${LOCAL_ROOT}/output"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"

module purge
module load "${MINIFORGE_MODULE}"

mkdir -p \
  "${LOCAL_BASE_DATASET}" \
  "${LOCAL_RECOVERY_DATASET}" \
  "${LOCAL_OUTPUT}" \
  "${LOCAL_ROOT}/numba" \
  "${SHARED_OUTPUT}"

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

echo "Staging V2 Clean, V3 recovery data, and the frozen V2 policy..."
rsync -a "${SHARED_BASE_DATASET}/" "${LOCAL_BASE_DATASET}/"
rsync -a "${SHARED_RECOVERY_DATASET}/" "${LOCAL_RECOVERY_DATASET}/"
cp "${SHARED_INIT_POLICY}" "${LOCAL_INIT_POLICY}"

cd "${PROJECT_ROOT}"
echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/preflight_v3.py \
    --base-data-dir "${LOCAL_BASE_DATASET}" \
    --recovery-data-dir "${LOCAL_RECOVERY_DATASET}" \
    --expected-base-episodes 1200 \
    --expected-recovery-episodes 600 \
    --init-policy "${LOCAL_INIT_POLICY}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --local-files-only \
    --output "${LOCAL_OUTPUT}/preflight_report_v3.json"

train_args=(
  python scripts/train_v3.py
  --base-data-dir "${LOCAL_BASE_DATASET}"
  --recovery-data-dir "${LOCAL_RECOVERY_DATASET}"
  --output-dir "${LOCAL_OUTPUT}"
  --init-policy "${LOCAL_INIT_POLICY}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --local-files-only
)

if [[ -n "${RESUME:-}" ]]; then
  train_args+=(--resume "${RESUME}")
fi

srun conda run --no-capture-output -n "${ENV_NAME}" "${train_args[@]}"

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/evaluate_policy_v2.py \
    --policy "${LOCAL_OUTPUT}/mini_vla_v3_policy.pth" \
    --num-episodes 6 \
    --max-steps 200 \
    --seed 20261219 \
    --replan-interval 1 \
    --ensemble-mode temporal \
    --temporal-profile robust \
    --visual-perturbation clean \
    --log-every 50 \
    --local-files-only \
    --output-prefix "${LOCAL_OUTPUT}/postflight_clean_v3"

echo "V3 artifacts will be synchronized to ${SHARED_OUTPUT}"
