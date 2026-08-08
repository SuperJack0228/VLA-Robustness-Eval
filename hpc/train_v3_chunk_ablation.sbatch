#!/usr/bin/env bash
#SBATCH --job-name=v3-chunk
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/users/3153782y/slurm-logs/%x-%A_%a.out

set -euo pipefail

CHUNK_SIZES=(1 5 10 20)
CHUNK_SIZE="${CHUNK_SIZES[${SLURM_ARRAY_TASK_ID}]}"

ENV_NAME="${ENV_NAME:-vla310}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/25.3.0-3/none-none/a-4lhn6xy}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/VLA-Robustness-Eval}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/scratch/users/${USER}/VLA-Robustness-Eval}"
SHARED_BASE_DATASET="${SHARED_BASE_DATASET:-${SHARED_ROOT}/datasets/dataset_v2_clean}"
SHARED_RECOVERY_DATASET="${SHARED_RECOVERY_DATASET:-${SHARED_ROOT}/datasets/dataset_v3_recovery}"
SHARED_INIT_POLICY="${SHARED_INIT_POLICY:-${SHARED_ROOT}/checkpoints/v2_clean/mini_vla_v2_clean_policy.pth}"
RUN_GROUP="${RUN_GROUP:-v3_chunk_ablation_${SLURM_ARRAY_JOB_ID}}"
SHARED_OUTPUT="${SHARED_ROOT}/runs/${RUN_GROUP}/chunk_${CHUNK_SIZE}"
LOCAL_ROOT="/tmp/users/${USER}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_BASE_DATASET="${LOCAL_ROOT}/dataset_v2_clean"
LOCAL_RECOVERY_DATASET="${LOCAL_ROOT}/dataset_v3_recovery"
LOCAL_INIT_POLICY="${LOCAL_ROOT}/mini_vla_v2_clean_policy.pth"
LOCAL_OUTPUT="${LOCAL_ROOT}/output"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"
TRAINING_SEED="${TRAINING_SEED:-20261201}"

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

echo "V3 ACT chunk-size ablation"
echo "Array job: ${SLURM_ARRAY_JOB_ID} | task: ${SLURM_ARRAY_TASK_ID}"
echo "Chunk size: ${CHUNK_SIZE} | seed: ${TRAINING_SEED}"
echo "Output: ${SHARED_OUTPUT}"
echo "Staging the shared V2 Clean and V3 recovery datasets..."

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
    --chunk-size "${CHUNK_SIZE}" \
    --reinitialize-action-queries \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --local-files-only \
    --output "${LOCAL_OUTPUT}/preflight_report_v3.json"

srun conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/train_v3.py \
    --base-data-dir "${LOCAL_BASE_DATASET}" \
    --recovery-data-dir "${LOCAL_RECOVERY_DATASET}" \
    --output-dir "${LOCAL_OUTPUT}" \
    --init-policy "${LOCAL_INIT_POLICY}" \
    --chunk-size "${CHUNK_SIZE}" \
    --reinitialize-action-queries \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${TRAINING_SEED}" \
    --patience 100 \
    --local-files-only

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

sha256sum "${LOCAL_OUTPUT}/mini_vla_v3_policy.pth" \
  > "${LOCAL_OUTPUT}/SHA256SUMS"

echo "Chunk ${CHUNK_SIZE} training and postflight complete."
echo "Artifacts will be synchronized to ${SHARED_OUTPUT}"
