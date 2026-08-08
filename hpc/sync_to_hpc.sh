#!/usr/bin/env bash
set -euo pipefail

HPC_LOGIN="${HPC_LOGIN:-3153782y@cognition.cose.gla.ac.uk}"
REMOTE_CODE="${REMOTE_CODE:-/users/3153782y/VLA-Robustness-Eval}"
REMOTE_SHARED="${REMOTE_SHARED:-/mnt/scratch/users/3153782y/VLA-Robustness-Eval}"
MODE="${1:---all}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Unix-domain socket paths are limited to roughly 104 bytes on macOS.
# $TMPDIR is unusually long there, so keep the multiplexing socket in /tmp.
CONTROL_PATH="/tmp/vla-hpc-%C"
SSH_OPTIONS=(
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=${CONTROL_PATH}"
)
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=${CONTROL_PATH}"
RSYNC=(rsync -a --human-readable --partial -e "${RSYNC_SSH}")
if rsync --help 2>&1 | grep -q -- "--info"; then
  RSYNC+=(--info=progress2)
else
  # macOS ships openrsync / rsync 2.6.9, which has no --info option.
  RSYNC+=(--progress)
fi

case "${MODE}" in
  --code|--dataset|--dataset-v3|--policy|--caches|--all) ;;
  *)
    echo "Usage: $0 [--code|--dataset|--dataset-v3|--policy|--caches|--all]" >&2
    exit 2
    ;;
esac

close_control_connection() {
  ssh "${SSH_OPTIONS[@]}" -O exit "${HPC_LOGIN}" >/dev/null 2>&1 || true
}
trap close_control_connection EXIT

ssh "${SSH_OPTIONS[@]}" "${HPC_LOGIN}" \
  "mkdir -p '${REMOTE_CODE}' \
    '${REMOTE_SHARED}/datasets/dataset_v2_clean' \
    '${REMOTE_SHARED}/datasets/dataset_v3_recovery' \
    '${REMOTE_SHARED}/checkpoints/v2_clean' \
    '${REMOTE_SHARED}/runs' \
    ~/.cache/huggingface/hub \
    ~/.cache/torch/hub/checkpoints \
    ~/slurm-logs"

if [[ "${MODE}" == "--code" || "${MODE}" == "--all" ]]; then
  "${RSYNC[@]}" \
    --exclude='data/' \
    --exclude='artifacts/' \
    --exclude='results/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "${PROJECT_ROOT}/" \
    "${HPC_LOGIN}:${REMOTE_CODE}/"
fi

if [[ "${MODE}" == "--dataset" || "${MODE}" == "--all" ]]; then
  "${RSYNC[@]}" \
    "${PROJECT_ROOT}/data/dataset_v2_clean/" \
    "${HPC_LOGIN}:${REMOTE_SHARED}/datasets/dataset_v2_clean/"
fi

if [[ "${MODE}" == "--dataset-v3" || "${MODE}" == "--all" ]]; then
  if [[ ! -d "${PROJECT_ROOT}/data/dataset_v3_recovery" ]]; then
    echo "Missing local V3 recovery dataset: data/dataset_v3_recovery" >&2
    exit 1
  fi
  "${RSYNC[@]}" \
    "${PROJECT_ROOT}/data/dataset_v3_recovery/" \
    "${HPC_LOGIN}:${REMOTE_SHARED}/datasets/dataset_v3_recovery/"
fi

if [[ "${MODE}" == "--policy" || "${MODE}" == "--all" ]]; then
  policy_files=(
    "${PROJECT_ROOT}/artifacts/v2-clean-rc1/mini_vla_v2_clean_policy.pth"
    "${PROJECT_ROOT}/artifacts/v2-clean-rc1/training_metadata_v2_clean.json"
    "${PROJECT_ROOT}/artifacts/v2-clean-rc1/preflight_report_v2_clean.json"
    "${PROJECT_ROOT}/artifacts/v2-clean-rc1/postflight_v2_clean.json"
  )
  "${RSYNC[@]}" \
    "${policy_files[@]}" \
    "${HPC_LOGIN}:${REMOTE_SHARED}/checkpoints/v2_clean/"
fi

if [[ "${MODE}" == "--caches" || "${MODE}" == "--all" ]]; then
  "${RSYNC[@]}" \
    "${HOME}/.cache/huggingface/hub/models--distilbert-base-uncased/" \
    "${HPC_LOGIN}:~/.cache/huggingface/hub/models--distilbert-base-uncased/"
  "${RSYNC[@]}" \
    "${HOME}/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth" \
    "${HPC_LOGIN}:~/.cache/torch/hub/checkpoints/"
fi

echo "HPC synchronization completed: ${MODE}"
