#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vla310}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/25.3.0-3/none-none/a-4lhn6xy}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"

if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi

module purge
module load "${MINIFORGE_MODULE}"

mkdir -p \
  "${HOME}/slurm-logs" \
  "${HOME}/.cache/huggingface/hub" \
  "${HOME}/.cache/torch/hub/checkpoints" \
  "/mnt/scratch/users/${USER}/VLA-Robustness-Eval/datasets" \
  "/mnt/scratch/users/${USER}/VLA-Robustness-Eval/checkpoints" \
  "/mnt/scratch/users/${USER}/VLA-Robustness-Eval/runs"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --upgrade pip setuptools wheel

torch_args=(torch==2.13.0 torchvision==0.28.0)
if [[ -n "${PYTORCH_INDEX_URL}" ]]; then
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" "${torch_args[@]}"
else
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install "${torch_args[@]}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install -r "${SCRIPT_DIR}/requirements-hpc.txt"

# robosuite 1.5.2 requires mujoco>=3.3.0 and mink 0.0.5 requires
# mujoco>=3.1.6. V2 Clean is a frozen mujoco 3.1.1 baseline, so dependency
# resolution is intentionally bypassed only for these two packages after all
# of their runtime dependencies are pinned.
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --no-deps --force-reinstall \
    mink==0.0.5 robosuite==1.5.2

if [[ "${DOWNLOAD_MODEL_ASSETS:-0}" == "1" ]]; then
  conda run --no-capture-output -n "${ENV_NAME}" python -c \
    "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModel.from_pretrained('distilbert-base-uncased')"
  conda run --no-capture-output -n "${ENV_NAME}" python -c \
    "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"
fi

conda run --no-capture-output -n "${ENV_NAME}" python -c \
  "import torch, mujoco, robosuite, transformers, cv2; assert mujoco.__version__ == '3.1.1'; assert robosuite.__version__ == '1.5.2'; print('torch', torch.__version__, 'runtime CUDA', torch.version.cuda); print('mujoco', mujoco.__version__, 'robosuite', robosuite.__version__, 'transformers', transformers.__version__, 'opencv', cv2.__version__)"

echo "Environment ${ENV_NAME} is installed."
echo "CUDA availability must be verified inside a Slurm GPU job."
