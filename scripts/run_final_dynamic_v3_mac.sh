#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-results/benchmarks/dynamic_displacement/v3_new_run}"
EPISODES_PER_LEVEL="${EPISODES_PER_LEVEL:-60}"
SCENE_SEED="${SCENE_SEED:-20260801}"
PERTURBATION_SEED="${PERTURBATION_SEED:-20260802}"

cd "${PROJECT_ROOT}"

if [[ -f "${OUTPUT_DIR}/benchmark_summary.json" ]]; then
  echo "Refusing to overwrite completed benchmark: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Starting V3 dynamic-displacement benchmark"
echo "Episodes per level: ${EPISODES_PER_LEVEL}"
echo "Levels: 0 1 2 3 4 5 6 7 8 cm"
echo "Output: ${OUTPUT_DIR}"

caffeinate -dimsu env \
  -u MUJOCO_GL \
  -u PYOPENGL_PLATFORM \
  PYTHONUNBUFFERED=1 \
  NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache \
  conda run --no-capture-output -n mujoco310 \
  python scripts/benchmark_robustness_v2.py \
    --policy artifacts/v3-clean-rc1/mini_vla_v3_policy.pth \
    --perturbation target-displacement \
    --levels 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
    --episodes-per-level "${EPISODES_PER_LEVEL}" \
    --max-steps 200 \
    --seed "${SCENE_SEED}" \
    --perturbation-seed "${PERTURBATION_SEED}" \
    --replan-interval 1 \
    --temporal-profile robust \
    --ensemble-modes temporal latest-only \
    --log-every 100 \
    --local-files-only \
    --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/full_console.log"

echo "V3 dynamic-displacement benchmark complete."
