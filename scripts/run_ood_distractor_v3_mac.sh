#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/benchmarks/ood_distractors}"
REPORT_DIR="${REPORT_DIR:-final_report/08_ood_distractors}"
EPISODES_PER_COUNT="${EPISODES_PER_COUNT:-60}"
SEEDS=(20262101 20262111)

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"

echo "Starting paired V3 OOD-distractor benchmark"
echo "Counts: 0 1 2 3"
echo "Seeds: ${SEEDS[*]}"
echo "Episodes per count and seed: ${EPISODES_PER_COUNT}"
echo "Expected total episodes: $((4 * EPISODES_PER_COUNT * ${#SEEDS[@]}))"

for seed in "${SEEDS[@]}"; do
  run_dir="${OUTPUT_ROOT}/seed_${seed}"
  mkdir -p "${run_dir}"
  echo ""
  echo "=== Scene seed ${seed} ==="
  caffeinate -dimsu env \
    -u MUJOCO_GL \
    -u PYOPENGL_PLATFORM \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache \
    conda run --no-capture-output -n mujoco310 \
    python scripts/benchmark_ood_distractors_v3.py \
      --policy artifacts/v3-clean-rc1/mini_vla_v3_policy.pth \
      --counts 0 1 2 3 \
      --episodes-per-count "${EPISODES_PER_COUNT}" \
      --max-steps 200 \
      --seed "${seed}" \
      --perturbation-seed "$((seed + 6000011))" \
      --replan-interval 1 \
      --temporal-profile robust \
      --ensemble-mode temporal \
      --local-files-only \
      --resume \
      --output-dir "${run_dir}" \
    2>&1 | tee -a "${run_dir}/console.log"
done

conda run --no-capture-output -n mujoco310 \
  python scripts/analyze_ood_distractors_v3.py \
    --input-dirs \
      "${OUTPUT_ROOT}/seed_${SEEDS[0]}" \
      "${OUTPUT_ROOT}/seed_${SEEDS[1]}" \
    --output-dir "${REPORT_DIR}"

echo "OOD-distractor benchmark and analysis complete."
echo "Raw results: ${OUTPUT_ROOT}"
echo "Paper outputs: ${REPORT_DIR}"
