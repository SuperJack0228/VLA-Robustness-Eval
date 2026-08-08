#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-mujoco310}"
EPISODES_PER_LEVEL="${EPISODES_PER_LEVEL:-60}"
MAX_STEPS="${MAX_STEPS:-200}"
EVAL_SEED="${EVAL_SEED:-20262001}"
PERTURBATION_SEED="${PERTURBATION_SEED:-20262002}"
RUN_TAG="${RUN_TAG:-seed_${EVAL_SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/benchmarks/chunk_size_ablation/${RUN_TAG}}"

CHUNKS=(1 5 10 20)
POLICIES=(
  "artifacts/chunk-ablation-v3/chunk_1/mini_vla_v3_policy.pth"
  "artifacts/chunk-ablation-v3/chunk_5/mini_vla_v3_policy.pth"
  "artifacts/chunk-ablation-v3/chunk_10/mini_vla_v3_policy.pth"
  "artifacts/v3-clean-rc1/mini_vla_v3_policy.pth"
)

mkdir -p "${OUTPUT_ROOT}" /tmp/robosuite_numba_cache

echo "ACT chunk-size ablation evaluation"
echo "Episodes per level: ${EPISODES_PER_LEVEL}"
echo "Levels: clean and 4 cm target displacement"
echo "Modes: legacy temporal and latest-only"
echo "Evaluation seed: ${EVAL_SEED}"
echo "Perturbation seed: ${PERTURBATION_SEED}"
echo "Output root: ${OUTPUT_ROOT}"

for index in "${!CHUNKS[@]}"; do
  chunk="${CHUNKS[${index}]}"
  policy="${POLICIES[${index}]}"
  output_dir="${OUTPUT_ROOT}/chunk_${chunk}"
  log_path="${OUTPUT_ROOT}/chunk_${chunk}_console.log"
  summary_path="${output_dir}/benchmark_summary.json"

  if [[ ! -f "${policy}" ]]; then
    echo "Missing policy for chunk_size=${chunk}: ${policy}" >&2
    exit 1
  fi

  if [[ "${FORCE_RERUN:-0}" != "1" ]] && python - \
    "${summary_path}" "${policy}" "${EPISODES_PER_LEVEL}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
policy = sys.argv[2]
episodes_per_level = int(sys.argv[3])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    raise SystemExit(1)

complete = (
    summary.get("status") == "complete"
    and summary.get("policy") == policy
    and summary.get("episodes_per_level") == episodes_per_level
    and summary.get("levels_m") == [0.0, 0.04]
    and summary.get("ensemble_modes") == ["temporal", "latest-only"]
    and summary.get("temporal_profile") == "legacy"
)
raise SystemExit(0 if complete else 1)
PY
  then
    echo
    echo "===== chunk_size=${chunk} already complete; skipping ====="
    echo "Summary: ${summary_path}"
    continue
  fi

  echo
  echo "===== chunk_size=${chunk} ====="
  echo "Policy: ${policy}"

  NUMBA_CACHE_DIR=/tmp/robosuite_numba_cache \
  PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "${ENV_NAME}" \
    python scripts/benchmark_robustness_v2.py \
      --policy "${policy}" \
      --perturbation target-displacement \
      --levels 0 0.04 \
      --episodes-per-level "${EPISODES_PER_LEVEL}" \
      --max-steps "${MAX_STEPS}" \
      --seed "${EVAL_SEED}" \
      --perturbation-seed "${PERTURBATION_SEED}" \
      --replan-interval 1 \
      --temporal-profile legacy \
      --ensemble-modes temporal latest-only \
      --local-files-only \
      --log-every 20 \
      --output-dir "${output_dir}" \
      2>&1 | tee "${log_path}"
done

echo
echo "Chunk-size ablation evaluation complete: ${OUTPUT_ROOT}"
