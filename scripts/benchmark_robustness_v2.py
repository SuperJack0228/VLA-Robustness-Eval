"""Paired robustness sweeps for the frozen MiniVLA V2 Clean policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.evaluation_core_v2 import (
    DEFAULT_POLICY_PATH,
    ENSEMBLE_MODES,
    MAX_STEPS,
    EvaluationConfig,
    EvaluationCore,
    get_device,
    load_policy,
    make_environment,
)
from utils.perturbations_v2 import (
    DynamicTargetDisplacement,
    PerturbationManager,
)


PERTURBATIONS = ("target-displacement",)
DEFAULT_LEVELS = (0.0, 0.02, 0.04, 0.06, 0.08)
DEFAULT_OUTPUT_DIR = "results/robustness/target_displacement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched scenes across perturbation levels and action "
            "ensemble modes."
        )
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--perturbation",
        choices=PERTURBATIONS,
        default="target-displacement",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Perturbation levels in meters.",
    )
    parser.add_argument("--episodes-per-level", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--perturbation-seed", type=int, default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--ensemble-modes",
        nargs="+",
        choices=ENSEMBLE_MODES,
        default=list(ENSEMBLE_MODES),
        help="Run both modes by default using identical scene seeds.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes_per_level <= 0:
        raise ValueError("episodes-per-level must be positive")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.replan_interval <= 0:
        raise ValueError("replan-interval must be positive")
    if not args.levels:
        raise ValueError("At least one perturbation level is required")
    if any(not np.isfinite(level) or level < 0.0 for level in args.levels):
        raise ValueError("Perturbation levels must be finite and non-negative")
    if len(set(args.levels)) != len(args.levels):
        raise ValueError("Perturbation levels must be unique")
    if len(set(args.ensemble_modes)) != len(args.ensemble_modes):
        raise ValueError("Ensemble modes must be unique")


def level_slug(level: float) -> str:
    return f"{int(round(level * 1000.0)):03d}mm"


def atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def finite_mean(values: list[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    return float(np.mean(finite)) if finite else None


def nonnegative_mean(values: list[int]) -> float | None:
    valid = [value for value in values if value >= 0]
    return float(np.mean(valid)) if valid else None


def compact_run_summary(rows: list[dict], report: dict) -> dict:
    return {
        "episodes": len(rows),
        "task_success_rate": report["overall"]["task_success_rate"],
        "clean_success_rate": report["overall"]["clean_success_rate"],
        "wrong_contact_rate": report["overall"]["wrong_contact_rate"],
        "failure_counts": dict(
            Counter(row["failure_category"] for row in rows)
        ),
        "injection_rate": float(np.mean([row["injected"] for row in rows])),
        "mean_actual_delta_m": finite_mean(
            [float(row["actual_delta_norm_m"]) for row in rows]
        ),
        "mean_reacquisition_latency": nonnegative_mean(
            [int(row["reacquisition_latency"]) for row in rows]
        ),
        "mean_post_injection_contact_latency": nonnegative_mean(
            [int(row["post_injection_contact_latency"]) for row in rows]
        ),
        "mean_grounding_error_after_injection_cm": finite_mean(
            [row["grounding_error_after_injection_cm"] for row in rows]
        ),
        "buckets": report["buckets"],
    }


def build_manager(
    perturbation: str,
    level: float,
    perturbation_seed: int,
) -> PerturbationManager:
    if perturbation == "target-displacement":
        return PerturbationManager(
            [DynamicTargetDisplacement(level, perturbation_seed)]
        )
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    perturbation_seed = (
        args.perturbation_seed
        if args.perturbation_seed is not None
        else args.seed + 2_000_003
    )
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(
        "Robustness execution: raw policy | privileged execution assistance: "
        "False | paired scenes: True",
        flush=True,
    )
    model, stats = load_policy(args.policy, device, args.local_files_only)
    all_rows: list[dict] = []
    run_records: list[dict] = []
    clean_reference: dict[str, float] = {}

    # Mode outermost keeps every mode's level sweep easy to resume and inspect.
    for mode in args.ensemble_modes:
        for level in args.levels:
            slug = level_slug(level)
            run_name = f"{args.perturbation}_{slug}_{mode}"
            output_prefix = os.path.join(args.output_dir, "runs", run_name)
            print(
                f"\n=== {args.perturbation} | level={level:.3f}m | "
                f"ensemble={mode} ===",
                flush=True,
            )
            manager = build_manager(
                args.perturbation,
                level,
                perturbation_seed,
            )
            config = EvaluationConfig(
                policy_path=args.policy,
                num_episodes=args.episodes_per_level,
                max_steps=args.max_steps,
                seed=args.seed,
                instruction=args.instruction,
                output_prefix=output_prefix,
                replan_interval=args.replan_interval,
                render=args.render,
                log_every=args.log_every,
                local_files_only=args.local_files_only,
                visual_perturbation="clean",
                ensemble_mode=mode,
                perturbation_label=args.perturbation,
                uses_privileged_perturbation_oracle=level > 0.0,
            )
            env = make_environment(args.render, args.max_steps)
            core = EvaluationCore(
                model=model,
                stats=stats,
                env=env,
                device=device,
                config=config,
                perturbation_manager=manager,
            )
            result = core.run(close_environment=True)
            for row in result.rows:
                row["benchmark_level"] = level
                row["benchmark_mode"] = mode
                all_rows.append(row)
            summary = compact_run_summary(result.rows, result.report)
            if level == 0.0:
                clean_reference[mode] = summary["clean_success_rate"]
            run_record = {
                "perturbation": args.perturbation,
                "level": level,
                "ensemble_mode": mode,
                "scene_seed": args.seed,
                "perturbation_seed": perturbation_seed,
                "run_csv": result.csv_path,
                "run_json": result.json_path,
                **summary,
            }
            run_records.append(run_record)
            atomic_write_csv(
                os.path.join(args.output_dir, "benchmark_episodes.csv"),
                all_rows,
            )
            partial = {
                "status": "running",
                "policy": args.policy,
                "perturbation": args.perturbation,
                "levels_m": args.levels,
                "ensemble_modes": args.ensemble_modes,
                "episodes_per_level": args.episodes_per_level,
                "seed": args.seed,
                "perturbation_seed": perturbation_seed,
                "paired_scenes": True,
                "runs": run_records,
            }
            atomic_write_json(
                os.path.join(args.output_dir, "benchmark_summary.json"),
                partial,
            )

    for record in run_records:
        baseline = clean_reference.get(record["ensemble_mode"])
        record["absolute_decay_from_clean"] = (
            None
            if baseline is None
            else baseline - record["clean_success_rate"]
        )
        record["relative_retention_from_clean"] = (
            None
            if baseline is None or baseline == 0.0
            else record["clean_success_rate"] / baseline
        )
    final_report = {
        "status": "complete",
        "policy": args.policy,
        "perturbation": args.perturbation,
        "levels_m": args.levels,
        "ensemble_modes": args.ensemble_modes,
        "episodes_per_level": args.episodes_per_level,
        "seed": args.seed,
        "perturbation_seed": perturbation_seed,
        "paired_scenes": True,
        "uses_privileged_execution_assistance": False,
        "uses_privileged_perturbation_oracle": any(
            level > 0.0 for level in args.levels
        ),
        "runs": run_records,
    }
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    episode_path = os.path.join(args.output_dir, "benchmark_episodes.csv")
    atomic_write_json(summary_path, final_report)
    atomic_write_csv(episode_path, all_rows)
    print(json.dumps(final_report, indent=2), flush=True)
    print(
        f"Robustness benchmark complete: {summary_path} and {episode_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
