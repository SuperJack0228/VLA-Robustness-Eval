#!/usr/bin/env python3
"""Run paired MiniVLA V3 visual-degradation robustness curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation_core_v2 import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    ENSEMBLE_DECAY,
    ENSEMBLE_MODES,
    GROUNDING_SHIFT_RESET_THRESHOLD_M,
    MAX_STEPS,
    MAX_TEMPORAL_PREDICTION_AGE,
    TEMPORAL_PROFILES,
    EvaluationConfig,
    EvaluationCore,
    get_device,
    load_policy,
    make_environment,
)
from utils.perturbations_v2 import (  # noqa: E402
    PerturbationManager,
    VisualDegradation,
)


PROTOCOL_VERSION = "visual-degradation.v1"
DEFAULT_POLICY = "artifacts/v3-clean-rc1/mini_vla_v3_policy.pth"
DEFAULT_OUTPUT_DIR = "results/benchmarks/visual_degradation"
# The first portion preserves the completed mild-regime protocol. The final four
# levels deliberately extend each curve into severe corruption so the benchmark
# can identify the policy's practical failure boundary.
DEFAULT_NOISE_LEVELS = (0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0)
DEFAULT_BLUR_LEVELS = (0.0, 0.75, 1.5, 2.25, 3.0, 4.0, 6.0, 10.0)
DEFAULT_BRIGHTNESS_LEVELS = (
    1.0,
    0.8,
    0.6,
    0.4,
    0.3,
    0.2,
    0.1,
    0.05,
    0.02,
    0.01,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_NOISE_LEVELS),
        help="Gaussian RGB noise standard deviations in 0-255 pixel units.",
    )
    parser.add_argument(
        "--blur-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_BLUR_LEVELS),
        help="Gaussian blur sigma values in image pixels.",
    )
    parser.add_argument(
        "--brightness-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_BRIGHTNESS_LEVELS),
        help="Multiplicative RGB intensity gains.",
    )
    parser.add_argument("--episodes-per-level", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=20262001)
    parser.add_argument("--perturbation-seed", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--task-type", choices=("pick", "push"), default=None)
    parser.add_argument("--target-id", choices=("A", "B", "C"), default=None)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument(
        "--ensemble-mode",
        choices=ENSEMBLE_MODES,
        default="temporal",
    )
    parser.add_argument(
        "--temporal-profile",
        choices=TEMPORAL_PROFILES,
        default="robust",
    )
    parser.add_argument("--ensemble-decay", type=float, default=ENSEMBLE_DECAY)
    parser.add_argument(
        "--max-prediction-age",
        type=int,
        default=MAX_TEMPORAL_PREDICTION_AGE,
    )
    parser.add_argument(
        "--grounding-reset-threshold",
        type=float,
        default=GROUNDING_SHIFT_RESET_THRESHOLD_M,
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--diagnostic-trace", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete per-condition CSV/JSON outputs in output-dir.",
    )
    return parser.parse_args()


def validate_levels(name: str, levels: list[float], identity: float) -> None:
    if not levels:
        raise ValueError(f"{name} requires at least one level")
    if any(not np.isfinite(level) for level in levels):
        raise ValueError(f"{name} levels must be finite")
    if name == "brightness":
        if any(level <= 0.0 for level in levels):
            raise ValueError("brightness levels must be positive")
    elif any(level < 0.0 for level in levels):
        raise ValueError(f"{name} levels must be non-negative")
    if len(set(levels)) != len(levels):
        raise ValueError(f"{name} levels must be unique")
    if not any(np.isclose(level, identity) for level in levels):
        raise ValueError(f"{name} levels must include clean identity {identity}")


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes_per_level <= 0:
        raise ValueError("episodes-per-level must be positive")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.replan_interval <= 0:
        raise ValueError("replan-interval must be positive")
    if args.log_every < 0:
        raise ValueError("log-every must be non-negative")
    if not np.isfinite(args.ensemble_decay) or args.ensemble_decay < 0.0:
        raise ValueError("ensemble-decay must be finite and non-negative")
    if args.max_prediction_age < 0:
        raise ValueError("max-prediction-age must be non-negative")
    if (
        not np.isfinite(args.grounding_reset_threshold)
        or args.grounding_reset_threshold < 0.0
    ):
        raise ValueError(
            "grounding-reset-threshold must be finite and non-negative"
        )
    validate_levels("gaussian-noise", args.noise_levels, 0.0)
    validate_levels("gaussian-blur", args.blur_levels, 0.0)
    validate_levels("brightness", args.brightness_levels, 1.0)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def level_token(level: float) -> str:
    return f"{level:g}".replace("-", "m").replace(".", "p")


def build_conditions(args: argparse.Namespace) -> list[dict]:
    conditions = [
        {
            "corruption": "clean",
            "level": 0.0,
            "unit": "identity",
            "condition": "clean",
        }
    ]
    specifications = (
        ("gaussian-noise", args.noise_levels, 0.0, "pixel_std_0_255"),
        ("gaussian-blur", args.blur_levels, 0.0, "sigma_pixels"),
        ("brightness", args.brightness_levels, 1.0, "rgb_gain"),
    )
    for corruption, levels, identity, unit in specifications:
        for level in levels:
            if np.isclose(level, identity):
                continue
            conditions.append(
                {
                    "corruption": corruption,
                    "level": float(level),
                    "unit": unit,
                    "condition": f"{corruption}_{level_token(level)}",
                }
            )
    return conditions


def read_completed_run(
    output_prefix: Path,
    expected_episodes: int,
) -> tuple[list[dict], dict] | None:
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    if not csv_path.is_file() or not json_path.is_file():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        return None
    with json_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    return rows, report


def numeric(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, "", "None"):
        return float("nan")
    return float(value)


def finite_mean(rows: list[dict], key: str) -> float | None:
    values = [numeric(row, key) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def compact_summary(rows: list[dict], report: dict) -> dict:
    return {
        "episodes": len(rows),
        "task_success_rate": report["overall"]["task_success_rate"],
        "clean_success_rate": report["overall"]["clean_success_rate"],
        "wrong_contact_rate": report["overall"]["wrong_contact_rate"],
        "strict_target_contact_rate": report["overall"][
            "strict_target_contact_rate"
        ],
        "mean_steps": finite_mean(rows, "steps"),
        "mean_action_clip_rate": finite_mean(rows, "action_clip_rate"),
        "mean_grounding_error_cm": finite_mean(
            rows,
            "mean_grounding_error_cm",
        ),
        "mean_target_class_accuracy": finite_mean(
            rows,
            "mean_target_class_accuracy",
        ),
        "mean_absolute_pixel_delta": finite_mean(
            rows,
            "mean_absolute_pixel_delta",
        ),
        "mean_input_psnr_db": finite_mean(rows, "input_psnr_db"),
        "mean_frames_changed": finite_mean(rows, "frames_changed"),
        "failure_counts": dict(
            Counter(str(row["failure_category"]) for row in rows)
        ),
        "buckets": report["buckets"],
    }


def scene_signature(rows: list[dict]) -> list[tuple]:
    return [
        (
            int(row["episode"]),
            str(row["task_type"]),
            str(row["target_id"]),
            int(row["scene_seed"]),
        )
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perturbation_seed = (
        args.perturbation_seed
        if args.perturbation_seed is not None
        else args.seed + 3_000_007
    )
    conditions = build_conditions(args)

    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(
        "Visual robustness: paired scenes | both policy cameras | "
        "raw frozen policy | privileged assistance: False",
        flush=True,
    )
    model, stats = load_policy(args.policy, device, args.local_files_only)
    all_rows: list[dict] = []
    run_records: list[dict] = []
    reference_signature: list[tuple] | None = None
    clean_success_rate: float | None = None

    for index, condition in enumerate(conditions, start=1):
        corruption = condition["corruption"]
        level = condition["level"]
        condition_name = condition["condition"]
        output_prefix = output_dir / "runs" / condition_name
        print(
            f"\n=== Visual condition {index}/{len(conditions)} | "
            f"{corruption} | level={level:g} {condition['unit']} ===",
            flush=True,
        )

        completed = (
            read_completed_run(output_prefix, args.episodes_per_level)
            if args.resume
            else None
        )
        if completed is not None:
            rows, report = completed
            print(
                f"[Resume] Reusing {len(rows)} episodes from {output_prefix}",
                flush=True,
            )
        else:
            degradation = VisualDegradation(
                degradation=corruption,
                level=level,
                base_seed=perturbation_seed,
            )
            manager = PerturbationManager([degradation])
            config = EvaluationConfig(
                policy_path=args.policy,
                num_episodes=args.episodes_per_level,
                max_steps=args.max_steps,
                seed=args.seed,
                instruction=args.instruction,
                task_type=args.task_type,
                target_id=args.target_id,
                output_prefix=str(output_prefix),
                replan_interval=args.replan_interval,
                render=args.render,
                log_every=args.log_every,
                local_files_only=args.local_files_only,
                visual_perturbation="clean",
                ensemble_mode=args.ensemble_mode,
                temporal_profile=args.temporal_profile,
                ensemble_decay=args.ensemble_decay,
                max_prediction_age=args.max_prediction_age,
                grounding_shift_reset_threshold_m=(
                    args.grounding_reset_threshold
                ),
                perturbation_label=f"visual:{corruption}",
                uses_privileged_perturbation_oracle=False,
                diagnostic_trace=args.diagnostic_trace,
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
            rows, report = result.rows, result.report

        signature = scene_signature(rows)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError(
                "Paired-scene protocol violation across visual conditions"
            )

        for row in rows:
            enriched = dict(row)
            enriched["benchmark_corruption"] = corruption
            enriched["benchmark_level"] = level
            enriched["benchmark_level_unit"] = condition["unit"]
            enriched["benchmark_condition"] = condition_name
            all_rows.append(enriched)

        summary = compact_summary(rows, report)
        if corruption == "clean":
            clean_success_rate = float(summary["task_success_rate"])
        run_records.append(
            {
                **condition,
                "scene_seed": args.seed,
                "perturbation_seed": perturbation_seed,
                "run_csv": str(output_prefix.with_suffix(".csv")),
                "run_json": str(output_prefix.with_suffix(".json")),
                **summary,
            }
        )
        atomic_write_csv(output_dir / "benchmark_episodes.csv", all_rows)
        atomic_write_json(
            output_dir / "benchmark_summary.json",
            {
                "status": "running",
                "protocol_version": PROTOCOL_VERSION,
                "policy": args.policy,
                "seed": args.seed,
                "perturbation_seed": perturbation_seed,
                "episodes_per_level": args.episodes_per_level,
                "ensemble_mode": args.ensemble_mode,
                "temporal_profile": args.temporal_profile,
                "paired_scenes": True,
                "paired_scene_validation_passed": True,
                "affected_cameras": [
                    "agentview",
                    "robot0_eye_in_hand",
                ],
                "curves": {
                    "gaussian-noise": args.noise_levels,
                    "gaussian-blur": args.blur_levels,
                    "brightness": args.brightness_levels,
                },
                "completed_conditions": len(run_records),
                "total_conditions": len(conditions),
                "runs": run_records,
            },
        )

    for record in run_records:
        rate = float(record["task_success_rate"])
        record["absolute_decay_from_clean"] = (
            None if clean_success_rate is None else clean_success_rate - rate
        )
        record["relative_retention_from_clean"] = (
            None
            if clean_success_rate in (None, 0.0)
            else rate / clean_success_rate
        )

    final_report = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "policy": args.policy,
        "seed": args.seed,
        "perturbation_seed": perturbation_seed,
        "episodes_per_level": args.episodes_per_level,
        "ensemble_mode": args.ensemble_mode,
        "temporal_profile": args.temporal_profile,
        "replan_interval": args.replan_interval,
        "paired_scenes": True,
        "paired_scene_validation_passed": True,
        "uses_privileged_execution_assistance": False,
        "uses_privileged_perturbation_oracle": False,
        "affected_cameras": ["agentview", "robot0_eye_in_hand"],
        "curves": {
            "gaussian-noise": args.noise_levels,
            "gaussian-blur": args.blur_levels,
            "brightness": args.brightness_levels,
        },
        "completed_conditions": len(run_records),
        "total_conditions": len(conditions),
        "runs": run_records,
    }
    atomic_write_csv(output_dir / "benchmark_episodes.csv", all_rows)
    atomic_write_json(output_dir / "benchmark_summary.json", final_report)
    print(
        f"Visual benchmark complete: {output_dir / 'benchmark_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
