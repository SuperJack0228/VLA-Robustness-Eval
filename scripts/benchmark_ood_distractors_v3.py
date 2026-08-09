#!/usr/bin/env python3
"""Run paired MiniVLA V3 evaluation with 0-3 unseen OOD distractors."""

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
import robosuite

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_data_v2 import (  # noqa: E402
    CONTROL_FREQ,
    build_osc_pose_controller_config_v2,
)
from utils.evaluation_core_v2 import (  # noqa: E402
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
)
from utils.ood_distractors_v3 import (  # noqa: E402
    OOD_DISTRACTOR_IDS,
    MultiObjectVLAOODEnvV3,
    OODDistractorInjection,
)
from utils.perturbations_v2 import PerturbationManager  # noqa: E402
from utils.v2_schema import ACTION_DIM, IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402


PROTOCOL_VERSION = "ood-distractor.v1"
DEFAULT_POLICY = "artifacts/v3-clean-rc1/mini_vla_v3_policy.pth"
DEFAULT_OUTPUT_DIR = "results/benchmarks/ood_distractors"
DEFAULT_COUNTS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
        help="Number of active OOD distractors; supported values are 0-3.",
    )
    parser.add_argument("--episodes-per-count", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=20262101)
    parser.add_argument("--perturbation-seed", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--task-type", choices=("pick", "push"), default=None)
    parser.add_argument("--target-id", choices=("A", "B", "C"), default=None)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--ensemble-mode", choices=ENSEMBLE_MODES, default="temporal")
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
        help="Reuse complete per-count CSV/JSON outputs in output-dir.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.counts or len(set(args.counts)) != len(args.counts):
        raise ValueError("counts must be non-empty and unique")
    unsupported = sorted(set(args.counts) - set(DEFAULT_COUNTS))
    if unsupported:
        raise ValueError(f"Unsupported OOD distractor counts: {unsupported}")
    if 0 not in args.counts:
        raise ValueError("counts must include the clean count 0")
    if args.episodes_per_count <= 0:
        raise ValueError("episodes-per-count must be positive")
    if args.max_steps <= 0 or args.replan_interval <= 0:
        raise ValueError("max-steps and replan-interval must be positive")
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
        raise ValueError("grounding-reset-threshold must be finite and non-negative")


def make_ood_environment(render: bool, horizon: int):
    _ = MultiObjectVLAOODEnvV3  # Import registers the environment with robosuite.
    camera_names = ["agentview", "robot0_eye_in_hand"]
    camera_heights = [IMAGE_HEIGHT, IMAGE_HEIGHT]
    camera_widths = [IMAGE_WIDTH, IMAGE_WIDTH]
    if render:
        camera_names.append("frontview")
        camera_heights.append(512)
        camera_widths.append(512)
    env = robosuite.make(
        env_name="MultiObjectVLAOODEnvV3",
        robots="Panda",
        controller_configs=build_osc_pose_controller_config_v2(),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        horizon=horizon,
        ignore_done=True,
        control_freq=CONTROL_FREQ,
        hard_reset=False,
    )
    low, high = env.action_spec
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        env.close()
        raise RuntimeError(f"Expected 7D OSC_POSE action space, got {low.shape}")
    return env


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


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true"}


def compact_summary(rows: list[dict], report: dict) -> dict:
    collision_aware = [
        as_bool(row, "task_success")
        and not as_bool(row, "wrong_object_contact")
        and not as_bool(row, "ood_wrong_object_contact")
        for row in rows
    ]
    return {
        "episodes": len(rows),
        "task_success_rate": report["overall"]["task_success_rate"],
        "collision_aware_success_rate": float(np.mean(collision_aware)),
        "wrong_trained_object_contact_rate": report["overall"]["wrong_contact_rate"],
        "ood_wrong_object_contact_rate": finite_mean(rows, "ood_wrong_object_contact"),
        "ood_collision_rate": finite_mean(rows, "ood_collision"),
        "ood_target_selection_failure_rate": finite_mean(
            rows,
            "ood_target_selection_failure",
        ),
        "mean_grounding_error_cm": finite_mean(rows, "mean_grounding_error_cm"),
        "mean_steps": finite_mean(rows, "steps"),
        "failure_counts": dict(Counter(str(row["failure_category"]) for row in rows)),
        "buckets": report["buckets"],
    }


def paired_signature(rows: list[dict]) -> list[tuple]:
    return [
        (
            int(row["episode"]),
            str(row["task_type"]),
            str(row["target_id"]),
            int(row["scene_seed"]),
            str(row["ood_layout_signature"]),
        )
        for row in rows
    ]


def enrich_row(row: dict, count: int) -> dict:
    enriched = dict(row)
    ood_contact = as_bool(row, "ood_wrong_object_contact")
    trained_contact = as_bool(row, "wrong_object_contact")
    task_success = as_bool(row, "task_success")
    enriched["ood_clean_success"] = int(
        task_success and not ood_contact and not trained_contact
    )
    if task_success and not ood_contact and not trained_contact:
        ood_category = "success"
    elif ood_contact:
        ood_category = "ood_distractor_contact"
    elif as_bool(row, "ood_target_selection_failure"):
        ood_category = "ood_target_selection_failure"
    else:
        ood_category = str(row["failure_category"])
    enriched.update(
        {
            "ood_failure_category": ood_category,
            "benchmark_count": count,
            "benchmark_condition": f"ood-distractors_count_{count}",
        }
    )
    return enriched


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perturbation_seed = (
        args.perturbation_seed
        if args.perturbation_seed is not None
        else args.seed + 6_000_011
    )

    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(
        "OOD distractors: unseen static geometry | safe paired layouts | "
        "raw frozen policy | privileged assistance: False",
        flush=True,
    )
    model, stats = load_policy(args.policy, device, args.local_files_only)
    all_rows: list[dict] = []
    run_records: list[dict] = []
    reference_signature: list[tuple] | None = None

    for index, count in enumerate(args.counts, start=1):
        condition = f"ood-distractors_count_{count}"
        output_prefix = output_dir / "runs" / condition
        print(
            f"\n=== OOD count {index}/{len(args.counts)} | active={count} ===",
            flush=True,
        )
        completed = (
            read_completed_run(output_prefix, args.episodes_per_count)
            if args.resume
            else None
        )
        if completed is not None:
            rows, report = completed
            print(f"[Resume] Reusing {len(rows)} episodes from {output_prefix}", flush=True)
        else:
            manager = PerturbationManager(
                [
                    OODDistractorInjection(
                        count=count,
                        base_seed=perturbation_seed,
                        validation_count=max(DEFAULT_COUNTS),
                    )
                ]
            )
            config = EvaluationConfig(
                policy_path=args.policy,
                num_episodes=args.episodes_per_count,
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
                grounding_shift_reset_threshold_m=args.grounding_reset_threshold,
                perturbation_label=f"ood-distractors:count-{count}",
                uses_privileged_perturbation_oracle=False,
                diagnostic_trace=args.diagnostic_trace,
            )
            env = make_ood_environment(args.render, args.max_steps)
            result = EvaluationCore(
                model=model,
                stats=stats,
                env=env,
                device=device,
                config=config,
                perturbation_manager=manager,
            ).run(close_environment=True)
            rows, report = result.rows, result.report

        signature = paired_signature(rows)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError("Paired-scene or paired-layout protocol violation")
        expected_active = "|".join(OOD_DISTRACTOR_IDS[:count])
        if any(str(row["ood_active_ids"]) != expected_active for row in rows):
            raise RuntimeError(f"Active distractor prefix mismatch at count {count}")

        enriched_rows = [enrich_row(row, count) for row in rows]
        all_rows.extend(enriched_rows)
        run_records.append(
            {
                "count": count,
                "condition": condition,
                "scene_seed": args.seed,
                "perturbation_seed": perturbation_seed,
                "run_csv": str(output_prefix.with_suffix(".csv")),
                "run_json": str(output_prefix.with_suffix(".json")),
                **compact_summary(rows, report),
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
                "episodes_per_count": args.episodes_per_count,
                "counts": args.counts,
                "paired_scenes": True,
                "paired_layouts": True,
                "layout_prefix_activation": True,
                "completed_conditions": len(run_records),
                "total_conditions": len(args.counts),
                "runs": run_records,
            },
        )

    final_report = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "policy": args.policy,
        "seed": args.seed,
        "perturbation_seed": perturbation_seed,
        "episodes_per_count": args.episodes_per_count,
        "ensemble_mode": args.ensemble_mode,
        "temporal_profile": args.temporal_profile,
        "replan_interval": args.replan_interval,
        "counts": args.counts,
        "paired_scenes": True,
        "paired_scene_validation_passed": True,
        "paired_layouts": True,
        "paired_layout_validation_passed": True,
        "layout_prefix_activation": True,
        "uses_privileged_execution_assistance": False,
        "uses_privileged_perturbation_oracle": False,
        "geometry_labels": {
            "D1": "magenta horizontal capsule",
            "D2": "yellow ellipsoid",
            "D3": "cyan crossed capsules",
        },
        "completed_conditions": len(run_records),
        "total_conditions": len(args.counts),
        "runs": run_records,
    }
    atomic_write_csv(output_dir / "benchmark_episodes.csv", all_rows)
    atomic_write_json(output_dir / "benchmark_summary.json", final_report)
    print(
        f"OOD distractor benchmark complete: {output_dir / 'benchmark_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
