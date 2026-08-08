"""Stage 0 gate for the deterministic six-task clean scripted oracle."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from collect_data_v2 import (
    TASK_BUCKETS,
    make_environment,
    object_poses,
    reseed_environment,
    rollout_oracle,
    schedule_task_v2,
)


DEFAULT_EPISODES_PER_TASK = 100
DEFAULT_SEED = 20260817
MIN_CLEAN_SUCCESS_RATE = 0.98
MAX_SATURATION_RATE = 0.05


def new_metrics() -> dict:
    return {
        "episodes": 0,
        "successes": 0,
        "clean_successes": 0,
        "wrong_object_contacts": 0,
        "scene_rejections": 0,
        "saturation_steps": 0,
        "trajectory_steps": 0,
        "total_retries": 0,
        "success_steps": [],
        "forced_push_recoveries": 0,
    }


def finalize_metrics(metrics: dict) -> dict:
    episodes = metrics["episodes"]
    trajectory_steps = metrics["trajectory_steps"]
    success_steps = metrics.pop("success_steps")
    metrics["success_rate"] = metrics["successes"] / episodes
    metrics["clean_success_rate"] = metrics["clean_successes"] / episodes
    metrics["wrong_contact_rate"] = metrics["wrong_object_contacts"] / episodes
    metrics["saturation_rate"] = (
        metrics["saturation_steps"] / trajectory_steps if trajectory_steps else 0.0
    )
    metrics["mean_retries"] = metrics["total_retries"] / episodes
    metrics["mean_success_step"] = (
        float(np.mean(success_steps)) if success_steps else None
    )
    metrics["passed"] = bool(
        metrics["clean_success_rate"] >= MIN_CLEAN_SUCCESS_RATE
        and metrics["saturation_rate"] <= MAX_SATURATION_RATE
    )
    return metrics


def run_benchmark(episodes_per_task: int, seed: int, quiet: bool = False) -> dict:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    env = make_environment(camera_observations=False)
    rng = np.random.default_rng(seed)
    buckets: dict[str, dict] = {}
    try:
        for task_type, target_id in TASK_BUCKETS:
            key = f"{task_type}_{target_id}"
            metrics = new_metrics()
            while metrics["episodes"] < episodes_per_task:
                scene_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                reseed_environment(env, scene_seed)
                obs = env.reset()
                _ = object_poses(env)
                try:
                    task = schedule_task_v2(
                        env,
                        obs,
                        task_type,
                        target_id,
                    )
                except RuntimeError:
                    metrics["scene_rejections"] += 1
                    continue

                result = rollout_oracle(
                    env,
                    task,
                    obs,
                    seed=scene_seed,
                    capture_images=False,
                )
                metrics["episodes"] += 1
                metrics["successes"] += int(result.success)
                metrics["clean_successes"] += int(result.clean_success)
                metrics["wrong_object_contacts"] += int(
                    result.wrong_object_contact
                )
                metrics["saturation_steps"] += result.saturation_steps
                metrics["trajectory_steps"] += result.trajectory_length
                metrics["total_retries"] += result.retry_count
                metrics["forced_push_recoveries"] += int(
                    result.forced_push_recovery
                )
                if result.success:
                    metrics["success_steps"].append(result.success_step)
                if not quiet:
                    print(
                        f"[{key}] {metrics['episodes']}/{episodes_per_task} | "
                        f"clean={result.clean_success} | "
                        f"steps={result.trajectory_length} | "
                        f"retry={result.retry_count} | "
                        f"phases={result.phase_counts} | "
                        f"pos_err={result.final_position_error:.3f} | "
                        f"rot_err={result.final_orientation_error:.3f} | "
                        f"push=({result.push_forward_displacement:.3f}, "
                        f"{result.push_lateral_displacement:.3f})",
                        flush=True,
                    )
            buckets[key] = finalize_metrics(metrics)
    finally:
        env.close()

    overall_episodes = sum(item["episodes"] for item in buckets.values())
    overall_steps = sum(item["trajectory_steps"] for item in buckets.values())
    overall = {
        "episodes": overall_episodes,
        "success_rate": sum(item["successes"] for item in buckets.values())
        / overall_episodes,
        "clean_success_rate": sum(
            item["clean_successes"] for item in buckets.values()
        )
        / overall_episodes,
        "wrong_contact_rate": sum(
            item["wrong_object_contacts"] for item in buckets.values()
        )
        / overall_episodes,
        "saturation_rate": sum(
            item["saturation_steps"] for item in buckets.values()
        )
        / overall_steps,
    }
    overall["passed"] = bool(
        all(item["passed"] for item in buckets.values())
        and overall["clean_success_rate"] >= MIN_CLEAN_SUCCESS_RATE
        and overall["saturation_rate"] <= MAX_SATURATION_RATE
    )
    return {
        "seed": seed,
        "episodes_per_task": episodes_per_task,
        "thresholds": {
            "minimum_clean_success_rate": MIN_CLEAN_SUCCESS_RATE,
            "maximum_saturation_rate": MAX_SATURATION_RATE,
        },
        "buckets": buckets,
        "overall": overall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=DEFAULT_EPISODES_PER_TASK,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        default="results/benchmarks/oracle/oracle_benchmark_v2_clean.json",
    )
    parser.add_argument(
        "--no-enforce",
        action="store_true",
        help="Write metrics without failing the process when gates are missed.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.episodes_per_task, args.seed, quiet=args.quiet)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if not report["overall"]["passed"] and not args.no_enforce:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
