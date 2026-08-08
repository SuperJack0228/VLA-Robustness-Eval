"""Select the blue-ball push height with matched deterministic scenes."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from collect_data_v2 import (
    BALL_PUSH_HEIGHT_ABOVE_TABLE,
    BALL_PUSH_HEIGHT_CANDIDATES,
    make_environment,
    reseed_environment,
    rollout_oracle,
    schedule_task_v2,
)


def benchmark(episodes: int, seed: int) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = make_environment(camera_observations=False)
    rng = np.random.default_rng(seed)
    scene_seeds: list[int] = []
    try:
        while len(scene_seeds) < episodes:
            scene_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            reseed_environment(env, scene_seed)
            obs = env.reset()
            try:
                schedule_task_v2(env, obs, "push", "B")
            except RuntimeError:
                continue
            scene_seeds.append(scene_seed)

        reports = {}
        for height in BALL_PUSH_HEIGHT_CANDIDATES:
            results = []
            for scene_seed in scene_seeds:
                reseed_environment(env, scene_seed)
                obs = env.reset()
                task = schedule_task_v2(env, obs, "push", "B")
                results.append(
                    rollout_oracle(
                        env,
                        task,
                        obs,
                        seed=scene_seed,
                        capture_images=False,
                        ball_push_height_above_table=height,
                    )
                )
            successes = [result for result in results if result.clean_success]
            reports[f"{height:.3f}"] = {
                "height_above_table_m": height,
                "clean_success_rate": len(successes) / episodes,
                "wrong_contact_rate": sum(
                    result.wrong_object_contact for result in results
                )
                / episodes,
                "mean_success_step": (
                    float(np.mean([result.success_step for result in successes]))
                    if successes
                    else None
                ),
                "mean_forward_displacement_m": float(
                    np.mean([result.push_forward_displacement for result in results])
                ),
                "saturation_rate": sum(
                    result.saturation_steps for result in results
                )
                / max(sum(result.trajectory_length for result in results), 1),
            }
    finally:
        env.close()

    def score(item: tuple[str, dict]) -> tuple[float, float, float]:
        values = item[1]
        success_step = values["mean_success_step"]
        return (
            values["clean_success_rate"],
            -(success_step if success_step is not None else float("inf")),
            -values["saturation_rate"],
        )

    selected = max(reports.items(), key=score)[1]["height_above_table_m"]
    return {
        "episodes_per_height": episodes,
        "seed": seed,
        "matched_scene_seeds": scene_seeds,
        "candidates": reports,
        "selected_height_above_table_m": selected,
        "configured_height_above_table_m": BALL_PUSH_HEIGHT_ABOVE_TABLE,
        "configuration_matches_benchmark": bool(
            np.isclose(selected, BALL_PUSH_HEIGHT_ABOVE_TABLE)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--output",
        default="results/benchmarks/diagnostics/blue_push_height_v2_clean.json",
    )
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args()
    report = benchmark(args.episodes, args.seed)
    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if not report["configuration_matches_benchmark"] and not args.no_enforce:
        raise SystemExit(
            "Configured blue-ball contact height does not match benchmark winner"
        )


if __name__ == "__main__":
    main()
