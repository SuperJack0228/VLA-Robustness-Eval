"""Collect additive recovery demonstrations for MiniVLA V3.

The frozen V2 Clean archives remain untouched. This collector writes a separate,
V2-schema-compatible supplement containing only clean successful recoveries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import collect_data_v2 as v2
from utils.perturbations_v2 import (
    DynamicTargetDisplacement,
    PerturbationManager,
    PerturbationSceneRejected,
)
from utils.v2_schema import TASK_BUCKETS, read_metadata


DEFAULT_NUM_EPISODES = 600
DEFAULT_DATA_DIR = "data/dataset_v3_recovery"
DEFAULT_SEED = 20261201
RECOVERY_VERSION = "v3.recovery.1"
DYNAMIC_MIN_DISTANCE = 0.02
DYNAMIC_MAX_DISTANCE = 0.05
DYNAMIC_DISTANCE_LEVELS = (0.02, 0.03, 0.04, 0.05)
FORCED_PICK_MISS_MIN = 0.038
FORCED_PICK_MISS_MAX = 0.050
RECOVERY_TYPES = {
    "pick": ("target_displacement", "forced_pick_miss"),
    "push": ("target_displacement", "forced_push_miss"),
}


class RecoveryOracleV3(v2.ScriptedOracleV2):
    """V2 oracle with a deterministic first-attempt miss when requested."""

    def __init__(
        self,
        env,
        task,
        obs: dict,
        seed: int,
        recovery_type: str,
    ) -> None:
        super().__init__(env, task, obs, seed)
        self.recovery_type = recovery_type
        recovery_rng = np.random.default_rng(seed + 91_337)
        angle = float(recovery_rng.uniform(0.0, 2.0 * np.pi))
        magnitude = float(
            recovery_rng.uniform(
                FORCED_PICK_MISS_MIN,
                FORCED_PICK_MISS_MAX,
            )
        )
        self.forced_pick_offset = magnitude * np.asarray(
            [np.cos(angle), np.sin(angle)],
            dtype=np.float64,
        )
        if recovery_type == "forced_push_miss":
            self.force_push_recovery = True

    def _pick_object_position(self) -> np.ndarray:
        position = super()._pick_object_position().copy()
        if (
            self.recovery_type == "forced_pick_miss"
            and self.retry_count == 0
            and self.phase in {0, 1, 2}
        ):
            position[:2] += self.forced_pick_offset
        return position


def recovery_type_for(bucket: tuple[str, str], accepted: int) -> str:
    options = RECOVERY_TYPES[bucket[0]]
    return options[accepted % len(options)]


def _metadata_from_archive(path: str) -> dict:
    metadata = read_metadata(path)
    with np.load(path, allow_pickle=False) as episode:
        if "v3_recovery_version" not in episode.files:
            raise ValueError(f"{path} is not a V3 recovery archive")
        metadata.update(
            {
                "recovery_type": str(episode["recovery_type"].item()),
                "v3_recovery_version": str(
                    episode["v3_recovery_version"].item()
                ),
                "perturbation_step": int(
                    episode["perturbation_step"].item()
                ),
                "perturbation_delta_m": float(
                    np.linalg.norm(episode["perturbation_delta"])
                ),
                "retry_count": int(
                    episode["retry_count"][
                        : int(episode["trajectory_length"].item())
                    ].max(initial=0)
                ),
            }
        )
    if metadata["v3_recovery_version"] != RECOVERY_VERSION:
        raise ValueError(f"{path} has an unexpected recovery version")
    return metadata


def _existing_state(data_dir: str) -> tuple[dict, set[int], int, list[dict]]:
    counts = {bucket: 0 for bucket in TASK_BUCKETS}
    seeds: set[int] = set()
    max_id = 0
    rows = []
    if not os.path.isdir(data_dir):
        return counts, seeds, max_id, rows
    for filename in sorted(os.listdir(data_dir)):
        if not filename.startswith("ep_") or not filename.endswith(".npz"):
            continue
        path = os.path.join(data_dir, filename)
        v2.verify_archive(path)
        metadata = _metadata_from_archive(path)
        bucket = (metadata["task_type"], metadata["target_id"])
        counts[bucket] += 1
        if metadata["scene_seed"] in seeds:
            raise ValueError(f"Duplicate scene seed in {path}")
        seeds.add(metadata["scene_seed"])
        rows.append(metadata)
        max_id = max(max_id, int(filename[3:8]))
    return counts, seeds, max_id, rows


def _write_manifest(data_dir: str, rows: list[dict]) -> None:
    path = os.path.join(data_dir, "dataset_manifest_v3_recovery.csv")
    temporary = f"{path}.tmp"
    fieldnames = [
        "filename",
        "split",
        "task_type",
        "target_id",
        "scene_seed",
        "collection_seed",
        "recovery_type",
        "perturbation_step",
        "perturbation_delta_m",
        "retry_count",
        "v3_recovery_version",
    ]
    with open(temporary, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: value["filename"]):
            writer.writerow({key: row[key] for key in fieldnames})
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_summary(
    data_dir: str,
    requested_total: int,
    collection_seed: int,
    attempts: int,
    rows: list[dict],
) -> None:
    bucket_counts = Counter(
        f"{row['task_type']}_{row['target_id']}" for row in rows
    )
    recovery_counts = Counter(row["recovery_type"] for row in rows)
    payload = {
        "recovery_version": RECOVERY_VERSION,
        "requested_total": requested_total,
        "accepted_total": len(rows),
        "attempts": attempts,
        "collection_seed": collection_seed,
        "base_dataset": "data/dataset_v2_clean",
        "buckets": dict(sorted(bucket_counts.items())),
        "recovery_types": dict(sorted(recovery_counts.items())),
    }
    path = os.path.join(data_dir, "collection_summary_v3_recovery.json")
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _resume_attempt_count(data_dir: str, accepted_total: int) -> int:
    path = os.path.join(
        data_dir,
        "collection_summary_v3_recovery.json",
    )
    if not os.path.isfile(path):
        return accepted_total
    try:
        with open(path, "r", encoding="utf-8") as summary_file:
            payload = json.load(summary_file)
        return max(int(payload.get("attempts", 0)), accepted_total)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return accepted_total


def verify_dataset(data_dir: str, expected_total: int | None = None) -> int:
    counts, _, _, rows = _existing_state(data_dir)
    if not rows:
        raise FileNotFoundError(f"No V3 recovery episodes found in {data_dir}")
    if expected_total is not None and len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} episodes, found {len(rows)}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"V3 recovery data is not task-balanced: {counts}")
    recovery_counts = Counter(
        (
            row["task_type"],
            row["target_id"],
            row["recovery_type"],
        )
        for row in rows
    )
    for bucket in TASK_BUCKETS:
        expected_types = RECOVERY_TYPES[bucket[0]]
        type_counts = [
            recovery_counts[(bucket[0], bucket[1], recovery_type)]
            for recovery_type in expected_types
        ]
        if len(set(type_counts)) != 1:
            raise ValueError(
                f"{bucket} recovery modes are imbalanced: "
                f"{dict(zip(expected_types, type_counts))}"
            )
        dynamic_distances = [
            row["perturbation_delta_m"]
            for row in rows
            if (
                row["task_type"],
                row["target_id"],
                row["recovery_type"],
            )
            == (bucket[0], bucket[1], "target_displacement")
        ]
        if len(dynamic_distances) >= 8:
            nearest_levels = {
                min(
                    DYNAMIC_DISTANCE_LEVELS,
                    key=lambda level: abs(level - distance),
                )
                for distance in dynamic_distances
            }
            if nearest_levels != set(DYNAMIC_DISTANCE_LEVELS):
                raise ValueError(
                    f"{bucket} does not cover all displacement levels: "
                    f"{sorted(nearest_levels)}"
                )
    for row in rows:
        recovery_type = row["recovery_type"]
        if recovery_type == "target_displacement":
            if not (
                DYNAMIC_MIN_DISTANCE - 1e-4
                <= row["perturbation_delta_m"]
                <= DYNAMIC_MAX_DISTANCE + 1e-4
            ):
                raise ValueError(
                    f"{row['filename']} has an invalid displacement"
                )
            if row["perturbation_step"] <= 0:
                raise ValueError(
                    f"{row['filename']} never injected its displacement"
                )
        elif recovery_type in {"forced_pick_miss", "forced_push_miss"}:
            if row["retry_count"] < 1:
                raise ValueError(
                    f"{row['filename']} lacks a completed retry"
                )
        else:
            raise ValueError(f"Unknown recovery type: {recovery_type}")
    _write_manifest(data_dir, rows)
    print(f"Verified {len(rows)} V3 recovery archives.", flush=True)
    for bucket in TASK_BUCKETS:
        print(
            f"  {bucket[0]}_{bucket[1]}: {counts[bucket]}",
            flush=True,
        )
    return len(rows)


def _dynamic_manager(
    task,
    obs: dict,
    env,
    scene_seed: int,
    episode_id: int,
    distance: float,
) -> PerturbationManager:
    manager = PerturbationManager(
        [
            DynamicTargetDisplacement(
                distance_m=distance,
                base_seed=scene_seed + 7_919,
                validation_distances_m=(DYNAMIC_MAX_DISTANCE,),
            )
        ]
    )
    manager.on_episode_start(episode_id, scene_seed, env, task, obs)
    return manager


def collect(
    num_episodes: int,
    data_dir: str,
    collection_seed: int,
) -> None:
    if num_episodes % (2 * len(TASK_BUCKETS)):
        raise ValueError(
            "num-episodes must be divisible by 12 for exact task and "
            "recovery-mode balance"
        )
    os.makedirs(data_dir, exist_ok=True)
    quota = num_episodes // len(TASK_BUCKETS)
    counts, used_seeds, episode_id, rows = _existing_state(data_dir)
    if any(count > quota for count in counts.values()):
        raise ValueError(f"Existing data exceeds requested quota: {counts}")
    if sum(counts.values()) == num_episodes:
        verify_dataset(data_dir, expected_total=num_episodes)
        return
    split_schedules = {
        bucket: v2.split_schedule(quota, bucket, collection_seed)
        for bucket in TASK_BUCKETS
    }
    rng = np.random.default_rng(collection_seed)
    attempts = _resume_attempt_count(data_dir, len(rows))
    if rows:
        print(
            f"[Resume] Found {len(rows)}/{num_episodes} accepted episodes; "
            f"continuing after recorded attempt {attempts}.",
            flush=True,
        )
    env = v2.make_environment(camera_observations=True)
    try:
        while sum(counts.values()) < num_episodes:
            underfilled = [
                bucket for bucket in TASK_BUCKETS if counts[bucket] < quota
            ]
            bucket = underfilled[int(rng.integers(len(underfilled)))]
            recovery_type = recovery_type_for(bucket, counts[bucket])
            attempts += 1
            while True:
                scene_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                if scene_seed not in used_seeds:
                    break
            v2.reseed_environment(env, scene_seed)
            obs = env.reset()
            initial_pose = v2.object_poses(env)
            try:
                task = v2.schedule_task_v2(env, obs, bucket[0], bucket[1])
            except RuntimeError as error:
                print(
                    f"[Scene rejected] Attempt {attempts} | "
                    f"{bucket} | {error}",
                    flush=True,
                )
                continue

            manager = None
            displacement_distance = 0.0
            if recovery_type == "target_displacement":
                dynamic_episode_index = counts[bucket] // len(
                    RECOVERY_TYPES[bucket[0]]
                )
                displacement_distance = DYNAMIC_DISTANCE_LEVELS[
                    dynamic_episode_index % len(DYNAMIC_DISTANCE_LEVELS)
                ]
                try:
                    manager = _dynamic_manager(
                        task,
                        obs,
                        env,
                        scene_seed,
                        episode_id + 1,
                        displacement_distance,
                    )
                except PerturbationSceneRejected as error:
                    print(
                        f"[Perturbation scene rejected] Attempt {attempts} | "
                        f"{bucket} | {error}",
                        flush=True,
                    )
                    continue

            oracle = RecoveryOracleV3(
                env,
                task,
                obs,
                seed=scene_seed + 1,
                recovery_type=recovery_type,
            )
            perturbation_step = 0
            perturbation_delta = np.zeros(3, dtype=np.float32)

            def before_step(step, active_oracle, active_obs, action) -> None:
                nonlocal perturbation_step, perturbation_delta
                if manager is None:
                    return
                previous_delta = manager.actual_target_delta
                manager.before_step(
                    step,
                    active_oracle.last_phase,
                    active_oracle.last_phase,
                    active_obs,
                    action,
                    0.0,
                )
                injected = manager.actual_target_delta - previous_delta
                if not np.any(injected):
                    return
                perturbation_step = step
                perturbation_delta = injected.astype(np.float32)
                task.target_goal[:2] += injected[:2]
                if task.task_type == "push":
                    env.initial_target_position[:2] += injected[:2]

            rollout_error = None
            try:
                result = v2.rollout_oracle(
                    env,
                    task,
                    obs,
                    seed=scene_seed,
                    capture_images=True,
                    oracle=oracle,
                    before_step_hook=before_step,
                )
            except RuntimeError as error:
                rollout_error = error
                result = None
            finally:
                if manager is not None:
                    manager.on_episode_end()
            if rollout_error is not None:
                print(
                    f"[Rollout rejected] Attempt {attempts} | {bucket} | "
                    f"type={recovery_type} | {rollout_error}",
                    flush=True,
                )
                continue
            assert result is not None
            valid_recovery = (
                perturbation_step > 0
                if recovery_type == "target_displacement"
                else result.retry_count >= 1
            )
            if not result.clean_success or not valid_recovery:
                print(
                    f"[Rejected] Attempt {attempts} | {bucket} | "
                    f"type={recovery_type} | success={result.clean_success} | "
                    f"retry={result.retry_count} | inject={perturbation_step}",
                    flush=True,
                )
                continue

            split = split_schedules[bucket][counts[bucket]]
            archive_initial_pose = initial_pose.copy()
            if (
                recovery_type == "target_displacement"
                and task.task_type == "push"
            ):
                target_index = tuple(v2.OBJECT_SPECS).index(task.target_id)
                archive_initial_pose[
                    target_index, :3
                ] += perturbation_delta
            try:
                arrays = v2.build_episode_arrays(
                    task,
                    result,
                    archive_initial_pose,
                    scene_seed,
                    collection_seed,
                    split,
                )
            except ValueError as error:
                print(
                    f"[Quality rejected] Attempt {attempts} | {bucket} | "
                    f"type={recovery_type} | {error}",
                    flush=True,
                )
                continue
            arrays.update(
                {
                    "v3_recovery_version": np.asarray(RECOVERY_VERSION),
                    "recovery_type": np.asarray(recovery_type),
                    "perturbation_step": np.asarray(
                        perturbation_step,
                        dtype=np.int64,
                    ),
                    "perturbation_delta": perturbation_delta.astype(np.float32),
                }
            )
            episode_id += 1
            output_path = os.path.join(data_dir, f"ep_{episode_id:05d}.npz")
            v2.save_episode(output_path, arrays)
            v2.verify_archive(output_path)
            metadata = _metadata_from_archive(output_path)
            rows.append(metadata)
            counts[bucket] += 1
            used_seeds.add(scene_seed)
            _write_manifest(data_dir, rows)
            _write_summary(
                data_dir,
                num_episodes,
                collection_seed,
                attempts,
                rows,
            )
            print(
                f"[Success] Attempt {attempts} | "
                f"Episode {len(rows)}/{num_episodes} | "
                f"{bucket[0]}_{bucket[1]} {counts[bucket]}/{quota} | "
                f"type={recovery_type} | split={split} | "
                f"steps={result.success_step} | retries={result.retry_count} | "
                f"delta={np.linalg.norm(perturbation_delta):.3f}m",
                flush=True,
            )
    finally:
        env.close()
    verify_dataset(data_dir, expected_total=num_episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_dataset(args.data_dir, expected_total=args.num_episodes)
        return
    collect(args.num_episodes, args.data_dir, args.seed)


if __name__ == "__main__":
    main()
