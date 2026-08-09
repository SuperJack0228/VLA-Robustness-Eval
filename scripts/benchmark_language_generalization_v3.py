#!/usr/bin/env python3
"""Run a paired canonical-versus-held-out language benchmark for MiniVLA V3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_data_v2 import reseed_environment, schedule_task_v2  # noqa: E402
from utils.evaluation_core_v2 import (  # noqa: E402
    ENSEMBLE_DECAY,
    GROUNDING_SHIFT_RESET_THRESHOLD_M,
    MAX_STEPS,
    MAX_TEMPORAL_PREDICTION_AGE,
    EvaluationConfig,
    EvaluationCore,
    get_device,
    load_policy,
    make_environment,
)
from utils.language_augmentation_v3 import (  # noqa: E402
    DEFAULT_LANGUAGE_CATALOG,
    LanguageAugmentationCatalog,
)
from utils.v2_schema import OBJECT_LABELS, TASK_BUCKETS, instruction_for  # noqa: E402


PROTOCOL_VERSION = "language-generalization.v1"
DEFAULT_POLICY = "artifacts/v3-clean-rc1/mini_vla_v3_policy.pth"
DEFAULT_OUTPUT_DIR = "results/benchmarks/language_generalization/seed_20262201"
EXPECTED_EVAL_EXPRESSIONS_PER_TASK = 30
CONDITIONS = ("canonical", "held_out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--language-catalog", default=DEFAULT_LANGUAGE_CATALOG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20262201)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--replan-interval", type=int, default=1)
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
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pilot-pairs",
        type=int,
        default=None,
        help="Run only the first N pairs for a protocol smoke test.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.replan_interval <= 0:
        raise ValueError("max-steps and replan-interval must be positive")
    if args.log_every < 0:
        raise ValueError("log-every must be non-negative")
    if args.pilot_pairs is not None and args.pilot_pairs <= 0:
        raise ValueError("pilot-pairs must be positive")
    if not np.isfinite(args.ensemble_decay) or args.ensemble_decay < 0.0:
        raise ValueError("ensemble-decay must be finite and non-negative")
    if args.max_prediction_age < 0:
        raise ValueError("max-prediction-age must be non-negative")
    if (
        not np.isfinite(args.grounding_reset_threshold)
        or args.grounding_reset_threshold < 0.0
    ):
        raise ValueError("grounding-reset-threshold must be finite and non-negative")


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
        return
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


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scene_signature(env, obs: dict, task) -> str:
    digest = hashlib.sha256()
    arrays = [
        np.stack([env.get_object_position(object_id) for object_id in OBJECT_LABELS]),
        np.stack([env.get_object_quaternion(object_id) for object_id in OBJECT_LABELS]),
        np.asarray(obs["robot0_eef_pos"]),
        np.asarray(obs["robot0_eef_quat"]),
        np.asarray(task.push_direction),
        np.asarray(task.target_goal),
        np.asarray(obs["agentview_image"]),
        np.asarray(obs["robot0_eye_in_hand_image"]),
    ]
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def reset_scheduled_task(env, scene_seed: int, task_type: str, target_id: str):
    reseed_environment(env, scene_seed)
    obs = env.reset()
    task = schedule_task_v2(env, obs, task_type, target_id)
    env.set_task(task)
    return obs, task, scene_signature(env, obs, task)


def expected_pair_specs(catalog: LanguageAugmentationCatalog, seed: int) -> list[dict]:
    specs = []
    for task_type, target_id in TASK_BUCKETS:
        expressions = catalog.expressions(task_type, target_id, "eval")
        if len(expressions) != EXPECTED_EVAL_EXPRESSIONS_PER_TASK:
            raise ValueError(
                f"{task_type}_{target_id} must have exactly "
                f"{EXPECTED_EVAL_EXPRESSIONS_PER_TASK} evaluation expressions, "
                f"found {len(expressions)}"
            )
        for variant_index, instruction in enumerate(expressions):
            specs.append(
                {
                    "task_type": task_type,
                    "target_id": target_id,
                    "task_bucket": f"{task_type}_{target_id}",
                    "canonical_instruction": instruction_for(task_type, target_id),
                    "held_out_instruction": instruction,
                    "variant_id": f"{task_type}_{target_id}:eval:{variant_index:03d}",
                }
            )
    rng = np.random.default_rng(seed + 71_003)
    rng.shuffle(specs)
    return specs


def build_pair_plan(
    env,
    catalog: LanguageAugmentationCatalog,
    seed: int,
) -> dict:
    specs = expected_pair_specs(catalog, seed)
    scene_rng = np.random.default_rng(seed)
    pairs = []
    for pair_id, spec in enumerate(specs, start=1):
        rejections = 0
        while True:
            scene_seed = int(scene_rng.integers(0, np.iinfo(np.int32).max))
            try:
                _, _, signature = reset_scheduled_task(
                    env,
                    scene_seed,
                    spec["task_type"],
                    spec["target_id"],
                )
            except RuntimeError:
                rejections += 1
                continue
            break
        pairs.append(
            {
                "pair_id": pair_id,
                **spec,
                "scene_seed": scene_seed,
                "scene_rejections": rejections,
                "scene_signature": signature,
                "execution_order": (
                    ["canonical", "held_out"]
                    if pair_id % 2
                    else ["held_out", "canonical"]
                ),
            }
        )
        if pair_id == 1 or pair_id % 30 == 0 or pair_id == len(specs):
            print(
                f"[Plan] Pair {pair_id}/{len(specs)} | "
                f"latest={spec['task_bucket']} | rejections={rejections}",
                flush=True,
            )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "catalog_version": catalog.catalog_version,
        "catalog_sha256": catalog.digest,
        "pair_count": len(pairs),
        "episode_count": 2 * len(pairs),
        "pairs": pairs,
    }


def load_or_create_plan(
    path: Path,
    env,
    catalog: LanguageAugmentationCatalog,
    seed: int,
    resume: bool,
) -> dict:
    if resume and path.is_file():
        with path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Existing pair plan uses a different protocol")
        if int(plan.get("seed", -1)) != seed:
            raise ValueError("Existing pair plan uses a different seed")
        if plan.get("catalog_sha256") != catalog.digest:
            raise ValueError("Existing pair plan uses a different language catalog")
        if len(plan.get("pairs", [])) != 180:
            raise ValueError("Existing pair plan does not contain 180 pairs")
        return plan
    plan = build_pair_plan(env, catalog, seed)
    atomic_write_json(path, plan)
    return plan


def verify_checkpoint_catalog(policy_path: str, catalog: LanguageAugmentationCatalog) -> dict:
    checkpoint = torch.load(policy_path, map_location="cpu")
    metadata = checkpoint.get("language_catalog")
    if not isinstance(metadata, dict):
        raise ValueError("Policy checkpoint has no language catalog provenance")
    if metadata.get("sha256") != catalog.digest:
        raise ValueError("Policy and evaluation language catalogs do not match")
    for task_type, target_id in TASK_BUCKETS:
        key = f"{task_type}_{target_id}"
        if int(metadata["tasks"][key]["eval"]) != 30:
            raise ValueError(f"Checkpoint does not declare 30 eval variants for {key}")
    return metadata


def exact_mcnemar_p(canonical_only: int, held_out_only: int) -> float:
    discordant = canonical_only + held_out_only
    if discordant == 0:
        return 1.0
    smaller = min(canonical_only, held_out_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true"}


def confusion(rows: list[dict], prediction_key: str) -> dict[str, dict[str, int]]:
    labels = [f"{task}_{target}" for task, target in TASK_BUCKETS]
    prediction_labels = [*labels, "unknown"]
    matrix = {
        truth: {prediction: 0 for prediction in prediction_labels}
        for truth in labels
    }
    for row in rows:
        truth = str(row["task_bucket"])
        prediction = str(row[prediction_key])
        if prediction not in matrix[truth]:
            prediction = "unknown"
        matrix[truth][prediction] += 1
    return matrix


def summarize(rows: list[dict], plan: dict, policy: str, catalog: dict) -> dict:
    by_condition = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["language_condition"] == condition]
        by_task = {}
        for task_type, target_id in TASK_BUCKETS:
            bucket = f"{task_type}_{target_id}"
            task_rows = [row for row in selected if row["task_bucket"] == bucket]
            if task_rows:
                by_task[bucket] = {
                    "episodes": len(task_rows),
                    "task_success_rate": float(
                        np.mean([as_bool(row, "task_success") for row in task_rows])
                    ),
                    "initial_task_bucket_accuracy": float(
                        np.mean(
                            [
                                row["initial_predicted_task_bucket"] == bucket
                                for row in task_rows
                            ]
                        )
                    ),
                }
        by_condition[condition] = {
            "episodes": len(selected),
            "task_success_rate": float(
                np.mean([as_bool(row, "task_success") for row in selected])
            ),
            "clean_success_rate": float(
                np.mean([as_bool(row, "clean_success") for row in selected])
            ),
            "initial_task_type_accuracy": float(
                np.mean(
                    [
                        row["initial_predicted_task_type"] == row["task_type"]
                        for row in selected
                    ]
                )
            ),
            "initial_target_accuracy": float(
                np.mean(
                    [
                        row["initial_predicted_target_id"] == row["target_id"]
                        for row in selected
                    ]
                )
            ),
            "initial_task_bucket_accuracy": float(
                np.mean(
                    [
                        row["initial_predicted_task_bucket"] == row["task_bucket"]
                        for row in selected
                    ]
                )
            ),
            "failure_counts": dict(Counter(row["failure_category"] for row in selected)),
            "by_task": by_task,
            "initial_task_confusion": confusion(
                selected,
                "initial_predicted_task_bucket",
            ),
        }

    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["pair_id"])][row["language_condition"]] = row
    both = canonical_only = held_out_only = neither = 0
    for pair in grouped.values():
        if set(pair) != set(CONDITIONS):
            continue
        canonical_success = as_bool(pair["canonical"], "task_success")
        held_out_success = as_bool(pair["held_out"], "task_success")
        both += int(canonical_success and held_out_success)
        canonical_only += int(canonical_success and not held_out_success)
        held_out_only += int(held_out_success and not canonical_success)
        neither += int(not canonical_success and not held_out_success)

    complete = len(rows) == int(plan["episode_count"])
    canonical_rate = by_condition["canonical"]["task_success_rate"]
    held_out_rate = by_condition["held_out"]["task_success_rate"]
    return {
        "status": "complete" if complete else "partial",
        "protocol_version": PROTOCOL_VERSION,
        "policy": policy,
        "seed": plan["seed"],
        "catalog": catalog,
        "pair_count": len(grouped),
        "expected_pair_count": plan["pair_count"],
        "episode_count": len(rows),
        "expected_episode_count": plan["episode_count"],
        "canonical_and_held_out_paired_scenes": True,
        "raw_instruction_passthrough": True,
        "instruction_resolver_used": False,
        "ground_truth_task_scope": "environment_generation_and_scoring_only",
        "execution_task_source": "model-prediction",
        "uses_privileged_execution_assistance": False,
        "conditions": by_condition,
        "paired_outcomes": {
            "both_success": both,
            "canonical_only_success": canonical_only,
            "held_out_only_success": held_out_only,
            "neither_success": neither,
            "exact_mcnemar_p": exact_mcnemar_p(canonical_only, held_out_only),
        },
        "held_out_success_delta": held_out_rate - canonical_rate,
        "held_out_success_retention": (
            held_out_rate / canonical_rate if canonical_rate > 0 else None
        ),
    }


def completed_pairs(rows: list[dict]) -> set[int]:
    grouped: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        grouped[int(row["pair_id"])].add(str(row["language_condition"]))
    incomplete = [pair_id for pair_id, values in grouped.items() if values != set(CONDITIONS)]
    if incomplete:
        raise ValueError(f"Existing result contains incomplete pairs: {incomplete[:5]}")
    return set(grouped)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "benchmark_episodes.csv"
    summary_path = output_dir / "benchmark_summary.json"
    plan_path = output_dir / "pair_plan.json"

    catalog = LanguageAugmentationCatalog(args.language_catalog)
    checkpoint_catalog = verify_checkpoint_catalog(args.policy, catalog)
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(
        "Language protocol: raw held-out text | resolver=False | "
        "ground-truth task available to scene/scoring only",
        flush=True,
    )
    model, stats = load_policy(args.policy, device, args.local_files_only)
    env = make_environment(render=False, horizon=args.max_steps)
    low, high = env.action_spec

    try:
        plan = load_or_create_plan(plan_path, env, catalog, args.seed, args.resume)
        selected_pairs = list(plan["pairs"])
        if args.pilot_pairs is not None:
            selected_pairs = selected_pairs[: args.pilot_pairs]

        rows = read_csv(episodes_path) if args.resume else []
        done = completed_pairs(rows)
        for pair_number, pair in enumerate(selected_pairs, start=1):
            pair_id = int(pair["pair_id"])
            if pair_id in done:
                print(f"[Resume] Pair {pair_id} already complete", flush=True)
                continue
            pair_rows = []
            for condition in pair["execution_order"]:
                instruction = (
                    pair["canonical_instruction"]
                    if condition == "canonical"
                    else pair["held_out_instruction"]
                )
                obs, task, signature = reset_scheduled_task(
                    env,
                    int(pair["scene_seed"]),
                    pair["task_type"],
                    pair["target_id"],
                )
                if signature != pair["scene_signature"]:
                    raise RuntimeError(
                        f"Scene reproducibility failure for pair {pair_id}"
                    )
                task = replace(task, instruction=instruction)
                env.set_task(task)
                config = EvaluationConfig(
                    policy_path=args.policy,
                    num_episodes=1,
                    max_steps=args.max_steps,
                    seed=args.seed,
                    instruction=instruction,
                    task_type=pair["task_type"],
                    target_id=pair["target_id"],
                    output_prefix=str(output_dir / "unused"),
                    replan_interval=args.replan_interval,
                    render=False,
                    log_every=args.log_every,
                    local_files_only=args.local_files_only,
                    visual_perturbation="clean",
                    ensemble_mode="temporal",
                    temporal_profile="robust",
                    ensemble_decay=args.ensemble_decay,
                    max_prediction_age=args.max_prediction_age,
                    grounding_shift_reset_threshold_m=args.grounding_reset_threshold,
                    perturbation_label=f"language:{condition}",
                    diagnostic_trace=False,
                    write_outputs=False,
                    execution_task_source="model-prediction",
                )
                core = EvaluationCore(model, stats, env, device, config)
                visual_rng = np.random.default_rng(
                    args.seed + pair_id * 10_007
                )
                global_episode_id = (pair_id - 1) * 2 + CONDITIONS.index(condition) + 1
                row, stop_requested = core.run_episode(
                    global_episode_id,
                    int(pair["scene_seed"]),
                    task,
                    obs,
                    visual_rng,
                    low,
                    high,
                    int(pair["scene_rejections"]),
                )
                if stop_requested:
                    raise RuntimeError("Unexpected user stop in headless benchmark")
                row.update(
                    {
                        "pair_id": pair_id,
                        "pair_order_index": pair_number,
                        "language_condition": condition,
                        "variant_id": pair["variant_id"],
                        "task_bucket": pair["task_bucket"],
                        "canonical_instruction": pair["canonical_instruction"],
                        "held_out_instruction": pair["held_out_instruction"],
                        "raw_policy_instruction": instruction,
                        "instruction_transform_applied": 0,
                        "instruction_resolver_used": 0,
                        "scene_signature": signature,
                        "ground_truth_task_scope": (
                            "environment_generation_and_scoring_only"
                        ),
                    }
                )
                if row["instruction"] != instruction:
                    raise RuntimeError("Policy instruction was transformed")
                pair_rows.append(row)
                print(
                    f"[Pair {pair_number}/{len(selected_pairs)}] {condition:9s} | "
                    f"{pair['task_bucket']} | success={row['task_success']} | "
                    f"pred={row['initial_predicted_task_bucket']} | "
                    f"{instruction}",
                    flush=True,
                )

            if {row["language_condition"] for row in pair_rows} != set(CONDITIONS):
                raise RuntimeError(f"Pair {pair_id} did not produce both conditions")
            if len({row["scene_signature"] for row in pair_rows}) != 1:
                raise RuntimeError(f"Pair {pair_id} did not share one scene")
            rows.extend(pair_rows)
            rows.sort(key=lambda row: (int(row["pair_id"]), CONDITIONS.index(row["language_condition"])))
            atomic_write_csv(episodes_path, rows)
            atomic_write_json(
                summary_path,
                summarize(rows, plan, args.policy, checkpoint_catalog),
            )

        report = summarize(rows, plan, args.policy, checkpoint_catalog)
        atomic_write_csv(episodes_path, rows)
        atomic_write_json(summary_path, report)
        expected_pairs = (
            len(selected_pairs) if args.pilot_pairs is not None else plan["pair_count"]
        )
        print(
            f"Language benchmark finished: {len(completed_pairs(rows))}/"
            f"{expected_pairs} selected pairs | status={report['status']}",
            flush=True,
        )
        print(f"Results: {episodes_path} and {summary_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
