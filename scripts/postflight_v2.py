"""Full-trajectory deployment gate for the clean MiniVLA policy."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mini_vla_v2 import MiniVLAV2
from scripts.train_v2 import (
    CHECKPOINT_FORMAT_VERSION,
    get_device,
    validate,
)
from utils.training_dataset_v2 import (
    ActionChunkDatasetV2,
    NormalizationStats,
    V2EpisodeStore,
)
from utils.v2_schema import DATASET_VERSION


DEFAULT_POLICY = "results/v2_clean/mini_vla_v2_clean_policy.pth"
DEFAULT_OUTPUT = "results/v2_clean/postflight_v2_clean.json"
TARGET_CLASS_ACCURACY_MIN = 0.85
TARGET_SELECTION_ACCURACY_MIN = 0.80
GROUNDING_CM_MAX = 4.0
XYZ_MAE_MAX = 0.08
RPY_MAE_MAX = 0.08
BUCKET_TARGET_CLASS_ACCURACY_MIN = 0.70
BUCKET_TARGET_SELECTION_ACCURACY_MIN = 0.65
BUCKET_GROUNDING_CM_MAX = 7.0
BUCKET_FIRST_XYZ_MAE_MAX = 0.07
BUCKET_FIRST_RPY_MAE_MAX = 0.15
CONTACT_ACCURACY_MIN = 0.95
GRASP_ACCURACY_MIN = 0.95
LIFT_TRANSITION_JOINT_ACCURACY_MIN = 0.95
BUCKET_LIFT_TRANSITION_JOINT_ACCURACY_MIN = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--data-dir", default="results/dataset_v2_clean")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--enforce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_policy(
    path: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[MiniVLAV2, NormalizationStats, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("Checkpoint is not a compatible MiniVLA policy")
    if checkpoint.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError(
            f"Policy must be trained on {DATASET_VERSION} data"
        )
    config = dict(checkpoint["model_config"])
    config["local_files_only"] = local_files_only
    model = MiniVLAV2(**config)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    stats = NormalizationStats.from_checkpoint(checkpoint["normalization"])
    return model, stats, checkpoint


def split_gate(metrics: dict, by_task: dict) -> tuple[bool, list[str]]:
    failures = []
    aggregate_checks = (
        (
            metrics["target_class_accuracy"]
            >= TARGET_CLASS_ACCURACY_MIN,
            "target class accuracy",
        ),
        (
            metrics["target_selection_accuracy"]
            >= TARGET_SELECTION_ACCURACY_MIN,
            "spatial target selection accuracy",
        ),
        (metrics["grounding_cm"] <= GROUNDING_CM_MAX, "grounding error"),
        (metrics["xyz_mae"] <= XYZ_MAE_MAX, "XYZ action MAE"),
        (metrics["rpy_mae"] <= RPY_MAE_MAX, "RPY action MAE"),
        (
            metrics["contact_accuracy"] >= CONTACT_ACCURACY_MIN,
            "target contact accuracy",
        ),
        (
            metrics["grasp_accuracy"] >= GRASP_ACCURACY_MIN,
            "target grasp accuracy",
        ),
        (
            metrics["lift_transition_joint_accuracy"]
            >= LIFT_TRANSITION_JOINT_ACCURACY_MIN,
            "Pick grasp-to-lift joint transition accuracy",
        ),
    )
    failures.extend(name for passed, name in aggregate_checks if not passed)
    for bucket, values in by_task.items():
        if values["target_class_accuracy"] < BUCKET_TARGET_CLASS_ACCURACY_MIN:
            failures.append(f"{bucket} target class accuracy")
        if (
            values["target_selection_accuracy"]
            < BUCKET_TARGET_SELECTION_ACCURACY_MIN
        ):
            failures.append(f"{bucket} spatial target selection accuracy")
        if values["grounding_cm"] > BUCKET_GROUNDING_CM_MAX:
            failures.append(f"{bucket} grounding error")
        if values["first_xyz_mae"] > BUCKET_FIRST_XYZ_MAE_MAX:
            failures.append(f"{bucket} first XYZ action MAE")
        if values["first_rpy_mae"] > BUCKET_FIRST_RPY_MAE_MAX:
            failures.append(f"{bucket} first RPY action MAE")
        if (
            bucket.startswith("pick_")
            and values["lift_transition_joint_accuracy"]
            < BUCKET_LIFT_TRANSITION_JOINT_ACCURACY_MIN
        ):
            failures.append(f"{bucket} grasp-to-lift transition accuracy")
    return not failures, failures


def serializable_metrics(metrics: dict, by_task: dict) -> dict:
    return {
        "aggregate": {key: float(value) for key, value in metrics.items()},
        "by_task": by_task,
    }


def write_report(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_release_loader(
    data_dir: str,
    split: str,
    stats: NormalizationStats,
    batch_size: int,
    num_workers: int,
) -> tuple[ActionChunkDatasetV2, DataLoader]:
    dataset = ActionChunkDatasetV2(
        V2EpisodeStore(data_dir, split, cache_size=16),
        stats,
        samples_per_episode=48,
        history_dropout_probability=0.0,
        state_noise_std=0.0,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return dataset, DataLoader(**kwargs)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    device = get_device()
    print(f"Using device: {device}", flush=True)
    model, stats, checkpoint = load_policy(
        args.policy,
        device,
        args.local_files_only,
    )
    split_results = {}
    all_passed = True
    for split in ("val", "test"):
        dataset, loader = build_release_loader(
            args.data_dir,
            split,
            stats,
            args.batch_size,
            args.num_workers,
        )
        metrics, by_task = validate(model, loader, device, stats)
        passed, failures = split_gate(metrics, by_task)
        all_passed &= passed
        split_results[split] = {
            "episodes": len(dataset),
            "passed": passed,
            "failures": failures,
            **serializable_metrics(metrics, by_task),
        }
        print(
            f"{split}: Target Class {metrics['target_class_accuracy']:.2%} | "
            f"Spatial Target {metrics['target_selection_accuracy']:.2%} | "
            f"Ground {metrics['grounding_cm']:.2f}cm | "
            f"XYZ MAE {metrics['xyz_mae']:.4f} | "
            f"Contact {metrics['contact_accuracy']:.2%} | "
            f"Grasp {metrics['grasp_accuracy']:.2%} | "
            f"Lift Gate {metrics['lift_transition_joint_accuracy']:.2%} | "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    report = {
        "passed": all_passed,
        "policy": args.policy,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "thresholds": {
            "target_class_accuracy_min": TARGET_CLASS_ACCURACY_MIN,
            "target_selection_accuracy_min": TARGET_SELECTION_ACCURACY_MIN,
            "grounding_cm_max": GROUNDING_CM_MAX,
            "xyz_mae_max": XYZ_MAE_MAX,
            "rpy_mae_max": RPY_MAE_MAX,
            "bucket_target_class_accuracy_min": (
                BUCKET_TARGET_CLASS_ACCURACY_MIN
            ),
            "bucket_target_selection_accuracy_min": (
                BUCKET_TARGET_SELECTION_ACCURACY_MIN
            ),
            "bucket_grounding_cm_max": BUCKET_GROUNDING_CM_MAX,
            "bucket_first_xyz_mae_max": BUCKET_FIRST_XYZ_MAE_MAX,
            "bucket_first_rpy_mae_max": BUCKET_FIRST_RPY_MAE_MAX,
            "contact_accuracy_min": CONTACT_ACCURACY_MIN,
            "grasp_accuracy_min": GRASP_ACCURACY_MIN,
            "lift_transition_joint_accuracy_min": (
                LIFT_TRANSITION_JOINT_ACCURACY_MIN
            ),
            "bucket_lift_transition_joint_accuracy_min": (
                BUCKET_LIFT_TRANSITION_JOINT_ACCURACY_MIN
            ),
        },
        "splits": split_results,
    }
    write_report(args.output, report)
    print(f"Postflight report: {args.output}", flush=True)
    print(
        f"V2 CLEAN FULL-TRAJECTORY GATE: {'PASS' if all_passed else 'FAIL'}",
        flush=True,
    )
    if args.enforce and not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
