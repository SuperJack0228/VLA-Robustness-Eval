"""Conservative V3 fine-tuning over V2 Clean plus recovery demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mini_vla_v2 import MiniVLAV2
from scripts.train_v2 import (
    EPISODES_PER_BATCH,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP_NORM,
    METRIC_NAMES,
    atomic_json_dump,
    atomic_torch_save,
    build_initial_loader,
    build_loader,
    compute_objective,
    forward_batch,
    get_device,
    move_batch,
    seed_worker,
    validate,
)
from utils.language_augmentation_v3 import (
    DEFAULT_LANGUAGE_CATALOG,
    LanguageAugmentationCatalog,
)
from utils.training_dataset_v2 import NormalizationStats
from utils.v2_schema import SCHEMA_VERSION


DATASET_VERSION = "v3.robust-language"
CHECKPOINT_FORMAT_VERSION = 8
CHUNK_SIZE = 20
LEARNING_RATE = 2e-5
VISION_LEARNING_RATE = 5e-7
MIN_LEARNING_RATE = 5e-7
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 2
PHASE_FAMILY_WEIGHT = 0.35
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_PATIENCE = 8
DEFAULT_TRAIN_SAMPLES = 64
DEFAULT_VAL_SAMPLES = 48
V3_LANGUAGE_MAX_LENGTH = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-data-dir",
        default="results/dataset_v2_clean",
        help="Frozen V2 Clean demonstrations. This directory is never modified.",
    )
    parser.add_argument(
        "--recovery-data-dir",
        default="results/dataset_v3_recovery",
        help="Additive V3 recovery demonstrations.",
    )
    parser.add_argument("--output-dir", default="results/v3")
    parser.add_argument(
        "--init-policy",
        default="results/v2_clean/mini_vla_v2_clean_policy.pth",
        help="Frozen V2 Clean policy used for full-weight warm-start.",
    )
    parser.add_argument(
        "--language-catalog",
        default=DEFAULT_LANGUAGE_CATALOG,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--train-samples-per-episode",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
    )
    parser.add_argument(
        "--val-samples-per-episode",
        type=int,
        default=DEFAULT_VAL_SAMPLES,
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def optimizer_for(model: MiniVLAV2) -> optim.AdamW:
    vision_parameters = [
        parameter
        for parameter in model.vision_encoder.layer3.parameters()
        if parameter.requires_grad
    ]
    vision_ids = {id(parameter) for parameter in vision_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in vision_ids
    ]
    return optim.AdamW(
        [
            {"params": main_parameters, "lr": LEARNING_RATE},
            {"params": vision_parameters, "lr": VISION_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def scheduler_for(
    optimizer: optim.Optimizer,
    epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    minimum_ratio = MIN_LEARNING_RATE / LEARNING_RATE

    def schedule(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return 0.35 + 0.65 * epoch / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(epochs - WARMUP_EPOCHS, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, schedule)


def checkpoint_payload(
    model: MiniVLAV2,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    stats: NormalizationStats,
    catalog: LanguageAugmentationCatalog,
    epoch: int,
    best_selection_score: float,
    epochs_without_improvement: int,
    metrics: dict,
    data_dirs: list[str],
    init_policy: str,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "model_state_dict": model.trainable_state_dict(),
        "model_config": model.model_config(),
        "normalization": stats.to_checkpoint(),
        "language_catalog": catalog.summary(),
        "training_data_dirs": data_dirs,
        "init_policy": init_policy,
        "phase_family_weight": PHASE_FAMILY_WEIGHT,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_selection_score": best_selection_score,
        "epochs_without_improvement": epochs_without_improvement,
        "metrics": metrics,
    }


def policy_payload(
    model: MiniVLAV2,
    stats: NormalizationStats,
    catalog: LanguageAugmentationCatalog,
    epoch: int,
    metrics: dict,
    data_dirs: list[str],
    init_policy: str,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "model_state_dict": model.trainable_state_dict(),
        "model_config": model.model_config(),
        "normalization": stats.to_checkpoint(),
        "language_catalog": catalog.summary(),
        "training_data_dirs": data_dirs,
        "init_policy": init_policy,
        "phase_family_weight": PHASE_FAMILY_WEIGHT,
        "epoch": epoch,
        "metrics": metrics,
    }


def write_log_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        output.flush()
        os.fsync(output.fileno())


def build_model_from_checkpoint(
    checkpoint: dict,
    local_files_only: bool,
) -> MiniVLAV2:
    config = dict(checkpoint["model_config"])
    if int(config.get("chunk_size", -1)) != CHUNK_SIZE:
        raise RuntimeError(
            f"V3 requires the frozen {CHUNK_SIZE}-step V2 Clean architecture"
        )
    config["language_max_length"] = max(
        int(config.get("language_max_length", 16)),
        V3_LANGUAGE_MAX_LENGTH,
    )
    config["local_files_only"] = local_files_only
    model = MiniVLAV2(**config)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    if args.batch_size % EPISODES_PER_BATCH:
        raise ValueError("batch-size must be divisible by 16")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    if not args.resume and not os.path.isfile(args.init_policy):
        raise FileNotFoundError(args.init_policy)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    catalog = LanguageAugmentationCatalog(args.language_catalog)
    print(
        "Language catalog: "
        f"{catalog.catalog_version} | sha256={catalog.digest[:12]} | "
        "60 train + 30 held-out eval expressions per task",
        flush=True,
    )
    shutil.copyfile(
        args.language_catalog,
        os.path.join(args.output_dir, "language_augmentations_v3.json"),
    )

    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu") if args.resume else None
    )
    if resume_checkpoint is not None:
        if resume_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError("Resume checkpoint is not a V3 checkpoint")
        if resume_checkpoint.get("dataset_version") != DATASET_VERSION:
            raise RuntimeError("Resume checkpoint has the wrong dataset version")
        source_checkpoint = resume_checkpoint
    else:
        source_checkpoint = torch.load(args.init_policy, map_location="cpu")
        if source_checkpoint.get("format_version") != 7:
            raise RuntimeError("V3 must warm-start from a final V2 Clean policy")
        if source_checkpoint.get("dataset_version") != "v2.clean":
            raise RuntimeError("init-policy is not the frozen V2 Clean policy")

    stats = NormalizationStats.from_checkpoint(
        source_checkpoint["normalization"]
    )
    stats_path = os.path.join(args.output_dir, "dataset_stats_v3.npz")
    stats.save(stats_path)
    print(
        f"Reusing frozen V2 Clean normalization: {stats_path}",
        flush=True,
    )

    data_dirs = [args.base_data_dir, args.recovery_data_dir]
    train_dataset, train_sampler, train_loader = build_loader(
        data_dirs,
        "train",
        stats,
        args.train_samples_per_episode,
        args.batch_size,
        args.num_workers,
        args.seed,
        True,
        language_catalog=catalog,
    )
    combined_val_dataset, combined_val_sampler, combined_val_loader = build_loader(
        data_dirs,
        "val",
        stats,
        args.val_samples_per_episode,
        args.batch_size,
        args.num_workers,
        args.seed + 1,
        False,
        language_catalog=catalog,
    )
    clean_val_dataset, clean_val_sampler, clean_val_loader = build_loader(
        args.base_data_dir,
        "val",
        stats,
        args.val_samples_per_episode,
        args.batch_size,
        args.num_workers,
        args.seed + 2,
        False,
        language_catalog=catalog,
    )
    initial_dataset, initial_loader = build_initial_loader(
        args.base_data_dir,
        "val",
        stats,
        args.batch_size,
        min(args.num_workers, 2),
        language_catalog=catalog,
    )
    print(
        f"Train: {len(train_dataset)} windows from "
        f"{len(train_dataset.episodes)} episodes | "
        f"Combined Val: {len(combined_val_dataset)} windows | "
        f"Clean Val: {len(clean_val_dataset)} windows | "
        f"Initial Clean Val: {len(initial_dataset)} episodes",
        flush=True,
    )

    model = build_model_from_checkpoint(
        source_checkpoint,
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = optimizer_for(model)
    scheduler = scheduler_for(optimizer, args.epochs)
    start_epoch = 1
    best_selection_score = float("inf")
    epochs_without_improvement = 0
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_selection_score = float(
            resume_checkpoint["best_selection_score"]
        )
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"Resumed V3 from epoch {start_epoch - 1}", flush=True)
    else:
        print(
            f"Warm-started all V2 Clean policy weights from {args.init_policy}.",
            flush=True,
        )

    best_path = os.path.join(args.output_dir, "mini_vla_v3_best.pth")
    policy_path = os.path.join(args.output_dir, "mini_vla_v3_policy.pth")
    last_path = os.path.join(args.output_dir, "mini_vla_v3_last.pth")
    interrupted_path = os.path.join(
        args.output_dir,
        "mini_vla_v3_interrupted.pth",
    )
    csv_path = os.path.join(args.output_dir, "training_log_v3.csv")
    metadata_path = os.path.join(args.output_dir, "training_metadata_v3.json")
    current_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            train_sampler.set_epoch(epoch)
            combined_val_sampler.set_epoch(0)
            clean_val_sampler.set_epoch(0)
            model.train()
            totals = defaultdict(float)
            batches = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                batch = move_batch(raw_batch, device)
                output = forward_batch(model, batch)
                metrics = compute_objective(
                    output,
                    batch,
                    stats,
                    phase_family_weight=PHASE_FAMILY_WEIGHT,
                )
                (metrics["total"] / GRADIENT_ACCUMULATION_STEPS).backward()
                should_step = (
                    batch_index % GRADIENT_ACCUMULATION_STEPS == 0
                    or batch_index == len(train_loader)
                    or (
                        args.max_train_batches is not None
                        and batch_index >= args.max_train_batches
                    )
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRADIENT_CLIP_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for name in METRIC_NAMES:
                    totals[name] += metrics[name].item()
                batches += 1
                if batch_index == 1 or batch_index % 50 == 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}] | "
                        f"Batch [{batch_index}/{len(train_loader)}] | "
                        f"Total {metrics['total'].item():.5f} | "
                        f"Pose {metrics['pose'].item():.5f} | "
                        f"Grip {metrics['gripper_bce'].item():.5f} | "
                        f"Phase Family {metrics['phase_family_accuracy']:.2%} | "
                        f"Ground {metrics['grounding_cm'].item():.2f}cm | "
                        f"Target {metrics['target_class_accuracy']:.2%}",
                        flush=True,
                    )
                if (
                    args.max_train_batches is not None
                    and batch_index >= args.max_train_batches
                ):
                    break

            train_metrics = {
                name: totals[name] / max(batches, 1)
                for name in METRIC_NAMES
            }
            combined_val_metrics, combined_val_by_task = validate(
                model,
                combined_val_loader,
                device,
                stats,
                max_batches=args.max_val_batches,
                phase_family_weight=PHASE_FAMILY_WEIGHT,
            )
            clean_val_metrics, clean_val_by_task = validate(
                model,
                clean_val_loader,
                device,
                stats,
                max_batches=args.max_val_batches,
                phase_family_weight=PHASE_FAMILY_WEIGHT,
            )
            initial_metrics, initial_by_task = validate(
                model,
                initial_loader,
                device,
                stats,
                max_batches=args.max_val_batches,
                phase_family_weight=PHASE_FAMILY_WEIGHT,
            )
            selection_score = (
                0.50 * clean_val_metrics["total"]
                + 0.30 * combined_val_metrics["total"]
                + 0.20 * initial_metrics["total"]
                + 0.30
                * (
                    1.0
                    - clean_val_metrics[
                        "lift_transition_joint_accuracy"
                    ]
                )
            )
            improved = selection_score < best_selection_score
            if improved:
                best_selection_score = selection_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            scheduler.step()

            epoch_metrics = {
                "train": train_metrics,
                "combined_val": combined_val_metrics,
                "combined_val_by_task": combined_val_by_task,
                "clean_val": clean_val_metrics,
                "clean_val_by_task": clean_val_by_task,
                "initial_clean_val": initial_metrics,
                "initial_clean_val_by_task": initial_by_task,
                "selection_score": selection_score,
            }
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                stats,
                catalog,
                epoch,
                best_selection_score,
                epochs_without_improvement,
                epoch_metrics,
                data_dirs,
                args.init_policy,
            )
            atomic_torch_save(payload, last_path)
            if improved:
                atomic_torch_save(payload, best_path)
                atomic_torch_save(
                    policy_payload(
                        model,
                        stats,
                        catalog,
                        epoch,
                        epoch_metrics,
                        data_dirs,
                        args.init_policy,
                    ),
                    policy_path,
                )

            log_row = {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_phase_family_accuracy": train_metrics[
                    "phase_family_accuracy"
                ],
                "combined_val_total": combined_val_metrics["total"],
                "combined_val_xyz_mae": combined_val_metrics["xyz_mae"],
                "combined_val_gripper_accuracy": combined_val_metrics[
                    "gripper_accuracy"
                ],
                "combined_val_phase_accuracy": combined_val_metrics[
                    "phase_accuracy"
                ],
                "combined_val_phase_family_accuracy": combined_val_metrics[
                    "phase_family_accuracy"
                ],
                "combined_val_grounding_cm": combined_val_metrics[
                    "grounding_cm"
                ],
                "combined_val_target_selection_accuracy": combined_val_metrics[
                    "target_selection_accuracy"
                ],
                "clean_val_total": clean_val_metrics["total"],
                "clean_val_xyz_mae": clean_val_metrics["xyz_mae"],
                "clean_val_gripper_accuracy": clean_val_metrics[
                    "gripper_accuracy"
                ],
                "clean_val_phase_accuracy": clean_val_metrics[
                    "phase_accuracy"
                ],
                "clean_val_phase_family_accuracy": clean_val_metrics[
                    "phase_family_accuracy"
                ],
                "clean_val_lift_transition_joint_accuracy": clean_val_metrics[
                    "lift_transition_joint_accuracy"
                ],
                "clean_val_grounding_cm": clean_val_metrics["grounding_cm"],
                "clean_val_target_selection_accuracy": clean_val_metrics[
                    "target_selection_accuracy"
                ],
                "initial_clean_total": initial_metrics["total"],
                "initial_clean_grounding_cm": initial_metrics["grounding_cm"],
                "initial_clean_target_selection_accuracy": initial_metrics[
                    "target_selection_accuracy"
                ],
                "initial_clean_phase_family_accuracy": initial_metrics[
                    "phase_family_accuracy"
                ],
                "selection_score": selection_score,
                "main_lr": optimizer.param_groups[0]["lr"],
                "vision_lr": optimizer.param_groups[1]["lr"],
                "best_selection_score": best_selection_score,
            }
            for bucket, values in clean_val_by_task.items():
                log_row[f"clean_{bucket}_first_action_mae"] = values[
                    "first_action_mae"
                ]
                log_row[f"clean_{bucket}_grounding_cm"] = values[
                    "grounding_cm"
                ]
            write_log_row(csv_path, log_row)
            atomic_json_dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_version": DATASET_VERSION,
                    "last_completed_epoch": epoch,
                    "best_selection_score": best_selection_score,
                    "epochs_without_improvement": epochs_without_improvement,
                    "model_config": model.model_config(),
                    "language_catalog": catalog.summary(),
                    "training_data_dirs": data_dirs,
                    "init_policy": args.init_policy,
                    "latest_metrics": epoch_metrics,
                },
                metadata_path,
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] | "
                f"Train {train_metrics['total']:.5f} | "
                f"Combined Val {combined_val_metrics['total']:.5f} | "
                f"Clean Val {clean_val_metrics['total']:.5f} | "
                f"XYZ {clean_val_metrics['xyz_mae']:.4f} | "
                f"Grip {clean_val_metrics['gripper_accuracy']:.2%} | "
                f"Phase {clean_val_metrics['phase_accuracy']:.2%} | "
                f"Family {clean_val_metrics['phase_family_accuracy']:.2%} | "
                f"Ground {clean_val_metrics['grounding_cm']:.2f}cm | "
                f"Target {clean_val_metrics['target_selection_accuracy']:.2%} | "
                f"Select {selection_score:.5f} | "
                f"Patience {epochs_without_improvement}/{args.patience}",
                flush=True,
            )
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break
    except KeyboardInterrupt:
        last_completed_epoch = max(current_epoch - 1, 0)
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            stats,
            catalog,
            last_completed_epoch,
            best_selection_score,
            epochs_without_improvement,
            {"interrupted": True},
            data_dirs,
            args.init_policy,
        )
        atomic_torch_save(payload, interrupted_path)
        print(
            f"Interrupted checkpoint saved to {interrupted_path}.",
            flush=True,
        )
        raise

    print(f"Best checkpoint: {best_path}", flush=True)
    print(f"Evaluation policy: {policy_path}", flush=True)
    print(f"Last checkpoint: {last_path}", flush=True)
    print(f"Persistent training log: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
