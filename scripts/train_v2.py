"""Transition-aware trainer for the clean MiniVLA release policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mini_vla_v2 import (
    DEFAULT_LANGUAGE_MODEL,
    MiniVLAV2,
)
from utils.training_dataset_v2 import (
    ActionChunkDatasetV2,
    InterleavedTaskBatchSampler,
    NormalizationStats,
    V2EpisodeStore,
    compute_normalization_stats,
)
from utils.language_augmentation_v3 import LanguageAugmentationCatalog
from utils.v2_schema import DATASET_VERSION, SCHEMA_VERSION, TASK_BUCKETS


CHUNK_SIZE = 20
HISTORY_LENGTH = 5
BATCH_SIZE = 32
EPISODES_PER_BATCH = 16
TRAIN_SAMPLES_PER_EPISODE = 64
VAL_SAMPLES_PER_EPISODE = 48
NUM_WORKERS = 4
EPISODE_CACHE_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 3e-5
VISION_LEARNING_RATE = 1e-6
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-3
GRADIENT_ACCUMULATION_STEPS = 2
GRADIENT_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 10

POSE_WEIGHT = 1.0
GRIPPER_WEIGHT = 0.5
SMOOTHNESS_WEIGHT = 0.1
PHASE_WEIGHT = 0.30
INTERACTION_WEIGHT = 0.30
LIFT_TRANSITION_WEIGHT = 0.50
GROUNDING_WEIGHT = 0.50
TARGET_CLASS_WEIGHT = 0.30
CHECKPOINT_FORMAT_VERSION = 7
INITIAL_SELECTION_WEIGHT = 0.30
TRANSITION_SELECTION_WEIGHT = 0.50
PUSH_B_BUCKET_INDEX = TASK_BUCKETS.index(("push", "B"))
PERCEPTION_WARMSTART_PREFIXES = (
    "vision_encoder.",
    "language_projection.",
    "grounding_decoder.",
    "grounding_query",
    "target_position_head.",
    "target_class_head.",
    "modality_embedding",
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results/dataset_v2_clean")
    parser.add_argument("--output-dir", default="results/v2_clean")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--language-model", default=DEFAULT_LANGUAGE_MODEL)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-policy",
        default=None,
        help="Warm-start model weights from a V2 policy with a fresh optimizer.",
    )
    parser.add_argument(
        "--init-scope",
        choices=("perception", "all"),
        default="perception",
        help=(
            "Warm-start the complete stable policy by default; use "
            "'perception' only for an intentional decoder reset."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--train-samples-per-episode",
        type=int,
        default=TRAIN_SAMPLES_PER_EPISODE,
    )
    parser.add_argument(
        "--val-samples-per-episode",
        type=int,
        default=VAL_SAMPLES_PER_EPISODE,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug only: stop each training epoch after this many batches.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Debug only: stop validation after this many batches.",
    )
    return parser.parse_args()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def build_loader(
    data_dir: str | Sequence[str],
    split: str,
    stats: NormalizationStats,
    samples_per_episode: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    language_catalog: LanguageAugmentationCatalog | None = None,
) -> tuple[ActionChunkDatasetV2, InterleavedTaskBatchSampler, DataLoader]:
    episodes = V2EpisodeStore(
        data_dir,
        split,
        cache_size=EPISODE_CACHE_SIZE,
    )
    dataset = ActionChunkDatasetV2(
        episodes,
        stats,
        chunk_size=CHUNK_SIZE,
        history_length=HISTORY_LENGTH,
        samples_per_episode=samples_per_episode,
        language_catalog=language_catalog,
    )
    sampler = InterleavedTaskBatchSampler(
        dataset,
        batch_size=batch_size,
        episodes_per_batch=EPISODES_PER_BATCH,
        shuffle=shuffle,
        seed=seed,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": False,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return dataset, sampler, DataLoader(**loader_kwargs)


def build_initial_loader(
    data_dir: str | Sequence[str],
    split: str,
    stats: NormalizationStats,
    batch_size: int,
    num_workers: int,
    language_catalog: LanguageAugmentationCatalog | None = None,
) -> tuple[ActionChunkDatasetV2, DataLoader]:
    episodes = V2EpisodeStore(
        data_dir,
        split,
        cache_size=2,
    )
    dataset = ActionChunkDatasetV2(
        episodes,
        stats,
        chunk_size=CHUNK_SIZE,
        history_length=HISTORY_LENGTH,
        samples_per_episode=1,
        initial_only=True,
        initial_repeats=1,
        history_dropout_probability=0.0,
        state_noise_std=0.0,
        language_catalog=language_catalog,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return dataset, DataLoader(**loader_kwargs)


def move_batch(batch: dict, device: torch.device) -> dict:
    tensor_keys = (
        "image_agentview",
        "image_wrist",
        "state_history",
        "pose_target",
        "pose_raw",
        "gripper_target",
        "phase_target",
        "action_mask",
        "target_position",
        "target_position_raw",
        "object_positions_raw",
        "target_index",
        "target_contact",
        "target_grasp",
        "is_pick_lift_transition",
        "is_push_recovery",
        "previous_gripper",
        "bucket_index",
        "timestep",
        "is_initial",
        "task_is_pick",
    )
    moved = {key: batch[key].to(device) for key in tensor_keys}
    moved["instruction"] = list(batch["instruction"])
    return moved


def forward_batch(model: MiniVLAV2, batch: dict) -> dict[str, torch.Tensor]:
    return model(
        image_agentview=batch["image_agentview"],
        image_wrist=batch["image_wrist"],
        state_history=batch["state_history"],
        instructions=batch["instruction"],
    )


def compute_objective(
    output: dict[str, torch.Tensor],
    batch: dict,
    stats: NormalizationStats,
    phase_family_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    mask = batch["action_mask"]
    policy_sample_weight = 1.0 + batch["is_initial"]
    steps = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
    temporal_weight = torch.exp(-0.03 * steps).unsqueeze(0)
    temporal_weight[:, :5] *= 1.5
    weighted_mask = mask * temporal_weight * policy_sample_weight.unsqueeze(1)
    denominator = weighted_mask.sum().clamp_min(1.0)

    pose_absolute_error = torch.abs(output["pose"] - batch["pose_target"])
    push_b_contact = batch["bucket_index"].eq(PUSH_B_BUCKET_INDEX).unsqueeze(1) & (
        batch["phase_target"].eq(6) | batch["phase_target"].eq(7)
    )
    pose_dimension_weight = torch.ones_like(pose_absolute_error)
    pose_dimension_weight[:, :, 2] += 2.0 * push_b_contact.to(
        pose_dimension_weight.dtype
    )
    pose_error = (
        pose_absolute_error * pose_dimension_weight
    ).sum(dim=-1) / pose_dimension_weight.sum(dim=-1)
    pose_loss = (pose_error * weighted_mask).sum() / denominator

    pair_mask = mask[:, 1:] * mask[:, :-1]
    pair_weight = (
        temporal_weight[:, 1:]
        * pair_mask
        * policy_sample_weight.unsqueeze(1)
    )
    contact_phase = torch.zeros_like(pair_weight, dtype=torch.bool)
    for phase_id in (2, 3, 6, 7):
        contact_phase |= (
            batch["phase_target"][:, 1:].eq(phase_id)
            | batch["phase_target"][:, :-1].eq(phase_id)
        )
    pair_weight = pair_weight * torch.where(
        contact_phase,
        torch.full_like(pair_weight, 0.25),
        torch.ones_like(pair_weight),
    )
    pair_denominator = pair_weight.sum().clamp_min(1.0)
    predicted_delta = output["pose"][:, 1:] - output["pose"][:, :-1]
    target_delta = batch["pose_target"][:, 1:] - batch["pose_target"][:, :-1]
    smoothness_error = torch.abs(predicted_delta - target_delta).mean(dim=-1)
    smoothness_loss = (smoothness_error * pair_weight).sum() / pair_denominator

    previous_gripper = torch.cat(
        [batch["previous_gripper"].unsqueeze(1), batch["gripper_target"][:, :-1]],
        dim=1,
    )
    transition = batch["gripper_target"].ne(previous_gripper).float()
    closed_hold = (
        batch["gripper_target"].gt(0.5) & previous_gripper.gt(0.5)
    ).float()
    gripper_weight = weighted_mask * (
        1.0 + 2.0 * transition + 1.5 * closed_hold
    )
    gripper_denominator = gripper_weight.sum().clamp_min(1.0)
    gripper_bce = F.binary_cross_entropy_with_logits(
        output["gripper_logits"],
        batch["gripper_target"],
        reduction="none",
    )
    gripper_loss = (gripper_bce * gripper_weight).sum() / gripper_denominator

    phase_error = F.cross_entropy(
        output["phase_logits"].transpose(1, 2),
        batch["phase_target"],
        reduction="none",
    )
    phase_loss = (phase_error * weighted_mask).sum() / denominator
    phase_log_probability = F.log_softmax(
        output["phase_logits"],
        dim=-1,
    )
    pick_family_log_probability = torch.logsumexp(
        phase_log_probability[:, :, :5],
        dim=-1,
    )
    push_family_log_probability = torch.logsumexp(
        phase_log_probability[:, :, 5:],
        dim=-1,
    )
    task_is_pick = batch["task_is_pick"].bool().unsqueeze(1)
    correct_family_log_probability = torch.where(
        task_is_pick,
        pick_family_log_probability,
        push_family_log_probability,
    )
    phase_family_loss = (
        -correct_family_log_probability * weighted_mask
    ).sum() / denominator
    interaction_target = torch.stack(
        [batch["target_contact"], batch["target_grasp"]],
        dim=-1,
    )
    interaction_bce = F.binary_cross_entropy_with_logits(
        output["interaction_logits"],
        interaction_target,
        pos_weight=torch.as_tensor(
            [2.0, 4.0],
            device=output["interaction_logits"].device,
        ),
    )
    grounding_error = F.smooth_l1_loss(
        output["target_position"],
        batch["target_position"],
        reduction="none",
    ).mean(dim=-1)
    recovery_sample = (
        batch["phase_target"][:, 0].eq(4)
        | batch["phase_target"][:, 0].eq(9)
    ).float()
    grounding_sample_weight = (
        1.0 + 4.0 * batch["is_initial"] + 3.0 * recovery_sample
    )
    grounding_loss = (
        grounding_error * grounding_sample_weight
    ).sum() / grounding_sample_weight.sum().clamp_min(1.0)
    target_class_loss = F.cross_entropy(
        output["target_class_logits"],
        batch["target_index"],
    )

    action_mean = torch.as_tensor(
        stats.action_pose_mean,
        device=output["pose"].device,
    )
    action_std = torch.as_tensor(
        stats.action_pose_std,
        device=output["pose"].device,
    )
    pose_raw = output["pose"] * action_std + action_mean
    lift_transition_mask = batch["is_pick_lift_transition"]
    lift_transition_denominator = lift_transition_mask.sum().clamp_min(1.0)
    lift_z_error = F.relu(0.20 - pose_raw[:, 0, 2])
    lift_grip_error = F.binary_cross_entropy_with_logits(
        output["gripper_logits"][:, 0],
        torch.ones_like(output["gripper_logits"][:, 0]),
        reduction="none",
    )
    lift_transition_loss = (
        (lift_z_error + 0.25 * lift_grip_error) * lift_transition_mask
    ).sum() / lift_transition_denominator
    total_loss = (
        POSE_WEIGHT * pose_loss
        + GRIPPER_WEIGHT * gripper_loss
        + SMOOTHNESS_WEIGHT * smoothness_loss
        + PHASE_WEIGHT * phase_loss
        + INTERACTION_WEIGHT * interaction_bce
        + LIFT_TRANSITION_WEIGHT * lift_transition_loss
        + GROUNDING_WEIGHT * grounding_loss
        + TARGET_CLASS_WEIGHT * target_class_loss
        + phase_family_weight * phase_family_loss
    )

    gripper_prediction = output["gripper_logits"].gt(0.0)
    gripper_target = batch["gripper_target"].gt(0.5)
    gripper_correct = gripper_prediction.eq(gripper_target).float()
    gripper_accuracy = (gripper_correct * mask).sum() / mask.sum().clamp_min(1.0)
    phase_prediction = output["phase_logits"].argmax(dim=-1)
    phase_accuracy = (
        phase_prediction.eq(batch["phase_target"]).float() * mask
    ).sum() / mask.sum().clamp_min(1.0)
    phase_family_prediction = phase_prediction.lt(5)
    phase_family_accuracy = (
        phase_family_prediction.eq(task_is_pick).float() * mask
    ).sum() / mask.sum().clamp_min(1.0)

    interaction_prediction = output["interaction_logits"].gt(0.0)
    interaction_truth = interaction_target.gt(0.5)
    contact_accuracy = interaction_prediction[:, 0].eq(
        interaction_truth[:, 0]
    ).float().mean()
    grasp_accuracy = interaction_prediction[:, 1].eq(
        interaction_truth[:, 1]
    ).float().mean()
    lift_transition_prediction = (
        pose_raw[:, 0, 2].gt(0.20)
        & output["gripper_logits"][:, 0].gt(0.0)
        & phase_prediction[:, 0].eq(3)
        & interaction_prediction[:, 1]
    )
    lift_transition_count = lift_transition_mask.sum()
    lift_transition_correct_count = (
        lift_transition_prediction.float() * lift_transition_mask
    ).sum()
    raw_error = torch.abs(pose_raw - batch["pose_raw"])
    xyz_mae = (
        raw_error[:, :, :3].mean(dim=-1) * mask
    ).sum() / mask.sum().clamp_min(1.0)
    rpy_mae = (
        raw_error[:, :, 3:].mean(dim=-1) * mask
    ).sum() / mask.sum().clamp_min(1.0)

    target_mean = torch.as_tensor(
        stats.target_position_mean,
        device=output["pose"].device,
    )
    target_std = torch.as_tensor(
        stats.target_position_std,
        device=output["pose"].device,
    )
    predicted_target_raw = output["target_position"] * target_std + target_mean
    grounding_cm = (
        torch.linalg.vector_norm(
            predicted_target_raw - batch["target_position_raw"],
            dim=-1,
        ).mean()
        * 100.0
    )
    nearest_object = torch.linalg.vector_norm(
        batch["object_positions_raw"] - predicted_target_raw.unsqueeze(1),
        dim=-1,
    ).argmin(dim=1)
    target_selection_accuracy = nearest_object.eq(
        batch["target_index"]
    ).float().mean()
    target_class_accuracy = output["target_class_logits"].argmax(dim=-1).eq(
        batch["target_index"]
    ).float().mean()
    return {
        "total": total_loss,
        "pose": pose_loss,
        "gripper_bce": gripper_loss,
        "smoothness": smoothness_loss,
        "phase": phase_loss,
        "phase_family": phase_family_loss,
        "interaction_bce": interaction_bce,
        "lift_transition": lift_transition_loss,
        "grounding": grounding_loss,
        "target_class": target_class_loss,
        "gripper_accuracy": gripper_accuracy,
        "phase_accuracy": phase_accuracy,
        "phase_family_accuracy": phase_family_accuracy,
        "contact_accuracy": contact_accuracy,
        "grasp_accuracy": grasp_accuracy,
        "xyz_mae": xyz_mae,
        "rpy_mae": rpy_mae,
        "grounding_cm": grounding_cm,
        "target_selection_accuracy": target_selection_accuracy,
        "target_class_accuracy": target_class_accuracy,
        "lift_transition_count": lift_transition_count,
        "lift_transition_correct_count": lift_transition_correct_count,
    }


METRIC_NAMES = (
    "total",
    "pose",
    "gripper_bce",
    "smoothness",
    "phase",
    "phase_family",
    "interaction_bce",
    "lift_transition",
    "grounding",
    "target_class",
    "gripper_accuracy",
    "phase_accuracy",
    "phase_family_accuracy",
    "contact_accuracy",
    "grasp_accuracy",
    "xyz_mae",
    "rpy_mae",
    "grounding_cm",
    "target_selection_accuracy",
    "target_class_accuracy",
)


def validate(
    model: MiniVLAV2,
    loader: DataLoader,
    device: torch.device,
    stats: NormalizationStats,
    max_batches: int | None = None,
    phase_family_weight: float = 0.0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    totals = defaultdict(float)
    batches = 0
    samples = 0
    bucket_pose = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_xyz = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_rpy = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_grip = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_ground = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_target = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_target_class = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_transition_correct = np.zeros(len(TASK_BUCKETS), dtype=np.float64)
    bucket_transition_count = np.zeros(len(TASK_BUCKETS), dtype=np.int64)
    bucket_count = np.zeros(len(TASK_BUCKETS), dtype=np.int64)
    transition_correct = 0.0
    transition_count = 0
    action_mean = torch.as_tensor(stats.action_pose_mean, device=device)
    action_std = torch.as_tensor(stats.action_pose_std, device=device)
    target_mean = torch.as_tensor(stats.target_position_mean, device=device)
    target_std = torch.as_tensor(stats.target_position_std, device=device)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = forward_batch(model, batch)
            metrics = compute_objective(
                output,
                batch,
                stats,
                phase_family_weight=phase_family_weight,
            )
            batch_size = int(batch["state_history"].shape[0])
            for name in METRIC_NAMES:
                totals[name] += metrics[name].item() * batch_size
            transition_correct += metrics[
                "lift_transition_correct_count"
            ].item()
            transition_count += int(metrics["lift_transition_count"].item())
            batches += 1
            samples += batch_size

            first_pose_raw = output["pose"][:, 0] * action_std + action_mean
            first_pose_difference = torch.abs(
                first_pose_raw - batch["pose_raw"][:, 0]
            )
            first_pose_error = first_pose_difference.mean(dim=-1)
            first_xyz_error = first_pose_difference[:, :3].mean(dim=-1)
            first_rpy_error = first_pose_difference[:, 3:].mean(dim=-1)
            first_grip_correct = output["gripper_logits"][:, 0].gt(0).eq(
                batch["gripper_target"][:, 0].gt(0.5)
            )
            first_interaction_grasp = output["interaction_logits"][:, 1].gt(0)
            first_phase = output["phase_logits"][:, 0].argmax(dim=-1)
            transition_correct_mask = (
                first_pose_raw[:, 2].gt(0.20)
                & first_grip_correct
                & first_interaction_grasp
                & first_phase.eq(3)
            )
            target_raw = output["target_position"] * target_std + target_mean
            target_error_cm = torch.linalg.vector_norm(
                target_raw - batch["target_position_raw"],
                dim=-1,
            ) * 100.0
            target_correct = torch.linalg.vector_norm(
                batch["object_positions_raw"] - target_raw.unsqueeze(1),
                dim=-1,
            ).argmin(dim=1).eq(batch["target_index"])
            target_class_correct = output["target_class_logits"].argmax(
                dim=-1
            ).eq(batch["target_index"])
            for bucket_index in range(len(TASK_BUCKETS)):
                selected = batch["bucket_index"].eq(bucket_index)
                count = int(selected.sum().item())
                if count:
                    bucket_pose[bucket_index] += first_pose_error[selected].sum().item()
                    bucket_xyz[bucket_index] += first_xyz_error[selected].sum().item()
                    bucket_rpy[bucket_index] += first_rpy_error[selected].sum().item()
                    bucket_grip[bucket_index] += first_grip_correct[selected].sum().item()
                    bucket_ground[bucket_index] += target_error_cm[selected].sum().item()
                    bucket_target[bucket_index] += target_correct[selected].sum().item()
                    bucket_target_class[bucket_index] += target_class_correct[
                        selected
                    ].sum().item()
                    transition_selected = selected & batch[
                        "is_pick_lift_transition"
                    ].bool()
                    bucket_transition_correct[bucket_index] += (
                        transition_correct_mask & transition_selected
                    ).sum().item()
                    bucket_transition_count[bucket_index] += int(
                        transition_selected.sum().item()
                    )
                    bucket_count[bucket_index] += count
            if max_batches is not None and batches >= max_batches:
                break
    if samples == 0:
        raise RuntimeError("Validation loader emitted no samples")
    aggregate = {
        name: totals[name] / max(samples, 1) for name in METRIC_NAMES
    }
    aggregate["lift_transition_joint_accuracy"] = float(
        transition_correct / max(transition_count, 1)
    )
    aggregate["lift_transition_samples"] = int(transition_count)
    by_task = {}
    for index, bucket in enumerate(TASK_BUCKETS):
        key = f"{bucket[0]}_{bucket[1]}"
        count = max(int(bucket_count[index]), 1)
        by_task[key] = {
            "first_action_mae": float(bucket_pose[index] / count),
            "first_xyz_mae": float(bucket_xyz[index] / count),
            "first_rpy_mae": float(bucket_rpy[index] / count),
            "first_gripper_accuracy": float(bucket_grip[index] / count),
            "grounding_cm": float(bucket_ground[index] / count),
            "target_selection_accuracy": float(bucket_target[index] / count),
            "target_class_accuracy": float(
                bucket_target_class[index] / count
            ),
            "lift_transition_joint_accuracy": float(
                bucket_transition_correct[index]
                / max(int(bucket_transition_count[index]), 1)
            ),
            "lift_transition_samples": int(bucket_transition_count[index]),
            "samples": int(bucket_count[index]),
        }
    return aggregate, by_task


def optimizer_for(model: MiniVLAV2) -> optim.AdamW:
    vision_parameters = [
        parameter
        for parameter in model.vision_encoder.layer3.parameters()
        if parameter.requires_grad
    ]
    vision_parameter_ids = {id(parameter) for parameter in vision_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in vision_parameter_ids
    ]
    return optim.AdamW(
        [
            {"params": main_parameters, "lr": LEARNING_RATE},
            {"params": vision_parameters, "lr": VISION_LEARNING_RATE},
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def scheduler_for(
    optimizer: optim.Optimizer,
    epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    minimum_ratio = MIN_LEARNING_RATE / LEARNING_RATE

    def schedule(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return 0.2 + 0.8 * epoch / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(epochs - WARMUP_EPOCHS, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, schedule)


def load_perception_warmstart(
    model: MiniVLAV2,
    source_state: dict[str, torch.Tensor],
) -> int:
    """Import target perception without carrying over obsolete policy behavior."""
    current_state = model.trainable_state_dict()
    selected = {
        key: value
        for key, value in source_state.items()
        if key.startswith(PERCEPTION_WARMSTART_PREFIXES)
    }
    expected = {
        key for key in current_state if key.startswith(PERCEPTION_WARMSTART_PREFIXES)
    }
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(selected) - set(current_state))
    mismatched = sorted(
        key
        for key in selected.keys() & current_state.keys()
        if selected[key].shape != current_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Perception warm-start mismatch. "
            f"Missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    model.load_state_dict(selected, strict=False)
    return len(selected)


def checkpoint_payload(
    model: MiniVLAV2,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    stats: NormalizationStats,
    epoch: int,
    best_selection_score: float,
    epochs_without_improvement: int,
    metrics: dict,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "model_state_dict": model.trainable_state_dict(),
        "model_config": model.model_config(),
        "normalization": stats.to_checkpoint(),
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
    epoch: int,
    metrics: dict,
) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "model_state_dict": model.trainable_state_dict(),
        "model_config": model.model_config(),
        "normalization": stats.to_checkpoint(),
        "epoch": epoch,
        "metrics": metrics,
    }


def write_log_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        log_file.flush()
        os.fsync(log_file.fileno())


def atomic_torch_save(payload: dict, path: str) -> None:
    temporary_path = f"{path}.tmp"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_json_dump(payload: dict, path: str) -> None:
    temporary_path = f"{path}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    args = parse_args()
    if args.resume and args.init_policy:
        raise ValueError("Use either --resume or --init-policy, not both")
    if args.batch_size % EPISODES_PER_BATCH:
        raise ValueError("batch-size must be divisible by 16")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    resume_checkpoint = None
    initialization = None
    if args.init_policy:
        initialization = torch.load(args.init_policy, map_location="cpu")
        if initialization.get("format_version") not in {3, 4, 5, 6, 7}:
            raise RuntimeError("init-policy is not a compatible MiniVLA policy")
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        if resume_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError("Resume checkpoint is not a compatible MiniVLA checkpoint")
        if resume_checkpoint.get("dataset_version") != DATASET_VERSION:
            raise RuntimeError(
                "Only final-format checkpoints can be resumed. Use --init-policy "
                "to reuse an older policy with a fresh optimizer."
            )
        stats = NormalizationStats.from_checkpoint(
            resume_checkpoint["normalization"]
        )
    elif initialization is not None and args.init_scope == "all":
        stats = NormalizationStats.from_checkpoint(
            initialization["normalization"]
        )
        print(
            f"Reusing normalization from {args.init_policy} to preserve its "
            "physical action mapping.",
            flush=True,
        )
    else:
        stats = compute_normalization_stats(args.data_dir, split="train")
    stats_path = os.path.join(args.output_dir, "dataset_stats_v2_clean.npz")
    stats.save(stats_path)
    print(f"Normalization statistics: {stats_path}", flush=True)

    train_dataset, train_sampler, train_loader = build_loader(
        args.data_dir,
        "train",
        stats,
        args.train_samples_per_episode,
        args.batch_size,
        args.num_workers,
        args.seed,
        True,
    )
    val_dataset, val_sampler, val_loader = build_loader(
        args.data_dir,
        "val",
        stats,
        args.val_samples_per_episode,
        args.batch_size,
        args.num_workers,
        args.seed + 1,
        False,
    )
    initial_val_dataset, initial_val_loader = build_initial_loader(
        args.data_dir,
        "val",
        stats,
        args.batch_size,
        min(args.num_workers, 2),
    )
    print(
        f"Train: {len(train_dataset)} windows from "
        f"{len(train_dataset.episodes)} episodes | "
        f"Val: {len(val_dataset)} windows from "
        f"{len(val_dataset.episodes)} episodes | "
        f"Initial-state Val: {len(initial_val_dataset)} episodes",
        flush=True,
    )

    if resume_checkpoint:
        model_config = dict(resume_checkpoint["model_config"])
        model_config["local_files_only"] = args.local_files_only
    else:
        model_config = {
            "language_model_name": args.language_model,
            "local_files_only": args.local_files_only,
        }
    model = MiniVLAV2(**model_config).to(device)
    if args.init_policy:
        assert initialization is not None
        initial_config = dict(initialization["model_config"])
        initial_config.pop("local_files_only", None)
        current_config = model.model_config()
        comparable_current = dict(current_config)
        if args.init_scope == "perception":
            initial_config.pop("architecture_version", None)
            initial_config.pop("visual_history_length", None)
            comparable_current.pop("architecture_version", None)
            comparable_current.pop("visual_history_length", None)
        if initial_config != comparable_current:
            raise RuntimeError(
                "init-policy model configuration does not match the current model"
            )
        if args.init_scope == "perception":
            loaded_count = load_perception_warmstart(
                model,
                initialization["model_state_dict"],
            )
            print(
                f"Warm-started {loaded_count} perception/grounding tensors from "
                f"{args.init_policy}; policy decoder and action heads are fresh.",
                flush=True,
            )
        else:
            model.load_trainable_state_dict(initialization["model_state_dict"])
            print(
                f"Warm-started all model weights from {args.init_policy}. "
                "This also imports old policy behavior.",
                flush=True,
            )
    optimizer = optimizer_for(model)
    scheduler = scheduler_for(optimizer, args.epochs)
    start_epoch = 1
    best_selection_score = float("inf")
    epochs_without_improvement = 0
    if resume_checkpoint:
        model.load_trainable_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_selection_score = float(
            resume_checkpoint["best_selection_score"]
        )
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"Resumed from epoch {start_epoch - 1}", flush=True)

    best_path = os.path.join(args.output_dir, "mini_vla_v2_clean_best.pth")
    policy_path = os.path.join(args.output_dir, "mini_vla_v2_clean_policy.pth")
    last_path = os.path.join(args.output_dir, "mini_vla_v2_clean_last.pth")
    interrupted_path = os.path.join(
        args.output_dir,
        "mini_vla_v2_clean_interrupted.pth",
    )
    csv_path = os.path.join(args.output_dir, "training_log_v2_clean.csv")
    metadata_path = os.path.join(args.output_dir, "training_metadata_v2_clean.json")
    current_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(0)
            model.train()
            totals = defaultdict(float)
            batches = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                batch = move_batch(raw_batch, device)
                output = forward_batch(model, batch)
                metrics = compute_objective(output, batch, stats)
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
                        f"Ground {metrics['grounding_cm'].item():.2f}cm | "
                        f"Target {metrics['target_class_accuracy']:.1%}",
                        flush=True,
                    )
                if (
                    args.max_train_batches is not None
                    and batch_index >= args.max_train_batches
                ):
                    break

            train_metrics = {
                name: totals[name] / max(batches, 1) for name in METRIC_NAMES
            }
            val_metrics, val_by_task = validate(
                model,
                val_loader,
                device,
                stats,
                max_batches=args.max_val_batches,
            )
            initial_metrics, initial_by_task = validate(
                model,
                initial_val_loader,
                device,
                stats,
                max_batches=args.max_val_batches,
            )
            selection_score = (
                INITIAL_SELECTION_WEIGHT * initial_metrics["total"]
                + (1.0 - INITIAL_SELECTION_WEIGHT) * val_metrics["total"]
                + TRANSITION_SELECTION_WEIGHT
                * (1.0 - val_metrics["lift_transition_joint_accuracy"])
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
                "val": val_metrics,
                "val_by_task": val_by_task,
                "initial_val": initial_metrics,
                "initial_val_by_task": initial_by_task,
                "selection_score": selection_score,
            }
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                stats,
                epoch,
                best_selection_score,
                epochs_without_improvement,
                epoch_metrics,
            )
            atomic_torch_save(payload, last_path)
            if improved:
                atomic_torch_save(payload, best_path)
                atomic_torch_save(
                    policy_payload(model, stats, epoch, epoch_metrics),
                    policy_path,
                )

            log_row = {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "val_total": val_metrics["total"],
                "val_pose": val_metrics["pose"],
                "val_gripper_bce": val_metrics["gripper_bce"],
                "val_gripper_accuracy": val_metrics["gripper_accuracy"],
                "val_phase_accuracy": val_metrics["phase_accuracy"],
                "val_contact_accuracy": val_metrics["contact_accuracy"],
                "val_grasp_accuracy": val_metrics["grasp_accuracy"],
                "val_lift_transition_joint_accuracy": val_metrics[
                    "lift_transition_joint_accuracy"
                ],
                "val_lift_transition_samples": val_metrics[
                    "lift_transition_samples"
                ],
                "val_xyz_mae": val_metrics["xyz_mae"],
                "val_rpy_mae": val_metrics["rpy_mae"],
                "val_grounding_cm": val_metrics["grounding_cm"],
                "val_target_selection_accuracy": val_metrics[
                    "target_selection_accuracy"
                ],
                "val_target_class_accuracy": val_metrics[
                    "target_class_accuracy"
                ],
                "initial_val_total": initial_metrics["total"],
                "initial_val_xyz_mae": initial_metrics["xyz_mae"],
                "initial_val_grounding_cm": initial_metrics["grounding_cm"],
                "initial_val_target_selection_accuracy": initial_metrics[
                    "target_selection_accuracy"
                ],
                "initial_val_target_class_accuracy": initial_metrics[
                    "target_class_accuracy"
                ],
                "selection_score": selection_score,
                "main_lr": optimizer.param_groups[0]["lr"],
                "vision_lr": optimizer.param_groups[1]["lr"],
                "best_selection_score": best_selection_score,
            }
            for bucket, values in val_by_task.items():
                log_row[f"{bucket}_first_action_mae"] = values[
                    "first_action_mae"
                ]
                log_row[f"{bucket}_first_grip_acc"] = values[
                    "first_gripper_accuracy"
                ]
                log_row[f"initial_{bucket}_grounding_cm"] = initial_by_task[
                    bucket
                ]["grounding_cm"]
                log_row[f"initial_{bucket}_first_xyz_mae"] = initial_by_task[
                    bucket
                ]["first_xyz_mae"]
                log_row[f"initial_{bucket}_first_rpy_mae"] = initial_by_task[
                    bucket
                ]["first_rpy_mae"]
                log_row[f"initial_{bucket}_target_acc"] = initial_by_task[
                    bucket
                ]["target_selection_accuracy"]
                log_row[f"initial_{bucket}_target_class_acc"] = initial_by_task[
                    bucket
                ]["target_class_accuracy"]
            write_log_row(csv_path, log_row)
            atomic_json_dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_version": DATASET_VERSION,
                    "last_completed_epoch": epoch,
                    "best_selection_score": best_selection_score,
                    "epochs_without_improvement": epochs_without_improvement,
                    "model_config": model.model_config(),
                    "train_windows": len(train_dataset),
                    "val_windows": len(val_dataset),
                    "initial_val_episodes": len(initial_val_dataset),
                    "init_policy": args.init_policy,
                    "init_scope": args.init_scope if args.init_policy else None,
                    "latest_metrics": epoch_metrics,
                },
                metadata_path,
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] | "
                f"Train {train_metrics['total']:.5f} | "
                f"Val {val_metrics['total']:.5f} | "
                f"XYZ MAE {val_metrics['xyz_mae']:.4f} | "
                f"Grip Acc {val_metrics['gripper_accuracy']:.2%} | "
                f"Phase Acc {val_metrics['phase_accuracy']:.2%} | "
                f"Contact {val_metrics['contact_accuracy']:.2%} | "
                f"Grasp {val_metrics['grasp_accuracy']:.2%} | "
                f"Lift Gate {val_metrics['lift_transition_joint_accuracy']:.2%} | "
                f"Init XYZ {initial_metrics['xyz_mae']:.4f} | "
                f"Init Ground {initial_metrics['grounding_cm']:.2f}cm | "
                f"Init Target {initial_metrics['target_selection_accuracy']:.2%} | "
                f"Init Class {initial_metrics['target_class_accuracy']:.2%} | "
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
            last_completed_epoch,
            best_selection_score,
            epochs_without_improvement,
            {"interrupted": True},
        )
        atomic_torch_save(payload, interrupted_path)
        print(
            f"Interrupted checkpoint saved to {interrupted_path}; "
            f"resume will restart epoch {last_completed_epoch + 1}.",
            flush=True,
        )
        raise

    print(f"Best checkpoint: {best_path}", flush=True)
    print(f"Evaluation policy: {policy_path}", flush=True)
    print(f"Last checkpoint: {last_path}", flush=True)
    print(f"Persistent training log: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
