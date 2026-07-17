"""Transition-aware windows for the release MiniVLA policy."""

from __future__ import annotations

import os
import random
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from utils.v2_schema import (
    ACTION_DIM,
    EPISODE_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_LABELS,
    STATE_DIM,
    TASK_BUCKETS,
    read_metadata,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_ID_TO_INDEX = {target_id: index for index, target_id in enumerate(OBJECT_LABELS)}
NON_TARGET_MOTION_LIMIT = 0.008
NON_TARGET_MOTION_BUFFER_STEPS = 3
MINIMUM_SUFFIX_STEPS = 10
PICK_PHASE_WEIGHTS = {0: 0.15, 1: 0.20, 2: 0.20, 3: 0.40, 4: 0.05}
PUSH_PHASE_WEIGHTS = {5: 0.15, 6: 0.20, 7: 0.40, 8: 0.10, 9: 0.15}


@dataclass
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    previous_action_mean: np.ndarray
    previous_action_std: np.ndarray
    action_pose_mean: np.ndarray
    action_pose_std: np.ndarray
    target_position_mean: np.ndarray
    target_position_std: np.ndarray

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary_path = f"{path}.tmp.npz"
        try:
            np.savez(temporary_path, **self.as_numpy_dict())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @classmethod
    def load(cls, path: str) -> "NormalizationStats":
        with np.load(path, allow_pickle=False) as archive:
            return cls(**{key: archive[key].astype(np.float32) for key in archive.files})

    def as_numpy_dict(self) -> dict[str, np.ndarray]:
        return {
            "state_mean": self.state_mean.astype(np.float32),
            "state_std": self.state_std.astype(np.float32),
            "previous_action_mean": self.previous_action_mean.astype(np.float32),
            "previous_action_std": self.previous_action_std.astype(np.float32),
            "action_pose_mean": self.action_pose_mean.astype(np.float32),
            "action_pose_std": self.action_pose_std.astype(np.float32),
            "target_position_mean": self.target_position_mean.astype(np.float32),
            "target_position_std": self.target_position_std.astype(np.float32),
        }

    def to_checkpoint(self) -> dict[str, list[float]]:
        return {key: value.tolist() for key, value in self.as_numpy_dict().items()}

    @classmethod
    def from_checkpoint(cls, payload: dict) -> "NormalizationStats":
        return cls(
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in payload.items()
            }
        )


class RunningMoments:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_square = np.zeros(dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.total.size)
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.total_square += np.square(values).sum(axis=0)

    def finalize(self, minimum_std: float) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty running moments")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), minimum_std)
        return mean.astype(np.float32), std.astype(np.float32)


def episode_paths(data_dir: str, split: str) -> list[str]:
    paths = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.startswith("ep_") or not filename.endswith(".npz"):
            continue
        path = os.path.join(data_dir, filename)
        if read_metadata(path)["split"] == split:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No {split!r} V2 episodes found in {data_dir}")
    return paths


def successful_suffix_start(episode: np.lib.npyio.NpzFile, length: int) -> int:
    """Keep the final recovery and successful attempt, excluding the bad action."""
    retries = episode["retry_count"][:length]
    final_retry = int(retries.max(initial=0))
    if final_retry == 0:
        return 0
    indices = np.flatnonzero(retries == final_retry)
    if not len(indices):
        raise ValueError("retry_count has no samples for its final retry value")
    candidate = int(indices[0])
    phases = episode["expert_phase"][:length]
    recovery_phase = 4 if int(phases[0]) < 5 else 9
    recovery = np.flatnonzero(
        (np.arange(length) >= candidate)
        & (retries == final_retry)
        & (phases == recovery_phase)
    )
    return int(recovery[0]) if len(recovery) else candidate


def successful_suffix_bounds(
    episode: np.lib.npyio.NpzFile,
    length: int,
    target_index: int,
) -> tuple[int, int]:
    """Return the final successful attempt, stopping before wrong-object motion."""
    start = successful_suffix_start(episode, length)
    positions = episode["object_pose"][start:length, :, :2]
    displacement = np.linalg.norm(positions - positions[0:1], axis=2)
    non_target_displacement = np.delete(displacement, target_index, axis=1)
    contaminated = np.flatnonzero(
        np.max(non_target_displacement, axis=1) > NON_TARGET_MOTION_LIMIT
    )
    if not len(contaminated):
        return start, length
    safe_relative_end = max(
        MINIMUM_SUFFIX_STEPS,
        int(contaminated[0]) - NON_TARGET_MOTION_BUFFER_STEPS,
    )
    return start, min(length, start + safe_relative_end)


def compute_normalization_stats(
    data_dir: str,
    split: str = "train",
) -> NormalizationStats:
    state_moments = RunningMoments(STATE_DIM)
    previous_action_moments = RunningMoments(ACTION_DIM)
    action_pose_moments = RunningMoments(6)
    target_position_moments = RunningMoments(3)
    for path in episode_paths(data_dir, split):
        metadata = read_metadata(path)
        target_index = TARGET_ID_TO_INDEX[metadata["target_id"]]
        with np.load(path, allow_pickle=False) as episode:
            length = int(episode["trajectory_length"].item())
            start, end = successful_suffix_bounds(
                episode, length, target_index
            )
            state_moments.update(episode["state"][start:end])
            previous_action_moments.update(episode["previous_action"][start:end])
            action_pose_moments.update(episode["action"][start:end, :6])
            target_position_moments.update(
                episode["object_pose"][start:end, target_index, :3]
            )
    state_mean, state_std = state_moments.finalize(1e-4)
    previous_action_mean, previous_action_std = previous_action_moments.finalize(0.02)
    action_pose_mean, action_pose_std = action_pose_moments.finalize(0.02)
    target_position_mean, target_position_std = target_position_moments.finalize(0.01)
    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        previous_action_mean=previous_action_mean,
        previous_action_std=previous_action_std,
        action_pose_mean=action_pose_mean,
        action_pose_std=action_pose_std,
        target_position_mean=target_position_mean,
        target_position_std=target_position_std,
    )


class V2EpisodeStore:
    """Per-worker LRU store that keeps compressed images off RAM until needed."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        cache_size: int = 32,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.paths = episode_paths(data_dir, split)
        self.split = split
        self.cache_size = max(int(cache_size), 0)
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.metadata = []
        for path in self.paths:
            metadata = read_metadata(path)
            with np.load(path, allow_pickle=False) as episode:
                length = int(episode["trajectory_length"].item())
                metadata["trajectory_length"] = length
                target_index = TARGET_ID_TO_INDEX[metadata["target_id"]]
                training_start, training_end = successful_suffix_bounds(
                    episode, length, target_index
                )
                metadata["training_start"] = training_start
                metadata["training_end"] = training_end
            self.metadata.append(metadata)

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, index: int) -> dict:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as episode:
            data = {
                key: episode[key].copy()
                for key in (
                    "image_agentview",
                    "image_wrist",
                    "state",
                    "previous_action",
                    "action",
                    "object_pose",
                    "object_contact",
                    "object_grasped",
                    "expert_phase",
                    "valid_mask",
                )
            }
        self._validate_loaded(path, data)
        if self.cache_size:
            self.cache[index] = data
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return data

    @staticmethod
    def _validate_loaded(path: str, episode: dict) -> None:
        image_shape = (EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if episode["image_agentview"].shape != image_shape:
            raise ValueError(f"{path}: invalid agentview shape")
        if episode["image_wrist"].shape != image_shape:
            raise ValueError(f"{path}: invalid wrist shape")
        if episode["image_agentview"].dtype != np.uint8:
            raise ValueError(f"{path}: images must remain uint8")
        if episode["state"].shape != (EPISODE_STEPS, STATE_DIM):
            raise ValueError(f"{path}: invalid state shape")
        if episode["previous_action"].shape != (EPISODE_STEPS, ACTION_DIM):
            raise ValueError(f"{path}: invalid previous_action shape")
        if episode["action"].shape != (EPISODE_STEPS, ACTION_DIM):
            raise ValueError(f"{path}: invalid action shape")
        expected_flags = (EPISODE_STEPS, len(OBJECT_LABELS))
        for key in ("object_contact", "object_grasped"):
            if episode[key].shape != expected_flags:
                raise ValueError(f"{path}: invalid {key} shape")


def _training_timesteps(
    phases: np.ndarray,
    gripper: np.ndarray,
    length: int,
    sample_count: int,
    initial_repeats: int = 1,
) -> list[int]:
    if sample_count <= 0:
        raise ValueError("samples_per_episode must be positive")
    if initial_repeats <= 0 or initial_repeats > sample_count:
        raise ValueError("initial_repeats must be in [1, samples_per_episode]")
    changes = np.flatnonzero(
        (phases[1:length] != phases[: length - 1])
        | (gripper[1:length] != gripper[: length - 1])
    ) + 1
    priority = []
    for transition in changes:
        priority.extend(
            step
            for step in range(transition - 3, transition + 4)
            if 0 <= step < length
        )
    for phase in np.unique(phases[:length]):
        indices = np.flatnonzero(phases[:length] == phase)
        count = min(4, len(indices))
        priority.extend(
            indices[
                np.linspace(0, len(indices) - 1, count).round().astype(int)
            ].tolist()
        )
    priority.extend(
        np.linspace(0, length - 1, min(sample_count, length)).round().astype(int)
    )
    unique = list(dict.fromkeys(int(step) for step in priority))
    selected = [0] * initial_repeats
    selected.extend(step for step in unique if step != 0)
    if len(selected) < sample_count:
        selected_set = set(selected)
        selected.extend(
            step for step in range(1, length) if step not in selected_set
        )
    if len(selected) < sample_count:
        repeated = np.linspace(0, length - 1, sample_count - len(selected))
        selected.extend(repeated.round().astype(int).tolist())
    return selected[:sample_count]


def _phase_balanced_timesteps(
    phases: np.ndarray,
    length: int,
    sample_count: int,
    initial_repeats: int,
) -> list[int]:
    """Allocate training windows to contact, recovery, and hold phases."""
    if sample_count <= 0:
        raise ValueError("samples_per_episode must be positive")
    if initial_repeats <= 0 or initial_repeats > sample_count:
        raise ValueError("initial_repeats must be in [1, samples_per_episode]")
    phase_weights = (
        PICK_PHASE_WEIGHTS if int(phases[0]) < 5 else PUSH_PHASE_WEIGHTS
    )
    available = {
        phase: np.flatnonzero(phases[:length] == phase)
        for phase in phase_weights
        if np.any(phases[:length] == phase)
    }
    if not available:
        raise ValueError("Episode contains no recognized expert phases")

    selected = [0] * initial_repeats
    phase_changes = np.flatnonzero(
        phases[1:length] != phases[: length - 1]
    ) + 1
    for transition in phase_changes:
        for timestep in range(transition - 2, transition + 4):
            if 0 <= timestep < length and timestep not in selected:
                selected.append(int(timestep))
    if len(selected) >= sample_count:
        return selected[:sample_count]

    budget = sample_count - len(selected)
    weight_sum = sum(phase_weights[phase] for phase in available)
    exact_counts = {
        phase: budget * phase_weights[phase] / weight_sum
        for phase in available
    }
    counts = {phase: int(np.floor(value)) for phase, value in exact_counts.items()}
    remainder = budget - sum(counts.values())
    for phase in sorted(
        available,
        key=lambda value: exact_counts[value] - counts[value],
        reverse=True,
    )[:remainder]:
        counts[phase] += 1

    selected_set = set(selected)
    for phase, indices in available.items():
        count = counts[phase]
        if count:
            candidates = np.asarray(
                [index for index in indices if int(index) not in selected_set]
            )
            if not len(candidates):
                continue
            positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)))
            chosen = candidates[positions.round().astype(int)].tolist()
            selected.extend(chosen)
            selected_set.update(int(index) for index in chosen)
    if len(selected) < sample_count:
        remaining = [
            timestep for timestep in range(length) if timestep not in selected_set
        ]
        if remaining:
            positions = np.linspace(
                0,
                len(remaining) - 1,
                min(sample_count - len(selected), len(remaining)),
            )
            selected.extend(
                np.asarray(remaining)[positions.round().astype(int)].tolist()
            )
    while len(selected) < sample_count:
        selected.append(selected[len(selected) % max(len(selected), 1)])
    return selected[:sample_count]


class ActionChunkDatasetV2(Dataset):
    """Return causal action windows from each episode's successful suffix."""

    def __init__(
        self,
        episodes: V2EpisodeStore,
        stats: NormalizationStats,
        chunk_size: int = 20,
        history_length: int = 5,
        samples_per_episode: int = 64,
        initial_only: bool = False,
        initial_repeats: int | None = None,
        history_dropout_probability: float = 0.10,
        state_noise_std: float = 0.005,
    ) -> None:
        self.episodes = episodes
        self.stats = stats
        self.chunk_size = chunk_size
        self.history_length = history_length
        self.training = episodes.split == "train"
        self.initial_only = initial_only
        self.initial_repeats = (
            4 if self.training else 1
        ) if initial_repeats is None else int(initial_repeats)
        self.history_dropout_probability = float(history_dropout_probability)
        self.state_noise_std = float(state_noise_std)
        self.image_augment = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.08,
                    hue=0.02,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.15,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03, 0.03),
                    fill=0,
                ),
            ]
        )
        self.image_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.sample_index: list[tuple[int, int]] = []
        self.sample_buckets: list[tuple[str, str]] = []
        self.episode_sample_indices: list[list[int]] = []
        for episode_index, metadata in enumerate(episodes.metadata):
            if self.initial_only and metadata["training_start"] != 0:
                self.episode_sample_indices.append([])
                continue
            with np.load(episodes.paths[episode_index], allow_pickle=False) as episode:
                phases = episode["expert_phase"]
                gripper = episode["action"][:, 6]
            start = metadata["training_start"]
            suffix_length = metadata["training_end"] - start
            if self.initial_only:
                timesteps = [0]
            elif self.training:
                timesteps = _phase_balanced_timesteps(
                    phases[start:],
                    suffix_length,
                    samples_per_episode,
                    initial_repeats=self.initial_repeats,
                )
                timesteps = [start + timestep for timestep in timesteps]
            else:
                timesteps = _training_timesteps(
                    phases[start:],
                    gripper[start:],
                    suffix_length,
                    samples_per_episode,
                    initial_repeats=self.initial_repeats,
                )
                timesteps = [start + timestep for timestep in timesteps]
            bucket = (metadata["task_type"], metadata["target_id"])
            episode_indices = []
            for timestep in timesteps:
                episode_indices.append(len(self.sample_index))
                self.sample_index.append((episode_index, timestep))
                self.sample_buckets.append(bucket)
            self.episode_sample_indices.append(episode_indices)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict:
        episode_index, timestep = self.sample_index[index]
        episode = self.episodes.get(episode_index)
        metadata = self.episodes.metadata[episode_index]
        length = metadata["training_end"]
        history_floor = 0 if self.initial_only else metadata["training_start"]

        history_indices = np.arange(
            timestep - self.history_length + 1,
            timestep + 1,
        )
        history_indices = np.clip(history_indices, history_floor, None)
        state_history = episode["state"][history_indices]
        state_history = (
            state_history - self.stats.state_mean
        ) / self.stats.state_std
        if self.training and np.random.random() < self.history_dropout_probability:
            state_history = np.repeat(
                state_history[-1:],
                self.history_length,
                axis=0,
            )
        if self.training and self.state_noise_std > 0:
            state_history = state_history + np.random.normal(
                0.0,
                self.state_noise_std,
                size=state_history.shape,
            )

        end = min(timestep + self.chunk_size, length)
        action = episode["action"][timestep:end]
        phase = episode["expert_phase"][timestep:end]
        valid_steps = len(action)
        action_mask = np.zeros(self.chunk_size, dtype=np.float32)
        action_mask[:valid_steps] = 1.0
        if valid_steps < self.chunk_size:
            padding = self.chunk_size - valid_steps
            action = np.concatenate(
                [action, np.repeat(episode["action"][length - 1 : length], padding, axis=0)],
                axis=0,
            )
            phase = np.concatenate(
                [phase, np.repeat(episode["expert_phase"][length - 1], padding)],
                axis=0,
            )

        pose_raw = action[:, :6].astype(np.float32)
        pose_target = (
            pose_raw - self.stats.action_pose_mean
        ) / self.stats.action_pose_std
        gripper_target = ((action[:, 6] + 1.0) / 2.0).astype(np.float32)
        target_index = TARGET_ID_TO_INDEX[metadata["target_id"]]
        target_contact = float(
            episode["object_contact"][timestep, target_index]
        )
        target_grasp = float(
            episode["object_grasped"][timestep, target_index]
        )
        current_phase = int(episode["expert_phase"][timestep])
        previous_phase = (
            int(episode["expert_phase"][timestep - 1])
            if timestep > history_floor
            else current_phase
        )
        target_position_raw = episode["object_pose"][timestep, target_index, :3]
        target_position = (
            target_position_raw - self.stats.target_position_mean
        ) / self.stats.target_position_std

        agentview = self._prepare_image(episode["image_agentview"][timestep])
        wrist = self._prepare_image(episode["image_wrist"][timestep])
        previous_gripper = (
            action[0, 6]
            if timestep == metadata["training_start"]
            else episode["previous_action"][timestep, 6]
        )
        return {
            "image_agentview": agentview,
            "image_wrist": wrist,
            "state_history": torch.from_numpy(state_history.astype(np.float32)),
            "pose_target": torch.from_numpy(pose_target.astype(np.float32)),
            "pose_raw": torch.from_numpy(pose_raw),
            "gripper_target": torch.from_numpy(gripper_target),
            "phase_target": torch.from_numpy(phase.astype(np.int64)),
            "action_mask": torch.from_numpy(action_mask),
            "target_position": torch.from_numpy(target_position.astype(np.float32)),
            "target_position_raw": torch.from_numpy(
                target_position_raw.astype(np.float32)
            ),
            "object_positions_raw": torch.from_numpy(
                episode["object_pose"][timestep, :, :3].astype(np.float32)
            ),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "target_contact": torch.tensor(target_contact, dtype=torch.float32),
            "target_grasp": torch.tensor(target_grasp, dtype=torch.float32),
            "is_pick_lift_transition": torch.tensor(
                metadata["task_type"] == "pick"
                and current_phase == 3
                and previous_phase != 3,
                dtype=torch.float32,
            ),
            "is_push_recovery": torch.tensor(
                metadata["task_type"] == "push" and current_phase == 9,
                dtype=torch.float32,
            ),
            "previous_gripper": torch.tensor(
                (previous_gripper + 1.0) / 2.0,
                dtype=torch.float32,
            ),
            "instruction": metadata["instruction"],
            "task_type": metadata["task_type"],
            "target_id": metadata["target_id"],
            "bucket_index": torch.tensor(TASK_BUCKETS.index(self.sample_buckets[index])),
            "episode_index": torch.tensor(episode_index),
            "timestep": torch.tensor(timestep),
            "is_initial": torch.tensor(
                self.initial_only or timestep == metadata["training_start"],
                dtype=torch.float32,
            ),
        }

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        tensor = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        if self.training:
            tensor = self.image_augment(tensor)
        return self.image_normalize(tensor)


class InterleavedTaskBatchSampler(Sampler[list[int]]):
    """Mix 16 episodes and all six tasks in every 32-sample batch."""

    def __init__(
        self,
        dataset: ActionChunkDatasetV2,
        batch_size: int = 32,
        episodes_per_batch: int = 16,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if not 8 <= episodes_per_batch <= 16:
            raise ValueError("episodes_per_batch must be between 8 and 16")
        if batch_size % episodes_per_batch:
            raise ValueError("batch_size must be divisible by episodes_per_batch")
        self.dataset = dataset
        self.batch_size = batch_size
        self.episodes_per_batch = episodes_per_batch
        self.samples_per_selected_episode = batch_size // episodes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        for indices in dataset.episode_sample_indices:
            if len(indices) % self.samples_per_selected_episode:
                raise ValueError(
                    "samples_per_episode must be divisible by samples drawn per batch"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        chunks_by_episode: dict[int, deque[list[int]]] = {}
        queues_by_bucket: dict[tuple[str, str], deque[int]] = {
            bucket: deque() for bucket in TASK_BUCKETS
        }
        for episode_index, indices in enumerate(
            self.dataset.episode_sample_indices
        ):
            indices = indices.copy()
            if self.shuffle:
                rng.shuffle(indices)
            chunks_by_episode[episode_index] = deque(
                indices[start : start + self.samples_per_selected_episode]
                for start in range(0, len(indices), self.samples_per_selected_episode)
            )
            metadata = self.dataset.episodes.metadata[episode_index]
            queues_by_bucket[(metadata["task_type"], metadata["target_id"])].append(
                episode_index
            )
        if self.shuffle:
            for queue in queues_by_bucket.values():
                values = list(queue)
                rng.shuffle(values)
                queue.clear()
                queue.extend(values)

        batch_number = 0
        while all(queues_by_bucket[bucket] for bucket in TASK_BUCKETS):
            selected: set[int] = set()
            batch_chunks: list[list[int]] = []
            offset = batch_number % len(TASK_BUCKETS)
            bucket_order = TASK_BUCKETS[offset:] + TASK_BUCKETS[:offset]
            while len(selected) < self.episodes_per_batch:
                made_progress = False
                for bucket in bucket_order:
                    queue = queues_by_bucket[bucket]
                    if not queue:
                        continue
                    episode_index = queue.popleft()
                    if episode_index in selected:
                        queue.append(episode_index)
                        continue
                    selected.add(episode_index)
                    batch_chunks.append(
                        chunks_by_episode[episode_index].popleft()
                    )
                    if chunks_by_episode[episode_index]:
                        queue.append(episode_index)
                    made_progress = True
                    if len(selected) == self.episodes_per_batch:
                        break
                if not made_progress:
                    break
            if self.shuffle:
                rng.shuffle(batch_chunks)
            batch = [index for chunk in batch_chunks for index in chunk]
            if len(batch) != self.batch_size:
                break
            yield batch
            batch_number += 1

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
