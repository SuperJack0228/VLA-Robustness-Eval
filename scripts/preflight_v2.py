"""Strict release gate before clean MiniVLA fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mini_vla_v2 import MiniVLAV2
from scripts.train_v2 import (
    CHECKPOINT_FORMAT_VERSION,
    compute_objective,
    forward_batch,
    move_batch,
    optimizer_for,
    policy_payload,
)
from utils.training_dataset_v2 import (
    ActionChunkDatasetV2,
    InterleavedTaskBatchSampler,
    NormalizationStats,
    V2EpisodeStore,
    compute_normalization_stats,
)
from utils.v2_schema import (
    DATASET_VERSION,
    EPISODE_STEPS,
    OBJECT_LABELS,
    TASK_BUCKETS,
    validate_episode_arrays,
)


DEFAULT_DATA_DIR = "results/dataset_v2_clean"
DEFAULT_REPORT = "results/v2_clean/preflight_report_v2_clean.json"
EXPECTED_PHASES = {
    "pick": {0, 1, 2, 3},
    "push": {5, 6, 7, 8, 9},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--expected-episodes", type=int, default=1200)
    parser.add_argument("--output", default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GateReport:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def pass_check(self, name: str, details=None) -> None:
        self.checks.append({"name": name, "passed": True, "details": details})
        print(f"[PASS] {name}", flush=True)

    def fail_check(self, name: str, error: Exception | str) -> None:
        message = str(error)
        self.checks.append({"name": name, "passed": False, "error": message})
        print(f"[FAIL] {name}: {message}", flush=True)

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)

    def payload(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def archive_paths(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    return sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.startswith("ep_") and filename.endswith(".npz")
    )


def audit_archives(data_dir: str, expected_total: int) -> dict:
    paths = archive_paths(data_dir)
    require(len(paths) == expected_total, f"Expected {expected_total} archives, found {len(paths)}")
    require(expected_total % len(TASK_BUCKETS) == 0, "Episode total must be divisible by six")
    quota = expected_total // len(TASK_BUCKETS)
    expected_split = {
        "train": int(quota * 0.8),
        "val": int(quota * 0.1),
    }
    expected_split["test"] = quota - sum(expected_split.values())

    counts = Counter()
    split_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    scene_seeds = set()
    phase_sets: dict[str, set[int]] = defaultdict(set)
    gripper_values: dict[str, set[float]] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)
    image_std = []
    view_differences = []
    saturation_steps = 0
    valid_steps = 0
    quaternion_errors = []
    retries: dict[str, Counter] = defaultdict(Counter)
    stable_success_lengths: dict[str, list[int]] = defaultdict(list)
    pick_open_while_grasped = 0
    push_recovery_forward = []
    final_push_forward = []
    blue_push_relative_height = []
    grasp_to_lift_delays = []
    grasp_confirmation_z = []
    lift_transition_z = []
    blue_recovery_episodes = 0

    for index, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as episode:
            arrays = {key: episode[key] for key in episode.files}
        validate_episode_arrays(arrays)
        task_type = str(arrays["task_type"].item())
        target_id = str(arrays["target_id"].item())
        split = str(arrays["split"].item())
        bucket = (task_type, target_id)
        key = f"{task_type}_{target_id}"
        length = int(arrays["trajectory_length"])
        scene_seed = int(arrays["scene_seed"])
        success_step = int(arrays["success_step"])
        stable_success_lengths[task_type].append(length - success_step + 1)
        require(scene_seed not in scene_seeds, f"Duplicate scene seed in {path}")
        scene_seeds.add(scene_seed)
        counts[bucket] += 1
        split_counts[bucket][split] += 1
        lengths[key].append(length)
        phase_sets[task_type].update(
            int(value) for value in np.unique(arrays["expert_phase"][:length])
        )
        gripper_values[task_type].update(
            float(value) for value in np.unique(arrays["action"][:length, 6])
        )
        retries[key].update([int(arrays["retry_count"][length - 1])])
        target_index = tuple(OBJECT_LABELS).index(target_id)
        phase = arrays["expert_phase"][:length]
        if task_type == "pick":
            grasped = arrays["object_grasped"][:length, target_index].astype(bool)
            pick_open_while_grasped += int(
                np.sum(grasped & (arrays["action"][:length, 6] < 0.0))
            )
            first_grasp = int(np.flatnonzero(grasped)[0])
            causal_lift = np.flatnonzero(
                (phase == 3) & (np.arange(length) >= first_grasp)
            )
            require(len(causal_lift) > 0, f"{path} has no causal lift phase")
            first_lift = int(causal_lift[0])
            grasp_to_lift_delays.append(first_lift - first_grasp)
            grasp_confirmation_z.extend(
                arrays["action"][:length, 2][grasped & (phase == 2)].tolist()
            )
            lift_transition_z.append(float(arrays["action"][first_lift, 2]))
        else:
            target_xy = arrays["object_pose"][:length, target_index, :2]
            initial_xy = arrays["initial_object_pose"][target_index, :2]
            forward = (target_xy - initial_xy) @ arrays["push_direction"]
            final_push_forward.append(float(forward[-1]))
            push_recovery_forward.extend(forward[phase == 9].tolist())
            if target_id == "B":
                blue_recovery_episodes += int(np.any(phase == 9))
                target_z = arrays["object_pose"][:length, target_index, 2]
                blue_push_relative_height.extend(
                    (arrays["state"][:length, 2] - target_z)[phase == 7].tolist()
                )
        saturation_steps += int(
            np.any(np.abs(arrays["action"][:length, :6]) >= 0.99, axis=1).sum()
        )
        valid_steps += length
        quaternion_norm = np.linalg.norm(arrays["state"][:length, 3:7], axis=1)
        quaternion_errors.append(float(np.max(np.abs(quaternion_norm - 1.0))))

        frame_indices = sorted({0, length // 2, length - 1})
        for frame_index in frame_indices:
            agent = arrays["image_agentview"][frame_index].astype(np.float32)
            wrist = arrays["image_wrist"][frame_index].astype(np.float32)
            image_std.extend([float(agent.std()), float(wrist.std())])
            view_differences.append(float(np.mean(np.abs(agent - wrist))))
        if index % 100 == 0 or index == len(paths):
            print(f"Validated archives: {index}/{len(paths)}", flush=True)

    for bucket in TASK_BUCKETS:
        require(counts[bucket] == quota, f"Task imbalance for {bucket}: {counts[bucket]} != {quota}")
        actual_split = {
            split: split_counts[bucket][split]
            for split in ("train", "val", "test")
        }
        require(
            actual_split == expected_split,
            f"Split imbalance for {bucket}: {actual_split} != {expected_split}",
        )
    for task_type, required_phases in EXPECTED_PHASES.items():
        require(required_phases.issubset(phase_sets[task_type]), f"{task_type} is missing phases {sorted(required_phases - phase_sets[task_type])}")
    require(gripper_values["pick"] == {-1.0, 1.0}, "Pick data must contain open and closed gripper actions")
    require(1.0 in gripper_values["push"], "Push data must contain closed gripper actions")
    require(min(image_std) > 5.0, f"A sampled image is nearly blank; minimum std={min(image_std):.3f}")
    require(min(view_differences) > 1.0, "Agent and wrist camera frames appear duplicated")
    saturation_rate = saturation_steps / max(valid_steps, 1)
    require(saturation_rate < 0.05, f"Action saturation rate is too high: {saturation_rate:.2%}")
    require(max(quaternion_errors) < 1e-3, f"EEF quaternion normalization error is too high: {max(quaternion_errors):.6f}")
    require(max(max(values) for values in lengths.values()) <= EPISODE_STEPS, "A trajectory exceeds the fixed horizon")
    require(
        min(stable_success_lengths["pick"]) >= 10,
        "A pick trajectory has fewer than 10 stable post-success steps",
    )
    require(
        min(stable_success_lengths["push"]) >= 5,
        "A push trajectory has fewer than 5 stable post-success steps",
    )
    require(
        pick_open_while_grasped == 0,
        "Pick demonstrations contain open-gripper labels while grasped",
    )
    require(
        max(grasp_to_lift_delays) <= 2,
        "A Pick demonstration waits too long between grasp and lift",
    )
    require(
        not grasp_confirmation_z or min(grasp_confirmation_z) >= -0.03,
        "A Pick demonstration presses downward after bilateral grasp",
    )
    require(
        min(lift_transition_z) >= 0.20,
        "A Pick lift transition lacks decisive positive Z",
    )
    require(
        not push_recovery_forward or max(push_recovery_forward) <= 0.010,
        "Push recovery starts after meaningful target motion",
    )
    require(
        min(final_push_forward) >= 0.064,
        "A successful push trajectory ends below the 6.5 cm task threshold",
    )
    blue_height_p90 = float(np.quantile(blue_push_relative_height, 0.90))
    require(
        blue_height_p90 <= 0.015,
        f"Blue-ball push contact is too high: p90={blue_height_p90:.4f}m",
    )
    require(
        blue_recovery_episodes >= max(1, int(quota * 0.05)),
        "Blue-ball data lacks enough explicit re-localization recoveries",
    )
    return {
        "episodes": len(paths),
        "task_counts": {f"{task}_{target}": counts[(task, target)] for task, target in TASK_BUCKETS},
        "split_counts": {
            f"{task}_{target}": {
                split: split_counts[(task, target)][split]
                for split in ("train", "val", "test")
            }
            for task, target in TASK_BUCKETS
        },
        "mean_lengths": {key: float(np.mean(value)) for key, value in lengths.items()},
        "phase_sets": {key: sorted(value) for key, value in phase_sets.items()},
        "gripper_values": {key: sorted(value) for key, value in gripper_values.items()},
        "action_saturation_rate": saturation_rate,
        "minimum_image_std": min(image_std),
        "minimum_view_difference": min(view_differences),
        "maximum_quaternion_error": max(quaternion_errors),
        "minimum_stable_success_steps": {
            key: min(value) for key, value in stable_success_lengths.items()
        },
        "pick_open_while_grasped_steps": pick_open_while_grasped,
        "maximum_push_recovery_forward_m": (
            max(push_recovery_forward) if push_recovery_forward else None
        ),
        "minimum_final_push_forward_m": min(final_push_forward),
        "blue_push_relative_height_p90_m": blue_height_p90,
        "maximum_grasp_to_lift_delay_steps": max(grasp_to_lift_delays),
        "minimum_lift_transition_z_action": min(lift_transition_z),
        "blue_recovery_episodes": blue_recovery_episodes,
        "dataset_version": DATASET_VERSION,
        "unique_scene_seeds": len(scene_seeds),
        "retry_histograms": {
            key: {str(retry): count for retry, count in value.items()}
            for key, value in retries.items()
        },
    }


def audit_stats(data_dir: str) -> tuple[NormalizationStats, dict]:
    stats = compute_normalization_stats(data_dir, split="train")
    arrays = stats.as_numpy_dict()
    for key, value in arrays.items():
        require(np.isfinite(value).all(), f"{key} contains non-finite values")
        if key.endswith("_std"):
            require(np.all(value > 0), f"{key} contains zero or negative scales")
    return stats, {key: value.tolist() for key, value in arrays.items()}


def audit_sampler_and_batch(
    data_dir: str,
    stats: NormalizationStats,
    batch_size: int,
    num_workers: int,
) -> tuple[dict, dict]:
    require(batch_size % 16 == 0, "Preflight batch size must be divisible by 16")
    train_store = V2EpisodeStore(data_dir, "train", cache_size=32)
    dataset = ActionChunkDatasetV2(
        train_store,
        stats,
        samples_per_episode=64,
    )
    sampler = InterleavedTaskBatchSampler(
        dataset,
        batch_size=batch_size,
        episodes_per_batch=16,
        shuffle=True,
        seed=20260714,
    )
    batch_count = 0
    for indices in sampler:
        episodes = {dataset.sample_index[index][0] for index in indices}
        buckets = {dataset.sample_buckets[index] for index in indices}
        require(len(indices) == batch_size, "Sampler emitted an incomplete batch")
        require(len(episodes) == 16, "A batch does not contain 16 distinct episodes")
        require(buckets == set(TASK_BUCKETS), "A batch does not cover all six task buckets")
        batch_count += 1
    require(batch_count == len(sampler), f"Sampler length mismatch: yielded {batch_count}, reports {len(sampler)}")

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if num_workers:
        loader_kwargs["prefetch_factor"] = 2
    raw_batch = next(iter(DataLoader(**loader_kwargs)))
    require(
        tuple(raw_batch["image_agentview"].shape[1:]) == (3, 112, 112),
        "Invalid single-frame agentview batch shape",
    )
    require(
        tuple(raw_batch["image_wrist"].shape[1:]) == (3, 112, 112),
        "Invalid single-frame wrist batch shape",
    )
    require(tuple(raw_batch["state_history"].shape[1:]) == (5, 17), "Invalid state history batch shape")
    require(
        "previous_action_history" not in raw_batch,
        "Expert action history leaked into the causal policy batch",
    )
    require(tuple(raw_batch["pose_target"].shape[1:]) == (20, 6), "Invalid action chunk shape")
    require(
        tuple(raw_batch["object_positions_raw"].shape[1:]) == (3, 3),
        "Invalid object position batch shape",
    )
    require(
        tuple(raw_batch["target_contact"].shape) == (batch_size,),
        "Invalid target-contact batch shape",
    )
    require(
        tuple(raw_batch["target_grasp"].shape) == (batch_size,),
        "Invalid target-grasp batch shape",
    )
    sampled_pick_transition = False
    sampled_push_recovery = False
    for episode_index, timestep in dataset.sample_index:
        episode = dataset.episodes.get(episode_index)
        current_phase = int(episode["expert_phase"][timestep])
        previous_phase = (
            int(episode["expert_phase"][timestep - 1])
            if timestep > dataset.episodes.metadata[episode_index]["training_start"]
            else current_phase
        )
        sampled_pick_transition |= current_phase == 3 and previous_phase != 3
        sampled_push_recovery |= current_phase == 9
        if sampled_pick_transition and sampled_push_recovery:
            break
    require(
        sampled_pick_transition,
        "Dataset sampler contains no Pick grasp-to-lift transition",
    )
    require(
        sampled_push_recovery,
        "Dataset sampler contains no Push recovery window",
    )
    require(
        all(
            timestep >= dataset.episodes.metadata[episode]["training_start"]
            for episode, timestep in dataset.sample_index
        ),
        "A policy window includes a scripted failed-attempt prefix",
    )
    for key in (
        "image_agentview",
        "image_wrist",
        "state_history",
        "pose_target",
        "object_positions_raw",
    ):
        require(torch.isfinite(raw_batch[key]).all().item(), f"Batch field {key} contains non-finite values")

    initial_store = V2EpisodeStore(data_dir, "val", cache_size=2)
    initial_dataset = ActionChunkDatasetV2(
        initial_store,
        stats,
        samples_per_episode=1,
        initial_only=True,
        initial_repeats=1,
        history_dropout_probability=0.0,
        state_noise_std=0.0,
    )
    expected_clean_initials = sum(
        metadata["training_start"] == 0
        for metadata in initial_store.metadata
    )
    require(
        len(initial_dataset) == expected_clean_initials,
        "Initial-state validation must exclude scripted failed attempts",
    )
    require(
        all(timestep == 0 for _, timestep in initial_dataset.sample_index),
        "Initial-state validation contains a nonzero timestep",
    )
    return {
        "batches": batch_count,
        "batch_size": batch_size,
        "episodes_per_batch": 16,
        "task_buckets_per_batch": 6,
        "train_windows": len(dataset),
        "initial_validation_episodes": len(initial_dataset),
        "expert_action_history_in_policy_input": False,
    }, raw_batch


def take_batch_prefix(batch: dict, count: int) -> dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value[:count]
        elif isinstance(value, (list, tuple)):
            output[key] = list(value[:count])
        else:
            output[key] = value
    return output


def model_smoke(
    device: torch.device,
    allow_download: bool,
    stats: NormalizationStats | None = None,
    raw_batch: dict | None = None,
) -> dict:
    model = MiniVLAV2(local_files_only=not allow_download).to(device)
    language_frozen = all(
        not parameter.requires_grad for parameter in model.language_model.parameters()
    )
    require(language_frozen, "Language encoder is not fully frozen")
    frozen_stages = (
        model.vision_encoder.stem,
        model.vision_encoder.layer1,
        model.vision_encoder.layer2,
    )
    early_vision_frozen = all(
        not parameter.requires_grad
        for stage in frozen_stages
        for parameter in stage.parameters()
    )
    layer3_trainable = all(
        parameter.requires_grad
        for parameter in model.vision_encoder.layer3.parameters()
    )
    require(early_vision_frozen, "The early ResNet feature stages are not frozen")
    require(layer3_trainable, "ResNet layer3 is not trainable")

    model.eval()
    invariant_agent = torch.randn(2, 3, 112, 112, device=device)
    invariant_wrist = torch.randn(2, 3, 112, 112, device=device)
    invariant_instructions = [
        "Pick up the red cube",
        "Push the blue ball away from the robot",
    ]
    with torch.no_grad():
        grounding_a = model(
            image_agentview=invariant_agent,
            image_wrist=invariant_wrist,
            state_history=torch.zeros(2, 5, 17, device=device),
            instructions=invariant_instructions,
        )
        grounding_b = model(
            image_agentview=invariant_agent,
            image_wrist=invariant_wrist,
            state_history=torch.full(
                (2, 5, 17),
                100.0,
                device=device,
            ),
            instructions=invariant_instructions,
        )
    grounding_state_invariant = bool(
        torch.allclose(
            grounding_a["target_position"],
            grounding_b["target_position"],
            atol=1e-6,
            rtol=0.0,
        )
        and torch.allclose(
            grounding_a["target_class_logits"],
            grounding_b["target_class_logits"],
            atol=1e-6,
            rtol=0.0,
        )
    )
    require(
        grounding_state_invariant,
        "Grounding output depends on privileged proprioceptive state",
    )
    model.train()
    if raw_batch is None:
        instructions = ["Pick up the red cube", "Push the blue ball away from the robot"]
        output = model(
            image_agentview=torch.randn(2, 3, 112, 112, device=device),
            image_wrist=torch.randn(2, 3, 112, 112, device=device),
            state_history=torch.randn(2, 5, 17, device=device),
            instructions=instructions,
        )
        loss = sum(value.float().square().mean() for value in output.values())
    else:
        require(stats is not None, "Real-batch model smoke requires normalization stats")
        small_batch = move_batch(take_batch_prefix(raw_batch, 2), device)
        output = forward_batch(model, small_batch)
        metrics = compute_objective(output, small_batch, stats)
        loss = metrics["total"]
    require(torch.isfinite(loss).item(), "Model loss is non-finite")
    optimizer = optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temporary:
        checkpoint_path = temporary.name
    try:
        torch.save(policy_payload(model, stats or identity_stats(), 0, {"preflight": True}), checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        require(
            checkpoint.get("format_version") == CHECKPOINT_FORMAT_VERSION,
            "Checkpoint format version was lost",
        )
        require("normalization" in checkpoint, "Checkpoint is missing normalization stats")
        require("model_state_dict" in checkpoint, "Checkpoint is missing model weights")
    finally:
        os.unlink(checkpoint_path)
    return {
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "pose_shape": list(output["pose"].shape),
        "gripper_shape": list(output["gripper_logits"].shape),
        "phase_shape": list(output["phase_logits"].shape),
        "interaction_shape": list(output["interaction_logits"].shape),
        "target_position_shape": list(output["target_position"].shape),
        "target_class_shape": list(output["target_class_logits"].shape),
        "language_frozen": language_frozen,
        "early_vision_frozen": early_vision_frozen,
        "vision_layer3_trainable": layer3_trainable,
        "grounding_state_invariant": grounding_state_invariant,
    }


def identity_stats() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(17, dtype=np.float32),
        state_std=np.ones(17, dtype=np.float32),
        previous_action_mean=np.zeros(7, dtype=np.float32),
        previous_action_std=np.ones(7, dtype=np.float32),
        action_pose_mean=np.zeros(6, dtype=np.float32),
        action_pose_std=np.ones(6, dtype=np.float32),
        target_position_mean=np.zeros(3, dtype=np.float32),
        target_position_std=np.ones(3, dtype=np.float32),
    )


def write_report(path: str, report: GateReport) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump(report.payload(), report_file, indent=2)


def main() -> None:
    args = parse_args()
    report = GateReport()
    device = get_device(args.device)
    stats = None
    raw_batch = None

    if not args.code_only:
        try:
            details = audit_archives(args.data_dir, args.expected_episodes)
            report.pass_check("archive_schema_balance_and_signal", details)
        except Exception as error:
            report.fail_check("archive_schema_balance_and_signal", error)
        try:
            stats, details = audit_stats(args.data_dir)
            report.pass_check("train_only_normalization", details)
        except Exception as error:
            report.fail_check("train_only_normalization", error)
        if stats is not None:
            try:
                details, raw_batch = audit_sampler_and_batch(
                    args.data_dir,
                    stats,
                    args.batch_size,
                    args.num_workers,
                )
                report.pass_check("interleaved_dataloader", details)
            except Exception as error:
                report.fail_check("interleaved_dataloader", error)

    if not args.skip_model:
        try:
            details = model_smoke(
                device,
                args.allow_download,
                stats=stats,
                raw_batch=raw_batch,
            )
            report.pass_check("model_optimizer_checkpoint", details)
        except Exception as error:
            report.fail_check("model_optimizer_checkpoint", error)

    write_report(args.output, report)
    print(f"Preflight report: {args.output}", flush=True)
    print(f"V2 CLEAN PREFLIGHT: {'PASS' if report.passed else 'FAIL'}", flush=True)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
