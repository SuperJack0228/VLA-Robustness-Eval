"""MuJoCo-independent schema contract for MiniVLA V2 demonstrations."""

from __future__ import annotations

import os

import numpy as np


SCHEMA_VERSION = 5
DATASET_VERSION = "v2.clean"
EPISODE_STEPS = 200
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
STATE_DIM = 17
ACTION_DIM = 7
OBJECT_COUNT = 3
OBJECT_LABELS = {
    "A": "red cube",
    "B": "blue ball",
    "C": "green cylinder",
}
STATE_LAYOUT = {
    "eef_position_xyz": [0, 3],
    "eef_quaternion_xyzw": [3, 7],
    "gripper_qpos": [7, 9],
    "gripper_qvel": [9, 11],
    "eef_linear_velocity": [11, 14],
    "eef_angular_velocity": [14, 17],
}
ACTION_LAYOUT = {
    "delta_position_xyz": [0, 3],
    "delta_orientation_rpy": [3, 6],
    "gripper": [6, 7],
}
TASK_BUCKETS = tuple(
    (task_type, target_id)
    for task_type in ("pick", "push")
    for target_id in OBJECT_LABELS
)


def instruction_for(task_type: str, target_id: str) -> str:
    label = OBJECT_LABELS[target_id]
    if task_type == "pick":
        return f"Pick up the {label}"
    if task_type == "push":
        return f"Push the {label} away from the robot"
    raise ValueError(f"Unsupported task type: {task_type}")


def validate_episode_arrays(arrays: dict[str, np.ndarray]) -> None:
    expected = {
        "image_agentview": ((EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8),
        "image_wrist": ((EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8),
        "state": ((EPISODE_STEPS, STATE_DIM), np.float32),
        "previous_action": ((EPISODE_STEPS, ACTION_DIM), np.float32),
        "action": ((EPISODE_STEPS, ACTION_DIM), np.float32),
        "object_pose": ((EPISODE_STEPS, OBJECT_COUNT, 7), np.float32),
        "object_contact": ((EPISODE_STEPS, OBJECT_COUNT), np.uint8),
        "object_grasped": ((EPISODE_STEPS, OBJECT_COUNT), np.uint8),
        "expert_phase": ((EPISODE_STEPS,), np.uint8),
        "retry_count": ((EPISODE_STEPS,), np.uint8),
        "success_after_action": ((EPISODE_STEPS,), np.uint8),
        "valid_mask": ((EPISODE_STEPS,), np.uint8),
        "initial_object_pose": ((OBJECT_COUNT, 7), np.float32),
        "push_direction": ((2,), np.float32),
        "target_goal": ((3,), np.float32),
    }
    scalar_fields = {
        "instruction",
        "target_id",
        "task_type",
        "split",
        "dataset_version",
        "schema_version",
        "success_step",
        "trajectory_length",
        "scene_seed",
        "collection_seed",
        "outcome",
        "wrong_object_contact",
    }
    missing = (set(expected) | scalar_fields) - set(arrays)
    if missing:
        raise ValueError(f"Archive is missing fields: {sorted(missing)}")
    for key, (shape, dtype) in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{key} must be {dtype} {shape}, got {value.dtype} {value.shape}"
            )
    for key in ("state", "previous_action", "action", "object_pose"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if np.any(np.abs(arrays["action"]) > 1.0001):
        raise ValueError("action exceeds OSC_POSE input bounds")

    for key in (
        "instruction",
        "target_id",
        "task_type",
        "split",
        "dataset_version",
    ):
        if arrays[key].shape != () or arrays[key].dtype.kind != "U":
            raise ValueError(f"{key} must be a scalar Unicode value")
    for key in (
        "schema_version",
        "success_step",
        "trajectory_length",
        "scene_seed",
        "collection_seed",
        "outcome",
        "wrong_object_contact",
    ):
        if arrays[key].shape != () or arrays[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must be a scalar integer")
    if int(arrays["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("Unexpected schema version")
    if str(arrays["dataset_version"].item()) != DATASET_VERSION:
        raise ValueError("Unexpected dataset version")

    length = int(arrays["trajectory_length"])
    if not 1 <= length <= EPISODE_STEPS:
        raise ValueError("trajectory_length is outside the episode horizon")
    expected_mask = np.arange(EPISODE_STEPS) < length
    if not np.array_equal(arrays["valid_mask"].astype(bool), expected_mask):
        raise ValueError("valid_mask does not match trajectory_length")

    outcome = bool(int(arrays["outcome"]))
    success_step = int(arrays["success_step"])
    if outcome and not 1 <= success_step <= length:
        raise ValueError("Successful trajectories need a valid success_step")
    if not outcome and success_step != 0:
        raise ValueError("Failed trajectories must use success_step=0")
    if outcome and not np.all(
        arrays["success_after_action"][success_step - 1 : length]
    ):
        raise ValueError("The final success streak is not stable")

    task_type = str(arrays["task_type"].item())
    target_id = str(arrays["target_id"].item())
    split = str(arrays["split"].item())
    if (task_type, target_id) not in TASK_BUCKETS:
        raise ValueError(f"Unknown task bucket: {(task_type, target_id)}")
    if str(arrays["instruction"].item()) != instruction_for(task_type, target_id):
        raise ValueError("Instruction is not aligned with task_type and target_id")
    if split not in {"train", "val", "test", "diagnostic"}:
        raise ValueError(f"Unknown dataset split: {split}")
    if split != "diagnostic" and (not outcome or arrays["wrong_object_contact"]):
        raise ValueError("Accepted splits may only contain clean successful episodes")

    previous_action = arrays["previous_action"]
    action = arrays["action"]
    if not np.allclose(previous_action[0], 0.0, atol=1e-6):
        raise ValueError("The first previous_action must be zero")
    if length > 1 and not np.allclose(
        previous_action[1:length],
        action[: length - 1],
        atol=1e-6,
    ):
        raise ValueError("previous_action is not temporally aligned with action")
    if not np.all(np.isin(action[:length, 6], (-1.0, 1.0))):
        raise ValueError("Expert gripper actions must be binary -1 or 1")

    target_index = tuple(OBJECT_LABELS).index(target_id)
    phase = arrays["expert_phase"][:length]
    target_grasped = arrays["object_grasped"][:length, target_index].astype(bool)
    target_z = arrays["object_pose"][:length, target_index, 2]
    initial_target_z = float(arrays["initial_object_pose"][target_index, 2])
    if task_type == "pick":
        if np.any(target_grasped & (action[:length, 6] < 0.0)):
            raise ValueError("Pick expert opens the gripper while target is grasped")
        unsafe_recovery = (phase == 4) & (
            target_grasped | (target_z > initial_target_z + 0.015)
        )
        if np.any(unsafe_recovery):
            raise ValueError(
                "Pick recovery may not release a grasped or airborne target"
            )
        if outcome:
            grasp_indices = np.flatnonzero(target_grasped)
            lift_indices = np.flatnonzero(phase == 3)
            if not len(grasp_indices) or not len(lift_indices):
                raise ValueError("Successful Pick trajectory lacks grasp or lift")
            first_grasp = int(grasp_indices[0])
            causal_lifts = lift_indices[lift_indices >= first_grasp]
            if not len(causal_lifts):
                raise ValueError("Pick lift phase occurs before the physical grasp")
            first_lift = int(causal_lifts[0])
            if first_lift - first_grasp > 2:
                raise ValueError(
                    "Pick grasp-to-lift delay exceeds two control steps"
                )
            grasp_confirmation = target_grasped & (phase == 2)
            if np.any(action[:length, 2][grasp_confirmation] < -0.03):
                raise ValueError(
                    "Pick expert presses downward after bilateral grasp"
                )
            if action[first_lift, 2] < 0.20:
                raise ValueError(
                    "Pick lift transition lacks a decisive positive Z action"
                )
            if action[first_lift, 6] < 0.0:
                raise ValueError("Pick lift transition must keep the gripper closed")
    else:
        initial_xy = arrays["initial_object_pose"][target_index, :2]
        displacement = (
            arrays["object_pose"][:length, target_index, :2] - initial_xy
        )
        forward = displacement @ arrays["push_direction"]
        if np.any((phase == 9) & (forward > 0.010)):
            raise ValueError(
                "Push recovery may not retreat after meaningful target motion"
            )
        prior_success = np.concatenate(
            [
                np.asarray([False]),
                np.maximum.accumulate(
                    arrays["success_after_action"][: length - 1].astype(bool)
                ),
            ]
        )
        if np.any((phase == 8) & ~prior_success):
            raise ValueError("Push hold phase started before physical success")
        if target_id == "B" and np.any(phase == 7):
            relative_eef_height = (
                arrays["state"][:length, 2] - target_z
            )[phase == 7]
            relative_height_p90 = float(
                np.quantile(relative_eef_height, 0.90)
            )
            if relative_height_p90 > 0.015:
                raise ValueError(
                    "Blue-ball push contact is systematically too high: "
                    f"p90={relative_height_p90:.4f}m"
                )
            push_actions = action[:length, :2][phase == 7]
            forward_actions = push_actions @ arrays["push_direction"]
            if float(np.quantile(forward_actions, 0.10)) < 0.05:
                raise ValueError(
                    "Blue-ball contact phase does not maintain forward drive"
                )


def schema_document() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "episode_steps": EPISODE_STEPS,
        "image_shape": [EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "image_dtype": "uint8",
        "camera_order": ["agentview", "robot0_eye_in_hand"],
        "state_dim": STATE_DIM,
        "state_layout": STATE_LAYOUT,
        "action_dim": ACTION_DIM,
        "action_layout": ACTION_LAYOUT,
        "object_order": list(OBJECT_LABELS),
        "object_labels": OBJECT_LABELS,
        "task_buckets": [list(bucket) for bucket in TASK_BUCKETS],
        "padding": "Repeat final sample; valid_mask marks real timesteps.",
    }


def read_metadata(path: str) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        if "dataset_version" not in episode.files:
            raise ValueError(
                f"{path} is a legacy dataset archive without dataset_version"
            )
        schema_version = int(episode["schema_version"].item())
        dataset_version = str(episode["dataset_version"].item())
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses schema {schema_version}, expected {SCHEMA_VERSION}"
            )
        if dataset_version != DATASET_VERSION:
            raise ValueError(
                f"{path} is {dataset_version}, expected {DATASET_VERSION}"
            )
        return {
            "filename": os.path.basename(path),
            "split": str(episode["split"].item()),
            "dataset_version": dataset_version,
            "task_type": str(episode["task_type"].item()),
            "target_id": str(episode["target_id"].item()),
            "instruction": str(episode["instruction"].item()),
            "scene_seed": int(episode["scene_seed"].item()),
            "collection_seed": int(episode["collection_seed"].item()),
            "success_step": int(episode["success_step"].item()),
            "wrong_object_contact": int(
                episode["wrong_object_contact"].item()
            ),
        }
