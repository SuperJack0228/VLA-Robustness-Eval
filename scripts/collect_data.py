"""Collect fixed-length multi-task, multi-view 7D VLA demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


NUM_EPISODES = 4800
EPISODE_STEPS = 150
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
POSITION_GAIN = 10.0
ORIENTATION_GAIN = 2.0
POSE_ACTION_NOISE_STD = 0.01
GRASP_HOLD_STEPS = 25
DEEP_GRASP_OFFSET = 0.012
MIN_OBJECT_CLEARANCE = 0.035
PUSH_DISTANCE = 0.18
PUSH_SUCCESS_DISTANCE = 0.065
DATASET_DIR = "results/dataset"
COLLECTION_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

OBJECT_SPECS = {
    "A": {"name": "red_cube", "label": "red cube"},
    "B": {"name": "blue_sphere", "label": "blue ball"},
    "C": {"name": "green_cylinder", "label": "green cylinder"},
}
BLUE_BALL_RADIUS = 0.026
TASK_BUCKETS = tuple(
    (task_type, target_id)
    for task_type in ("pick", "push")
    for target_id in OBJECT_SPECS
)


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    target_id: str
    instruction: str
    push_direction: np.ndarray | None = None


def build_osc_pose_controller_config() -> dict:
    controller_config = load_composite_controller_config(robot="Panda")
    arm_config = load_part_controller_config(default_controller="OSC_POSE")
    arm_config["input_type"] = "delta"
    arm_config["input_ref_frame"] = "base"
    arm_config["gripper"] = {"type": "GRIP"}
    controller_config["body_parts"]["right"] = arm_config
    return controller_config


class MultiObjectVLAEnv(Lift):
    """Panda tabletop environment containing three collision-enabled objects."""

    def __init__(self, *args, **kwargs) -> None:
        self.active_task: TaskSpec | None = None
        self.initial_target_position: np.ndarray | None = None
        self.objects_by_id: dict[str, object] = {}
        self.object_body_ids: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        # Skip Lift._load_model because it hard-codes a single cube.
        ManipulationEnv._load_model(self)
        base_position = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(base_position)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.objects_by_id = {
            "A": BoxObject(
                name=OBJECT_SPECS["A"]["name"],
                size=[0.022, 0.022, 0.022],
                rgba=[0.9, 0.05, 0.05, 1.0],
                rng=self.rng,
            ),
            "B": BallObject(
                name=OBJECT_SPECS["B"]["name"],
                size=[BLUE_BALL_RADIUS],
                rgba=[0.05, 0.2, 0.95, 1.0],
                friction=[0.9, 0.005, 0.0001],
                solref=[0.004, 1.0],
                solimp=[0.98, 0.995, 0.0005, 0.5, 2.0],
                joints=[{"type": "free", "damping": "0.001"}],
            ),
            "C": CylinderObject(
                name=OBJECT_SPECS["C"]["name"],
                size=[0.028, 0.022],
                rgba=[0.05, 0.75, 0.15, 1.0],
                friction=[0.3, 0.005, 0.0001],
                rng=self.rng,
            ),
        }
        objects = list(self.objects_by_id.values())
        self.placement_initializer = UniformRandomSampler(
            name="ThreeObjectSampler",
            mujoco_objects=objects,
            x_range=[-0.16, 0.16],
            y_range=[-0.16, 0.16],
            rotation=None,
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.002,
            rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self) -> None:
        ManipulationEnv._setup_references(self)
        self.object_body_ids = {
            object_id: self.sim.model.body_name2id(obj.root_body)
            for object_id, obj in self.objects_by_id.items()
        }

    def _setup_observables(self):
        observables = ManipulationEnv._setup_observables(self)
        if not self.use_object_obs:
            return observables

        for object_id in OBJECT_SPECS:
            position_sensor = self._make_object_position_sensor(object_id)
            quaternion_sensor = self._make_object_quaternion_sensor(object_id)
            for object_sensor in (position_sensor, quaternion_sensor):
                observables[object_sensor.__name__] = Observable(
                    name=object_sensor.__name__,
                    sensor=object_sensor,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _make_object_position_sensor(self, object_id: str):
        @sensor(modality="object")
        def object_position(_obs_cache):
            return self.get_object_position(object_id)

        object_position.__name__ = f"object_{object_id}_pos"
        return object_position

    def _make_object_quaternion_sensor(self, object_id: str):
        @sensor(modality="object")
        def object_quaternion(_obs_cache):
            body_id = self.object_body_ids[object_id]
            return T.convert_quat(
                np.array(self.sim.data.body_xquat[body_id]),
                to="xyzw",
            )

        object_quaternion.__name__ = f"object_{object_id}_quat"
        return object_quaternion

    def scene_generator(self) -> dict:
        """Sample three non-overlapping placements with an explicit safety margin."""
        for _ in range(200):
            placements = self.placement_initializer.sample()
            if self._placements_have_clearance(placements):
                return placements
        raise RuntimeError("Unable to generate a collision-safe three-object scene")

    def _placements_have_clearance(self, placements: dict) -> bool:
        placement_by_id = {
            object_id: placements[obj.name]
            for object_id, obj in self.objects_by_id.items()
        }
        for first_id, second_id in combinations(OBJECT_SPECS, 2):
            first_pos, _, first_obj = placement_by_id[first_id]
            second_pos, _, second_obj = placement_by_id[second_id]
            minimum_distance = (
                first_obj.horizontal_radius
                + second_obj.horizontal_radius
                + MIN_OBJECT_CLEARANCE
            )
            if np.linalg.norm(np.asarray(first_pos)[:2] - np.asarray(second_pos)[:2]) < minimum_distance:
                return False
        return True

    def _reset_internal(self) -> None:
        ManipulationEnv._reset_internal(self)
        if not self.deterministic_reset:
            for position, quaternion, obj in self.scene_generator().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.asarray(position), np.asarray(quaternion)]),
                )
        self.active_task = None
        self.initial_target_position = None

    def set_task(self, task: TaskSpec) -> None:
        self.active_task = task
        self.initial_target_position = self.get_object_position(task.target_id).copy()

    def get_object_position(self, object_id: str) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.object_body_ids[object_id]])

    def get_object_quaternion(self, object_id: str) -> np.ndarray:
        body_id = self.object_body_ids[object_id]
        return T.convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

    def reward(self, action=None) -> float:
        return float(self._check_success())

    def _check_success(self) -> bool:
        if self.active_task is None or self.initial_target_position is None:
            return False
        current_position = self.get_object_position(self.active_task.target_id)
        if self.active_task.task_type == "pick":
            table_height = self.model.mujoco_arena.table_offset[2]
            return bool(current_position[2] > table_height + 0.08)
        displacement = np.linalg.norm(
            current_position[:2] - self.initial_target_position[:2]
        )
        table_height = self.model.mujoco_arena.table_offset[2]
        remains_on_table = current_position[2] > table_height - 0.02
        return bool(displacement >= PUSH_SUCCESS_DISTANCE and remains_on_table)

    def visualize(self, vis_settings) -> None:
        ManipulationEnv.visualize(self, vis_settings=vis_settings)


def rotation_about_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def canonicalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def robot_state(obs: dict) -> np.ndarray:
    """Return XYZ and the matching end-effector site quaternion in xyzw order."""
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            canonicalize_quaternion(obs["robot0_eef_quat_site"]),
        ]
    ).astype(np.float32)


def distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator == 0.0:
        return float(np.linalg.norm(point - start))
    interpolation = np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0)
    projection = start + interpolation * segment
    return float(np.linalg.norm(point - projection))


def choose_push_direction(env: MultiObjectVLAEnv, target_id: str) -> np.ndarray:
    target_position = env.get_object_position(target_id)[:2]
    target_radius = env.objects_by_id[target_id].horizontal_radius
    angles = env.rng.permutation(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False))

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        start = target_position - direction * (target_radius + 0.07)
        end = target_position + direction * PUSH_DISTANCE
        if np.any(np.abs(start) > 0.28) or np.any(np.abs(end) > 0.30):
            continue

        path_is_clear = True
        for other_id in OBJECT_SPECS:
            if other_id == target_id:
                continue
            other_position = env.get_object_position(other_id)[:2]
            required_clearance = (
                target_radius
                + env.objects_by_id[other_id].horizontal_radius
                + 0.035
            )
            if distance_to_segment(other_position, start, end) < required_clearance:
                path_is_clear = False
                break
        if path_is_clear:
            return direction

    raise RuntimeError(f"No collision-safe push path available for target {target_id}")


def schedule_task(
    env: MultiObjectVLAEnv,
    task_type: str | None = None,
    target_id: str | None = None,
) -> TaskSpec:
    task_type = task_type or str(env.rng.choice(["pick", "push"]))
    target_id = target_id or str(env.rng.choice(list(OBJECT_SPECS)))
    object_label = OBJECT_SPECS[target_id]["label"]
    if task_type == "pick":
        instruction = f"Pick up the {object_label}"
        return TaskSpec(task_type, target_id, instruction)
    instruction = f"Push away the {object_label}"
    return TaskSpec(
        task_type,
        target_id,
        instruction,
        push_direction=choose_push_direction(env, target_id),
    )


class ScriptedOracle:
    """Privileged state-machine expert for picking and collision-safe pushing."""

    def __init__(
        self,
        env: MultiObjectVLAEnv,
        task: TaskSpec,
        obs: dict,
        seed: int,
    ) -> None:
        self.env = env
        self.task = task
        self.phase = 0
        self.hold_steps = 0
        self.push_hold_position: np.ndarray | None = None
        self.rng = np.random.default_rng(seed)
        self.initial_object_position = env.get_object_position(task.target_id).copy()
        self.initial_eef_orientation = T.quat2mat(obs["robot0_eef_quat_site"])
        self.target_orientation = self._target_orientation()

    def _target_orientation(self) -> np.ndarray:
        if self.task.task_type == "push":
            if self.task.push_direction is None:
                raise RuntimeError("Push task is missing its direction")
            push_yaw = np.arctan2(
                self.task.push_direction[1], self.task.push_direction[0]
            )
            push_yaw = (push_yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
            return rotation_about_z(push_yaw) @ self.initial_eef_orientation
        if self.task.target_id != "A":
            return self.initial_eef_orientation.copy()
        object_matrix = T.quat2mat(self.env.get_object_quaternion(self.task.target_id))
        object_yaw = np.arctan2(object_matrix[1, 0], object_matrix[0, 0])
        aligned_yaw = (object_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        return rotation_about_z(aligned_yaw) @ self.initial_eef_orientation

    def action(self, obs: dict) -> np.ndarray:
        if self.task.task_type == "pick":
            target_position, gripper = self._pick_target(obs)
        else:
            target_position, gripper = self._push_target(obs)
        return self._pose_action(obs, target_position, gripper)

    def _pick_target(self, obs: dict) -> tuple[np.ndarray, float]:
        object_position = self.env.get_object_position(self.task.target_id)
        eef_position = obs["robot0_eef_pos"]

        if self.phase == 0:
            target = object_position + np.array([0.0, 0.0, 0.14])
            gripper = -1.0
            position_error = np.linalg.norm(target - eef_position)
            current_orientation = T.quat2mat(obs["robot0_eef_quat_site"])
            rotation_error = np.linalg.norm(
                orientation_error(self.target_orientation, current_orientation)
            )
            if position_error < 0.025 and rotation_error < 0.05:
                self.phase = 1
        elif self.phase == 1:
            target = object_position.copy()
            target[2] -= DEEP_GRASP_OFFSET
            gripper = -1.0
            if np.linalg.norm(target - eef_position) < 0.018:
                self.phase = 2
        elif self.phase == 2:
            target = object_position.copy()
            target[2] -= DEEP_GRASP_OFFSET
            gripper = 1.0
            self.hold_steps += 1
            if self.hold_steps >= GRASP_HOLD_STEPS:
                self.phase = 3
        else:
            target = object_position.copy()
            target[2] = self.env.table_offset[2] + 0.20
            gripper = 1.0
        return target, gripper

    def _push_target(self, obs: dict) -> tuple[np.ndarray, float]:
        direction = self.task.push_direction
        if direction is None:
            raise RuntimeError("Push task is missing its direction")

        target_radius = self.env.objects_by_id[self.task.target_id].horizontal_radius
        behind_xy = self.initial_object_position[:2] - direction * (target_radius + 0.045)
        push_xy = self.initial_object_position[:2] + direction * PUSH_DISTANCE
        object_z = self.initial_object_position[2]
        contact_z = object_z - DEEP_GRASP_OFFSET
        eef_position = obs["robot0_eef_pos"]

        if self.phase == 0:
            target = np.array([behind_xy[0], behind_xy[1], object_z + 0.10])
            if np.linalg.norm(target - eef_position) < 0.025:
                self.phase = 1
        elif self.phase == 1:
            target = np.array([behind_xy[0], behind_xy[1], contact_z])
            if np.linalg.norm(target - eef_position) < 0.018:
                self.phase = 2
        elif self.phase == 2:
            target = np.array([push_xy[0], push_xy[1], contact_z])
            if self.env._check_success():
                self.phase = 3
                self.push_hold_position = eef_position.copy()
        else:
            if self.push_hold_position is None:
                self.push_hold_position = eef_position.copy()
            target = self.push_hold_position
        return target, 1.0

    def _pose_action(
        self,
        obs: dict,
        target_position: np.ndarray,
        gripper_action: float,
    ) -> np.ndarray:
        low, high = self.env.action_spec
        action = np.zeros(7, dtype=np.float32)
        if self.task.task_type == "push" and self.phase >= 2:
            position_gain = 3.0
        else:
            position_gain = POSITION_GAIN
        action[:3] = (target_position - obs["robot0_eef_pos"]) * position_gain
        current_orientation = T.quat2mat(obs["robot0_eef_quat_site"])
        action[3:6] = (
            orientation_error(self.target_orientation, current_orientation)
            * ORIENTATION_GAIN
        )
        action[:6] += self.rng.normal(0.0, POSE_ACTION_NOISE_STD, size=6)
        action[6] = gripper_action
        return np.clip(action, low, high)


def flip_camera_image(obs: dict, camera_name: str) -> np.ndarray:
    image = np.flipud(obs[f"{camera_name}_image"])
    image = np.ascontiguousarray(image)
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(f"Unexpected {camera_name} image shape: {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"{camera_name} must be uint8, got {image.dtype}")
    return image


def initial_object_poses(env: MultiObjectVLAEnv) -> np.ndarray:
    return np.stack(
        [
            np.concatenate(
                [
                    env.get_object_position(object_id),
                    canonicalize_quaternion(env.get_object_quaternion(object_id)),
                ]
            )
            for object_id in OBJECT_SPECS
        ],
        axis=0,
    ).astype(np.float32)


def validate_episode_arrays(
    image_agentview: np.ndarray,
    image_wrist: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    instruction: np.ndarray,
    valid_mask: np.ndarray,
    initial_object_pose: np.ndarray,
    success_step: np.ndarray,
    scene_seed: np.ndarray,
    collection_seed: np.ndarray,
    split: np.ndarray,
) -> None:
    expected_image_shape = (EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    if image_agentview.shape != expected_image_shape or image_agentview.dtype != np.uint8:
        raise ValueError(
            f"image_agentview must be uint8 {expected_image_shape}, got "
            f"{image_agentview.shape} {image_agentview.dtype}"
        )
    if image_wrist.shape != expected_image_shape or image_wrist.dtype != np.uint8:
        raise ValueError(
            f"image_wrist must be uint8 {expected_image_shape}, got "
            f"{image_wrist.shape} {image_wrist.dtype}"
        )
    if state.shape != (EPISODE_STEPS, 7) or state.dtype != np.float32:
        raise ValueError(f"state must be float32 (150, 7), got {state.shape} {state.dtype}")
    if action.shape != (EPISODE_STEPS, 7) or action.dtype != np.float32:
        raise ValueError(f"action must be float32 (150, 7), got {action.shape} {action.dtype}")
    if instruction.shape != () or instruction.dtype.kind != "U":
        raise ValueError("instruction must be a scalar Unicode string")
    if valid_mask.shape != (EPISODE_STEPS,) or valid_mask.dtype != np.uint8:
        raise ValueError("valid_mask must be uint8 (150,)")
    if initial_object_pose.shape != (3, 7) or initial_object_pose.dtype != np.float32:
        raise ValueError("initial_object_pose must be float32 (3, 7)")
    if success_step.shape != () or not 1 <= int(success_step) <= EPISODE_STEPS:
        raise ValueError("success_step must be a scalar in [1, 150]")
    expected_mask = np.arange(EPISODE_STEPS) < int(success_step)
    if not np.array_equal(valid_mask.astype(bool), expected_mask):
        raise ValueError("valid_mask must be one through success_step and zero afterwards")
    if scene_seed.shape != () or scene_seed.dtype != np.int64:
        raise ValueError("scene_seed must be a scalar int64")
    if collection_seed.shape != () or collection_seed.dtype != np.int64:
        raise ValueError("collection_seed must be a scalar int64")
    if split.shape != () or str(split) not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")


def save_episode(
    episode_id: int,
    image_agentview: list[np.ndarray],
    image_wrist: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    task: TaskSpec,
    object_poses: np.ndarray,
    success_step: int,
    scene_seed: int,
    collection_seed: int,
    split: str,
    data_dir: str = DATASET_DIR,
) -> str:
    valid_mask = (np.arange(EPISODE_STEPS) < success_step).astype(np.uint8)
    arrays = {
        "image_agentview": np.asarray(image_agentview, dtype=np.uint8),
        "image_wrist": np.asarray(image_wrist, dtype=np.uint8),
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "instruction": np.asarray(task.instruction),
        "valid_mask": valid_mask,
        "initial_object_pose": np.asarray(object_poses, dtype=np.float32),
        "success_step": np.asarray(success_step, dtype=np.int64),
        "scene_seed": np.asarray(scene_seed, dtype=np.int64),
        "collection_seed": np.asarray(collection_seed, dtype=np.int64),
        "split": np.asarray(split),
    }
    validate_episode_arrays(**arrays)
    output_path = os.path.join(data_dir, f"ep_{episode_id:05d}.npz")
    np.savez_compressed(
        output_path,
        **arrays,
        target_id=np.asarray(task.target_id),
        task_type=np.asarray(task.task_type),
    )
    return output_path


def verify_archive(path: str) -> None:
    with np.load(path, allow_pickle=False) as episode:
        required_keys = {
            "image_agentview",
            "image_wrist",
            "state",
            "action",
            "instruction",
            "valid_mask",
            "initial_object_pose",
            "success_step",
            "scene_seed",
            "collection_seed",
            "split",
            "target_id",
            "task_type",
        }
        missing_keys = required_keys.difference(episode.files)
        if missing_keys:
            raise KeyError(f"{path} is missing keys: {sorted(missing_keys)}")
        validate_episode_arrays(
            image_agentview=episode["image_agentview"],
            image_wrist=episode["image_wrist"],
            state=episode["state"],
            action=episode["action"],
            instruction=episode["instruction"],
            valid_mask=episode["valid_mask"],
            initial_object_pose=episode["initial_object_pose"],
            success_step=episode["success_step"],
            scene_seed=episode["scene_seed"],
            collection_seed=episode["collection_seed"],
            split=episode["split"],
        )


def read_episode_metadata(path: str) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        return {
            "filename": os.path.basename(path),
            "split": str(episode["split"].item()),
            "task_type": str(episode["task_type"].item()),
            "target_id": str(episode["target_id"].item()),
            "instruction": str(episode["instruction"].item()),
            "scene_seed": int(episode["scene_seed"].item()),
            "collection_seed": int(episode["collection_seed"].item()),
            "success_step": int(episode["success_step"].item()),
        }


def write_manifest(data_dir: str, rows: list[dict]) -> None:
    manifest_path = os.path.join(data_dir, "dataset_manifest.csv")
    fieldnames = [
        "filename",
        "split",
        "task_type",
        "target_id",
        "instruction",
        "scene_seed",
        "collection_seed",
        "success_step",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bucket_name(bucket: tuple[str, str]) -> str:
    return f"{bucket[0]}_{bucket[1]}"


def load_attempt_statistics(data_dir: str) -> tuple[dict, dict]:
    summary_path = os.path.join(data_dir, "collection_summary.json")
    if not os.path.exists(summary_path):
        return (
            {bucket: 0 for bucket in TASK_BUCKETS},
            {bucket: 0 for bucket in TASK_BUCKETS},
        )
    with open(summary_path, "r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    attempts = {
        bucket: int(summary.get("buckets", {}).get(bucket_name(bucket), {}).get("attempts", 0))
        for bucket in TASK_BUCKETS
    }
    failures = {
        bucket: int(summary.get("buckets", {}).get(bucket_name(bucket), {}).get("failures", 0))
        for bucket in TASK_BUCKETS
    }
    return attempts, failures


def write_collection_summary(
    data_dir: str,
    collection_seed: int,
    requested_total: int,
    accepted: dict,
    attempts: dict,
    failures: dict,
) -> None:
    summary = {
        "collection_seed": collection_seed,
        "requested_total": requested_total,
        "accepted_total": sum(accepted.values()),
        "buckets": {},
    }
    for bucket in TASK_BUCKETS:
        attempt_count = attempts[bucket]
        summary["buckets"][bucket_name(bucket)] = {
            "accepted": accepted[bucket],
            "attempts": attempt_count,
            "failures": failures[bucket],
            "success_rate": accepted[bucket] / attempt_count if attempt_count else None,
        }
    with open(
        os.path.join(data_dir, "collection_summary.json"),
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)


def verify_dataset(
    data_dir: str = DATASET_DIR,
    expected_total: int | None = None,
    require_balanced: bool = True,
) -> int:
    episode_paths = sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".npz")
    )
    if not episode_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    bucket_counts = {bucket: 0 for bucket in TASK_BUCKETS}
    split_counts = {
        bucket: {"train": 0, "val": 0, "test": 0}
        for bucket in TASK_BUCKETS
    }
    metadata_rows = []
    for episode_path in episode_paths:
        verify_archive(episode_path)
        metadata = read_episode_metadata(episode_path)
        bucket = (metadata["task_type"], metadata["target_id"])
        if bucket not in bucket_counts:
            raise ValueError(f"Unknown task bucket in {episode_path}: {bucket}")
        bucket_counts[bucket] += 1
        split_counts[bucket][metadata["split"]] += 1
        metadata_rows.append(metadata)

    if expected_total is not None and len(episode_paths) != expected_total:
        raise ValueError(f"Expected {expected_total} episodes, found {len(episode_paths)}")
    if require_balanced and len(set(bucket_counts.values())) != 1:
        raise ValueError(f"Dataset is not balanced: {bucket_counts}")
    if expected_total is not None:
        if expected_total % len(TASK_BUCKETS) != 0:
            raise ValueError("expected_total must be divisible by the number of task buckets")
        quota = expected_total // len(TASK_BUCKETS)
        expected_splits = {
            "train": int(quota * TRAIN_RATIO),
            "val": int(quota * VAL_RATIO),
        }
        expected_splits["test"] = quota - expected_splits["train"] - expected_splits["val"]
        for bucket in TASK_BUCKETS:
            if bucket_counts[bucket] != quota:
                raise ValueError(
                    f"Bucket {bucket} has {bucket_counts[bucket]} episodes, expected {quota}"
                )
            if split_counts[bucket] != expected_splits:
                raise ValueError(
                    f"Bucket {bucket} split mismatch: {split_counts[bucket]} "
                    f"!= {expected_splits}"
                )

    write_manifest(data_dir, metadata_rows)
    print(f"Verified {len(episode_paths)} fixed-length archives.", flush=True)
    for bucket in TASK_BUCKETS:
        print(
            f"  {bucket[0]:4s} target {bucket[1]}: {bucket_counts[bucket]} | "
            f"train={split_counts[bucket]['train']} "
            f"val={split_counts[bucket]['val']} "
            f"test={split_counts[bucket]['test']}",
            flush=True,
        )
    return len(episode_paths)


def split_schedule(quota: int, bucket: tuple[str, str], seed: int) -> list[str]:
    train_count = int(quota * TRAIN_RATIO)
    val_count = int(quota * VAL_RATIO)
    labels = (
        ["train"] * train_count
        + ["val"] * val_count
        + ["test"] * (quota - train_count - val_count)
    )
    bucket_offset = TASK_BUCKETS.index(bucket) * 10_007
    rng = np.random.default_rng(seed + bucket_offset)
    rng.shuffle(labels)
    return labels


def existing_collection_state(data_dir: str) -> tuple[dict, set[int], int, list[dict]]:
    counts = {bucket: 0 for bucket in TASK_BUCKETS}
    seeds: set[int] = set()
    metadata_rows: list[dict] = []
    max_episode_id = 0
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".npz"):
            continue
        path = os.path.join(data_dir, filename)
        verify_archive(path)
        metadata = read_episode_metadata(path)
        bucket = (metadata["task_type"], metadata["target_id"])
        if bucket not in counts:
            raise ValueError(f"Unknown task bucket in {path}: {bucket}")
        if metadata["scene_seed"] in seeds:
            raise ValueError(f"Duplicate scene_seed {metadata['scene_seed']} in {path}")
        counts[bucket] += 1
        seeds.add(metadata["scene_seed"])
        metadata_rows.append(metadata)
        try:
            max_episode_id = max(max_episode_id, int(os.path.splitext(filename)[0].split("_")[-1]))
        except ValueError:
            pass
    return counts, seeds, max_episode_id, metadata_rows


def reseed_environment(env: MultiObjectVLAEnv, scene_seed: int) -> None:
    env.rng = np.random.default_rng(scene_seed)
    env.placement_initializer.rng = env.rng


def make_environment() -> MultiObjectVLAEnv:
    env = robosuite.make(
        env_name="MultiObjectVLAEnv",
        robots="Panda",
        controller_configs=build_osc_pose_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[IMAGE_HEIGHT, IMAGE_HEIGHT],
        camera_widths=[IMAGE_WIDTH, IMAGE_WIDTH],
        horizon=EPISODE_STEPS,
        ignore_done=True,
        control_freq=20,
    )
    low, _ = env.action_spec
    if low.shape != (7,):
        env.close()
        raise RuntimeError(f"OSC_POSE must expose exactly 7 actions, got {low.shape}")
    return env


def collect(
    num_episodes: int,
    data_dir: str = DATASET_DIR,
    collection_seed: int = COLLECTION_SEED,
) -> None:
    if num_episodes % len(TASK_BUCKETS) != 0:
        raise ValueError(
            f"num_episodes must be divisible by {len(TASK_BUCKETS)} for exact balance"
        )
    os.makedirs(data_dir, exist_ok=True)
    quota = num_episodes // len(TASK_BUCKETS)
    counts, used_seeds, episode_id, metadata_rows = existing_collection_state(data_dir)
    existing_collection_seeds = {row["collection_seed"] for row in metadata_rows}
    if existing_collection_seeds and existing_collection_seeds != {collection_seed}:
        raise ValueError(
            f"Existing dataset uses collection_seed={sorted(existing_collection_seeds)}, "
            f"but this run requested {collection_seed}"
        )
    if any(count > quota for count in counts.values()):
        raise ValueError(f"Existing data exceeds the requested per-bucket quota {quota}: {counts}")
    write_manifest(data_dir, metadata_rows)
    attempt_counts, failure_counts = load_attempt_statistics(data_dir)

    schedules = {
        bucket: split_schedule(quota, bucket, collection_seed)
        for bucket in TASK_BUCKETS
    }
    collection_rng = np.random.default_rng(collection_seed)
    env = make_environment()
    collected = sum(counts.values())
    attempt = 0

    try:
        while collected < num_episodes:
            attempt += 1
            underfilled = [bucket for bucket in TASK_BUCKETS if counts[bucket] < quota]
            bucket = underfilled[int(collection_rng.integers(len(underfilled)))]
            attempt_counts[bucket] += 1
            while True:
                scene_seed = int(collection_rng.integers(0, np.iinfo(np.int32).max))
                if scene_seed not in used_seeds:
                    break
            reseed_environment(env, scene_seed)
            obs = env.reset()
            object_poses = initial_object_poses(env)
            try:
                task = schedule_task(env, task_type=bucket[0], target_id=bucket[1])
            except RuntimeError as error:
                failure_counts[bucket] += 1
                write_collection_summary(
                    data_dir,
                    collection_seed,
                    num_episodes,
                    counts,
                    attempt_counts,
                    failure_counts,
                )
                print(f"[Discarded] Attempt {attempt} | {error}", flush=True)
                continue

            env.set_task(task)
            oracle = ScriptedOracle(env, task, obs, seed=scene_seed + 1)
            image_agentview: list[np.ndarray] = []
            image_wrist: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            success_step: int | None = None

            for step in range(1, EPISODE_STEPS + 1):
                action = oracle.action(obs)
                image_agentview.append(flip_camera_image(obs, "agentview"))
                image_wrist.append(flip_camera_image(obs, "robot0_eye_in_hand"))
                states.append(robot_state(obs))
                actions.append(action.copy())
                obs, _, _, _ = env.step(action)
                if success_step is None and env._check_success():
                    success_step = step

            success = success_step is not None and bool(env._check_success())
            if not success:
                failure_counts[bucket] += 1
                write_collection_summary(
                    data_dir,
                    collection_seed,
                    num_episodes,
                    counts,
                    attempt_counts,
                    failure_counts,
                )
                print(
                    f"[Discarded] Attempt {attempt} | Task: {task.instruction} | Oracle failed",
                    flush=True,
                )
                continue

            split = schedules[bucket][counts[bucket]]
            episode_id += 1
            output_path = save_episode(
                episode_id,
                image_agentview,
                image_wrist,
                states,
                actions,
                task,
                object_poses=object_poses,
                success_step=success_step,
                scene_seed=scene_seed,
                collection_seed=collection_seed,
                split=split,
                data_dir=data_dir,
            )
            verify_archive(output_path)
            counts[bucket] += 1
            collected += 1
            used_seeds.add(scene_seed)
            write_collection_summary(
                data_dir,
                collection_seed,
                num_episodes,
                counts,
                attempt_counts,
                failure_counts,
            )
            print(
                f"[Success] Attempt {attempt} | Episode {collected}/{num_episodes} | "
                f"{bucket[0]}-{bucket[1]} {counts[bucket]}/{quota} | "
                f"split={split} | success_step={success_step}",
                flush=True,
            )
    finally:
        env.close()

    verify_dataset(data_dir, expected_total=num_episodes, require_balanced=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--data-dir", default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=COLLECTION_SEED)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_dataset(args.data_dir, expected_total=args.num_episodes)
        return
    collect(args.num_episodes, data_dir=args.data_dir, collection_seed=args.seed)


if __name__ == "__main__":
    main()
