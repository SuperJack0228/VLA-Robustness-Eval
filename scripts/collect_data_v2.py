"""Clean-label data pipeline for the release MiniVLA policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

import robosuite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.collect_data import (
    OBJECT_SPECS,
    MultiObjectVLAEnv,
    canonicalize_quaternion,
    distance_to_segment,
    rotation_about_z,
)
from utils.v2_schema import (
    ACTION_DIM,
    DATASET_VERSION,
    EPISODE_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_COUNT,
    SCHEMA_VERSION,
    STATE_DIM,
    TASK_BUCKETS,
    instruction_for,
    read_metadata,
    schema_document,
    validate_episode_arrays,
)


DEFAULT_NUM_EPISODES = 1200
CONTROL_FREQ = 20
DATASET_DIR = "data/dataset_v2_clean"
COLLECTION_SEED = 20260817
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

POSITION_GAIN = 6.0
PICK_LIFT_GAIN = 4.0
PUSH_POSITION_GAIN = 3.5
BALL_PUSH_POSITION_GAIN = 5.5
CYLINDER_PUSH_POSITION_GAIN = 3.5
ORIENTATION_GAIN = 2.0
POSE_ACTION_NOISE_STD = 0.008
ORACLE_ACTION_LIMIT = 0.90
MAX_GRASP_ATTEMPTS = 3
GRASP_CONFIRM_STEPS = 2
LOST_GRASP_CONFIRM_STEPS = 5
GRASP_CLOSE_TIMEOUT = 12
PICK_DESCEND_TIMEOUT = 55
PUSH_DESCEND_TIMEOUT = 55
PUSH_TIMEOUT = 45
PUSH_SUCCESS_LOSS_CONFIRM_STEPS = 3
SPHERE_PENETRATION_TOLERANCE = 0.006
PICK_HOVER_HEIGHT = 0.12
PUSH_HOVER_HEIGHT = 0.10
DEEP_GRASP_OFFSET = 0.010
PUSH_DISTANCE = 0.14
BALL_PUSH_DISTANCE = 0.18
PUSH_SUCCESS_DISTANCE = 0.065
PUSH_MAX_LATERAL_ERROR = 0.06
# These margins account for the Panda fingers, not just the target geometry.
# A collision-free placement can still produce a bad demonstration when a
# finger sweeps a distractor during approach or contact.
PICK_TARGET_CLEARANCE = 0.050
PUSH_PATH_CLEARANCE = {
    "A": 0.060,
    "B": 0.060,
    # A fallen cylinder has a wider, less predictable swept envelope.
    "C": 0.080,
}
PICK_STABLE_SUCCESS_STEPS = 10
PUSH_STABLE_SUCCESS_STEPS = 5
FORCED_PUSH_RECOVERY_PROBABILITY = {
    "A": 0.0,
    "B": 0.30,
    "C": 0.0,
}
FORCED_PUSH_MISS_STEPS = 8
FORCED_PUSH_MISS_LATERAL_OFFSET = 0.045
PUSH_CONTACT_ACQUIRE_TIMEOUT = 12
PUSH_MEANINGFUL_MOTION = 0.010
BALL_PUSH_HEIGHT_ABOVE_TABLE = 0.022
BALL_PUSH_HEIGHT_CANDIDATES = (0.014, 0.018, 0.022, 0.026)
STABLE_MANIPULATION_NOISE_STD = 0.002
CYLINDER_CONTACT_NOISE_STD = 0.001
PUSH_CONTACT_HEIGHT_ABOVE_TABLE = {
    "A": 0.016,
    "B": 0.023,
    "C": 0.010,
}

PHASE_NAMES = {
    0: "pick_approach",
    1: "pick_descend",
    2: "pick_close",
    3: "pick_lift",
    4: "pick_recover",
    5: "push_approach",
    6: "push_descend",
    7: "push_contact",
    8: "push_hold",
    9: "push_recover",
}


@dataclass(frozen=True)
class TaskSpecV2:
    task_type: str
    target_id: str
    instruction: str
    push_direction: np.ndarray
    target_goal: np.ndarray


@dataclass
class RolloutResult:
    success: bool
    clean_success: bool
    success_step: int
    trajectory_length: int
    wrong_object_contact: bool
    saturation_steps: int
    retry_count: int
    phase_counts: dict[str, int]
    final_position_error: float
    final_orientation_error: float
    push_forward_displacement: float
    push_lateral_displacement: float
    forced_push_recovery: bool
    physics_failure: str | None
    arrays: dict[str, np.ndarray] | None


def build_osc_pose_controller_config_v2() -> dict:
    """Return a lower-energy OSC_POSE controller shared by collection and evaluation."""
    controller_config = load_composite_controller_config(robot="Panda")
    arm_config = load_part_controller_config(default_controller="OSC_POSE")
    arm_config.update(
        {
            "input_type": "delta",
            "input_ref_frame": "base",
            "input_min": -1,
            "input_max": 1,
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "kp": 120,
            "damping_ratio": 1,
            "gripper": {"type": "GRIP"},
        }
    )
    controller_config["body_parts"]["right"] = arm_config
    return controller_config


class MultiObjectVLAEnvV2(MultiObjectVLAEnv):
    """V2 scene with task-aware success and contact diagnostics."""

    active_task: TaskSpecV2 | None

    def set_task(self, task: TaskSpecV2) -> None:
        self.active_task = task
        self.initial_target_position = self.get_object_position(task.target_id).copy()

    def is_grasping(self, object_id: str) -> bool:
        return bool(
            self._check_grasp(
                self.robots[0].gripper,
                self.objects_by_id[object_id],
            )
        )

    def object_contact_flags(self) -> np.ndarray:
        gripper = self.robots[0].gripper
        grippers = gripper.values() if isinstance(gripper, dict) else (gripper,)
        gripper_geoms = [
            geom
            for gripper_model in grippers
            for geom in gripper_model.contact_geoms
        ]
        return np.asarray(
            [
                self.check_contact(gripper_geoms, self.objects_by_id[object_id])
                for object_id in OBJECT_SPECS
            ],
            dtype=np.uint8,
        )

    def object_grasp_flags(self) -> np.ndarray:
        return np.asarray(
            [self.is_grasping(object_id) for object_id in OBJECT_SPECS],
            dtype=np.uint8,
        )

    def object_uprightness(self, object_id: str) -> float:
        if object_id == "B":
            return 1.0
        rotation = T.quat2mat(self.get_object_quaternion(object_id))
        return float(abs(rotation[2, 2]))

    def _check_success(self) -> bool:
        if self.active_task is None or self.initial_target_position is None:
            return False
        current = self.get_object_position(self.active_task.target_id)
        table_height = self.model.mujoco_arena.table_offset[2]
        if self.active_task.task_type == "pick":
            return bool(
                current[2] > table_height + 0.08
                and self.is_grasping(self.active_task.target_id)
            )

        displacement = current[:2] - self.initial_target_position[:2]
        direction = self.active_task.push_direction
        forward = float(np.dot(displacement, direction))
        lateral_vector = displacement - forward * direction
        lateral_error = float(np.linalg.norm(lateral_vector))
        remains_on_table = current[2] > table_height - 0.02
        # "Push away" is a displacement task. Cylinder uprightness remains a
        # stricter evaluation diagnostic, but is not part of the language task.
        return bool(
            forward >= PUSH_SUCCESS_DISTANCE
            and lateral_error <= PUSH_MAX_LATERAL_ERROR
            and remains_on_table
        )


def _path_is_clear(
    env: MultiObjectVLAEnvV2,
    target_id: str,
    start: np.ndarray,
    end: np.ndarray,
) -> bool:
    target_radius = env.objects_by_id[target_id].horizontal_radius
    for other_id in OBJECT_SPECS:
        if other_id == target_id:
            continue
        other_xy = env.get_object_position(other_id)[:2]
        clearance = (
            target_radius
            + env.objects_by_id[other_id].horizontal_radius
            + PUSH_PATH_CLEARANCE[target_id]
        )
        if distance_to_segment(other_xy, start, end) < clearance:
            return False
    return True


def _target_has_gripper_clearance(
    env: MultiObjectVLAEnvV2,
    target_id: str,
) -> bool:
    """Require enough free space for a centered, bilateral Panda grasp."""
    target_xy = env.get_object_position(target_id)[:2]
    target_radius = env.objects_by_id[target_id].horizontal_radius
    for other_id in OBJECT_SPECS:
        if other_id == target_id:
            continue
        other_xy = env.get_object_position(other_id)[:2]
        minimum_distance = (
            target_radius
            + env.objects_by_id[other_id].horizontal_radius
            + PICK_TARGET_CLEARANCE
        )
        if np.linalg.norm(target_xy - other_xy) < minimum_distance:
            return False
    return True


def deterministic_push_direction(
    env: MultiObjectVLAEnvV2,
    obs: dict,
    target_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Define "away" as the ray from the initial EEF through the target."""
    target = env.get_object_position(target_id)
    eef_xy = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)[:2]
    direction = target[:2] - eef_xy
    norm = float(np.linalg.norm(direction))
    if norm < 0.04:
        raise RuntimeError("Target is too close to the robot reference ray")
    direction /= norm

    target_radius = env.objects_by_id[target_id].horizontal_radius
    start = target[:2] - direction * (target_radius + 0.065)
    push_distance = BALL_PUSH_DISTANCE if target_id == "B" else PUSH_DISTANCE
    end = target[:2] + direction * push_distance
    if np.any(np.abs(start) > 0.28) or np.any(np.abs(end) > 0.30):
        raise RuntimeError("Deterministic push path leaves the safe workspace")
    if not _path_is_clear(env, target_id, start, end):
        raise RuntimeError("Deterministic push path is blocked by a distractor")

    goal = target.copy()
    goal[:2] = end
    return direction.astype(np.float32), goal.astype(np.float32)


def schedule_task_v2(
    env: MultiObjectVLAEnvV2,
    obs: dict,
    task_type: str,
    target_id: str,
) -> TaskSpecV2:
    if task_type not in {"pick", "push"}:
        raise ValueError(f"Unsupported task type: {task_type}")
    if target_id not in OBJECT_SPECS:
        raise ValueError(f"Unsupported target ID: {target_id}")
    target = env.get_object_position(target_id).astype(np.float32)
    if task_type == "pick":
        if not _target_has_gripper_clearance(env, target_id):
            raise RuntimeError("Pick target lacks bilateral gripper clearance")
        goal = target.copy()
        goal[2] = env.table_offset[2] + 0.20
        return TaskSpecV2(
            task_type="pick",
            target_id=target_id,
            instruction=instruction_for("pick", target_id),
            push_direction=np.zeros(2, dtype=np.float32),
            target_goal=goal,
        )

    direction, goal = deterministic_push_direction(env, obs, target_id)
    return TaskSpecV2(
        task_type="push",
        target_id=target_id,
        instruction=instruction_for("push", target_id),
        push_direction=direction,
        target_goal=goal,
    )


class ScriptedOracleV2:
    """Contact-aware oracle with deterministic goals and bounded recovery attempts."""

    def __init__(
        self,
        env: MultiObjectVLAEnvV2,
        task: TaskSpecV2,
        obs: dict,
        seed: int,
        ball_push_height_above_table: float = BALL_PUSH_HEIGHT_ABOVE_TABLE,
    ) -> None:
        self.env = env
        self.task = task
        self.phase = 0 if task.task_type == "pick" else 5
        self.last_phase = self.phase
        self.phase_steps = 0
        self.confirmed_grasp_steps = 0
        self.lost_grasp_steps = 0
        self.lost_push_success_steps = 0
        self.retry_count = 0
        self.rng = np.random.default_rng(seed)
        self.ball_push_height_above_table = float(ball_push_height_above_table)
        self.force_push_recovery = bool(
            task.task_type == "push"
            and self.rng.random()
            < FORCED_PUSH_RECOVERY_PROBABILITY.get(task.target_id, 0.0)
        )
        push_lateral = np.array(
            [-task.push_direction[1], task.push_direction[0]],
            dtype=np.float64,
        )
        push_sign = -1.0 if self.rng.random() < 0.5 else 1.0
        self.forced_push_offset = (
            push_lateral * push_sign * FORCED_PUSH_MISS_LATERAL_OFFSET
        )
        self.forced_push_initial_position = self.env.get_object_position(
            task.target_id
        ).copy()
        self.initial_eef_orientation = T.quat2mat(obs["robot0_eef_quat_site"])
        self.target_orientation = self._target_orientation()
        self.push_hold_position: np.ndarray | None = None
        self.push_contact_seen = False

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES[self.last_phase]

    def _set_phase(self, phase: int) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_steps = 0
            self.confirmed_grasp_steps = 0
            self.lost_grasp_steps = 0
            self.lost_push_success_steps = 0

    def _target_orientation(self) -> np.ndarray:
        if self.task.task_type == "push":
            yaw = np.arctan2(
                self.task.push_direction[1],
                self.task.push_direction[0],
            )
            yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
            return rotation_about_z(yaw) @ self.initial_eef_orientation
        if self.task.target_id != "A":
            return self.initial_eef_orientation.copy()
        object_matrix = T.quat2mat(self.env.get_object_quaternion("A"))
        object_yaw = np.arctan2(object_matrix[1, 0], object_matrix[0, 0])
        aligned_yaw = (object_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        return rotation_about_z(aligned_yaw) @ self.initial_eef_orientation

    def _pick_object_position(self) -> np.ndarray:
        """Return the live target pose; V3 recovery oracles may override it."""
        return self.env.get_object_position(self.task.target_id)

    def action(self, obs: dict) -> np.ndarray:
        # A real grasp always wins over a scripted recovery. This transition is
        # performed before the phase label is recorded, keeping labels causal.
        if (
            self.task.task_type == "pick"
            and self.phase == 4
            and self.env.is_grasping(self.task.target_id)
        ):
            self._set_phase(3)
        self.last_phase = self.phase
        self.phase_steps += 1
        if self.task.task_type == "pick":
            target, gripper, gain = self._pick_target(obs)
        else:
            target, gripper, gain = self._push_target(obs)
        return self._pose_action(obs, target, gripper, gain)

    def _pick_target(self, obs: dict) -> tuple[np.ndarray, float, float]:
        object_position = self._pick_object_position()
        eef_position = np.asarray(obs["robot0_eef_pos"])
        hover = object_position + np.array([0.0, 0.0, PICK_HOVER_HEIGHT])

        if self.phase == 0:
            target = hover
            gripper = -1.0
            current_orientation = T.quat2mat(obs["robot0_eef_quat_site"])
            if (
                np.linalg.norm(target - eef_position) < 0.018
                and np.linalg.norm(
                    orientation_error(self.target_orientation, current_orientation)
                )
                < 0.04
            ):
                self._set_phase(1)
        elif self.phase == 1:
            target = object_position.copy()
            target[2] -= DEEP_GRASP_OFFSET
            gripper = -1.0
            if np.linalg.norm(target - eef_position) < 0.012:
                self._set_phase(2)
            elif self.phase_steps >= PICK_DESCEND_TIMEOUT:
                self.retry_count += 1
                self._set_phase(4)
        elif self.phase == 2:
            target = object_position.copy()
            target[2] -= DEEP_GRASP_OFFSET
            gripper = 1.0
            if self.env.is_grasping(self.task.target_id):
                self.confirmed_grasp_steps += 1
                # Do not press a grasped object into the table while confirming
                # bilateral contact. The next recorded phase will be lift.
                target = eef_position.copy()
            else:
                self.confirmed_grasp_steps = 0
            if self.confirmed_grasp_steps >= GRASP_CONFIRM_STEPS:
                self._set_phase(3)
            elif self.phase_steps >= GRASP_CLOSE_TIMEOUT:
                self.retry_count += 1
                self._set_phase(4)
        elif self.phase == 3:
            target = object_position.copy()
            target[2] = self.env.table_offset[2] + 0.20
            gripper = 1.0
            if self.env.is_grasping(self.task.target_id):
                self.lost_grasp_steps = 0
            else:
                self.lost_grasp_steps += 1
            if (
                self.lost_grasp_steps >= LOST_GRASP_CONFIRM_STEPS
                and self.retry_count < MAX_GRASP_ATTEMPTS
            ):
                self.retry_count += 1
                self._set_phase(4)
        else:
            if self.env.is_grasping(self.task.target_id):
                self._set_phase(3)
                target = object_position.copy()
                target[2] = self.env.table_offset[2] + 0.20
                gripper = 1.0
            else:
                gripper = -1.0
                clearance_z = object_position[2] + PICK_HOVER_HEIGHT
                if eef_position[2] < clearance_z - 0.02:
                    target = eef_position.copy()
                    target[2] = clearance_z
                else:
                    target = hover
                    if np.linalg.norm(target - eef_position) < 0.025:
                        self._set_phase(0)
        gain = PICK_LIFT_GAIN if self.last_phase == 3 else POSITION_GAIN
        return target, gripper, gain

    def _push_target(self, obs: dict) -> tuple[np.ndarray, float, float]:
        object_position = self.env.get_object_position(self.task.target_id)
        eef_position = np.asarray(obs["robot0_eef_pos"])
        radius = self.env.objects_by_id[self.task.target_id].horizontal_radius
        behind_xy = object_position[:2] - self.task.push_direction * (radius + 0.055)
        forcing_recovery = self.force_push_recovery and self.retry_count == 0
        approach_offset = (
            self.forced_push_offset if forcing_recovery else np.zeros(2)
        )
        approach_xy = behind_xy + approach_offset
        hover = np.array(
            [approach_xy[0], approach_xy[1], object_position[2] + PUSH_HOVER_HEIGHT]
        )
        if self.task.target_id == "B":
            contact_z = (
                self.env.table_offset[2] + self.ball_push_height_above_table
            )
        else:
            contact_z = self.env.table_offset[2] + PUSH_CONTACT_HEIGHT_ABOVE_TABLE[
                self.task.target_id
            ]

        if self.phase == 5:
            target = hover
            if np.linalg.norm(target - eef_position) < 0.018:
                self._set_phase(6)
        elif self.phase == 6:
            target = np.array([approach_xy[0], approach_xy[1], contact_z])
            if np.linalg.norm(target - eef_position) < 0.012:
                self._set_phase(7)
            elif self.phase_steps >= PUSH_DESCEND_TIMEOUT:
                self.retry_count += 1
                self._set_phase(9)
        elif self.phase == 7:
            target = self.task.target_goal.copy()
            target[2] = contact_z
            target_index = tuple(OBJECT_SPECS).index(self.task.target_id)
            target_contact = bool(self.env.object_contact_flags()[target_index])
            target_motion = float(
                np.linalg.norm(
                    object_position - self.forced_push_initial_position
                )
            )
            if target_contact or target_motion > 0.003:
                self.push_contact_seen = True
            if forcing_recovery:
                target[:2] += self.forced_push_offset
                if target_contact or target_motion > 0.003:
                    self.force_push_recovery = False
                    forcing_recovery = False
                    target = self.task.target_goal.copy()
                    target[2] = contact_z
            push_timeout = PUSH_TIMEOUT
            if self.env._check_success():
                self.push_hold_position = eef_position.copy()
                self._set_phase(8)
            elif forcing_recovery and self.phase_steps >= FORCED_PUSH_MISS_STEPS:
                self.retry_count += 1
                self._set_phase(9)
            elif (
                not self.push_contact_seen
                and self.phase_steps >= PUSH_CONTACT_ACQUIRE_TIMEOUT
            ):
                self.retry_count += 1
                self._set_phase(9)
            elif self.phase_steps >= push_timeout:
                displacement = object_position[:2] - self.env.initial_target_position[:2]
                forward = float(np.dot(displacement, self.task.push_direction))
                self.retry_count += 1
                if forward <= PUSH_MEANINGFUL_MOTION:
                    self._set_phase(9)
                else:
                    # Reacquire the moved object directly; do not teach a retreat
                    # after a useful partial push.
                    self._set_phase(5)
        elif self.phase == 8:
            if self.push_hold_position is None:
                self.push_hold_position = eef_position.copy()
            target = self.push_hold_position
            if self.env._check_success():
                self.lost_push_success_steps = 0
            else:
                self.lost_push_success_steps += 1
                if (
                    self.lost_push_success_steps
                    >= PUSH_SUCCESS_LOSS_CONFIRM_STEPS
                ):
                    self.retry_count += 1
                    self.push_hold_position = None
                    self._set_phase(5)
        else:
            retreat_z = object_position[2] + PUSH_HOVER_HEIGHT
            if eef_position[2] < retreat_z - 0.02:
                target = eef_position.copy()
                target[2] = retreat_z
            else:
                # Both the behind point and height are recomputed from the live
                # object pose, so the retry cannot reuse a stale contact line.
                target = hover
                if np.linalg.norm(target - eef_position) < 0.025:
                    self.force_push_recovery = False
                    self.push_contact_seen = False
                    self._set_phase(5)
        if self.last_phase == 7 and self.task.target_id == "B":
            gain = BALL_PUSH_POSITION_GAIN
        elif self.last_phase == 7 and self.task.target_id == "C":
            gain = CYLINDER_PUSH_POSITION_GAIN
        elif self.last_phase == 7:
            gain = PUSH_POSITION_GAIN
        else:
            gain = POSITION_GAIN
        return target, 1.0, gain

    def _pose_action(
        self,
        obs: dict,
        target_position: np.ndarray,
        gripper_action: float,
        position_gain: float,
    ) -> np.ndarray:
        low, high = self.env.action_spec
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:3] = (
            target_position - np.asarray(obs["robot0_eef_pos"])
        ) * position_gain
        current_orientation = T.quat2mat(obs["robot0_eef_quat_site"])
        action[3:6] = (
            orientation_error(self.target_orientation, current_orientation)
            * ORIENTATION_GAIN
        )
        if self.last_phase in {3, 8}:
            noise_std = STABLE_MANIPULATION_NOISE_STD
        elif self.last_phase == 7 and self.task.target_id == "C":
            noise_std = CYLINDER_CONTACT_NOISE_STD
        else:
            noise_std = POSE_ACTION_NOISE_STD
        action[:6] += self.rng.normal(
            0.0,
            noise_std,
            size=6,
        )
        action[:6] = np.clip(
            action[:6],
            -ORACLE_ACTION_LIMIT,
            ORACLE_ACTION_LIMIT,
        )
        action[6] = gripper_action
        return np.clip(action, low, high)


class ProprioceptionTracker:
    """Build a 17D state with finite-difference EEF velocities."""

    def __init__(self, control_freq: int = CONTROL_FREQ) -> None:
        self.control_freq = float(control_freq)
        self.previous_position: np.ndarray | None = None
        self.previous_orientation: np.ndarray | None = None

    def extract(self, obs: dict) -> np.ndarray:
        position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        quaternion = canonicalize_quaternion(obs["robot0_eef_quat_site"])
        orientation = T.quat2mat(quaternion)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        gripper_qvel = np.asarray(obs["robot0_gripper_qvel"], dtype=np.float32)
        if gripper_qpos.shape != (2,) or gripper_qvel.shape != (2,):
            raise ValueError(
                "Panda gripper observations must be 2D qpos and qvel, got "
                f"{gripper_qpos.shape} and {gripper_qvel.shape}"
            )

        if self.previous_position is None:
            linear_velocity = np.zeros(3, dtype=np.float32)
            angular_velocity = np.zeros(3, dtype=np.float32)
        else:
            linear_velocity = (
                (position - self.previous_position) * self.control_freq
            ).astype(np.float32)
            angular_velocity = (
                orientation_error(orientation, self.previous_orientation)
                * self.control_freq
            ).astype(np.float32)
        self.previous_position = position.copy()
        self.previous_orientation = orientation.copy()

        state = np.concatenate(
            [
                position,
                quaternion,
                gripper_qpos,
                gripper_qvel,
                linear_velocity,
                angular_velocity,
            ]
        ).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected {STATE_DIM}D state, got {state.shape}")
        return state


def object_poses(env: MultiObjectVLAEnvV2) -> np.ndarray:
    return np.stack(
        [
            np.concatenate(
                [
                    env.get_object_position(object_id),
                    canonicalize_quaternion(env.get_object_quaternion(object_id)),
                ]
            )
            for object_id in OBJECT_SPECS
        ]
    ).astype(np.float32)


def camera_image(obs: dict, camera_name: str) -> np.ndarray:
    image = np.ascontiguousarray(np.flipud(obs[f"{camera_name}_image"]))
    expected = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    if image.shape != expected or image.dtype != np.uint8:
        raise ValueError(
            f"{camera_name} must be uint8 {expected}, got {image.shape} {image.dtype}"
        )
    return image


def _pad(values: list[np.ndarray | int], length: int) -> np.ndarray:
    if not values:
        raise ValueError("Cannot pad an empty trajectory")
    array = np.asarray(values)
    if len(array) > length:
        raise ValueError(f"Trajectory length {len(array)} exceeds {length}")
    if len(array) == length:
        return array
    padding = np.repeat(array[-1:], length - len(array), axis=0)
    return np.concatenate([array, padding], axis=0)


def rollout_oracle(
    env: MultiObjectVLAEnvV2,
    task: TaskSpecV2,
    obs: dict,
    seed: int,
    capture_images: bool,
    ball_push_height_above_table: float = BALL_PUSH_HEIGHT_ABOVE_TABLE,
    oracle: ScriptedOracleV2 | None = None,
    before_step_hook: (
        Callable[[int, ScriptedOracleV2, dict, np.ndarray], None] | None
    ) = None,
) -> RolloutResult:
    env.set_task(task)
    if oracle is None:
        oracle = ScriptedOracleV2(
            env,
            task,
            obs,
            seed=seed + 1,
            ball_push_height_above_table=ball_push_height_above_table,
        )
    tracker = ProprioceptionTracker()
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    target_index = tuple(OBJECT_SPECS).index(task.target_id)
    records: dict[str, list] = {
        "state": [],
        "previous_action": [],
        "action": [],
        "object_pose": [],
        "object_contact": [],
        "object_grasped": [],
        "expert_phase": [],
        "retry_count": [],
        "success_after_action": [],
    }
    if capture_images:
        records["image_agentview"] = []
        records["image_wrist"] = []

    wrong_object_contact = False
    saturation_steps = 0
    success_step = 0
    success_streak = 0
    physics_failure: str | None = None
    required_success_steps = (
        PICK_STABLE_SUCCESS_STEPS
        if task.task_type == "pick"
        else PUSH_STABLE_SUCCESS_STEPS
    )

    for step in range(1, EPISODE_STEPS + 1):
        state = tracker.extract(obs)
        action = oracle.action(obs)
        contacts_before = env.object_contact_flags()
        grasps_before = env.object_grasp_flags()
        records["state"].append(state)
        records["previous_action"].append(previous_action.copy())
        records["action"].append(action.copy())
        records["object_pose"].append(object_poses(env))
        records["object_contact"].append(contacts_before)
        records["object_grasped"].append(grasps_before)
        records["expert_phase"].append(oracle.last_phase)
        records["retry_count"].append(oracle.retry_count)
        if capture_images:
            records["image_agentview"].append(camera_image(obs, "agentview"))
            records["image_wrist"].append(
                camera_image(obs, "robot0_eye_in_hand")
            )

        saturation_steps += int(np.any(np.abs(action[:6]) >= 0.99))
        if before_step_hook is not None:
            before_step_hook(step, oracle, obs, action)
        obs, _, _, _ = env.step(action)
        contacts_after = env.object_contact_flags()
        wrong_object_contact |= bool(
            np.any(np.delete(contacts_after.astype(bool), target_index))
        )
        if task.task_type == "push" and task.target_id == "B":
            sphere_floor = (
                env.table_offset[2]
                + float(env.objects_by_id["B"].horizontal_radius)
                - SPHERE_PENETRATION_TOLERANCE
            )
            sphere_z = float(env.get_object_position("B")[2])
            if sphere_z < sphere_floor:
                physics_failure = (
                    f"blue_ball_penetration(z={sphere_z:.4f}, "
                    f"floor={sphere_floor:.4f})"
                )
        success = bool(env._check_success()) and physics_failure is None
        records["success_after_action"].append(int(success))
        previous_action = action
        if success:
            if success_streak == 0:
                success_step = step
            success_streak += 1
        else:
            success_step = 0
            success_streak = 0
        if success_streak >= required_success_steps:
            break
        if physics_failure is not None:
            break

    trajectory_length = len(records["action"])
    success = success_streak >= required_success_steps
    if not success:
        # A transient success near the horizon is not a stable accepted outcome.
        success_step = 0
    clean_success = success and not wrong_object_contact and physics_failure is None
    phase_values, phase_frequencies = np.unique(
        np.asarray(records["expert_phase"]),
        return_counts=True,
    )
    phase_counts = {
        PHASE_NAMES[int(phase)]: int(count)
        for phase, count in zip(phase_values, phase_frequencies)
    }
    final_position_error = float(
        np.linalg.norm(task.target_goal - np.asarray(obs["robot0_eef_pos"]))
    )
    final_orientation_error = float(
        np.linalg.norm(
            orientation_error(
                oracle.target_orientation,
                T.quat2mat(obs["robot0_eef_quat_site"]),
            )
        )
    )
    target_displacement = (
        env.get_object_position(task.target_id)[:2]
        - env.initial_target_position[:2]
    )
    push_forward_displacement = float(
        np.dot(target_displacement, task.push_direction)
    )
    push_lateral_displacement = float(
        np.linalg.norm(
            target_displacement
            - push_forward_displacement * task.push_direction
        )
    )
    arrays = None
    if capture_images:
        arrays = {
            key: _pad(values, EPISODE_STEPS)
            for key, values in records.items()
        }
        arrays["image_agentview"] = arrays["image_agentview"].astype(np.uint8)
        arrays["image_wrist"] = arrays["image_wrist"].astype(np.uint8)
        for key in ("state", "previous_action", "action", "object_pose"):
            arrays[key] = arrays[key].astype(np.float32)
        for key in (
            "object_contact",
            "object_grasped",
            "expert_phase",
            "retry_count",
            "success_after_action",
        ):
            arrays[key] = arrays[key].astype(np.uint8)

    return RolloutResult(
        success=success,
        clean_success=clean_success,
        success_step=success_step,
        trajectory_length=trajectory_length,
        wrong_object_contact=wrong_object_contact,
        saturation_steps=saturation_steps,
        retry_count=oracle.retry_count,
        phase_counts=phase_counts,
        final_position_error=final_position_error,
        final_orientation_error=final_orientation_error,
        push_forward_displacement=push_forward_displacement,
        push_lateral_displacement=push_lateral_displacement,
        forced_push_recovery=bool(
            task.task_type == "push"
            and task.target_id == "B"
            and np.any(np.asarray(records["expert_phase"]) == 9)
        ),
        physics_failure=physics_failure,
        arrays=arrays,
    )


def build_episode_arrays(
    task: TaskSpecV2,
    result: RolloutResult,
    initial_pose: np.ndarray,
    scene_seed: int,
    collection_seed: int,
    split: str,
) -> dict[str, np.ndarray]:
    if result.arrays is None:
        raise ValueError("Image capture is required before saving an episode")
    arrays = dict(result.arrays)
    arrays.update(
        {
            "valid_mask": (
                np.arange(EPISODE_STEPS) < result.trajectory_length
            ).astype(np.uint8),
            "initial_object_pose": initial_pose.astype(np.float32),
            "push_direction": task.push_direction.astype(np.float32),
            "target_goal": task.target_goal.astype(np.float32),
            "instruction": np.asarray(task.instruction),
            "target_id": np.asarray(task.target_id),
            "task_type": np.asarray(task.task_type),
            "split": np.asarray(split),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "dataset_version": np.asarray(DATASET_VERSION),
            "success_step": np.asarray(result.success_step, dtype=np.int64),
            "trajectory_length": np.asarray(
                result.trajectory_length,
                dtype=np.int64,
            ),
            "scene_seed": np.asarray(scene_seed, dtype=np.int64),
            "collection_seed": np.asarray(collection_seed, dtype=np.int64),
            "outcome": np.asarray(int(result.success), dtype=np.uint8),
            "wrong_object_contact": np.asarray(
                int(result.wrong_object_contact),
                dtype=np.uint8,
            ),
        }
    )
    validate_episode_arrays(arrays)
    return arrays


def save_episode(path: str, arrays: dict[str, np.ndarray]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp.npz"
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def verify_archive(path: str) -> None:
    with np.load(path, allow_pickle=False) as episode:
        arrays = {key: episode[key] for key in episode.files}
    validate_episode_arrays(arrays)


def write_manifest(data_dir: str, rows: list[dict]) -> None:
    fieldnames = [
        "filename",
        "dataset_version",
        "split",
        "task_type",
        "target_id",
        "instruction",
        "scene_seed",
        "collection_seed",
        "success_step",
        "wrong_object_contact",
    ]
    path = os.path.join(data_dir, "dataset_manifest.csv")
    temporary_path = f"{path}.tmp"
    with open(
        temporary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["filename"]))
        manifest.flush()
        os.fsync(manifest.fileno())
    os.replace(temporary_path, path)


def write_schema_document(data_dir: str) -> None:
    path = os.path.join(data_dir, "dataset_schema_v2_clean.json")
    temporary_path = f"{path}.tmp"
    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as schema_file:
        json.dump(schema_document(), schema_file, indent=2)
        schema_file.flush()
        os.fsync(schema_file.fileno())
    os.replace(temporary_path, path)


def split_schedule(quota: int, bucket: tuple[str, str], seed: int) -> list[str]:
    train_count = int(quota * TRAIN_RATIO)
    val_count = int(quota * VAL_RATIO)
    labels = (
        ["train"] * train_count
        + ["val"] * val_count
        + ["test"] * (quota - train_count - val_count)
    )
    rng = np.random.default_rng(seed + TASK_BUCKETS.index(bucket) * 10_007)
    rng.shuffle(labels)
    return labels


def existing_state(data_dir: str) -> tuple[dict, set[int], int, list[dict]]:
    counts = {bucket: 0 for bucket in TASK_BUCKETS}
    seeds: set[int] = set()
    max_id = 0
    rows: list[dict] = []
    if not os.path.isdir(data_dir):
        return counts, seeds, max_id, rows
    for filename in sorted(os.listdir(data_dir)):
        if not filename.startswith("ep_") or not filename.endswith(".npz"):
            continue
        path = os.path.join(data_dir, filename)
        verify_archive(path)
        metadata = read_metadata(path)
        bucket = (metadata["task_type"], metadata["target_id"])
        if bucket not in counts:
            raise ValueError(f"Unknown task bucket in {path}: {bucket}")
        if metadata["scene_seed"] in seeds:
            raise ValueError(f"Duplicate scene seed in {path}")
        counts[bucket] += 1
        seeds.add(metadata["scene_seed"])
        rows.append(metadata)
        max_id = max(max_id, int(filename[3:8]))
    return counts, seeds, max_id, rows


def make_environment(camera_observations: bool) -> MultiObjectVLAEnvV2:
    kwargs = {
        "env_name": "MultiObjectVLAEnvV2",
        "robots": "Panda",
        "controller_configs": build_osc_pose_controller_config_v2(),
        "has_renderer": False,
        "has_offscreen_renderer": camera_observations,
        "use_camera_obs": camera_observations,
        "use_object_obs": True,
        "horizon": EPISODE_STEPS,
        "ignore_done": True,
        "control_freq": CONTROL_FREQ,
        "hard_reset": False,
    }
    if camera_observations:
        kwargs.update(
            {
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "camera_heights": [IMAGE_HEIGHT, IMAGE_HEIGHT],
                "camera_widths": [IMAGE_WIDTH, IMAGE_WIDTH],
            }
        )
    env = robosuite.make(**kwargs)
    low, high = env.action_spec
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        env.close()
        raise RuntimeError(f"Expected 7D OSC_POSE actions, got {low.shape}")
    return env


def reseed_environment(env: MultiObjectVLAEnvV2, scene_seed: int) -> None:
    env.rng = np.random.default_rng(scene_seed)
    env.placement_initializer.rng = env.rng


def _summary_template() -> dict:
    return {
        f"{task}_{target}": {
            "accepted": 0,
            "attempts": 0,
            "scene_rejections": 0,
            "oracle_failures": 0,
            "wrong_contact_failures": 0,
            "quality_rejections": 0,
            "physics_failures": 0,
        }
        for task, target in TASK_BUCKETS
    }


def _load_summary(data_dir: str) -> dict:
    path = os.path.join(data_dir, "collection_summary_v2_clean.json")
    if not os.path.exists(path):
        return _summary_template()
    with open(path, "r", encoding="utf-8") as summary_file:
        payload = json.load(summary_file)
    template = _summary_template()
    for key in template:
        template[key].update(payload.get("buckets", {}).get(key, {}))
    return template


def _write_summary(
    data_dir: str,
    collection_seed: int,
    requested_total: int,
    buckets: dict,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "collection_seed": collection_seed,
        "requested_total": requested_total,
        "accepted_total": sum(item["accepted"] for item in buckets.values()),
        "buckets": buckets,
    }
    for item in payload["buckets"].values():
        attempts = item["attempts"]
        item["clean_success_rate"] = (
            item["accepted"] / attempts if attempts else None
        )
    path = os.path.join(data_dir, "collection_summary_v2_clean.json")
    temporary_path = f"{path}.tmp"
    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(payload, summary_file, indent=2)
        summary_file.flush()
        os.fsync(summary_file.fileno())
    os.replace(temporary_path, path)


def verify_dataset(
    data_dir: str = DATASET_DIR,
    expected_total: int | None = None,
) -> int:
    paths = sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.startswith("ep_") and filename.endswith(".npz")
    )
    if not paths:
        raise FileNotFoundError(f"No clean V2 episodes found in {data_dir}")
    counts = {bucket: 0 for bucket in TASK_BUCKETS}
    split_counts = {
        bucket: {"train": 0, "val": 0, "test": 0}
        for bucket in TASK_BUCKETS
    }
    rows = []
    for path in paths:
        verify_archive(path)
        metadata = read_metadata(path)
        bucket = (metadata["task_type"], metadata["target_id"])
        counts[bucket] += 1
        split_counts[bucket][metadata["split"]] += 1
        if metadata["wrong_object_contact"]:
            raise ValueError(f"Accepted trajectory contains wrong contact: {path}")
        rows.append(metadata)
    if expected_total is not None and len(paths) != expected_total:
        raise ValueError(f"Expected {expected_total} episodes, found {len(paths)}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"Clean V2 dataset is not task-balanced: {counts}")
    if expected_total is not None:
        quota = expected_total // len(TASK_BUCKETS)
        expected_splits = {
            "train": int(quota * TRAIN_RATIO),
            "val": int(quota * VAL_RATIO),
        }
        expected_splits["test"] = quota - sum(expected_splits.values())
        for bucket in TASK_BUCKETS:
            if split_counts[bucket] != expected_splits:
                raise ValueError(
                    f"Split imbalance for {bucket}: {split_counts[bucket]} "
                    f"!= {expected_splits}"
                )
    write_manifest(data_dir, rows)
    write_schema_document(data_dir)
    print(f"Verified {len(paths)} clean V2 archives.", flush=True)
    for bucket in TASK_BUCKETS:
        print(
            f"  {bucket[0]:4s}-{bucket[1]}: {counts[bucket]} | "
            f"train={split_counts[bucket]['train']} "
            f"val={split_counts[bucket]['val']} "
            f"test={split_counts[bucket]['test']}",
            flush=True,
        )
    return len(paths)


def collect(
    num_episodes: int,
    data_dir: str,
    collection_seed: int,
    save_failures: bool,
) -> None:
    if num_episodes % len(TASK_BUCKETS) != 0:
        raise ValueError("num_episodes must be divisible by six")
    os.makedirs(data_dir, exist_ok=True)
    write_schema_document(data_dir)
    quota = num_episodes // len(TASK_BUCKETS)
    counts, used_seeds, episode_id, rows = existing_state(data_dir)
    if any(count > quota for count in counts.values()):
        raise ValueError(f"Existing data exceeds the requested quota: {counts}")
    schedules = {
        bucket: split_schedule(quota, bucket, collection_seed)
        for bucket in TASK_BUCKETS
    }
    summary = _load_summary(data_dir)
    for bucket, count in counts.items():
        summary[f"{bucket[0]}_{bucket[1]}"]["accepted"] = count
    write_manifest(data_dir, rows)
    if sum(counts.values()) == num_episodes:
        verify_dataset(data_dir, expected_total=num_episodes)
        return

    rng = np.random.default_rng(collection_seed)
    env = make_environment(camera_observations=True)
    collected = sum(counts.values())
    attempt = sum(item["attempts"] for item in summary.values())
    try:
        while collected < num_episodes:
            underfilled = [bucket for bucket in TASK_BUCKETS if counts[bucket] < quota]
            bucket = underfilled[int(rng.integers(len(underfilled)))]
            key = f"{bucket[0]}_{bucket[1]}"
            summary[key]["attempts"] += 1
            attempt += 1
            while True:
                scene_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                if scene_seed not in used_seeds:
                    break
            reseed_environment(env, scene_seed)
            obs = env.reset()
            initial_pose = object_poses(env)
            try:
                task = schedule_task_v2(env, obs, bucket[0], bucket[1])
            except RuntimeError as error:
                summary[key]["scene_rejections"] += 1
                _write_summary(data_dir, collection_seed, num_episodes, summary)
                print(
                    f"[Scene rejected] Attempt {attempt} | {key} | {error}",
                    flush=True,
                )
                continue

            result = rollout_oracle(
                env,
                task,
                obs,
                seed=scene_seed,
                capture_images=True,
            )
            split = schedules[bucket][counts[bucket]]
            if not result.clean_success:
                if result.physics_failure is not None:
                    summary[key]["physics_failures"] += 1
                if result.wrong_object_contact:
                    summary[key]["wrong_contact_failures"] += 1
                else:
                    summary[key]["oracle_failures"] += 1
                if save_failures:
                    try:
                        diagnostic_arrays = build_episode_arrays(
                            task,
                            result,
                            initial_pose,
                            scene_seed,
                            collection_seed,
                            "diagnostic",
                        )
                    except ValueError as error:
                        print(
                            f"[Diagnostic skipped] Attempt {attempt} | {error}",
                            flush=True,
                        )
                    else:
                        failure_path = os.path.join(
                            data_dir,
                            "failed",
                            f"attempt_{attempt:06d}.npz",
                        )
                        save_episode(failure_path, diagnostic_arrays)
                        verify_archive(failure_path)
                _write_summary(data_dir, collection_seed, num_episodes, summary)
                print(
                    f"[Oracle failed] Attempt {attempt} | {task.instruction} | "
                    f"wrong_contact={result.wrong_object_contact} | "
                    f"retries={result.retry_count} | "
                    f"physics={result.physics_failure or '-'}",
                    flush=True,
                )
                continue

            try:
                arrays = build_episode_arrays(
                    task,
                    result,
                    initial_pose,
                    scene_seed,
                    collection_seed,
                    split,
                )
            except ValueError as error:
                summary[key]["quality_rejections"] += 1
                _write_summary(
                    data_dir,
                    collection_seed,
                    num_episodes,
                    summary,
                )
                print(
                    f"[Quality rejected] Attempt {attempt} | "
                    f"{task.instruction} | {error}",
                    flush=True,
                )
                continue
            episode_id += 1
            output_path = os.path.join(data_dir, f"ep_{episode_id:05d}.npz")
            save_episode(output_path, arrays)
            verify_archive(output_path)
            metadata = read_metadata(output_path)
            rows.append(metadata)
            counts[bucket] += 1
            collected += 1
            used_seeds.add(scene_seed)
            summary[key]["accepted"] = counts[bucket]
            write_manifest(data_dir, rows)
            _write_summary(data_dir, collection_seed, num_episodes, summary)
            saturation_rate = result.saturation_steps / result.trajectory_length
            print(
                f"[Success] Attempt {attempt} | Episode {collected}/{num_episodes} | "
                f"{key} {counts[bucket]}/{quota} | split={split} | "
                f"steps={result.success_step} | retries={result.retry_count} | "
                f"saturation={saturation_rate:.2%}",
                flush=True,
            )
    finally:
        env.close()

    verify_dataset(data_dir, expected_total=num_episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--data-dir", default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=COLLECTION_SEED)
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_dataset(args.data_dir, expected_total=args.num_episodes)
        return
    collect(
        num_episodes=args.num_episodes,
        data_dir=args.data_dir,
        collection_seed=args.seed,
        save_failures=args.save_failures,
    )


if __name__ == "__main__":
    main()
