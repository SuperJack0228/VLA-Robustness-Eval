"""OOD distractor scene and diagnostics for paired MiniVLA V3 evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from robosuite.utils.mjcf_utils import new_body, new_geom

from scripts.collect_data import OBJECT_SPECS, distance_to_segment
from scripts.collect_data_v2 import (
    BALL_PUSH_DISTANCE,
    PICK_HOVER_HEIGHT,
    PUSH_DISTANCE,
    PUSH_HOVER_HEIGHT,
    MultiObjectVLAEnvV2,
)
from utils.perturbations_v2 import (
    Perturbation,
    PerturbationContext,
    PerturbationSceneRejected,
)


OOD_DISTRACTOR_IDS = ("D1", "D2", "D3")
HIDDEN_POSITION = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
ROBOT_PATH_CLEARANCE_M = 0.045
OBJECT_CLEARANCE_M = 0.035
DISTRACTOR_CLEARANCE_M = 0.025
VISIBILITY_CLEARANCE_M = 0.012
SELECTION_MARGIN_M = 0.005
SELECTION_CONFIRM_STEPS = 3
VISIBLE_WORKSPACE_HALF_EXTENT_M = 0.24
MIN_VISIBLE_CHANGED_PIXELS = 8


@dataclass(frozen=True)
class OODGeometrySpec:
    label: str
    bounding_radius: float
    center_height: float
    rgba: tuple[float, float, float, float]


OOD_GEOMETRY_SPECS = {
    "D1": OODGeometrySpec(
        label="magenta horizontal capsule",
        bounding_radius=0.040,
        center_height=0.015,
        rgba=(0.95, 0.05, 0.75, 1.0),
    ),
    "D2": OODGeometrySpec(
        label="yellow ellipsoid",
        bounding_radius=0.030,
        center_height=0.020,
        rgba=(1.0, 0.82, 0.02, 1.0),
    ),
    "D3": OODGeometrySpec(
        label="cyan crossed capsules",
        bounding_radius=0.040,
        center_height=0.010,
        rgba=(0.0, 0.9, 0.9, 1.0),
    ),
}


class MultiObjectVLAOODEnvV3(MultiObjectVLAEnvV2):
    """V2 task environment with three precompiled, optional OOD obstacles."""

    def __init__(self, *args, **kwargs) -> None:
        self.ood_body_ids: dict[str, int] = {}
        self.ood_geom_ids: dict[str, tuple[int, ...]] = {}
        self.ood_geom_names: dict[str, tuple[str, ...]] = {}
        self.ood_visual_geom_ids: dict[str, tuple[int, ...]] = {}
        self.ood_visual_geom_names: dict[str, tuple[str, ...]] = {}
        super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        super()._load_model()
        for distractor_id in OOD_DISTRACTOR_IDS:
            body = new_body(
                name=f"ood_{distractor_id}_body",
                pos=HIDDEN_POSITION,
            )
            spec = OOD_GEOMETRY_SPECS[distractor_id]
            geom_names: list[str] = []
            visual_geom_names: list[str] = []
            if distractor_id == "D1":
                geom_names.append("ood_D1_capsule")
                body.append(
                    new_geom(
                        name=geom_names[-1],
                        type="capsule",
                        size=[0.013, 0.025],
                        quat=[0.70710678, 0.0, 0.70710678, 0.0],
                        rgba=spec.rgba,
                        friction=[0.8, 0.005, 0.0001],
                        contype=1,
                        conaffinity=1,
                    )
                )
                visual_geom_names.append("ood_D1_capsule_vis")
                body.append(
                    new_geom(
                        name=visual_geom_names[-1],
                        type="capsule",
                        size=[0.013, 0.025],
                        quat=[0.70710678, 0.0, 0.70710678, 0.0],
                        rgba=spec.rgba,
                        group=1,
                        contype=0,
                        conaffinity=0,
                    )
                )
            elif distractor_id == "D2":
                geom_names.append("ood_D2_ellipsoid")
                body.append(
                    new_geom(
                        name=geom_names[-1],
                        type="ellipsoid",
                        size=[0.030, 0.016, 0.018],
                        rgba=spec.rgba,
                        friction=[0.65, 0.004, 0.0001],
                        contype=1,
                        conaffinity=1,
                    )
                )
                visual_geom_names.append("ood_D2_ellipsoid_vis")
                body.append(
                    new_geom(
                        name=visual_geom_names[-1],
                        type="ellipsoid",
                        size=[0.030, 0.016, 0.018],
                        rgba=spec.rgba,
                        group=1,
                        contype=0,
                        conaffinity=0,
                    )
                )
            else:
                for suffix, quaternion in (
                    ("x", [0.70710678, 0.0, 0.70710678, 0.0]),
                    ("y", [0.70710678, -0.70710678, 0.0, 0.0]),
                ):
                    geom_names.append(f"ood_D3_cross_{suffix}")
                    body.append(
                        new_geom(
                            name=geom_names[-1],
                            type="capsule",
                            size=[0.008, 0.028],
                            quat=quaternion,
                            rgba=spec.rgba,
                            friction=[0.7, 0.004, 0.0001],
                            contype=1,
                            conaffinity=1,
                        )
                    )
                    visual_geom_names.append(f"ood_D3_cross_{suffix}_vis")
                    body.append(
                        new_geom(
                            name=visual_geom_names[-1],
                            type="capsule",
                            size=[0.008, 0.028],
                            quat=quaternion,
                            rgba=spec.rgba,
                            group=1,
                            contype=0,
                            conaffinity=0,
                        )
                    )
            self.ood_geom_names[distractor_id] = tuple(geom_names)
            self.ood_visual_geom_names[distractor_id] = tuple(visual_geom_names)
            self.model.worldbody.append(body)

    def _setup_references(self) -> None:
        super()._setup_references()
        self.ood_body_ids = {
            distractor_id: self.sim.model.body_name2id(
                f"ood_{distractor_id}_body"
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        }
        self.ood_geom_ids = {
            distractor_id: tuple(
                self.sim.model.geom_name2id(name)
                for name in self.ood_geom_names[distractor_id]
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        }
        self.ood_visual_geom_ids = {
            distractor_id: tuple(
                self.sim.model.geom_name2id(name)
                for name in self.ood_visual_geom_names[distractor_id]
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        }


@dataclass
class OODDistractorInjection(Perturbation):
    """Place a paired prefix of safe, unseen collision-enabled distractors."""

    count: int
    base_seed: int
    validation_count: int = 3
    name: str = field(init=False, default="ood_distractor_injection")

    _env: Any = field(init=False, default=None)
    _layout: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _active_ids: tuple[str, ...] = field(init=False, default=())
    _robot_geom_ids: set[int] = field(init=False, default_factory=set)
    _object_geom_ids: dict[str, set[int]] = field(
        init=False,
        default_factory=dict,
    )
    _robot_contact_ids: set[str] = field(init=False, default_factory=set)
    _target_contact_ids: set[str] = field(init=False, default_factory=set)
    _other_object_contact_ids: set[str] = field(
        init=False,
        default_factory=set,
    )
    _collision_step_count: int = field(init=False, default=0)
    _collision_event_count: int = field(init=False, default=0)
    _first_collision_step: int = field(init=False, default=-1)
    _previous_collision_ids: set[str] = field(init=False, default_factory=set)
    _selection_failure_steps: int = field(init=False, default=0)
    _selection_prediction_steps: int = field(init=False, default=0)
    _selection_streak: int = field(init=False, default=0)
    _max_selection_streak: int = field(init=False, default=0)
    _first_selection_failure_step: int = field(init=False, default=-1)
    _selected_distractor_counts: dict[str, int] = field(
        init=False,
        default_factory=dict,
    )
    _application_verified: bool = field(init=False, default=False)
    _restoration_verified: bool = field(init=False, default=False)
    _visibility_clear: bool = field(init=False, default=False)
    _path_clear: bool = field(init=False, default=False)
    _collision_integrity: bool = field(init=False, default=False)
    _visible_cameras: dict[str, tuple[str, ...]] = field(
        init=False,
        default_factory=dict,
    )
    _initial_frame_mae: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.count not in range(4):
            raise ValueError("OOD distractor count must be one of 0, 1, 2, 3")
        if self.validation_count != 3:
            raise ValueError("validation_count must remain 3 for paired layouts")

    @staticmethod
    def _geom_ids(model, names: list[str] | tuple[str, ...]) -> set[int]:
        ids: set[int] = set()
        for name in names:
            try:
                geom_id = int(model.geom_name2id(name))
            except Exception:
                continue
            if geom_id >= 0:
                ids.add(geom_id)
        return ids

    def _cache_contact_ids(self, env) -> None:
        model = env.sim.model
        robot_names = list(env.robots[0].robot_model.contact_geoms)
        gripper = env.robots[0].gripper
        grippers = tuple(gripper.values()) if isinstance(gripper, dict) else (gripper,)
        for gripper_model in grippers:
            robot_names.extend(gripper_model.contact_geoms)
        self._robot_geom_ids = self._geom_ids(model, robot_names)
        self._object_geom_ids = {
            object_id: self._geom_ids(
                model,
                env.objects_by_id[object_id].contact_geoms,
            )
            for object_id in OBJECT_SPECS
        }

    @staticmethod
    def _camera_position(env, camera_name: str) -> np.ndarray:
        camera_id = int(env.sim.model.camera_name2id(camera_name))
        return np.asarray(env.sim.data.cam_xpos[camera_id], dtype=np.float64)

    def _path_segments(self, context: PerturbationContext) -> list[tuple[np.ndarray, np.ndarray]]:
        env = context.env
        task = context.task
        table_height = float(env.table_offset[2])
        target = env.get_object_position(task.target_id).astype(np.float64)
        eef = np.asarray(context.obs["robot0_eef_pos"], dtype=np.float64)
        target_radius = float(env.objects_by_id[task.target_id].horizontal_radius)
        if task.task_type == "pick":
            hover = target.copy()
            hover[2] = table_height + PICK_HOVER_HEIGHT
            contact = target.copy()
            contact[2] = max(target[2], table_height + 0.02)
            goal = np.asarray(task.target_goal, dtype=np.float64).copy()
            return [(eef, hover), (hover, contact), (contact, goal)]

        direction = np.asarray(task.push_direction, dtype=np.float64)
        push_distance = BALL_PUSH_DISTANCE if task.target_id == "B" else PUSH_DISTANCE
        start_xy = target[:2] - direction * (target_radius + 0.065)
        end_xy = target[:2] + direction * push_distance
        hover = np.asarray(
            [start_xy[0], start_xy[1], table_height + PUSH_HOVER_HEIGHT],
            dtype=np.float64,
        )
        contact_height = table_height + (0.023 if task.target_id == "B" else 0.018)
        contact = np.asarray([start_xy[0], start_xy[1], contact_height])
        end = np.asarray([end_xy[0], end_xy[1], contact_height])
        return [(eef, hover), (hover, contact), (contact, end)]

    def _candidate_is_safe(
        self,
        context: PerturbationContext,
        distractor_id: str,
        position: np.ndarray,
        accepted: dict[str, np.ndarray],
    ) -> tuple[bool, bool, bool]:
        env = context.env
        spec = OOD_GEOMETRY_SPECS[distractor_id]
        half_size = np.minimum(
            np.asarray(env.table_full_size[:2], dtype=np.float64) / 2.0,
            VISIBLE_WORKSPACE_HALF_EXTENT_M,
        )
        if np.any(
            np.abs(position[:2])
            > half_size - spec.bounding_radius - 0.025
        ):
            return False, False, False

        for object_id in OBJECT_SPECS:
            object_position = env.get_object_position(object_id)
            clearance = (
                spec.bounding_radius
                + float(env.objects_by_id[object_id].horizontal_radius)
                + OBJECT_CLEARANCE_M
            )
            if np.linalg.norm(position[:2] - object_position[:2]) < clearance:
                return False, False, False
        for other_id, other_position in accepted.items():
            clearance = (
                spec.bounding_radius
                + OOD_GEOMETRY_SPECS[other_id].bounding_radius
                + DISTRACTOR_CLEARANCE_M
            )
            if np.linalg.norm(position[:2] - other_position[:2]) < clearance:
                return False, False, False

        path_clear = all(
            distance_to_segment(position, start, end)
            > spec.bounding_radius + ROBOT_PATH_CLEARANCE_M
            for start, end in self._path_segments(context)
        )
        target = env.get_object_position(context.task.target_id).astype(np.float64)
        visibility_clear = True
        for camera_name in ("agentview", "robot0_eye_in_hand"):
            camera = self._camera_position(env, camera_name)
            if distance_to_segment(position, camera, target) <= (
                spec.bounding_radius + VISIBILITY_CLEARANCE_M
            ):
                visibility_clear = False
                break
        return path_clear and visibility_clear, visibility_clear, path_clear

    def _generate_layout(self, context: PerturbationContext) -> dict[str, np.ndarray]:
        env = context.env
        sequence = np.random.SeedSequence(
            [int(self.base_seed), int(context.scene_seed), 0x00D15A7]
        )
        rng = np.random.default_rng(sequence)
        table_height = float(env.table_offset[2])
        half_size = np.minimum(
            np.asarray(env.table_full_size[:2], dtype=np.float64) / 2.0,
            VISIBLE_WORKSPACE_HALF_EXTENT_M,
        )
        accepted: dict[str, np.ndarray] = {}
        for distractor_id in OOD_DISTRACTOR_IDS[: self.validation_count]:
            spec = OOD_GEOMETRY_SPECS[distractor_id]
            found = False
            for _ in range(2500):
                xy = rng.uniform(
                    -half_size + spec.bounding_radius + 0.025,
                    half_size - spec.bounding_radius - 0.025,
                )
                candidate = np.asarray(
                    [xy[0], xy[1], table_height + spec.center_height],
                    dtype=np.float64,
                )
                safe, _, _ = self._candidate_is_safe(
                    context,
                    distractor_id,
                    candidate,
                    accepted,
                )
                if safe:
                    accepted[distractor_id] = candidate
                    found = True
                    break
            if not found:
                raise PerturbationSceneRejected(
                    "No collision-safe, non-occluding OOD distractor layout"
                )
        self._visibility_clear = True
        self._path_clear = True
        return accepted

    def _set_enabled(self, distractor_id: str, enabled: bool) -> None:
        env = self._env
        model = env.sim.model
        body_id = env.ood_body_ids[distractor_id]
        geom_ids = env.ood_geom_ids[distractor_id]
        model.body_pos[body_id] = (
            self._layout[distractor_id] if enabled else HIDDEN_POSITION
        )
        for geom_id in geom_ids:
            model.geom_contype[geom_id] = 1 if enabled else 0
            model.geom_conaffinity[geom_id] = 1 if enabled else 0
            model.geom_rgba[geom_id, 3] = 1.0 if enabled else 0.0
        for geom_id in env.ood_visual_geom_ids[distractor_id]:
            model.geom_rgba[geom_id, 3] = 1.0 if enabled else 0.0

    def _hide_all(self) -> None:
        if self._env is None:
            return
        for distractor_id in OOD_DISTRACTOR_IDS:
            body_id = self._env.ood_body_ids[distractor_id]
            self._env.sim.model.body_pos[body_id] = HIDDEN_POSITION
            for geom_id in self._env.ood_geom_ids[distractor_id]:
                self._env.sim.model.geom_contype[geom_id] = 0
                self._env.sim.model.geom_conaffinity[geom_id] = 0
                self._env.sim.model.geom_rgba[geom_id, 3] = 0.0
            for geom_id in self._env.ood_visual_geom_ids[distractor_id]:
                self._env.sim.model.geom_rgba[geom_id, 3] = 0.0
        self._env.sim.forward()

    def _validate_render_visibility(
        self,
        context: PerturbationContext,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, np.ndarray]]:
        camera_names = ("agentview", "robot0_eye_in_hand")
        self._hide_all()
        baseline_obs = self._env._get_observations(force_update=True)
        baseline = {
            camera_name: np.asarray(
                baseline_obs[f"{camera_name}_image"],
                dtype=np.uint8,
            ).copy()
            for camera_name in camera_names
        }
        visible: dict[str, tuple[str, ...]] = {}
        for distractor_id in OOD_DISTRACTOR_IDS:
            self._set_enabled(distractor_id, True)
            self._env.sim.forward()
            candidate_obs = self._env._get_observations(force_update=True)
            cameras = []
            for camera_name in camera_names:
                candidate = np.asarray(
                    candidate_obs[f"{camera_name}_image"],
                    dtype=np.uint8,
                )
                changed_pixels = int(
                    np.count_nonzero(
                        np.any(candidate != baseline[camera_name], axis=-1)
                    )
                )
                if changed_pixels >= MIN_VISIBLE_CHANGED_PIXELS:
                    cameras.append(camera_name)
            visible[distractor_id] = tuple(cameras)
            self._set_enabled(distractor_id, False)
            self._env.sim.forward()
        self._hide_all()
        if any(not cameras for cameras in visible.values()):
            missing = [
                distractor_id
                for distractor_id, cameras in visible.items()
                if not cameras
            ]
            raise PerturbationSceneRejected(
                "OOD distractors are outside both policy views: "
                + ",".join(missing)
            )
        return visible, baseline

    def _contact_sets(self, target_id: str) -> tuple[set[str], set[str], set[str]]:
        robot_contacts: set[str] = set()
        target_contacts: set[str] = set()
        other_contacts: set[str] = set()
        target_geoms = self._object_geom_ids[target_id]
        other_geoms = set().union(
            *(
                geoms
                for object_id, geoms in self._object_geom_ids.items()
                if object_id != target_id
            )
        )
        for index in range(int(self._env.sim.data.ncon)):
            contact = self._env.sim.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for distractor_id in self._active_ids:
                distractor_geoms = set(self._env.ood_geom_ids[distractor_id])
                if not pair.intersection(distractor_geoms):
                    continue
                counterpart = pair - distractor_geoms
                if counterpart.intersection(self._robot_geom_ids):
                    robot_contacts.add(distractor_id)
                if counterpart.intersection(target_geoms):
                    target_contacts.add(distractor_id)
                if counterpart.intersection(other_geoms):
                    other_contacts.add(distractor_id)
        return robot_contacts, target_contacts, other_contacts

    def on_episode_start(self, context: PerturbationContext) -> None:
        env = context.env
        if not isinstance(env, MultiObjectVLAOODEnvV3):
            raise TypeError("OODDistractorInjection requires MultiObjectVLAOODEnvV3")
        self._env = env
        self._hide_all()
        self._cache_contact_ids(env)
        self._layout = self._generate_layout(context)
        try:
            self._visible_cameras, baseline_frames = self._validate_render_visibility(
                context
            )
        except PerturbationSceneRejected:
            self._hide_all()
            raise
        self._active_ids = OOD_DISTRACTOR_IDS[: self.count]
        self._robot_contact_ids = set()
        self._target_contact_ids = set()
        self._other_object_contact_ids = set()
        self._collision_step_count = 0
        self._collision_event_count = 0
        self._first_collision_step = -1
        self._previous_collision_ids = set()
        self._selection_failure_steps = 0
        self._selection_prediction_steps = 0
        self._selection_streak = 0
        self._max_selection_streak = 0
        self._first_selection_failure_step = -1
        self._selected_distractor_counts = {
            distractor_id: 0 for distractor_id in self._active_ids
        }
        self._restoration_verified = False
        self._initial_frame_mae = 0.0

        for distractor_id in OOD_DISTRACTOR_IDS:
            self._set_enabled(distractor_id, distractor_id in self._active_ids)
        env.sim.forward()
        robot, target, other = self._contact_sets(context.task.target_id)
        self._collision_integrity = not (robot or target or other)
        if not self._collision_integrity:
            self._hide_all()
            raise PerturbationSceneRejected(
                "OOD placement created forbidden initial physical contact"
            )
        self._application_verified = all(
            np.allclose(
                env.sim.model.body_pos[env.ood_body_ids[distractor_id]],
                self._layout[distractor_id]
                if distractor_id in self._active_ids
                else HIDDEN_POSITION,
                atol=1e-12,
                rtol=0.0,
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        ) and all(
            all(
                int(env.sim.model.geom_contype[geom_id])
                == int(distractor_id in self._active_ids)
                and int(env.sim.model.geom_conaffinity[geom_id])
                == int(distractor_id in self._active_ids)
                and np.isclose(
                    float(env.sim.model.geom_rgba[geom_id, 3]),
                    float(distractor_id in self._active_ids),
                )
                for geom_id in env.ood_geom_ids[distractor_id]
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        ) and all(
            all(
                int(env.sim.model.geom_group[geom_id]) == 1
                and int(env.sim.model.geom_contype[geom_id]) == 0
                and int(env.sim.model.geom_conaffinity[geom_id]) == 0
                and np.isclose(
                    float(env.sim.model.geom_rgba[geom_id, 3]),
                    float(distractor_id in self._active_ids),
                )
                for geom_id in env.ood_visual_geom_ids[distractor_id]
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        )
        if not self._application_verified:
            self._hide_all()
            raise RuntimeError("Failed to apply OOD distractor layout")

        refreshed = env._get_observations(force_update=True)
        pixel_differences = []
        for camera_name, baseline in baseline_frames.items():
            current = np.asarray(
                refreshed[f"{camera_name}_image"],
                dtype=np.float32,
            )
            pixel_differences.append(
                float(np.mean(np.abs(current - baseline.astype(np.float32))))
            )
        self._initial_frame_mae = float(np.mean(pixel_differences))
        if self.count > 0 and self._initial_frame_mae <= 0.0:
            self._hide_all()
            raise RuntimeError("Active OOD distractors did not alter policy images")
        context.obs.clear()
        context.obs.update(refreshed)

    def after_prediction(self, context: PerturbationContext) -> None:
        prediction = context.predicted_target_position
        if prediction is None or not self._active_ids:
            return
        prediction = np.asarray(prediction, dtype=np.float64)
        target = context.env.get_object_position(context.task.target_id)
        target_distance = float(np.linalg.norm(prediction - target))
        distances = {
            distractor_id: float(
                np.linalg.norm(prediction - self._layout[distractor_id])
            )
            for distractor_id in self._active_ids
        }
        closest_id = min(distances, key=distances.get)
        failed = distances[closest_id] + SELECTION_MARGIN_M < target_distance
        self._selection_prediction_steps += 1
        if failed:
            self._selection_failure_steps += 1
            self._selection_streak += 1
            self._selected_distractor_counts[closest_id] += 1
            if self._first_selection_failure_step < 0:
                self._first_selection_failure_step = context.step
        else:
            self._selection_streak = 0
        self._max_selection_streak = max(
            self._max_selection_streak,
            self._selection_streak,
        )

    def after_step(self, context: PerturbationContext) -> None:
        robot, target, other = self._contact_sets(context.task.target_id)
        collision_ids = robot | target | other
        if collision_ids:
            self._collision_step_count += 1
            if self._first_collision_step < 0:
                self._first_collision_step = context.step
        self._collision_event_count += len(collision_ids - self._previous_collision_ids)
        self._previous_collision_ids = collision_ids
        self._robot_contact_ids.update(robot)
        self._target_contact_ids.update(target)
        self._other_object_contact_ids.update(other)

    def on_episode_end(self, context: PerturbationContext) -> None:
        del context
        self._hide_all()
        self._restoration_verified = all(
            np.allclose(
                self._env.sim.model.body_pos[self._env.ood_body_ids[distractor_id]],
                HIDDEN_POSITION,
                atol=1e-12,
                rtol=0.0,
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        ) and all(
            all(
                int(self._env.sim.model.geom_contype[geom_id]) == 0
                and int(self._env.sim.model.geom_conaffinity[geom_id]) == 0
                and np.isclose(
                    float(self._env.sim.model.geom_rgba[geom_id, 3]),
                    0.0,
                )
                for geom_id in self._env.ood_geom_ids[distractor_id]
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        ) and all(
            all(
                np.isclose(
                    float(self._env.sim.model.geom_rgba[geom_id, 3]),
                    0.0,
                )
                for geom_id in self._env.ood_visual_geom_ids[distractor_id]
            )
            for distractor_id in OOD_DISTRACTOR_IDS
        )
        if not self._restoration_verified:
            raise RuntimeError("Failed to restore hidden OOD distractor state")

    @staticmethod
    def _position_string(position: np.ndarray) -> str:
        return "|".join(f"{float(value):.10g}" for value in position)

    def episode_metrics(self) -> dict:
        selected_id = ""
        if self._selected_distractor_counts:
            selected_id = max(
                self._selected_distractor_counts,
                key=self._selected_distractor_counts.get,
            )
            if self._selected_distractor_counts[selected_id] == 0:
                selected_id = ""
        metrics: dict[str, Any] = {
            "perturbation_type": self.name,
            "perturbation_protocol_version": "ood-distractor.v1",
            "ood_distractor_count": int(self.count),
            "ood_validation_count": int(self.validation_count),
            "ood_active_ids": "|".join(self._active_ids),
            "ood_geometry_labels_json": json.dumps(
                {
                    distractor_id: OOD_GEOMETRY_SPECS[distractor_id].label
                    for distractor_id in OOD_DISTRACTOR_IDS
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "ood_visible_cameras_json": json.dumps(
                {
                    distractor_id: list(self._visible_cameras[distractor_id])
                    for distractor_id in OOD_DISTRACTOR_IDS
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "ood_initial_policy_frame_mae": self._initial_frame_mae,
            "ood_layout_signature": ";".join(
                f"{distractor_id}:{self._position_string(self._layout[distractor_id])}"
                for distractor_id in OOD_DISTRACTOR_IDS
            ),
            "ood_initial_target_visibility_clear": int(self._visibility_clear),
            "ood_robot_path_clear": int(self._path_clear),
            "ood_initial_collision_integrity_passed": int(
                self._collision_integrity
            ),
            "ood_application_verified": int(self._application_verified),
            "ood_restoration_verified": int(self._restoration_verified),
            "ood_wrong_object_contact": int(bool(self._robot_contact_ids)),
            "ood_wrong_object_contact_ids": "|".join(
                sorted(self._robot_contact_ids)
            ),
            "ood_target_collision": int(bool(self._target_contact_ids)),
            "ood_target_collision_ids": "|".join(
                sorted(self._target_contact_ids)
            ),
            "ood_other_object_collision": int(
                bool(self._other_object_contact_ids)
            ),
            "ood_other_object_collision_ids": "|".join(
                sorted(self._other_object_contact_ids)
            ),
            "ood_collision": int(
                bool(
                    self._robot_contact_ids
                    or self._target_contact_ids
                    or self._other_object_contact_ids
                )
            ),
            "ood_collision_step_count": self._collision_step_count,
            "ood_collision_event_count": self._collision_event_count,
            "ood_first_collision_step": self._first_collision_step,
            "ood_target_selection_failure": int(
                self._max_selection_streak >= SELECTION_CONFIRM_STEPS
            ),
            "ood_target_selection_failure_steps": self._selection_failure_steps,
            "ood_target_selection_prediction_steps": self._selection_prediction_steps,
            "ood_target_selection_failure_rate": (
                self._selection_failure_steps / self._selection_prediction_steps
                if self._selection_prediction_steps
                else 0.0
            ),
            "ood_max_target_selection_failure_streak": self._max_selection_streak,
            "ood_first_target_selection_failure_step": (
                self._first_selection_failure_step
            ),
            "ood_most_selected_distractor_id": selected_id,
            "injected": int(self.count > 0),
        }
        for distractor_id in OOD_DISTRACTOR_IDS:
            position = self._layout.get(distractor_id, HIDDEN_POSITION)
            metrics[f"ood_{distractor_id}_x"] = float(position[0])
            metrics[f"ood_{distractor_id}_y"] = float(position[1])
            metrics[f"ood_{distractor_id}_z"] = float(position[2])
        return metrics
