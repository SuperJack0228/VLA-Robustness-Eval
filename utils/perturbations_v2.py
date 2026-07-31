"""Composable perturbations for MiniVLA V2 robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scripts.collect_data import distance_to_segment
from scripts.collect_data_v2 import (
    BALL_PUSH_DISTANCE,
    PICK_TARGET_CLEARANCE,
    PUSH_DISTANCE,
    PUSH_PATH_CLEARANCE,
)


GROUNDING_REACQUISITION_THRESHOLDS_CM = (0.5, 1.0, 2.0)
PLACEMENT_MARGIN_M = 0.015
PUSH_APPROACH_OFFSET_M = 0.065
PUSH_WORKSPACE_MARGIN_M = 0.015


class PerturbationSceneRejected(RuntimeError):
    """Raised when a scene cannot support a matched perturbation protocol."""


@dataclass
class PerturbationContext:
    """Mutable episode context shared with perturbation hooks."""

    episode_id: int
    scene_seed: int
    env: Any
    task: Any
    obs: dict
    step: int = 0
    predicted_phase: int = -1
    executed_phase: int = -1
    action: np.ndarray | None = None
    grounding_error_cm: float | None = None
    target_contact: bool = False
    current_target_position: np.ndarray | None = None


class Perturbation:
    """No-op base class defining all supported evaluation lifecycle hooks."""

    name = "clean"

    def on_episode_start(self, context: PerturbationContext) -> None:
        pass

    def transform_image(
        self,
        context: PerturbationContext,
        camera_name: str,
        image: np.ndarray,
    ) -> np.ndarray:
        return image

    def after_prediction(self, context: PerturbationContext) -> None:
        pass

    def before_step(self, context: PerturbationContext) -> dict | None:
        return None

    def after_step(self, context: PerturbationContext) -> None:
        pass

    def on_episode_end(self, context: PerturbationContext) -> None:
        pass

    @property
    def actual_target_delta(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def episode_metrics(self) -> dict:
        return {}


@dataclass
class DynamicTargetDisplacement(Perturbation):
    """Teleport the live target once during approach without steering policy."""

    distance_m: float
    base_seed: int
    validation_distances_m: tuple[float, ...] | None = None
    grounding_thresholds_cm: tuple[float, ...] = (
        GROUNDING_REACQUISITION_THRESHOLDS_CM
    )
    name: str = field(init=False, default="target_displacement")

    _rng: np.random.Generator | None = field(init=False, default=None)
    _fired: bool = field(init=False, default=False)
    _selected_direction: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(2, dtype=np.float64),
    )
    _injection_step: int = field(init=False, default=0)
    _injection_phase: int = field(init=False, default=-1)
    _actual_delta: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    _grounding_before_cm: float = field(init=False, default=float("nan"))
    _grounding_after_cm: float = field(init=False, default=float("nan"))
    _max_grounding_after_cm: float = field(init=False, default=float("nan"))
    _reacquisition_latencies: dict[float, int] = field(
        init=False,
        default_factory=dict,
    )
    _contact_latency: int = field(init=False, default=-1)
    _collision_integrity_passed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.distance_m < 0.0:
            raise ValueError("distance_m must be non-negative")
        distances = (
            (self.distance_m,)
            if self.validation_distances_m is None
            else tuple(float(value) for value in self.validation_distances_m)
        )
        if any(not np.isfinite(value) or value < 0.0 for value in distances):
            raise ValueError(
                "validation_distances_m must contain finite non-negative values"
            )
        self.validation_distances_m = tuple(sorted(set((*distances, self.distance_m))))
        thresholds = tuple(
            sorted(set(float(value) for value in self.grounding_thresholds_cm))
        )
        if not thresholds or any(
            not np.isfinite(value) or value <= 0.0 for value in thresholds
        ):
            raise ValueError(
                "grounding_thresholds_cm must contain positive finite values"
            )
        self.grounding_thresholds_cm = thresholds

    def on_episode_start(self, context: PerturbationContext) -> None:
        seed_sequence = np.random.SeedSequence(
            [int(self.base_seed), int(context.scene_seed)]
        )
        self._rng = np.random.default_rng(seed_sequence)
        self._fired = False
        self._selected_direction = np.zeros(2, dtype=np.float64)
        self._injection_step = 0
        self._injection_phase = -1
        self._actual_delta = np.zeros(3, dtype=np.float32)
        self._grounding_before_cm = float("nan")
        self._grounding_after_cm = float("nan")
        self._max_grounding_after_cm = float("nan")
        self._reacquisition_latencies = {
            threshold: -1 for threshold in self.grounding_thresholds_cm
        }
        self._contact_latency = -1
        self._collision_integrity_passed = False

        target = context.env.get_object_position(context.task.target_id).copy()
        for direction in self._candidate_directions(context):
            direction = np.asarray(direction, dtype=np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            if self._direction_supports_protocol(context, target, direction):
                self._selected_direction = direction
                self._collision_integrity_passed = True
                return
        levels = ", ".join(
            f"{distance:.3f}" for distance in self.validation_distances_m or ()
        )
        raise PerturbationSceneRejected(
            "No single collision-safe displacement ray supports all requested "
            f"levels [{levels}] for {context.task.task_type}_{context.task.target_id}"
        )

    def _candidate_directions(self, context: PerturbationContext) -> list[np.ndarray]:
        if self._rng is None:
            raise RuntimeError("Perturbation episode was not initialized")
        if context.task.task_type == "push":
            direction = np.asarray(context.task.push_direction, dtype=np.float64)
            perpendicular = np.asarray([-direction[1], direction[0]])
            if self._rng.random() < 0.5:
                perpendicular *= -1.0
            angles = self._rng.permutation(
                np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
            )
            fallback = [
                np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
                for angle in angles
            ]
            # Prefer lateral shifts, but retain exact-distance fallbacks for
            # scenes where both perpendicular directions leave the table.
            fallback.sort(key=lambda candidate: abs(float(np.dot(candidate, direction))))
            return [perpendicular, -perpendicular, *fallback]
        angles = self._rng.permutation(
            np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        )
        return [
            np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
            for angle in angles
        ]

    def _inside_table(
        self,
        context: PerturbationContext,
        xy: np.ndarray,
        margin: float,
    ) -> bool:
        env = context.env
        table_xy = np.asarray(env.table_offset[:2], dtype=np.float64)
        half_table = np.asarray(env.table_full_size[:2], dtype=np.float64) / 2.0
        return bool(np.all(np.abs(xy - table_xy) <= half_table - margin))

    @staticmethod
    def _gripper_models(env: Any) -> tuple[Any, ...]:
        gripper = env.robots[0].gripper
        if isinstance(gripper, dict):
            return tuple(gripper.values())
        return (gripper,)

    def _has_forbidden_physical_contact(
        self,
        context: PerturbationContext,
    ) -> bool:
        env = context.env
        target_id = context.task.target_id
        target_model = env.objects_by_id[target_id]
        if env.check_contact(target_model, env.robots[0].robot_model):
            return True
        if any(
            env.check_contact(target_model, gripper_model)
            for gripper_model in self._gripper_models(env)
        ):
            return True
        return any(
            env.check_contact(target_model, env.objects_by_id[other_id])
            for other_id in ("A", "B", "C")
            if other_id != target_id
        )

    def _temporary_contact_check(
        self,
        context: PerturbationContext,
        candidate_xy: np.ndarray,
    ) -> bool:
        env = context.env
        joint_name = env.objects_by_id[context.task.target_id].joints[0]
        original_qpos = env.sim.data.get_joint_qpos(joint_name).copy()
        original_qvel = env.sim.data.get_joint_qvel(joint_name).copy()
        candidate_qpos = original_qpos.copy()
        candidate_qpos[:2] = candidate_xy
        try:
            env.sim.data.set_joint_qpos(joint_name, candidate_qpos)
            env.sim.data.set_joint_qvel(
                joint_name,
                np.zeros_like(original_qvel),
            )
            env.sim.forward()
            return not self._has_forbidden_physical_contact(context)
        finally:
            env.sim.data.set_joint_qpos(joint_name, original_qpos)
            env.sim.data.set_joint_qvel(joint_name, original_qvel)
            env.sim.forward()

    def _placement_clear(
        self,
        context: PerturbationContext,
        candidate_xy: np.ndarray,
    ) -> bool:
        env = context.env
        target_id = context.task.target_id
        target_radius = float(env.objects_by_id[target_id].horizontal_radius)
        task_clearance = (
            PICK_TARGET_CLEARANCE
            if context.task.task_type == "pick"
            else PLACEMENT_MARGIN_M
        )
        if not self._inside_table(
            context,
            candidate_xy,
            target_radius + PLACEMENT_MARGIN_M,
        ):
            return False
        for other_id in ("A", "B", "C"):
            if other_id == target_id:
                continue
            other_position = env.get_object_position(other_id)[:2]
            clearance = (
                target_radius
                + float(env.objects_by_id[other_id].horizontal_radius)
                + task_clearance
            )
            if np.linalg.norm(candidate_xy - other_position) < clearance:
                return False
        return True

    def _push_corridor_clear(
        self,
        context: PerturbationContext,
        candidate_xy: np.ndarray,
    ) -> bool:
        env = context.env
        target_id = context.task.target_id
        direction = np.asarray(context.task.push_direction, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        target_radius = float(env.objects_by_id[target_id].horizontal_radius)
        start = candidate_xy - direction * (
            target_radius + PUSH_APPROACH_OFFSET_M
        )
        push_distance = BALL_PUSH_DISTANCE if target_id == "B" else PUSH_DISTANCE
        end = candidate_xy + direction * push_distance
        if not self._inside_table(
            context,
            start,
            target_radius + PUSH_WORKSPACE_MARGIN_M,
        ) or not self._inside_table(
            context,
            end,
            target_radius + PUSH_WORKSPACE_MARGIN_M,
        ):
            return False
        for other_id in ("A", "B", "C"):
            if other_id == target_id:
                continue
            other_xy = env.get_object_position(other_id)[:2]
            clearance = (
                target_radius
                + float(env.objects_by_id[other_id].horizontal_radius)
                + PUSH_PATH_CLEARANCE[target_id]
            )
            if distance_to_segment(other_xy, start, end) < clearance:
                return False
        return True

    def _direction_supports_protocol(
        self,
        context: PerturbationContext,
        target: np.ndarray,
        direction: np.ndarray,
    ) -> bool:
        for distance in self.validation_distances_m or (self.distance_m,):
            candidate_xy = target[:2] + direction * distance
            if not self._placement_clear(context, candidate_xy):
                return False
            if (
                context.task.task_type == "push"
                and not self._push_corridor_clear(context, candidate_xy)
            ):
                return False
            if distance > 0.0 and not self._temporary_contact_check(
                context,
                candidate_xy,
            ):
                return False
        return True

    def before_step(self, context: PerturbationContext) -> dict | None:
        trigger_phase = 1 if context.task.task_type == "pick" else 6
        if (
            self._fired
            or self.distance_m <= 0.0
            or context.executed_phase != trigger_phase
        ):
            return None

        env = context.env
        target_id = context.task.target_id
        target = env.get_object_position(target_id).copy()
        selected_xy = target[:2] + self._selected_direction * self.distance_m
        if not self._placement_clear(context, selected_xy):
            raise RuntimeError(
                "Target moved before injection and invalidated the preflighted "
                "displacement placement"
            )
        if (
            context.task.task_type == "push"
            and not self._push_corridor_clear(context, selected_xy)
        ):
            raise RuntimeError(
                "Target moved before injection and invalidated the preflighted "
                "push corridor"
            )
        joint_name = env.objects_by_id[target_id].joints[0]
        original_qpos = env.sim.data.get_joint_qpos(joint_name).copy()
        original_qvel = env.sim.data.get_joint_qvel(joint_name).copy()
        qpos = original_qpos.copy()
        qpos[:2] = selected_xy
        env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.data.set_joint_qvel(joint_name, np.zeros_like(original_qvel))
        env.sim.forward()
        if self._has_forbidden_physical_contact(context):
            env.sim.data.set_joint_qpos(joint_name, original_qpos)
            env.sim.data.set_joint_qvel(joint_name, original_qvel)
            env.sim.forward()
            raise RuntimeError(
                "Dynamic displacement created a forbidden physical contact"
            )

        self._actual_delta[:2] = selected_xy - target[:2]
        self._injection_step = context.step
        self._injection_phase = context.executed_phase
        self._grounding_before_cm = (
            float(context.grounding_error_cm)
            if context.grounding_error_cm is not None
            else float("nan")
        )
        self._fired = True
        return {
            "type": self.name,
            "step": self._injection_step,
            "phase": self._injection_phase,
            "actual_delta": self._actual_delta.copy(),
            "direction": self._selected_direction.copy(),
        }

    def after_prediction(self, context: PerturbationContext) -> None:
        if (
            not self._fired
            or context.step <= self._injection_step
            or context.grounding_error_cm is None
        ):
            return
        error = float(context.grounding_error_cm)
        if np.isnan(self._grounding_after_cm):
            self._grounding_after_cm = error
        if np.isnan(self._max_grounding_after_cm):
            self._max_grounding_after_cm = error
        else:
            self._max_grounding_after_cm = max(self._max_grounding_after_cm, error)
        for threshold in self.grounding_thresholds_cm:
            if (
                self._reacquisition_latencies[threshold] < 0
                and error <= threshold
            ):
                self._reacquisition_latencies[threshold] = (
                    context.step - self._injection_step
                )

    def after_step(self, context: PerturbationContext) -> None:
        if not self._fired or context.step < self._injection_step:
            return
        if self._contact_latency < 0 and context.target_contact:
            self._contact_latency = context.step - self._injection_step

    @property
    def actual_target_delta(self) -> np.ndarray:
        return self._actual_delta.copy()

    def episode_metrics(self) -> dict:
        def optional_number(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        metrics = {
            "perturbation_type": self.name,
            "perturbation_protocol_version": "target-displacement.v2",
            "perturbation_level": float(self.distance_m),
            "protocol_collision_integrity_passed": int(
                self._collision_integrity_passed
            ),
            "injected": int(self._fired),
            "injection_step": self._injection_step,
            "injection_phase": self._injection_phase,
            "requested_delta_m": float(self.distance_m),
            "selected_direction_x": float(self._selected_direction[0]),
            "selected_direction_y": float(self._selected_direction[1]),
            "actual_delta_x_m": float(self._actual_delta[0]),
            "actual_delta_y_m": float(self._actual_delta[1]),
            "actual_delta_z_m": float(self._actual_delta[2]),
            "actual_delta_norm_m": float(np.linalg.norm(self._actual_delta)),
            "grounding_error_before_injection_cm": optional_number(
                self._grounding_before_cm
            ),
            "grounding_error_after_injection_cm": optional_number(
                self._grounding_after_cm
            ),
            "max_grounding_error_after_injection_cm": optional_number(
                self._max_grounding_after_cm
            ),
            "post_injection_contact_latency": self._contact_latency,
        }
        for threshold, latency in self._reacquisition_latencies.items():
            slug = str(threshold).replace(".", "_")
            metrics[f"reacquired_within_{slug}cm"] = int(latency >= 0)
            metrics[f"reacquisition_latency_{slug}cm"] = latency
        # Keep the old 2 cm field for compatibility with existing analysis.
        metrics["reacquisition_latency"] = self._reacquisition_latencies.get(
            2.0,
            -1,
        )
        return metrics


class PerturbationManager:
    """Dispatch hooks and expose cumulative exogenous target displacement."""

    def __init__(self, perturbations: list[Perturbation] | None = None) -> None:
        self.perturbations = list(perturbations or [])
        self.context: PerturbationContext | None = None
        self.events: list[dict] = []

    def on_episode_start(
        self,
        episode_id: int,
        scene_seed: int,
        env: Any,
        task: Any,
        obs: dict,
    ) -> None:
        self.context = PerturbationContext(
            episode_id=episode_id,
            scene_seed=scene_seed,
            env=env,
            task=task,
            obs=obs,
        )
        self.events = []
        for perturbation in self.perturbations:
            perturbation.on_episode_start(self.context)

    def _require_context(self) -> PerturbationContext:
        if self.context is None:
            raise RuntimeError("Perturbation episode has not started")
        return self.context

    def transform_image(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        context = self._require_context()
        transformed = image
        for perturbation in self.perturbations:
            transformed = perturbation.transform_image(
                context,
                camera_name,
                transformed,
            )
        return np.ascontiguousarray(transformed)

    def after_prediction(
        self,
        step: int,
        predicted_phase: int,
        grounding_error_cm: float,
    ) -> None:
        context = self._require_context()
        context.step = step
        context.predicted_phase = predicted_phase
        context.grounding_error_cm = grounding_error_cm
        for perturbation in self.perturbations:
            perturbation.after_prediction(context)

    def before_step(
        self,
        step: int,
        predicted_phase: int,
        executed_phase: int,
        obs: dict,
        action: np.ndarray,
        grounding_error_cm: float,
    ) -> list[dict]:
        context = self._require_context()
        context.step = step
        context.predicted_phase = predicted_phase
        context.executed_phase = executed_phase
        context.obs = obs
        context.action = action
        context.grounding_error_cm = grounding_error_cm
        new_events = []
        for perturbation in self.perturbations:
            event = perturbation.before_step(context)
            if event is not None:
                self.events.append(event)
                new_events.append(event)
        return new_events

    def after_step(
        self,
        step: int,
        obs: dict,
        target_contact: bool,
        current_target_position: np.ndarray,
    ) -> None:
        context = self._require_context()
        context.step = step
        context.obs = obs
        context.target_contact = target_contact
        context.current_target_position = current_target_position
        for perturbation in self.perturbations:
            perturbation.after_step(context)

    def on_episode_end(self) -> None:
        context = self._require_context()
        for perturbation in reversed(self.perturbations):
            perturbation.on_episode_end(context)

    @property
    def actual_target_delta(self) -> np.ndarray:
        delta = np.zeros(3, dtype=np.float32)
        for perturbation in self.perturbations:
            delta += perturbation.actual_target_delta
        return delta

    def scoring_origin(self, initial_target_position: np.ndarray) -> np.ndarray:
        return np.asarray(initial_target_position) + self.actual_target_delta

    def episode_metrics(self) -> dict:
        metrics: dict[str, Any] = {
            "perturbation_event_count": len(self.events),
        }
        for perturbation in self.perturbations:
            for key, value in perturbation.episode_metrics().items():
                if key in metrics:
                    key = f"{perturbation.name}_{key}"
                metrics[key] = value
        return metrics
