"""Composable perturbations for MiniVLA V2 robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


GROUNDING_REACQUISITION_THRESHOLD_CM = 2.0


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
    grounding_threshold_cm: float = GROUNDING_REACQUISITION_THRESHOLD_CM
    name: str = field(init=False, default="target_displacement")

    _rng: np.random.Generator | None = field(init=False, default=None)
    _fired: bool = field(init=False, default=False)
    _injection_step: int = field(init=False, default=0)
    _injection_phase: int = field(init=False, default=-1)
    _actual_delta: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    _grounding_before_cm: float = field(init=False, default=float("nan"))
    _grounding_after_cm: float = field(init=False, default=float("nan"))
    _max_grounding_after_cm: float = field(init=False, default=float("nan"))
    _reacquisition_latency: int = field(init=False, default=-1)
    _contact_latency: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        if self.distance_m < 0.0:
            raise ValueError("distance_m must be non-negative")

    def on_episode_start(self, context: PerturbationContext) -> None:
        seed_sequence = np.random.SeedSequence(
            [int(self.base_seed), int(context.scene_seed)]
        )
        self._rng = np.random.default_rng(seed_sequence)
        self._fired = False
        self._injection_step = 0
        self._injection_phase = -1
        self._actual_delta = np.zeros(3, dtype=np.float32)
        self._grounding_before_cm = float("nan")
        self._grounding_after_cm = float("nan")
        self._max_grounding_after_cm = float("nan")
        self._reacquisition_latency = -1
        self._contact_latency = -1

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

    def _collision_safe_target_xy(
        self,
        context: PerturbationContext,
        target: np.ndarray,
    ) -> np.ndarray:
        env = context.env
        target_id = context.task.target_id
        target_radius = float(env.objects_by_id[target_id].horizontal_radius)
        table_xy = np.asarray(env.table_offset[:2], dtype=np.float64)
        half_table = np.asarray(env.table_full_size[:2], dtype=np.float64) / 2.0

        for direction in self._candidate_directions(context):
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            candidate = target[:2] + direction * self.distance_m
            if np.any(
                np.abs(candidate - table_xy)
                > half_table - target_radius - 0.015
            ):
                continue
            collision_free = True
            for other_id in ("A", "B", "C"):
                if other_id == target_id:
                    continue
                other_position = env.get_object_position(other_id)[:2]
                clearance = (
                    target_radius
                    + float(env.objects_by_id[other_id].horizontal_radius)
                    + 0.015
                )
                if np.linalg.norm(candidate - other_position) < clearance:
                    collision_free = False
                    break
            if collision_free:
                return candidate
        raise RuntimeError(
            f"No collision-safe {self.distance_m:.3f}m target displacement"
        )

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
        selected_xy = self._collision_safe_target_xy(context, target)
        joint_name = env.objects_by_id[target_id].joints[0]
        qpos = env.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:2] = selected_xy
        env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
        env.sim.forward()

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
        if self._reacquisition_latency < 0 and error <= self.grounding_threshold_cm:
            self._reacquisition_latency = context.step - self._injection_step

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

        return {
            "perturbation_type": self.name,
            "perturbation_level": float(self.distance_m),
            "injected": int(self._fired),
            "injection_step": self._injection_step,
            "injection_phase": self._injection_phase,
            "requested_delta_m": float(self.distance_m),
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
            "reacquisition_latency": self._reacquisition_latency,
            "post_injection_contact_latency": self._contact_latency,
        }


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
