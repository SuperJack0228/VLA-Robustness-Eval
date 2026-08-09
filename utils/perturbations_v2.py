"""Composable perturbations for MiniVLA V2 robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
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
    predicted_target_position: np.ndarray | None = None
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


VISUAL_DEGRADATIONS = (
    "clean",
    "gaussian-noise",
    "gaussian-blur",
    "brightness",
)
VISUAL_LEVEL_UNITS = {
    "clean": "identity",
    "gaussian-noise": "pixel_std_0_255",
    "gaussian-blur": "sigma_pixels",
    "brightness": "rgb_gain",
}


@dataclass
class VisualDegradation(Perturbation):
    """Apply a deterministic, paired corruption to both policy cameras."""

    degradation: str
    level: float
    base_seed: int
    camera_names: tuple[str, ...] = (
        "agentview",
        "robot0_eye_in_hand",
    )
    name: str = field(init=False, default="visual_degradation")

    _scene_seed: int = field(init=False, default=0)
    _frame_indices: dict[str, int] = field(init=False, default_factory=dict)
    _frames_seen: int = field(init=False, default=0)
    _changed_frames: int = field(init=False, default=0)
    _absolute_error_sum: float = field(init=False, default=0.0)
    _squared_error_sum: float = field(init=False, default=0.0)
    _pixel_value_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.degradation not in VISUAL_DEGRADATIONS:
            raise ValueError(
                f"Unsupported visual degradation: {self.degradation}"
            )
        if not np.isfinite(self.level):
            raise ValueError("Visual degradation level must be finite")
        if self.degradation in {"clean", "gaussian-noise", "gaussian-blur"}:
            if self.level < 0.0:
                raise ValueError(
                    f"{self.degradation} level must be non-negative"
                )
        elif self.level <= 0.0:
            raise ValueError("Brightness gain must be positive")
        if not self.camera_names or len(set(self.camera_names)) != len(
            self.camera_names
        ):
            raise ValueError("camera_names must be non-empty and unique")

    @property
    def active(self) -> bool:
        if self.degradation == "clean":
            return False
        if self.degradation in {"gaussian-noise", "gaussian-blur"}:
            return self.level > 0.0
        return not np.isclose(self.level, 1.0)

    def on_episode_start(self, context: PerturbationContext) -> None:
        self._scene_seed = int(context.scene_seed)
        self._frame_indices = {camera: 0 for camera in self.camera_names}
        self._frames_seen = 0
        self._changed_frames = 0
        self._absolute_error_sum = 0.0
        self._squared_error_sum = 0.0
        self._pixel_value_count = 0

    @staticmethod
    def _camera_code(camera_name: str) -> int:
        return sum(
            (index + 1) * byte
            for index, byte in enumerate(camera_name.encode("utf-8"))
        )

    def _noise_rng(
        self,
        camera_name: str,
        frame_index: int,
    ) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [
                int(self.base_seed),
                self._scene_seed,
                self._camera_code(camera_name),
                int(frame_index),
            ]
        )
        return np.random.default_rng(sequence)

    def transform_image(
        self,
        context: PerturbationContext,
        camera_name: str,
        image: np.ndarray,
    ) -> np.ndarray:
        del context
        if camera_name not in self.camera_names:
            return np.ascontiguousarray(image)

        frame_index = self._frame_indices[camera_name]
        self._frame_indices[camera_name] = frame_index + 1
        self._frames_seen += 1
        original = np.asarray(image, dtype=np.uint8)

        if not self.active:
            transformed = original.copy()
        elif self.degradation == "gaussian-noise":
            noise = self._noise_rng(camera_name, frame_index).normal(
                0.0,
                self.level,
                size=original.shape,
            )
            transformed = np.clip(
                original.astype(np.float32) + noise,
                0.0,
                255.0,
            ).astype(np.uint8)
        elif self.degradation == "gaussian-blur":
            transformed = cv2.GaussianBlur(
                original,
                (0, 0),
                sigmaX=self.level,
                sigmaY=self.level,
                borderType=cv2.BORDER_REFLECT_101,
            )
        elif self.degradation == "brightness":
            transformed = np.clip(
                original.astype(np.float32) * self.level,
                0.0,
                255.0,
            ).astype(np.uint8)
        else:
            raise RuntimeError(
                f"Unhandled visual degradation: {self.degradation}"
            )

        difference = transformed.astype(np.float32) - original.astype(np.float32)
        self._absolute_error_sum += float(np.abs(difference).sum())
        self._squared_error_sum += float(np.square(difference).sum())
        self._pixel_value_count += int(difference.size)
        self._changed_frames += int(np.any(difference != 0.0))
        return np.ascontiguousarray(transformed)

    def episode_metrics(self) -> dict:
        mean_absolute_error = (
            self._absolute_error_sum / self._pixel_value_count
            if self._pixel_value_count
            else 0.0
        )
        mean_squared_error = (
            self._squared_error_sum / self._pixel_value_count
            if self._pixel_value_count
            else 0.0
        )
        psnr_db = (
            None
            if mean_squared_error <= 0.0
            else float(10.0 * np.log10((255.0**2) / mean_squared_error))
        )
        return {
            "perturbation_type": self.name,
            "perturbation_protocol_version": "visual-degradation.v1",
            "visual_corruption": self.degradation,
            "visual_level": float(self.level),
            "visual_level_unit": VISUAL_LEVEL_UNITS[self.degradation],
            "affected_cameras": "|".join(self.camera_names),
            "injected": int(self.active and self._changed_frames > 0),
            "frames_transformed": self._frames_seen,
            "frames_changed": self._changed_frames,
            "mean_absolute_pixel_delta": float(mean_absolute_error),
            "mean_squared_pixel_delta": float(mean_squared_error),
            "input_psnr_db": psnr_db,
        }


@dataclass
class CameraExtrinsicShift(Perturbation):
    """Apply a deterministic scene-paired pose shift to a MuJoCo camera."""

    level: int
    translation_m: float
    rotation_deg: float
    base_seed: int
    camera_name: str = "agentview"
    name: str = field(init=False, default="camera_extrinsic_shift")

    _camera_id: int = field(init=False, default=-1)
    _original_position: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    _applied_position: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    _original_quaternion: np.ndarray = field(
        init=False,
        default_factory=lambda: np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )
    _applied_quaternion: np.ndarray = field(
        init=False,
        default_factory=lambda: np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )
    _translation_direction: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    _rotation_axis: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    _initial_frame_mae: float = field(init=False, default=0.0)
    _initial_frame_mse: float = field(init=False, default=0.0)
    _application_verified: bool = field(init=False, default=False)
    _restoration_verified: bool = field(init=False, default=False)
    _env: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Camera extrinsic level must be non-negative")
        if not np.isfinite(self.translation_m) or self.translation_m < 0.0:
            raise ValueError("Camera translation must be finite and non-negative")
        if not np.isfinite(self.rotation_deg) or self.rotation_deg < 0.0:
            raise ValueError("Camera rotation must be finite and non-negative")
        if not self.camera_name:
            raise ValueError("camera_name must be non-empty")
        if self.level == 0 and (
            not np.isclose(self.translation_m, 0.0)
            or not np.isclose(self.rotation_deg, 0.0)
        ):
            raise ValueError("Camera extrinsic level 0 must be an identity pose")

    @property
    def active(self) -> bool:
        return not (
            np.isclose(self.translation_m, 0.0)
            and np.isclose(self.rotation_deg, 0.0)
        )

    @staticmethod
    def _camera_code(camera_name: str) -> int:
        return sum(
            (index + 1) * byte
            for index, byte in enumerate(camera_name.encode("utf-8"))
        )

    @staticmethod
    def _unit_vector(rng: np.random.Generator) -> np.ndarray:
        while True:
            vector = rng.normal(size=3).astype(np.float64)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12:
                return vector / norm

    @staticmethod
    def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        product = np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )
        return product / np.linalg.norm(product)

    @staticmethod
    def _rotation_delta(axis: np.ndarray, angle_deg: float) -> np.ndarray:
        half_angle = 0.5 * np.deg2rad(angle_deg)
        return np.concatenate(
            [
                np.asarray([np.cos(half_angle)], dtype=np.float64),
                axis * np.sin(half_angle),
            ]
        )

    @staticmethod
    def _quaternion_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
        normalized_left = left / np.linalg.norm(left)
        normalized_right = right / np.linalg.norm(right)
        cosine = float(np.clip(abs(np.dot(normalized_left, normalized_right)), 0, 1))
        return float(np.rad2deg(2.0 * np.arccos(cosine)))

    def _paired_directions(self, scene_seed: int) -> tuple[np.ndarray, np.ndarray]:
        sequence = np.random.SeedSequence(
            [
                int(self.base_seed),
                int(scene_seed),
                self._camera_code(self.camera_name),
            ]
        )
        rng = np.random.default_rng(sequence)
        return self._unit_vector(rng), self._unit_vector(rng)

    def on_episode_start(self, context: PerturbationContext) -> None:
        self._env = context.env
        model = context.env.sim.model
        self._camera_id = int(model.camera_name2id(self.camera_name))
        self._original_position = np.asarray(
            model.cam_pos[self._camera_id],
            dtype=np.float64,
        ).copy()
        self._original_quaternion = np.asarray(
            model.cam_quat[self._camera_id],
            dtype=np.float64,
        ).copy()
        (
            self._translation_direction,
            self._rotation_axis,
        ) = self._paired_directions(context.scene_seed)
        self._applied_position = (
            self._original_position
            + self.translation_m * self._translation_direction
        )
        rotation_delta = self._rotation_delta(
            self._rotation_axis,
            self.rotation_deg,
        )
        self._applied_quaternion = self._quaternion_multiply(
            rotation_delta,
            self._original_quaternion,
        )
        self._initial_frame_mae = 0.0
        self._initial_frame_mse = 0.0
        self._restoration_verified = False

        original_frame = np.asarray(
            context.obs[f"{self.camera_name}_image"],
            dtype=np.uint8,
        ).copy()
        model.cam_pos[self._camera_id] = self._applied_position
        model.cam_quat[self._camera_id] = self._applied_quaternion
        context.env.sim.forward()
        self._application_verified = bool(
            np.allclose(
                model.cam_pos[self._camera_id],
                self._applied_position,
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                model.cam_quat[self._camera_id],
                self._applied_quaternion,
                rtol=0.0,
                atol=1e-12,
            )
        )
        if not self._application_verified:
            raise RuntimeError(
                f"Failed to apply camera extrinsic level {self.level}"
            )

        refreshed = context.env._get_observations(force_update=True)
        context.obs.clear()
        context.obs.update(refreshed)
        applied_frame = np.asarray(
            context.obs[f"{self.camera_name}_image"],
            dtype=np.uint8,
        )
        difference = (
            applied_frame.astype(np.float32) - original_frame.astype(np.float32)
        )
        self._initial_frame_mae = float(np.mean(np.abs(difference)))
        self._initial_frame_mse = float(np.mean(np.square(difference)))
        if self.active and self._initial_frame_mae <= 0.0:
            self.on_episode_end(context)
            raise RuntimeError(
                f"Camera level {self.level} did not alter the initial frame"
            )

    def on_episode_end(self, context: PerturbationContext) -> None:
        del context
        if self._env is None or self._camera_id < 0:
            return
        model = self._env.sim.model
        model.cam_pos[self._camera_id] = self._original_position
        model.cam_quat[self._camera_id] = self._original_quaternion
        self._env.sim.forward()
        self._restoration_verified = bool(
            np.allclose(
                model.cam_pos[self._camera_id],
                self._original_position,
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                model.cam_quat[self._camera_id],
                self._original_quaternion,
                rtol=0.0,
                atol=1e-12,
            )
        )
        if not self._restoration_verified:
            raise RuntimeError(
                f"Failed to restore camera {self.camera_name} extrinsics"
            )

    @staticmethod
    def _vector_string(values: np.ndarray) -> str:
        return "|".join(f"{float(value):.10g}" for value in values)

    def episode_metrics(self) -> dict:
        actual_translation_mm = 1000.0 * float(
            np.linalg.norm(self._applied_position - self._original_position)
        )
        actual_rotation_deg = self._quaternion_distance_deg(
            self._original_quaternion,
            self._applied_quaternion,
        )
        psnr_db = (
            None
            if self._initial_frame_mse <= 0.0
            else float(
                10.0 * np.log10((255.0**2) / self._initial_frame_mse)
            )
        )
        return {
            "perturbation_type": self.name,
            "perturbation_protocol_version": "camera-extrinsic.v1",
            "camera_name": self.camera_name,
            "camera_extrinsic_level": int(self.level),
            "nominal_translation_mm": 1000.0 * float(self.translation_m),
            "nominal_rotation_deg": float(self.rotation_deg),
            "actual_translation_mm": actual_translation_mm,
            "actual_rotation_deg": actual_rotation_deg,
            "translation_direction_x": float(self._translation_direction[0]),
            "translation_direction_y": float(self._translation_direction[1]),
            "translation_direction_z": float(self._translation_direction[2]),
            "rotation_axis_x": float(self._rotation_axis[0]),
            "rotation_axis_y": float(self._rotation_axis[1]),
            "rotation_axis_z": float(self._rotation_axis[2]),
            "original_camera_position": self._vector_string(
                self._original_position
            ),
            "applied_camera_position": self._vector_string(
                self._applied_position
            ),
            "original_camera_quaternion": self._vector_string(
                self._original_quaternion
            ),
            "applied_camera_quaternion": self._vector_string(
                self._applied_quaternion
            ),
            "initial_frame_mean_absolute_pixel_delta": self._initial_frame_mae,
            "initial_frame_psnr_db": psnr_db,
            "injected": int(self.active),
            "camera_application_verified": int(self._application_verified),
            "camera_restoration_verified": int(self._restoration_verified),
        }


PHYSICS_DRIFTS = ("clean", "target-mass", "target-friction")


@dataclass
class PhysicsParameterDrift(Perturbation):
    """Scale only the active target's mass/inertia or contact friction."""

    parameter: str
    multiplier: float
    name: str = field(init=False, default="physics_parameter_drift")

    _env: Any = field(init=False, default=None)
    _target_id: str = field(init=False, default="")
    _body_id: int = field(init=False, default=-1)
    _geom_ids: list[int] = field(init=False, default_factory=list)
    _original_mass: float = field(init=False, default=float("nan"))
    _applied_mass: float = field(init=False, default=float("nan"))
    _original_inertia: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    _applied_inertia: np.ndarray = field(
        init=False,
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    _original_frictions: np.ndarray = field(
        init=False,
        default_factory=lambda: np.empty((0, 3), dtype=np.float64),
    )
    _applied_frictions: np.ndarray = field(
        init=False,
        default_factory=lambda: np.empty((0, 3), dtype=np.float64),
    )
    _application_verified: bool = field(init=False, default=False)
    _restoration_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.parameter not in PHYSICS_DRIFTS:
            raise ValueError(f"Unsupported physics drift: {self.parameter}")
        if not np.isfinite(self.multiplier) or self.multiplier <= 0.0:
            raise ValueError("Physics multiplier must be finite and positive")
        if self.parameter == "clean" and not np.isclose(self.multiplier, 1.0):
            raise ValueError("Clean physics condition requires multiplier 1.0")

    @property
    def active(self) -> bool:
        return self.parameter != "clean" and not np.isclose(
            self.multiplier,
            1.0,
        )

    def on_episode_start(self, context: PerturbationContext) -> None:
        self._env = context.env
        self._target_id = str(context.task.target_id)
        model = context.env.sim.model
        self._body_id = int(context.env.object_body_ids[self._target_id])
        object_model = context.env.objects_by_id[self._target_id]
        self._geom_ids = [
            int(model.geom_name2id(name))
            for name in object_model.contact_geoms
        ]
        if not self._geom_ids:
            raise RuntimeError(
                f"Target {self._target_id} has no collision geoms"
            )

        self._original_mass = float(model.body_mass[self._body_id])
        self._original_inertia = np.asarray(
            model.body_inertia[self._body_id],
            dtype=np.float64,
        ).copy()
        self._original_frictions = np.asarray(
            model.geom_friction[self._geom_ids],
            dtype=np.float64,
        ).copy()
        self._applied_mass = self._original_mass
        self._applied_inertia = self._original_inertia.copy()
        self._applied_frictions = self._original_frictions.copy()
        self._application_verified = False
        self._restoration_verified = False

        if self.parameter == "target-mass":
            self._applied_mass = self._original_mass * self.multiplier
            self._applied_inertia = self._original_inertia * self.multiplier
            model.body_mass[self._body_id] = self._applied_mass
            model.body_inertia[self._body_id] = self._applied_inertia
        elif self.parameter == "target-friction":
            self._applied_frictions = (
                self._original_frictions * self.multiplier
            )
            model.geom_friction[self._geom_ids] = self._applied_frictions

        context.env.sim.forward()
        if self.parameter in {"clean", "target-mass"}:
            mass_matches = np.isclose(
                model.body_mass[self._body_id],
                self._applied_mass,
            ) and np.allclose(
                model.body_inertia[self._body_id],
                self._applied_inertia,
            )
        else:
            mass_matches = True
        if self.parameter in {"clean", "target-friction"}:
            friction_matches = np.allclose(
                model.geom_friction[self._geom_ids],
                self._applied_frictions,
            )
        else:
            friction_matches = True
        self._application_verified = bool(mass_matches and friction_matches)
        if not self._application_verified:
            raise RuntimeError(
                f"Failed to apply {self.parameter} multiplier {self.multiplier}"
            )

    def on_episode_end(self, context: PerturbationContext) -> None:
        if self._env is None:
            return
        model = self._env.sim.model
        model.body_mass[self._body_id] = self._original_mass
        model.body_inertia[self._body_id] = self._original_inertia
        model.geom_friction[self._geom_ids] = self._original_frictions
        self._env.sim.forward()
        self._restoration_verified = bool(
            np.isclose(model.body_mass[self._body_id], self._original_mass)
            and np.allclose(
                model.body_inertia[self._body_id],
                self._original_inertia,
            )
            and np.allclose(
                model.geom_friction[self._geom_ids],
                self._original_frictions,
            )
        )
        if not self._restoration_verified:
            raise RuntimeError(
                f"Failed to restore physics for target {self._target_id}"
            )

    @staticmethod
    def _vector_string(values: np.ndarray) -> str:
        return "|".join(f"{float(value):.10g}" for value in values.ravel())

    def episode_metrics(self) -> dict:
        original_sliding = (
            float(np.mean(self._original_frictions[:, 0]))
            if self._original_frictions.size
            else None
        )
        applied_sliding = (
            float(np.mean(self._applied_frictions[:, 0]))
            if self._applied_frictions.size
            else None
        )
        return {
            "perturbation_type": self.name,
            "perturbation_protocol_version": "physics-drift.v1",
            "physics_parameter": self.parameter,
            "physics_multiplier": float(self.multiplier),
            "physics_level_unit": "baseline_multiplier",
            "physics_target_id": self._target_id,
            "physics_body_id": self._body_id,
            "physics_geom_count": len(self._geom_ids),
            "injected": int(self.active),
            "physics_application_verified": int(self._application_verified),
            "physics_restoration_verified": int(self._restoration_verified),
            "original_body_mass": float(self._original_mass),
            "applied_body_mass": float(self._applied_mass),
            "original_body_inertia": self._vector_string(
                self._original_inertia
            ),
            "applied_body_inertia": self._vector_string(
                self._applied_inertia
            ),
            "original_mean_sliding_friction": original_sliding,
            "applied_mean_sliding_friction": applied_sliding,
            "original_geom_frictions": self._vector_string(
                self._original_frictions
            ),
            "applied_geom_frictions": self._vector_string(
                self._applied_frictions
            ),
        }


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
        predicted_target_position: np.ndarray | None = None,
    ) -> None:
        context = self._require_context()
        context.step = step
        context.predicted_phase = predicted_phase
        context.grounding_error_cm = grounding_error_cm
        context.predicted_target_position = (
            None
            if predicted_target_position is None
            else np.asarray(predicted_target_position, dtype=np.float32).copy()
        )
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
