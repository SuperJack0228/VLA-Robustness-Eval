"""Protocol-level regression tests for bounded ACT and target displacement."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from utils.evaluation_core_v2 import (
    TemporalActionEnsembler,
    classify_failure,
)
from utils.perturbations_v2 import (
    CameraExtrinsicShift,
    DynamicTargetDisplacement,
    PhysicsParameterDrift,
    Perturbation,
    PerturbationContext,
    PerturbationManager,
    VisualDegradation,
)
from utils.ood_distractors_v3 import (
    OOD_DISTRACTOR_IDS,
    OOD_GEOMETRY_SPECS,
    OODDistractorInjection,
)


class TemporalActionEnsemblerTests(unittest.TestCase):
    def _populate(self, ensembler: TemporalActionEnsembler, step: int) -> None:
        for created in range(step + 1):
            length = step - created + 1
            pose = np.full((length, 6), float(created), dtype=np.float32)
            gripper = np.full(length, created / 10.0, dtype=np.float32)
            phase = np.zeros(length, dtype=np.int64)
            ensembler.add(created, pose, gripper, phase)

    def test_temporal_mode_uses_only_bounded_recent_predictions(self) -> None:
        ensembler = TemporalActionEnsembler(decay=0.75, max_prediction_age=2)
        self._populate(ensembler, step=4)

        pose, _, _, latest_only = ensembler.action(
            step=4,
            latest_only_phases=frozenset(),
            mode="temporal",
        )

        ages = np.asarray([2, 1, 0], dtype=np.float64)
        weights = np.exp(-0.75 * ages)
        weights /= weights.sum()
        expected = float(np.dot(np.asarray([2.0, 3.0, 4.0]), weights))
        self.assertFalse(latest_only)
        self.assertEqual(ensembler.last_prediction_count, 3)
        self.assertEqual(ensembler.last_oldest_prediction_age, 2)
        np.testing.assert_allclose(pose, expected, atol=1e-6)

    def test_reactive_phase_forces_latest_prediction(self) -> None:
        ensembler = TemporalActionEnsembler(decay=0.75, max_prediction_age=3)
        self._populate(ensembler, step=4)
        for entries in ensembler.predictions.values():
            for index, entry in enumerate(entries):
                entries[index] = (*entry[:3], 1)

        pose, _, phase, latest_only = ensembler.action(
            step=4,
            latest_only_phases=frozenset({1}),
            mode="temporal",
        )

        self.assertTrue(latest_only)
        self.assertEqual(phase, 1)
        self.assertEqual(ensembler.last_prediction_count, 1)
        np.testing.assert_allclose(pose, 4.0)

    def test_grounding_reset_discards_pre_reset_chunks(self) -> None:
        ensembler = TemporalActionEnsembler(decay=0.75, max_prediction_age=3)
        self._populate(ensembler, step=4)
        removed = ensembler.discard_created_before(4)
        pose, _, _, _ = ensembler.action(4, frozenset(), "temporal")

        self.assertGreater(removed, 0)
        self.assertEqual(ensembler.last_prediction_count, 1)
        np.testing.assert_allclose(pose, 4.0)


class FailureTaxonomyTests(unittest.TestCase):
    def classify_pick(self, **overrides) -> str:
        arguments = {
            "task_type": "pick",
            "target_id": "A",
            "task_success": False,
            "wrong_object_contact": False,
            "termination": "horizon",
            "min_eef_target_distance": 0.02,
            "gripper_close_step": 20,
            "target_contact": True,
            "grasped_once": False,
            "final_grasp": False,
            "initial_target_height": 0.82,
            "max_target_height": 0.82,
            "final_target_height": 0.82,
            "min_target_uprightness": 1.0,
            "table_height": 0.80,
            "push_forward": 0.0,
            "push_lateral": 0.0,
        }
        arguments.update(overrides)
        return classify_failure(**arguments)

    def test_collision_launched_object_is_not_insufficient_lift(self) -> None:
        category = self.classify_pick(max_target_height=0.86)
        self.assertEqual(category, "object_launched")

    def test_lost_grasp_is_object_dropped(self) -> None:
        category = self.classify_pick(
            grasped_once=True,
            final_grasp=False,
            max_target_height=0.90,
        )
        self.assertEqual(category, "object_dropped")

    def test_contact_without_grasp_is_separate_failure(self) -> None:
        self.assertEqual(
            self.classify_pick(target_contact=True),
            "grasp_failed_after_contact",
        )
        self.assertEqual(
            self.classify_pick(target_contact=False),
            "missed_grasp",
        )


class _FakeData:
    def __init__(self, positions: dict[str, np.ndarray]) -> None:
        self.qpos = {
            f"{object_id}_joint": np.concatenate(
                [position.astype(np.float64), [1.0, 0.0, 0.0, 0.0]]
            )
            for object_id, position in positions.items()
        }
        self.qvel = {
            f"{object_id}_joint": np.zeros(6, dtype=np.float64)
            for object_id in positions
        }

    def get_joint_qpos(self, name: str) -> np.ndarray:
        return self.qpos[name]

    def set_joint_qpos(self, name: str, value: np.ndarray) -> None:
        self.qpos[name] = np.asarray(value, dtype=np.float64).copy()

    def get_joint_qvel(self, name: str) -> np.ndarray:
        return self.qvel[name]

    def set_joint_qvel(self, name: str, value: np.ndarray) -> None:
        self.qvel[name] = np.asarray(value, dtype=np.float64).copy()


class _FakeEnvironment:
    def __init__(self) -> None:
        positions = {
            "A": np.asarray([0.0, 0.0, 0.82]),
            "B": np.asarray([0.25, 0.25, 0.82]),
            "C": np.asarray([-0.25, 0.25, 0.82]),
        }
        self.sim = SimpleNamespace(data=_FakeData(positions), forward=lambda: None)
        self.objects_by_id = {
            object_id: SimpleNamespace(
                horizontal_radius=0.02,
                joints=[f"{object_id}_joint"],
            )
            for object_id in positions
        }
        self.table_offset = np.asarray([0.0, 0.0, 0.80])
        self.table_full_size = np.asarray([0.8, 0.8, 0.05])
        self.robots = [
            SimpleNamespace(
                robot_model=object(),
                gripper=object(),
            )
        ]

    def get_object_position(self, object_id: str) -> np.ndarray:
        return self.sim.data.qpos[f"{object_id}_joint"][:3].copy()

    def check_contact(self, *_args) -> bool:
        return False


class DynamicDisplacementProtocolTests(unittest.TestCase):
    def make_context(
        self,
        env: _FakeEnvironment,
        task_type: str,
    ) -> PerturbationContext:
        task = SimpleNamespace(
            task_type=task_type,
            target_id="A",
            push_direction=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        return PerturbationContext(
            episode_id=1,
            scene_seed=12345,
            env=env,
            task=task,
            obs={},
        )

    def test_all_levels_use_the_same_prevalidated_direction(self) -> None:
        levels = (0.0, 0.02, 0.04, 0.06, 0.08)
        for task_type in ("pick", "push"):
            directions = []
            for level in (0.02, 0.08):
                env = _FakeEnvironment()
                perturbation = DynamicTargetDisplacement(
                    distance_m=level,
                    base_seed=991,
                    validation_distances_m=levels,
                )
                perturbation.on_episode_start(
                    self.make_context(env, task_type)
                )
                metrics = perturbation.episode_metrics()
                directions.append(
                    np.asarray(
                        [
                            metrics["selected_direction_x"],
                            metrics["selected_direction_y"],
                        ]
                    )
                )
                self.assertEqual(
                    metrics["protocol_collision_integrity_passed"],
                    1,
                )

            np.testing.assert_allclose(
                directions[0],
                directions[1],
                atol=1e-12,
            )


class VisualDegradationProtocolTests(unittest.TestCase):
    def make_context(self, scene_seed: int = 12345) -> PerturbationContext:
        return PerturbationContext(
            episode_id=1,
            scene_seed=scene_seed,
            env=None,
            task=None,
            obs={},
        )

    @staticmethod
    def sample_image() -> np.ndarray:
        values = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
        return (values % 256).astype(np.uint8)

    def test_clean_visual_condition_is_exact_identity(self) -> None:
        perturbation = VisualDegradation("clean", 0.0, base_seed=77)
        context = self.make_context()
        perturbation.on_episode_start(context)
        image = self.sample_image()
        transformed = perturbation.transform_image(
            context,
            "agentview",
            image,
        )
        np.testing.assert_array_equal(transformed, image)
        metrics = perturbation.episode_metrics()
        self.assertEqual(metrics["injected"], 0)
        self.assertEqual(metrics["mean_absolute_pixel_delta"], 0.0)

    def test_gaussian_noise_is_reproducible_for_paired_scene(self) -> None:
        context = self.make_context()
        image = self.sample_image()
        outputs = []
        for _ in range(2):
            perturbation = VisualDegradation(
                "gaussian-noise",
                20.0,
                base_seed=99,
            )
            perturbation.on_episode_start(context)
            outputs.append(
                perturbation.transform_image(context, "agentview", image)
            )
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertFalse(np.array_equal(outputs[0], image))

    def test_blur_and_brightness_preserve_image_contract(self) -> None:
        context = self.make_context()
        image = self.sample_image()
        for degradation, level in (
            ("gaussian-blur", 1.5),
            ("brightness", 0.6),
        ):
            perturbation = VisualDegradation(
                degradation,
                level,
                base_seed=17,
            )
            perturbation.on_episode_start(context)
            transformed = perturbation.transform_image(
                context,
                "robot0_eye_in_hand",
                image,
            )
            self.assertEqual(transformed.shape, image.shape)
            self.assertEqual(transformed.dtype, np.uint8)
            self.assertTrue(transformed.flags.c_contiguous)
            self.assertEqual(perturbation.episode_metrics()["injected"], 1)


class _FakeCameraModel:
    def __init__(self) -> None:
        self.cam_pos = np.asarray([[0.7, 0.1, 1.4]], dtype=np.float64)
        self.cam_quat = np.asarray(
            [[0.70710678, 0.0, 0.0, 0.70710678]],
            dtype=np.float64,
        )

    @staticmethod
    def camera_name2id(name: str) -> int:
        if name != "agentview":
            raise KeyError(name)
        return 0


class _FakeCameraEnvironment:
    def __init__(self) -> None:
        self.model = _FakeCameraModel()
        self.forward_calls = 0
        self.sim = SimpleNamespace(
            model=self.model,
            forward=self._forward,
        )

    def _forward(self) -> None:
        self.forward_calls += 1

    def _get_observations(self, force_update: bool = False) -> dict:
        del force_update
        signature = float(
            np.dot(self.model.cam_pos[0], [3000.0, 5000.0, 7000.0])
            + np.dot(self.model.cam_quat[0], [1100.0, 1300.0, 1700.0, 1900.0])
        )
        value = int(abs(signature)) % 256
        return {
            "agentview_image": np.full(
                (8, 8, 3),
                value,
                dtype=np.uint8,
            ),
            "robot0_eye_in_hand_image": np.zeros(
                (8, 8, 3),
                dtype=np.uint8,
            ),
        }


class CameraExtrinsicProtocolTests(unittest.TestCase):
    @staticmethod
    def make_context(
        env: _FakeCameraEnvironment,
        scene_seed: int = 12345,
    ) -> PerturbationContext:
        return PerturbationContext(
            episode_id=1,
            scene_seed=scene_seed,
            env=env,
            task=SimpleNamespace(target_id="A"),
            obs=env._get_observations(force_update=True),
        )

    def test_pose_shift_refreshes_first_frame_and_restores(self) -> None:
        env = _FakeCameraEnvironment()
        context = self.make_context(env)
        original_position = env.model.cam_pos.copy()
        original_quaternion = env.model.cam_quat.copy()
        original_frame = context.obs["agentview_image"].copy()
        perturbation = CameraExtrinsicShift(
            level=4,
            translation_m=0.020,
            rotation_deg=4.0,
            base_seed=91,
        )

        perturbation.on_episode_start(context)

        self.assertAlmostEqual(
            np.linalg.norm(env.model.cam_pos[0] - original_position[0]),
            0.020,
            places=12,
        )
        self.assertFalse(
            np.array_equal(context.obs["agentview_image"], original_frame)
        )
        metrics = perturbation.episode_metrics()
        self.assertAlmostEqual(metrics["actual_translation_mm"], 20.0)
        self.assertAlmostEqual(metrics["actual_rotation_deg"], 4.0)
        self.assertEqual(metrics["camera_application_verified"], 1)
        self.assertEqual(metrics["injected"], 1)

        perturbation.on_episode_end(context)

        np.testing.assert_allclose(env.model.cam_pos, original_position)
        np.testing.assert_allclose(env.model.cam_quat, original_quaternion)
        self.assertEqual(
            perturbation.episode_metrics()["camera_restoration_verified"],
            1,
        )

    def test_all_levels_share_scene_paired_directions(self) -> None:
        metrics = []
        for level, translation_m, rotation_deg in (
            (1, 0.005, 1.0),
            (4, 0.020, 4.0),
        ):
            env = _FakeCameraEnvironment()
            context = self.make_context(env, scene_seed=777)
            perturbation = CameraExtrinsicShift(
                level=level,
                translation_m=translation_m,
                rotation_deg=rotation_deg,
                base_seed=19,
            )
            perturbation.on_episode_start(context)
            metrics.append(perturbation.episode_metrics())
            perturbation.on_episode_end(context)

        for prefix in ("translation_direction", "rotation_axis"):
            first = np.asarray(
                [metrics[0][f"{prefix}_{axis}"] for axis in "xyz"]
            )
            second = np.asarray(
                [metrics[1][f"{prefix}_{axis}"] for axis in "xyz"]
            )
            np.testing.assert_allclose(first, second, atol=1e-12)

    def test_level_zero_is_exact_pose_and_image_identity(self) -> None:
        env = _FakeCameraEnvironment()
        context = self.make_context(env)
        original_frame = context.obs["agentview_image"].copy()
        perturbation = CameraExtrinsicShift(
            level=0,
            translation_m=0.0,
            rotation_deg=0.0,
            base_seed=31,
        )
        perturbation.on_episode_start(context)
        metrics = perturbation.episode_metrics()
        np.testing.assert_array_equal(
            context.obs["agentview_image"],
            original_frame,
        )
        self.assertEqual(metrics["injected"], 0)
        self.assertEqual(metrics["initial_frame_mean_absolute_pixel_delta"], 0.0)
        perturbation.on_episode_end(context)


class _FakePhysicsModel:
    def __init__(self) -> None:
        self.body_mass = np.asarray([0.25], dtype=np.float64)
        self.body_inertia = np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64)
        self.geom_friction = np.asarray(
            [[0.9, 0.005, 0.0001], [0.8, 0.004, 0.0002]],
            dtype=np.float64,
        )
        self._geom_ids = {"target_g0": 0, "target_g1": 1}

    def geom_name2id(self, name: str) -> int:
        return self._geom_ids[name]


class _FakePhysicsEnvironment:
    def __init__(self) -> None:
        self.model = _FakePhysicsModel()
        self.forward_calls = 0
        self.sim = SimpleNamespace(
            model=self.model,
            forward=self._forward,
        )
        self.object_body_ids = {"A": 0}
        self.objects_by_id = {
            "A": SimpleNamespace(contact_geoms=["target_g0", "target_g1"])
        }

    def _forward(self) -> None:
        self.forward_calls += 1


class PhysicsParameterDriftTests(unittest.TestCase):
    @staticmethod
    def make_context(env: _FakePhysicsEnvironment) -> PerturbationContext:
        return PerturbationContext(
            episode_id=1,
            scene_seed=123,
            env=env,
            task=SimpleNamespace(target_id="A"),
            obs={},
        )

    def test_mass_scales_inertia_and_restores_baseline(self) -> None:
        env = _FakePhysicsEnvironment()
        original_mass = env.model.body_mass.copy()
        original_inertia = env.model.body_inertia.copy()
        perturbation = PhysicsParameterDrift("target-mass", 2.0)
        context = self.make_context(env)
        perturbation.on_episode_start(context)
        np.testing.assert_allclose(env.model.body_mass, 2.0 * original_mass)
        np.testing.assert_allclose(env.model.body_inertia, 2.0 * original_inertia)
        perturbation.on_episode_end(context)
        np.testing.assert_allclose(env.model.body_mass, original_mass)
        np.testing.assert_allclose(env.model.body_inertia, original_inertia)
        self.assertEqual(
            perturbation.episode_metrics()["physics_restoration_verified"],
            1,
        )

    def test_friction_scales_all_contact_components_and_restores(self) -> None:
        env = _FakePhysicsEnvironment()
        original = env.model.geom_friction.copy()
        perturbation = PhysicsParameterDrift("target-friction", 0.5)
        context = self.make_context(env)
        perturbation.on_episode_start(context)
        np.testing.assert_allclose(env.model.geom_friction, 0.5 * original)
        perturbation.on_episode_end(context)
        np.testing.assert_allclose(env.model.geom_friction, original)


class _PredictionCapture(Perturbation):
    def __init__(self) -> None:
        self.prediction = None

    def after_prediction(self, context: PerturbationContext) -> None:
        self.prediction = context.predicted_target_position


class OODDistractorProtocolTests(unittest.TestCase):
    def test_supported_counts_and_unseen_specs_are_explicit(self) -> None:
        for count in range(4):
            perturbation = OODDistractorInjection(count=count, base_seed=7)
            self.assertEqual(perturbation.count, count)
        with self.assertRaises(ValueError):
            OODDistractorInjection(count=4, base_seed=7)
        self.assertEqual(tuple(OOD_GEOMETRY_SPECS), OOD_DISTRACTOR_IDS)
        self.assertEqual(
            len({spec.label for spec in OOD_GEOMETRY_SPECS.values()}),
            3,
        )

    def test_manager_exposes_predicted_target_position_to_perturbations(self) -> None:
        capture = _PredictionCapture()
        manager = PerturbationManager([capture])
        manager.on_episode_start(
            episode_id=1,
            scene_seed=2,
            env=None,
            task=None,
            obs={},
        )
        prediction = np.asarray([0.1, -0.2, 0.8], dtype=np.float32)
        manager.after_prediction(3, 1, 0.5, prediction)
        np.testing.assert_array_equal(capture.prediction, prediction)
        self.assertIsNot(capture.prediction, prediction)


if __name__ == "__main__":
    unittest.main()
