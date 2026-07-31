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
    DynamicTargetDisplacement,
    PerturbationContext,
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


if __name__ == "__main__":
    unittest.main()
