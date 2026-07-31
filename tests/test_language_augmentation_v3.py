"""Regression tests for V3 language augmentation and explicit task routing."""

from __future__ import annotations

import unittest

import numpy as np

from utils.evaluation_core_v2 import EvaluationConfig, balanced_schedule
from utils.language_augmentation_v3 import LanguageAugmentationCatalog
from utils.v2_schema import TASK_BUCKETS


class LanguageCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = LanguageAugmentationCatalog()

    def test_every_task_has_train_and_held_out_language(self) -> None:
        for task_type, target_id in TASK_BUCKETS:
            train = set(
                self.catalog.expressions(task_type, target_id, "train")
            )
            evaluation = set(
                self.catalog.expressions(task_type, target_id, "eval")
            )
            self.assertGreaterEqual(len(train), 50)
            self.assertGreaterEqual(len(train | evaluation), 50)
            self.assertTrue(train.isdisjoint(evaluation))

    def test_training_sampling_is_not_fixed_to_one_sentence(self) -> None:
        rng = np.random.default_rng(7)
        samples = {
            self.catalog.sample(
                "pick",
                "A",
                training=True,
                rng=rng,
            ).text
            for _ in range(100)
        }
        self.assertGreaterEqual(len(samples), 30)

    def test_held_out_sampling_is_deterministic(self) -> None:
        first = self.catalog.sample(
            "push",
            "C",
            training=False,
            deterministic_index=17,
        )
        second = self.catalog.sample(
            "push",
            "C",
            training=False,
            deterministic_index=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.split, "eval")


class ExplicitTaskRoutingTests(unittest.TestCase):
    def test_noncanonical_instruction_uses_explicit_task(self) -> None:
        instruction = "Raise the crimson block from the tabletop"
        config = EvaluationConfig(
            instruction=instruction,
            task_type="pick",
            target_id="A",
        )
        config.validate()
        schedule = balanced_schedule(
            3,
            seed=1,
            fixed_instruction=instruction,
            fixed_task=("pick", "A"),
        )
        self.assertEqual(schedule, [("pick", "A")] * 3)

    def test_partial_explicit_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationConfig(
                instruction="Move the blue sphere away",
                task_type="push",
            ).validate()


if __name__ == "__main__":
    unittest.main()
