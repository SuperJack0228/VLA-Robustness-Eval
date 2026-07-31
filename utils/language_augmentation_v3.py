"""Validated dynamic language augmentation for MiniVLA V3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from utils.v2_schema import TASK_BUCKETS


DEFAULT_LANGUAGE_CATALOG = "configs/language_augmentations_v3.json"
MIN_EXPRESSIONS_PER_TASK = 50


@dataclass(frozen=True)
class LanguageVariant:
    text: str
    split: str
    variant_id: str


class LanguageAugmentationCatalog:
    """Expand and sample semantically aligned instruction paraphrases."""

    def __init__(self, path: str = DEFAULT_LANGUAGE_CATALOG) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8") as catalog_file:
            payload = json.load(catalog_file)
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported V3 language catalog schema")
        self.catalog_version = str(payload["catalog_version"])
        aliases = payload["object_aliases"]
        self.variants: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
        all_text: dict[str, tuple[str, str, str]] = {}
        for task_type, target_id in TASK_BUCKETS:
            if target_id not in aliases or len(aliases[target_id]) < 2:
                raise ValueError(f"Missing object aliases for target {target_id}")
            split_variants = {}
            for split, template_key in (
                ("train", "train_templates"),
                ("eval", "eval_templates"),
            ):
                templates = payload[template_key][task_type]
                rendered = tuple(
                    template.format(object=alias)
                    for template in templates
                    for alias in aliases[target_id]
                )
                if len(rendered) != len(set(rendered)):
                    raise ValueError(
                        f"Duplicate {split} expressions for {task_type}_{target_id}"
                    )
                split_variants[split] = rendered
                for text in rendered:
                    normalized = self._normalize(text)
                    owner = all_text.get(normalized)
                    current = (task_type, target_id, split)
                    if owner is not None and owner != current:
                        raise ValueError(
                            f"Instruction collision: {text!r} maps to both "
                            f"{owner} and {current}"
                        )
                    all_text[normalized] = current
            overlap = set(split_variants["train"]) & set(split_variants["eval"])
            if overlap:
                raise ValueError(
                    f"Train/eval language leakage for {task_type}_{target_id}"
                )
            total = len(split_variants["train"]) + len(
                split_variants["eval"]
            )
            if total < MIN_EXPRESSIONS_PER_TASK:
                raise ValueError(
                    f"{task_type}_{target_id} has only {total} expressions"
                )
            self.variants[(task_type, target_id)] = split_variants
        self.digest = self._digest(payload)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _digest(payload: dict) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def expressions(
        self,
        task_type: str,
        target_id: str,
        split: str,
    ) -> tuple[str, ...]:
        if split not in {"train", "eval"}:
            raise ValueError("Language split must be train or eval")
        try:
            return self.variants[(task_type, target_id)][split]
        except KeyError as error:
            raise ValueError(
                f"Unknown language task: {(task_type, target_id)}"
            ) from error

    def sample(
        self,
        task_type: str,
        target_id: str,
        training: bool,
        rng: np.random.Generator | None = None,
        deterministic_index: int = 0,
    ) -> LanguageVariant:
        split = "train" if training else "eval"
        candidates = self.expressions(task_type, target_id, split)
        if training:
            rng = rng or np.random.default_rng()
            index = int(rng.integers(len(candidates)))
        else:
            index = int(deterministic_index) % len(candidates)
        return LanguageVariant(
            text=candidates[index],
            split=split,
            variant_id=f"{task_type}_{target_id}:{split}:{index:03d}",
        )

    def summary(self) -> dict:
        tasks = {}
        for task_type, target_id in TASK_BUCKETS:
            train_count = len(
                self.expressions(task_type, target_id, "train")
            )
            eval_count = len(self.expressions(task_type, target_id, "eval"))
            tasks[f"{task_type}_{target_id}"] = {
                "train": train_count,
                "eval": eval_count,
                "total": train_count + eval_count,
            }
        return {
            "catalog_version": self.catalog_version,
            "sha256": self.digest,
            "tasks": tasks,
        }
