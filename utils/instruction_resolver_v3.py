"""Reliable finite-domain command parsing for the MiniVLA demo application."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


MODEL_INSTRUCTIONS = {
    ("pick", "A"): "Pick up the red cube",
    ("pick", "B"): "Pick up the blue ball",
    ("pick", "C"): "Pick up the green cylinder",
    ("push", "A"): "Push away the red cube",
    ("push", "B"): "Push away the blue ball",
    ("push", "C"): "Push away the green cylinder",
}

TASK_DISPLAY_NAMES = {
    ("pick", "A"): "PICK // RED CUBE",
    ("pick", "B"): "PICK // BLUE SPHERE",
    ("pick", "C"): "PICK // GREEN CYLINDER",
    ("push", "A"): "PUSH // RED CUBE",
    ("push", "B"): "PUSH // BLUE SPHERE",
    ("push", "C"): "PUSH // GREEN CYLINDER",
}

ACTION_TERMS = {
    "pick": {
        "pick",
        "pickup",
        "lift",
        "grab",
        "grasp",
        "raise",
        "elevate",
        "collect",
        "fetch",
        "remove",
    },
    "push": {
        "push",
        "shove",
        "slide",
        "nudge",
        "propel",
        "press",
        "drive",
    },
}

ACTION_PHRASES = {
    "pick": (
        "pick up",
        "take hold",
        "take off the table",
        "move upward",
    ),
    "push": (
        "move away",
        "move outward",
        "send away",
        "move farther",
        "across the table",
    ),
}

CHINESE_ACTION_TERMS = {
    "pick": ("拿起", "抓起", "拾起", "抬起", "夹起", "取下"),
    "push": ("推开", "推远", "推动", "往前推", "推"),
}

TARGET_TERMS = {
    "A": {
        "red",
        "crimson",
        "scarlet",
        "cube",
        "block",
        "box",
    },
    "B": {
        "blue",
        "azure",
        "cyan",
        "ball",
        "sphere",
        "orb",
        "round",
    },
    "C": {
        "green",
        "emerald",
        "cylinder",
        "cylindrical",
        "column",
        "tube",
    },
}

TARGET_PHRASES = {
    "A": ("red cube", "red block", "crimson cube", "scarlet block"),
    "B": ("blue ball", "blue sphere", "azure ball", "round object"),
    "C": (
        "green cylinder",
        "emerald cylinder",
        "green cylindrical object",
    ),
}

CHINESE_TARGET_TERMS = {
    "A": ("红色方块", "红方块", "红色积木", "红块"),
    "B": ("蓝色球", "蓝球", "蓝色圆球", "蓝色小球"),
    "C": ("绿色圆柱", "绿圆柱", "绿色圆柱体", "绿柱"),
}

FUZZY_THRESHOLD = 0.79


class InstructionResolutionError(ValueError):
    """Raised when a command cannot be mapped safely to one of six tasks."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ResolvedInstruction:
    original_text: str
    normalized_text: str
    task_type: str
    target_id: str
    model_instruction: str
    display_name: str
    confidence: float


def normalize_instruction(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _fuzzy_matches(tokens: list[str], vocabulary: set[str]) -> bool:
    return any(
        SequenceMatcher(None, token, candidate).ratio() >= FUZZY_THRESHOLD
        for token in tokens
        for candidate in vocabulary
        if len(token) >= 3 and len(candidate) >= 3
    )


def _resolve_action(normalized: str, tokens: list[str]) -> tuple[str | None, float]:
    candidates: dict[str, float] = {}
    for action, phrases in ACTION_PHRASES.items():
        if any(phrase in normalized for phrase in phrases):
            candidates[action] = 1.0
    for action, terms in ACTION_TERMS.items():
        if any(token in terms for token in tokens):
            candidates[action] = max(candidates.get(action, 0.0), 1.0)
        elif _fuzzy_matches(tokens, terms):
            candidates[action] = max(candidates.get(action, 0.0), 0.84)
    for action, phrases in CHINESE_ACTION_TERMS.items():
        if any(phrase in normalized for phrase in phrases):
            candidates[action] = max(candidates.get(action, 0.0), 1.0)
    if len(candidates) > 1:
        raise InstructionResolutionError(
            "ambiguous_action",
            "The command contains both pick and push intentions.",
        )
    if not candidates:
        return None, 0.0
    return next(iter(candidates.items()))


def _resolve_target(normalized: str, tokens: list[str]) -> tuple[str | None, float]:
    candidates: dict[str, float] = {}
    for target_id, phrases in TARGET_PHRASES.items():
        if any(phrase in normalized for phrase in phrases):
            candidates[target_id] = 1.0
    for target_id, terms in TARGET_TERMS.items():
        exact_count = sum(token in terms for token in tokens)
        if exact_count:
            candidates[target_id] = max(
                candidates.get(target_id, 0.0),
                min(1.0, 0.88 + 0.06 * exact_count),
            )
        elif _fuzzy_matches(tokens, terms):
            candidates[target_id] = max(candidates.get(target_id, 0.0), 0.82)
    for target_id, phrases in CHINESE_TARGET_TERMS.items():
        if any(phrase in normalized for phrase in phrases):
            candidates[target_id] = max(candidates.get(target_id, 0.0), 1.0)
    if len(candidates) > 1:
        strongest = max(candidates.values())
        strongest_ids = [
            target_id
            for target_id, score in candidates.items()
            if score >= strongest - 0.03
        ]
        if len(strongest_ids) == 1:
            return strongest_ids[0], candidates[strongest_ids[0]]
        raise InstructionResolutionError(
            "ambiguous_target",
            "The command refers to more than one target object.",
        )
    if not candidates:
        return None, 0.0
    return next(iter(candidates.items()))


class ReliableInstructionResolver:
    """Map flexible finite-domain language onto a reliable policy prompt."""

    def resolve(self, text: str) -> ResolvedInstruction:
        original = text.strip()
        if not original:
            raise InstructionResolutionError(
                "empty",
                "Enter a command before starting the robot.",
            )
        normalized = normalize_instruction(original)
        tokens = normalized.split()
        action, action_confidence = _resolve_action(normalized, tokens)
        target_id, target_confidence = _resolve_target(normalized, tokens)
        if action is None and target_id is None:
            raise InstructionResolutionError(
                "unsupported",
                "Use a pick or push command for the red cube, blue sphere, or green cylinder.",
            )
        if action is None:
            raise InstructionResolutionError(
                "missing_action",
                "The object is clear, but the command must say whether to pick or push it.",
            )
        if target_id is None:
            raise InstructionResolutionError(
                "missing_target",
                "The action is clear, but the target object is missing.",
            )
        key = (action, target_id)
        return ResolvedInstruction(
            original_text=original,
            normalized_text=normalized,
            task_type=action,
            target_id=target_id,
            model_instruction=MODEL_INSTRUCTIONS[key],
            display_name=TASK_DISPLAY_NAMES[key],
            confidence=min(action_confidence, target_confidence),
        )
