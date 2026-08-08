import pytest

from utils.instruction_resolver_v3 import (
    InstructionResolutionError,
    ReliableInstructionResolver,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Pick the azure ball up", ("pick", "B")),
        ("Shove the crimson block away", ("push", "A")),
        ("Raise the emerald cylinder", ("pick", "C")),
        ("slide the sphere away", ("push", "B")),
        ("pic up the red blok", ("pick", "A")),
        ("把蓝色球拿起来", ("pick", "B")),
        ("把绿色圆柱推远", ("push", "C")),
    ],
)
def test_resolves_flexible_demo_commands(text, expected):
    resolved = ReliableInstructionResolver().resolve(text)
    assert (resolved.task_type, resolved.target_id) == expected
    assert resolved.model_instruction
    assert 0.0 < resolved.confidence <= 1.0


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", "empty"),
        ("hello robot", "unsupported"),
        ("pick it up", "missing_target"),
        ("the blue sphere", "missing_action"),
        ("pick and push the red cube", "ambiguous_action"),
        ("push the red cube and blue sphere", "ambiguous_target"),
    ],
)
def test_rejects_incomplete_or_ambiguous_commands(text, code):
    with pytest.raises(InstructionResolutionError) as error:
        ReliableInstructionResolver().resolve(text)
    assert error.value.code == code


def test_uses_reliable_policy_prompt_instead_of_raw_demo_wording():
    resolved = ReliableInstructionResolver().resolve("Pick the azure ball up")
    assert resolved.original_text == "Pick the azure ball up"
    assert resolved.model_instruction == "Pick up the blue ball"
