"""Clean and legacy visual evaluation CLI for MiniVLA V2 policies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.evaluation_core_v2 import (
    DEFAULT_NUM_EPISODES,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_POLICY_PATH,
    ENSEMBLE_DECAY,
    ENSEMBLE_MODES,
    GROUNDING_SHIFT_RESET_THRESHOLD_M,
    MAX_STEPS,
    MAX_TEMPORAL_PREDICTION_AGE,
    REPLAN_INTERVAL,
    TEMPORAL_PROFILES,
    VISUAL_PERTURBATIONS,
    EvaluationConfig,
    EvaluationCore,
    get_device,
)


EXECUTION_MODES = ("raw",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--task-type",
        choices=("pick", "push"),
        default=None,
        help="Ground-truth task for a non-canonical V3 language instruction.",
    )
    parser.add_argument(
        "--target-id",
        choices=("A", "B", "C"),
        default=None,
        help="Ground-truth target for a non-canonical V3 language instruction.",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--replan-interval", type=int, default=REPLAN_INTERVAL)
    parser.add_argument("--ensemble-decay", type=float, default=ENSEMBLE_DECAY)
    parser.add_argument(
        "--max-prediction-age",
        type=int,
        default=MAX_TEMPORAL_PREDICTION_AGE,
    )
    parser.add_argument(
        "--grounding-reset-threshold",
        type=float,
        default=GROUNDING_SHIFT_RESET_THRESHOLD_M,
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enforce-80", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="raw",
    )
    parser.add_argument(
        "--visual-perturbation",
        choices=VISUAL_PERTURBATIONS,
        default="clean",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=ENSEMBLE_MODES,
        default="temporal",
    )
    parser.add_argument(
        "--temporal-profile",
        choices=TEMPORAL_PROFILES,
        default="robust",
        help=(
            "Use robust bounded temporal integration, or legacy to reproduce "
            "the frozen V2 Clean execution protocol."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(
        "Execution mode: raw | privileged execution assistance: False",
        flush=True,
    )
    perturbation_label = (
        "clean"
        if args.visual_perturbation == "clean"
        else f"visual:{args.visual_perturbation}"
    )
    config = EvaluationConfig(
        policy_path=args.policy,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        instruction=args.instruction,
        task_type=args.task_type,
        target_id=args.target_id,
        output_prefix=args.output_prefix,
        replan_interval=args.replan_interval,
        render=args.render,
        log_every=args.log_every,
        local_files_only=args.local_files_only,
        visual_perturbation=args.visual_perturbation,
        ensemble_mode=args.ensemble_mode,
        temporal_profile=args.temporal_profile,
        ensemble_decay=args.ensemble_decay,
        max_prediction_age=args.max_prediction_age,
        grounding_shift_reset_threshold_m=args.grounding_reset_threshold,
        perturbation_label=perturbation_label,
    )
    core = EvaluationCore.from_policy(config, perturbation_manager=None, device=device)
    result = core.run()
    print(json.dumps(result.report, indent=2), flush=True)
    print(
        f"Results: {result.csv_path} and {result.json_path}",
        flush=True,
    )
    if args.enforce_80 and not result.report["overall"]["passed_80_percent_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
