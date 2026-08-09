#!/usr/bin/env python3
"""Audit and plot the paired MiniVLA V3 language-generalization benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTOCOL_VERSION = "language-generalization.v1"
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
CONDITIONS = ("canonical", "held_out")
COLORS = {"canonical": "#36688D", "held_out": "#D95F59"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="final_report/09_language_generalization",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    columns = list(rows[0]) if rows else list(fieldnames or [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true"}


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def audit(rows: list[dict], summary: dict) -> dict:
    if summary.get("status") != "complete":
        raise ValueError("Language benchmark is not complete")
    if summary.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected language protocol version")
    if len(rows) != 360:
        raise ValueError(f"Expected 360 episode rows, found {len(rows)}")
    if any(row["execution_task_source"] != "model-prediction" for row in rows):
        raise ValueError("Ground-truth task leaked into action execution")
    if any(as_bool(row, "instruction_resolver_used") for row in rows):
        raise ValueError("Instruction resolver was used")
    if any(as_bool(row, "instruction_transform_applied") for row in rows):
        raise ValueError("An instruction transformation was applied")
    if any(row["instruction"] != row["raw_policy_instruction"] for row in rows):
        raise ValueError("Raw policy instruction does not match evaluator input")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["pair_id"])].append(row)
    if len(grouped) != 180:
        raise ValueError(f"Expected 180 paired scenes, found {len(grouped)}")
    for pair_id, pair in grouped.items():
        if {row["language_condition"] for row in pair} != set(CONDITIONS):
            raise ValueError(f"Pair {pair_id} lacks one language condition")
        if len({row["scene_signature"] for row in pair}) != 1:
            raise ValueError(f"Pair {pair_id} does not share the same scene")
        if len({row["scene_seed"] for row in pair}) != 1:
            raise ValueError(f"Pair {pair_id} does not share the same seed")

    for task in TASKS:
        held_out = [
            row
            for row in rows
            if row["task_bucket"] == task
            and row["language_condition"] == "held_out"
        ]
        canonical = [
            row
            for row in rows
            if row["task_bucket"] == task
            and row["language_condition"] == "canonical"
        ]
        if len(held_out) != 30 or len(canonical) != 30:
            raise ValueError(f"{task} is not a balanced 30-pair task bucket")
        if len({row["raw_policy_instruction"] for row in held_out}) != 30:
            raise ValueError(f"{task} does not contain 30 unique held-out texts")
        if len({row["raw_policy_instruction"] for row in canonical}) != 1:
            raise ValueError(f"{task} canonical condition is not canonical")

    return {
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "episode_rows": len(rows),
        "paired_scenes": len(grouped),
        "held_out_expressions": 180,
        "canonical_control_episodes": 180,
        "balanced_tasks": True,
        "paired_scene_validation_passed": True,
        "raw_instruction_passthrough_validated": True,
        "instruction_resolver_absent": True,
        "ground_truth_task_restricted_to_scene_and_scoring": True,
        "policy_execution_uses_model_predictions": True,
    }


def success_summary(rows: list[dict]) -> list[dict]:
    output = []
    for condition in CONDITIONS:
        for task in ("overall", *TASKS):
            selected = [
                row
                for row in rows
                if row["language_condition"] == condition
                and (task == "overall" or row["task_bucket"] == task)
            ]
            successes = sum(as_bool(row, "task_success") for row in selected)
            lower, upper = wilson_interval(successes, len(selected))
            output.append(
                {
                    "condition": condition,
                    "task": task,
                    "episodes": len(selected),
                    "successes": successes,
                    "success_rate": successes / len(selected),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "initial_task_type_accuracy": float(
                        np.mean(
                            [
                                row["initial_predicted_task_type"] == row["task_type"]
                                for row in selected
                            ]
                        )
                    ),
                    "initial_target_accuracy": float(
                        np.mean(
                            [
                                row["initial_predicted_target_id"] == row["target_id"]
                                for row in selected
                            ]
                        )
                    ),
                    "initial_task_bucket_accuracy": float(
                        np.mean(
                            [
                                row["initial_predicted_task_bucket"] == row["task_bucket"]
                                for row in selected
                            ]
                        )
                    ),
                }
            )
    return output


def paired_outcomes(rows: list[dict]) -> tuple[dict, list[dict]]:
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["pair_id"])][row["language_condition"]] = row
    details = []
    counts = Counter()
    for pair_id, pair in grouped.items():
        canonical = as_bool(pair["canonical"], "task_success")
        held_out = as_bool(pair["held_out"], "task_success")
        if canonical and held_out:
            outcome = "both_success"
        elif canonical:
            outcome = "canonical_only_success"
        elif held_out:
            outcome = "held_out_only_success"
        else:
            outcome = "neither_success"
        counts[outcome] += 1
        details.append(
            {
                "pair_id": pair_id,
                "task_bucket": pair["canonical"]["task_bucket"],
                "scene_seed": pair["canonical"]["scene_seed"],
                "variant_id": pair["held_out"]["variant_id"],
                "held_out_instruction": pair["held_out"]["raw_policy_instruction"],
                "canonical_success": int(canonical),
                "held_out_success": int(held_out),
                "paired_outcome": outcome,
            }
        )
    canonical_only = counts["canonical_only_success"]
    held_out_only = counts["held_out_only_success"]
    return (
        {
            **dict(counts),
            "discordant_pairs": canonical_only + held_out_only,
            "exact_mcnemar_p": exact_mcnemar_p(canonical_only, held_out_only),
        },
        sorted(details, key=lambda row: int(row["pair_id"])),
    )


def confusion_rows(rows: list[dict], condition: str) -> list[dict]:
    selected = [row for row in rows if row["language_condition"] == condition]
    output = []
    for truth in TASKS:
        truth_rows = [row for row in selected if row["task_bucket"] == truth]
        for prediction in TASKS:
            count = sum(
                row["initial_predicted_task_bucket"] == prediction
                for row in truth_rows
            )
            output.append(
                {
                    "condition": condition,
                    "true_task": truth,
                    "predicted_task": prediction,
                    "count": count,
                    "row_rate": count / len(truth_rows),
                }
            )
    return output


def plot_success(summary_rows: list[dict], output_dir: Path) -> None:
    labels = ["Overall", "Pick A", "Pick B", "Pick C", "Push A", "Push B", "Push C"]
    keys = ["overall", *TASKS]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for offset, condition in zip((-width / 2, width / 2), CONDITIONS):
        selected = {
            row["task"]: row
            for row in summary_rows
            if row["condition"] == condition
        }
        values = np.asarray([selected[key]["success_rate"] for key in keys])
        lower = np.maximum(
            0.0,
            values - np.asarray([selected[key]["ci95_lower"] for key in keys]),
        )
        upper = np.maximum(
            0.0,
            np.asarray([selected[key]["ci95_upper"] for key in keys]) - values,
        )
        bars = ax.bar(
            x + offset,
            values * 100.0,
            width,
            color=COLORS[condition],
            label="Canonical" if condition == "canonical" else "Held-out language",
            yerr=np.vstack([lower, upper]) * 100.0,
            capsize=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{value * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Task success rate (%)")
    ax.set_xticks(x, labels)
    ax.set_title("Canonical vs Held-out Language Generalization")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "01_language_success_comparison.png", dpi=220)
    fig.savefig(output_dir / "01_language_success_comparison.pdf")
    plt.close(fig)


def plot_confusions(confusions: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), constrained_layout=True)
    short = ["Pk-A", "Pk-B", "Pk-C", "Ps-A", "Ps-B", "Ps-C"]
    for ax, condition in zip(axes, CONDITIONS):
        selected = [row for row in confusions if row["condition"] == condition]
        lookup = {
            (row["true_task"], row["predicted_task"]): float(row["row_rate"])
            for row in selected
        }
        matrix = np.asarray(
            [[lookup[(truth, prediction)] for prediction in TASKS] for truth in TASKS]
        )
        image = ax.imshow(matrix * 100.0, vmin=0.0, vmax=100.0, cmap="Blues")
        for row_index in range(len(TASKS)):
            for column_index in range(len(TASKS)):
                value = matrix[row_index, column_index] * 100.0
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white" if value > 55 else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(len(TASKS)), short)
        ax.set_yticks(range(len(TASKS)), short)
        ax.set_xlabel("Initial predicted task")
        ax.set_ylabel("Ground-truth task")
        ax.set_title("Canonical" if condition == "canonical" else "Held-out language")
    fig.colorbar(image, ax=axes, label="Row-normalized rate (%)", shrink=0.82)
    fig.savefig(output_dir / "02_language_task_confusion.png", dpi=220)
    fig.savefig(output_dir / "02_language_task_confusion.pdf")
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary_rows: list[dict],
    paired: dict,
    audit_payload: dict,
) -> None:
    overall = {
        row["condition"]: row
        for row in summary_rows
        if row["task"] == "overall"
    }
    canonical = overall["canonical"]["success_rate"]
    held_out = overall["held_out"]["success_rate"]
    delta = held_out - canonical
    report = f"""# MiniVLA V3 Language Generalization

## Protocol

- 180 evaluation-only expressions: 30 for each of six task buckets.
- Each held-out instruction is paired with its canonical instruction on the exact same scene.
- The raw text is passed directly to DistilBERT; no UI resolver or canonical rewriting is used.
- Ground-truth task metadata is restricted to scene generation and scoring. Action execution uses model-predicted phase and target only.

## Main Result

- Canonical success: {canonical * 100:.2f}% ({overall['canonical']['successes']}/180)
- Held-out success: {held_out * 100:.2f}% ({overall['held_out']['successes']}/180)
- Held-out minus canonical: {delta * 100:+.2f} percentage points
- Held-out retention: {(held_out / canonical * 100 if canonical else float('nan')):.2f}%
- Paired McNemar p-value: {paired['exact_mcnemar_p']:.6g}
- Canonical-only successes: {paired.get('canonical_only_success', 0)}
- Held-out-only successes: {paired.get('held_out_only_success', 0)}

## Semantic Routing

- Canonical initial six-task accuracy: {overall['canonical']['initial_task_bucket_accuracy'] * 100:.2f}%
- Held-out initial six-task accuracy: {overall['held_out']['initial_task_bucket_accuracy'] * 100:.2f}%
- Held-out operation accuracy: {overall['held_out']['initial_task_type_accuracy'] * 100:.2f}%
- Held-out target accuracy: {overall['held_out']['initial_target_accuracy'] * 100:.2f}%

## Audit

Protocol audit: {audit_payload['status']}. See `protocol_audit.json` for machine-readable checks.
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_dir / "benchmark_episodes.csv")
    with (input_dir / "benchmark_summary.json").open(encoding="utf-8") as handle:
        benchmark_summary = json.load(handle)

    audit_payload = audit(rows, benchmark_summary)
    summaries = success_summary(rows)
    paired, pair_details = paired_outcomes(rows)
    confusions = [
        *confusion_rows(rows, "canonical"),
        *confusion_rows(rows, "held_out"),
    ]
    failure_rows = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["language_condition"] == condition]
        for category, count in sorted(Counter(row["failure_category"] for row in selected).items()):
            failure_rows.append(
                {
                    "condition": condition,
                    "failure_category": category,
                    "count": count,
                    "rate": count / len(selected),
                }
            )

    write_json(output_dir / "protocol_audit.json", audit_payload)
    write_json(output_dir / "paired_language_test.json", paired)
    write_csv(output_dir / "success_summary.csv", summaries)
    write_csv(output_dir / "paired_episode_outcomes.csv", pair_details)
    write_csv(output_dir / "task_confusion_matrix.csv", confusions)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)
    plot_success(summaries, output_dir)
    plot_confusions(confusions, output_dir)
    write_report(output_dir, summaries, paired, audit_payload)
    print(f"LANGUAGE GENERALIZATION AUDIT: {audit_payload['status']}", flush=True)
    print(f"Paper artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
