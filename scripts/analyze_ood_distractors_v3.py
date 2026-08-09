#!/usr/bin/env python3
"""Audit and plot paired MiniVLA V3 OOD-distractor benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
COUNTS = (0, 1, 2, 3)
COLOR = "#00876C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="final_report/08_ood_distractors",
    )
    return parser.parse_args()


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true"}


def as_float(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return float("nan")
    return float(value)


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
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Refusing to write headerless CSV: {path}")
    columns = list(rows[0]) if rows else list(fieldnames or [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rows_at_count(rows: list[dict], count: int) -> list[dict]:
    return [row for row in rows if int(row["benchmark_count"]) == count]


def episode_signature(row: dict) -> tuple:
    return (
        row["episode"],
        row["task_type"],
        row["target_id"],
        row["scene_seed"],
        row["ood_layout_signature"],
    )


def load_and_audit(input_dirs: list[str]) -> tuple[list[dict], dict, list[dict]]:
    all_rows: list[dict] = []
    sources: list[dict] = []
    reference_protocol: tuple | None = None

    for raw_dir in input_dirs:
        root = Path(raw_dir)
        summary_path = root / "benchmark_summary.json"
        episodes_path = root / "benchmark_episodes.csv"
        if not summary_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"Incomplete OOD benchmark: {root}")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if summary.get("status") != "complete":
            raise ValueError(f"OOD benchmark is not complete: {root}")
        if summary.get("protocol_version") != "ood-distractor.v1":
            raise ValueError(f"Unexpected OOD protocol version: {root}")
        if list(summary.get("counts", [])) != list(COUNTS):
            raise ValueError(f"Expected counts {COUNTS}: {root}")
        expected = int(summary["episodes_per_count"]) * len(COUNTS)
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} rows in {root}, found {len(rows)}")
        if not summary.get("paired_scene_validation_passed"):
            raise ValueError(f"Paired-scene validation failed: {root}")
        if not summary.get("paired_layout_validation_passed"):
            raise ValueError(f"Paired-layout validation failed: {root}")
        if any(as_bool(row, "uses_privileged_execution_assistance") for row in rows):
            raise ValueError(f"Privileged execution assistance detected: {root}")
        if any(
            not as_bool(row, "ood_initial_target_visibility_clear")
            or not as_bool(row, "ood_robot_path_clear")
            or not as_bool(row, "ood_initial_collision_integrity_passed")
            or not as_bool(row, "ood_application_verified")
            or not as_bool(row, "ood_restoration_verified")
            for row in rows
        ):
            raise ValueError(f"OOD placement safety audit failed: {root}")
        for row in rows:
            visible = json.loads(row["ood_visible_cameras_json"])
            if set(visible) != {"D1", "D2", "D3"} or any(
                not cameras for cameras in visible.values()
            ):
                raise ValueError(f"OOD render visibility audit failed: {root}")

        baseline = rows_at_count(rows, 0)
        expected_per_count = int(summary["episodes_per_count"])
        if len(baseline) != expected_per_count:
            raise ValueError(f"Invalid OOD count-0 baseline: {root}")
        baseline_signature = [episode_signature(row) for row in baseline]
        for count in COUNTS:
            group = rows_at_count(rows, count)
            if len(group) != expected_per_count:
                raise ValueError(f"Invalid OOD count {count}: {root}")
            if [episode_signature(row) for row in group] != baseline_signature:
                raise ValueError(f"Scene/layout mismatch at count {count}: {root}")
            expected_ids = "|".join(("D1", "D2", "D3")[:count])
            if any(row["ood_active_ids"] != expected_ids for row in group):
                raise ValueError(f"Prefix activation mismatch at count {count}: {root}")
            if any(int(float(row["ood_distractor_count"])) != count for row in group):
                raise ValueError(f"Count metric mismatch at count {count}: {root}")
            if count == 0 and any(
                as_bool(row, "injected")
                or as_bool(row, "ood_collision")
                or as_bool(row, "ood_target_selection_failure")
                or not np.isclose(
                    as_float(row, "ood_initial_policy_frame_mae"),
                    0.0,
                )
                for row in group
            ):
                raise ValueError(f"Count-0 identity audit failed: {root}")
            if count > 0 and any(not as_bool(row, "injected") for row in group):
                raise ValueError(f"OOD distractors not injected at count {count}: {root}")
            if count > 0 and any(
                as_float(row, "ood_initial_policy_frame_mae") <= 0.0
                for row in group
            ):
                raise ValueError(f"OOD distractors did not alter policy views: {root}")

        protocol = (
            summary["protocol_version"],
            summary["policy"],
            summary["episodes_per_count"],
            summary["ensemble_mode"],
            summary["temporal_profile"],
        )
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"Protocol mismatch in {root}")

        source = f"seed_{summary['seed']}"
        all_rows.extend({"source_run": source, **row} for row in rows)
        sources.append(
            {
                "source_run": source,
                "input_dir": str(root),
                "scene_seed": summary["seed"],
                "perturbation_seed": summary["perturbation_seed"],
                "episode_rows": len(rows),
            }
        )

    if len({source["scene_seed"] for source in sources}) != len(sources):
        raise ValueError("Input directories must use distinct scene seeds")
    audit = {
        "status": "PASS",
        "protocol_version": reference_protocol[0],
        "policy": reference_protocol[1],
        "seeds": len(sources),
        "episode_rows": len(all_rows),
        "paired_scene_validation_passed": True,
        "paired_layout_validation_passed": True,
        "prefix_activation_validation_passed": True,
        "initial_visibility_validation_passed": True,
        "robot_path_validation_passed": True,
        "initial_collision_integrity_passed": True,
        "render_visibility_validation_passed": True,
        "application_verification_rate": 1.0,
        "restoration_verification_rate": 1.0,
        "count_zero_identity_passed": True,
        "counts": list(COUNTS),
    }
    return all_rows, audit, sources


def summarize_success(rows: list[dict]) -> list[dict]:
    summaries = []
    for count in COUNTS:
        group = rows_at_count(rows, count)
        task_successes = sum(as_bool(row, "task_success") for row in group)
        clean_successes = sum(as_bool(row, "ood_clean_success") for row in group)
        lower, upper = wilson_interval(task_successes, len(group))
        clean_lower, clean_upper = wilson_interval(clean_successes, len(group))
        summaries.append(
            {
                "count": count,
                "episodes": len(group),
                "task_successes": task_successes,
                "task_success_rate": task_successes / len(group),
                "task_ci95_lower": lower,
                "task_ci95_upper": upper,
                "collision_aware_successes": clean_successes,
                "collision_aware_success_rate": clean_successes / len(group),
                "collision_aware_ci95_lower": clean_lower,
                "collision_aware_ci95_upper": clean_upper,
                "ood_wrong_object_contact_rate": float(
                    np.mean([as_bool(row, "ood_wrong_object_contact") for row in group])
                ),
                "ood_target_selection_failure_rate": float(
                    np.mean([as_bool(row, "ood_target_selection_failure") for row in group])
                ),
                "ood_target_collision_rate": float(
                    np.mean([as_bool(row, "ood_target_collision") for row in group])
                ),
                "ood_any_collision_rate": float(
                    np.mean([as_bool(row, "ood_collision") for row in group])
                ),
                "mean_grounding_error_cm": float(
                    np.mean([as_float(row, "mean_grounding_error_cm") for row in group])
                ),
                "mean_steps": float(np.mean([as_float(row, "steps") for row in group])),
            }
        )
    return summaries


def paired_tests(rows: list[dict], outcome: str) -> list[dict]:
    results = []
    raw_p = []
    for count in COUNTS[1:]:
        first_only = second_only = 0
        for source in sorted({row["source_run"] for row in rows}):
            source_rows = [row for row in rows if row["source_run"] == source]
            baseline = {
                row["episode"]: as_bool(row, outcome)
                for row in rows_at_count(source_rows, 0)
            }
            shifted = {
                row["episode"]: as_bool(row, outcome)
                for row in rows_at_count(source_rows, count)
            }
            for episode_id in baseline:
                first_only += int(baseline[episode_id] and not shifted[episode_id])
                second_only += int(shifted[episode_id] and not baseline[episode_id])
        p_value = exact_mcnemar_p(first_only, second_only)
        raw_p.append(p_value)
        results.append(
            {
                "outcome": outcome,
                "count": count,
                "count_zero_only_successes": first_only,
                "shifted_only_successes": second_only,
                "mcnemar_p": p_value,
            }
        )
    for row, adjusted in zip(results, holm_adjust(raw_p)):
        row["holm_p"] = adjusted
    return results


def task_summary(rows: list[dict]) -> list[dict]:
    output = []
    for count in COUNTS:
        group = rows_at_count(rows, count)
        for task in TASKS:
            task_type, target_id = task.split("_")
            selected = [
                row
                for row in group
                if row["task_type"] == task_type and row["target_id"] == target_id
            ]
            task_rate = (
                float(np.mean([as_bool(row, "task_success") for row in selected]))
                if selected
                else float("nan")
            )
            clean_rate = (
                float(np.mean([as_bool(row, "ood_clean_success") for row in selected]))
                if selected
                else float("nan")
            )
            output.append(
                {
                    "count": count,
                    "task": task,
                    "episodes": len(selected),
                    "task_success_rate": task_rate,
                    "collision_aware_success_rate": clean_rate,
                }
            )
    return output


def failure_summary(rows: list[dict]) -> list[dict]:
    output = []
    for count in COUNTS:
        group = rows_at_count(rows, count)
        failures = Counter(
            row["ood_failure_category"]
            for row in group
            if row["ood_failure_category"] != "success"
        )
        for category, value in sorted(failures.items()):
            output.append(
                {
                    "count": count,
                    "failure_category": category,
                    "count_value": value,
                    "rate_over_all_episodes": value / len(group),
                }
            )
    return output


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_success(summary: list[dict], output_dir: Path) -> None:
    x = np.asarray([row["count"] for row in summary])
    task = np.asarray([row["task_success_rate"] for row in summary]) * 100.0
    clean = np.asarray([row["collision_aware_success_rate"] for row in summary]) * 100.0
    lower = task - np.asarray([row["task_ci95_lower"] for row in summary]) * 100.0
    upper = np.asarray([row["task_ci95_upper"] for row in summary]) * 100.0 - task
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.errorbar(
        x,
        task,
        yerr=[lower, upper],
        marker="o",
        linewidth=2.5,
        capsize=5,
        color=COLOR,
        label="Task success",
    )
    ax.plot(x, clean, marker="s", linewidth=2.2, color="#D43D51", label="Collision-aware success")
    for index, value in enumerate(task):
        ax.text(x[index], value + 2.0, f"{value:.1f}", ha="center")
    ax.set_xticks(x, [f"{value} distractors" for value in x])
    ax.set_ylim(0, 107)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("OOD distractor count")
    ax.set_title("MiniVLA V3 Robustness to Unseen Distractors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "01_ood_distractor_success_decay")


def plot_diagnostics(summary: list[dict], output_dir: Path) -> None:
    x = np.asarray([row["count"] for row in summary])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for key, label, marker in (
        ("ood_target_selection_failure_rate", "Target-selection failure", "o"),
        ("ood_wrong_object_contact_rate", "Robot-distractor contact", "s"),
        ("ood_target_collision_rate", "Target-distractor collision", "^"),
    ):
        axes[0].plot(
            x,
            np.asarray([row[key] for row in summary]) * 100.0,
            marker=marker,
            linewidth=2.2,
            label=label,
        )
    axes[0].set_ylabel("Episode rate (%)")
    axes[0].set_xlabel("OOD distractor count")
    axes[0].set_xticks(x)
    axes[0].set_title("OOD-specific failure signals")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(
        x,
        [row["mean_grounding_error_cm"] for row in summary],
        marker="o",
        linewidth=2.5,
        color="#4C78A8",
    )
    axes[1].set_ylabel("Mean grounding error (cm)")
    axes[1].set_xlabel("OOD distractor count")
    axes[1].set_xticks(x)
    axes[1].set_title("Visual target grounding")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("OOD Distractor Diagnostics")
    fig.tight_layout()
    save_figure(fig, output_dir, "02_ood_distractor_diagnostics")


def plot_task_heatmap(summary: list[dict], output_dir: Path) -> None:
    lookup = {(row["task"], row["count"]): row for row in summary}
    matrix = np.asarray(
        [
            [100.0 * lookup[(task, count)]["task_success_rate"] for count in COUNTS]
            for task in TASKS
        ]
    )
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    image = ax.imshow(matrix, vmin=0.0, vmax=100.0, cmap="YlGnBu", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.0f}",
                ha="center",
                va="center",
                color="black",
            )
    ax.set_xticks(range(len(COUNTS)), [str(count) for count in COUNTS])
    ax.set_yticks(range(len(TASKS)), [task.replace("_", "-") for task in TASKS])
    ax.set_xlabel("OOD distractor count")
    ax.set_title("Per-Task Success Under OOD Distractors")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Task success rate (%)")
    fig.tight_layout()
    save_figure(fig, output_dir, "03_ood_distractor_task_heatmap")


def plot_failures(summary: list[dict], output_dir: Path) -> None:
    categories = sorted({row["failure_category"] for row in summary})
    lookup = {
        (row["count"], row["failure_category"]): 100.0 * row["rate_over_all_episodes"]
        for row in summary
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bottom = np.zeros(len(COUNTS))
    for category in categories:
        values = np.asarray([lookup.get((count, category), 0.0) for count in COUNTS])
        ax.bar(COUNTS, values, bottom=bottom, label=category.replace("_", " "))
        bottom += values
    ax.set_xticks(COUNTS)
    ax.set_xlabel("OOD distractor count")
    ax.set_ylabel("Failure rate over all episodes (%)")
    ax.set_title("Failure Taxonomy Under OOD Distractors")
    if categories:
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_ood_distractor_failure_taxonomy")


def write_report(
    output_dir: Path,
    audit: dict,
    summary: list[dict],
    tests: list[dict],
) -> None:
    tests_by_count = {
        row["count"]: row for row in tests if row["outcome"] == "task_success"
    }
    lines = [
        "# MiniVLA V3 OOD-Distractor Robustness",
        "",
        "## Protocol Audit",
        "",
        "- Status: PASS.",
        f"- Raw episode rows: {audit['episode_rows']}.",
        f"- Seeds: {audit['seeds']}.",
        "- All counts share the same task, scene seed, and three-candidate layout.",
        "- Higher counts activate a strict D1/D2/D3 prefix.",
        "- Initial target visibility, nominal robot path, and collision integrity verified.",
        "- No privileged execution assistance.",
        "",
        "## Success Rates",
        "",
        "| OOD objects | Episodes | Task success | Collision-aware | "
        "OOD contact | Selection failure | Holm p vs 0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        test = tests_by_count.get(row["count"])
        p_value = "reference" if test is None else f"{test['holm_p']:.4g}"
        lines.append(
            f"| {row['count']} | {row['episodes']} | "
            f"{100.0 * row['task_success_rate']:.2f}% | "
            f"{100.0 * row['collision_aware_success_rate']:.2f}% | "
            f"{100.0 * row['ood_wrong_object_contact_rate']:.2f}% | "
            f"{100.0 * row['ood_target_selection_failure_rate']:.2f}% | "
            f"{p_value} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Task success measures whether the commanded manipulation completed.",
            "- Collision-aware success additionally rejects contacts with OOD "
            "or wrong trained objects.",
            "- Target-selection failure requires at least three consecutive "
            "predictions closer to an OOD object than to the true target.",
            "- Static distractors isolate perception and path-selection "
            "robustness; they are not a dynamic-obstacle benchmark.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, audit, sources = load_and_audit(args.input_dirs)
    summary = summarize_success(rows)
    tests = paired_tests(rows, "task_success") + paired_tests(rows, "ood_clean_success")
    tasks = task_summary(rows)
    failures = failure_summary(rows)

    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "source_runs.csv", sources)
    write_csv(output_dir / "success_summary.csv", summary)
    write_csv(output_dir / "paired_count_zero_tests.csv", tests)
    write_csv(output_dir / "task_success_summary.csv", tasks)
    write_csv(
        output_dir / "failure_taxonomy.csv",
        failures,
        fieldnames=[
            "count",
            "failure_category",
            "count_value",
            "rate_over_all_episodes",
        ],
    )
    plot_success(summary, output_dir)
    plot_diagnostics(summary, output_dir)
    plot_task_heatmap(tasks, output_dir)
    plot_failures(failures, output_dir)
    write_report(output_dir, audit, summary, tests)
    print(
        f"OOD distractor protocol audit: PASS ({len(rows)} episode rows)",
        flush=True,
    )
    print(f"Analysis saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
