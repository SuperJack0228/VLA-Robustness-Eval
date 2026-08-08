#!/usr/bin/env python3
"""Aggregate paired V3 target-displacement benchmarks and make paper figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = ("temporal", "latest-only")
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
FAILURE_ORDER = (
    "insufficient_push_distance",
    "recovery_limit",
    "target_not_contacted",
    "gripper_never_closed",
    "insufficient_lift",
    "wrong_object_contact",
    "lateral_push_error",
    "grasp_failed_after_contact",
)
MODE_COLORS = {"temporal": "#16697A", "latest-only": "#E07A1F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Completed benchmark directories to combine.",
    )
    parser.add_argument(
        "--output-dir",
        default="final_report/02_dynamic_displacement/appendix_v3_only",
    )
    return parser.parse_args()


def as_bool(row: dict[str, str], key: str) -> bool:
    return row[key] in {"1", "True", "true"}


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def exact_mcnemar_p(temporal_only: int, latest_only: int) -> float:
    discordant = temporal_only + latest_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(temporal_only, latest_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate(input_dirs: list[str]) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    run_metadata: list[dict] = []
    reference_protocol: tuple | None = None

    for seed_index, raw_dir in enumerate(input_dirs, start=1):
        root = Path(raw_dir)
        summary_path = root / "benchmark_summary.json"
        episodes_path = root / "benchmark_episodes.csv"
        if not summary_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"Incomplete benchmark directory: {root}")

        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if summary["status"] != "complete":
            raise ValueError(f"Benchmark is not complete: {root}")
        if not summary["paired_scene_validation_passed"]:
            raise ValueError(f"Paired-scene validation failed: {root}")

        expected = (
            len(summary["levels_m"])
            * len(summary["ensemble_modes"])
            * int(summary["episodes_per_level"])
        )
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} rows in {root}, found {len(rows)}")
        if any(not as_bool(row, "protocol_collision_integrity_passed") for row in rows):
            raise ValueError(f"Collision-integrity failure in {root}")
        if any(
            float(row["benchmark_level"]) > 0.0 and not as_bool(row, "injected")
            for row in rows
        ):
            raise ValueError(f"A nonzero perturbation was not injected in {root}")
        if any(as_bool(row, "uses_privileged_execution_assistance") for row in rows):
            raise ValueError(f"Privileged execution assistance was enabled in {root}")
        if any(
            abs(float(row["actual_delta_norm_m"]) - float(row["requested_delta_m"]))
            > 1e-6
            or abs(float(row["actual_delta_z_m"])) > 1e-8
            for row in rows
        ):
            raise ValueError(f"Injected displacement does not match the protocol in {root}")

        protocol = (
            tuple(summary["levels_m"]),
            tuple(summary["ensemble_modes"]),
            summary["protocol_version"],
            summary["policy"],
            summary["temporal_profile"],
            summary["max_prediction_age"],
            summary["grounding_reset_threshold_m"],
        )
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"Protocol mismatch in {root}")

        source_id = f"seed_{seed_index}"
        for row in rows:
            enriched = {
                "source_run": source_id,
                "benchmark_seed": summary["seed"],
                "perturbation_seed": summary["perturbation_seed"],
                **row,
            }
            all_rows.append(enriched)
        run_metadata.append(
            {
                "source_run": source_id,
                "input_dir": str(root),
                "scene_seed": summary["seed"],
                "perturbation_seed": summary["perturbation_seed"],
                "episodes": len(rows),
            }
        )

    return all_rows, run_metadata


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    levels = sorted({float(row["benchmark_level"]) for row in rows})
    summary_rows: list[dict] = []
    task_rows: list[dict] = []
    failure_rows: list[dict] = []

    for level in levels:
        for mode in MODES:
            group = [
                row
                for row in rows
                if float(row["benchmark_level"]) == level
                and row["benchmark_mode"] == mode
            ]
            successes = sum(as_bool(row, "task_success") for row in group)
            clean_successes = sum(as_bool(row, "clean_success") for row in group)
            lower, upper = wilson_interval(successes, len(group))
            injected = [row for row in group if as_bool(row, "injected")]
            reacquired = (
                [row for row in injected if as_bool(row, "reacquired_within_0_5cm")]
                if injected
                else []
            )
            summary_rows.append(
                {
                    "level_cm": int(round(level * 100)),
                    "ensemble_mode": mode,
                    "episodes": len(group),
                    "successes": successes,
                    "task_success_rate": successes / len(group),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "clean_success_rate": clean_successes / len(group),
                    "strict_target_contact_rate": mean(
                        as_bool(row, "target_contact") for row in group
                    ),
                    "wrong_object_contact_rate": mean(
                        as_bool(row, "wrong_object_contact") for row in group
                    ),
                    "mean_steps": mean(float(row["steps"]) for row in group),
                    "mean_grounding_error_after_injection_cm": (
                        mean(float(row["grounding_error_after_injection_cm"]) for row in injected)
                        if injected
                        else ""
                    ),
                    "reacquisition_rate_0_5cm": (
                        len(reacquired) / len(injected) if injected else ""
                    ),
                    "mean_successful_reacquisition_latency_steps_0_5cm": (
                        mean(float(row["reacquisition_latency_0_5cm"]) for row in reacquired)
                        if reacquired
                        else ""
                    ),
                }
            )

            for task in TASKS:
                task_type, target_id = task.split("_")
                task_group = [
                    row
                    for row in group
                    if row["task_type"] == task_type and row["target_id"] == target_id
                ]
                task_successes = sum(as_bool(row, "task_success") for row in task_group)
                task_lower, task_upper = wilson_interval(task_successes, len(task_group))
                task_rows.append(
                    {
                        "level_cm": int(round(level * 100)),
                        "ensemble_mode": mode,
                        "task": task,
                        "episodes": len(task_group),
                        "successes": task_successes,
                        "task_success_rate": task_successes / len(task_group),
                        "ci95_lower": task_lower,
                        "ci95_upper": task_upper,
                    }
                )

            counts = Counter(
                row["failure_category"]
                for row in group
                if not as_bool(row, "task_success")
            )
            for category, count in sorted(counts.items()):
                failure_rows.append(
                    {
                        "level_cm": int(round(level * 100)),
                        "ensemble_mode": mode,
                        "failure_category": category,
                        "count": count,
                        "rate_over_all_episodes": count / len(group),
                    }
                )

    paired: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["source_run"], int(round(float(row["benchmark_level"]) * 100)), int(row["episode"]))
        paired[key][row["benchmark_mode"]] = row

    comparison_rows: list[dict] = []
    for level_cm in sorted({key[1] for key in paired}):
        temporal_only = latest_only = both_success = both_fail = 0
        for (source_run, paired_level, episode), outcomes in paired.items():
            if paired_level != level_cm:
                continue
            if set(outcomes) != set(MODES):
                raise ValueError(
                    f"Unpaired mode result at {level_cm} cm: "
                    f"{source_run}, episode {episode}"
                )
            temporal_row = outcomes["temporal"]
            latest_row = outcomes["latest-only"]
            paired_fields = (
                "scene_seed",
                "task_type",
                "target_id",
                "requested_delta_m",
                "selected_direction_x",
                "selected_direction_y",
                "injection_phase",
            )
            if any(temporal_row[field] != latest_row[field] for field in paired_fields):
                raise ValueError(
                    f"Paired protocol mismatch at {level_cm} cm: "
                    f"{source_run}, episode {episode}"
                )
            temporal_success = as_bool(outcomes["temporal"], "task_success")
            latest_success = as_bool(outcomes["latest-only"], "task_success")
            if temporal_success and not latest_success:
                temporal_only += 1
            elif latest_success and not temporal_success:
                latest_only += 1
            elif temporal_success:
                both_success += 1
            else:
                both_fail += 1
        total = temporal_only + latest_only + both_success + both_fail
        comparison_rows.append(
            {
                "level_cm": level_cm,
                "paired_episodes": total,
                "temporal_success_rate": (temporal_only + both_success) / total,
                "latest_only_success_rate": (latest_only + both_success) / total,
                "difference_percentage_points": 100.0 * (temporal_only - latest_only) / total,
                "temporal_only_successes": temporal_only,
                "latest_only_successes": latest_only,
                "both_success": both_success,
                "both_fail": both_fail,
                "exact_mcnemar_p": exact_mcnemar_p(temporal_only, latest_only),
            }
        )

    return summary_rows, task_rows, failure_rows, comparison_rows


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(
    summary_rows: list[dict],
    task_rows: list[dict],
    failure_rows: list[dict],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode in MODES:
        group = [row for row in summary_rows if row["ensemble_mode"] == mode]
        x = [row["level_cm"] for row in group]
        y = [100.0 * row["task_success_rate"] for row in group]
        lower = [100.0 * row["ci95_lower"] for row in group]
        upper = [100.0 * row["ci95_upper"] for row in group]
        label = "Temporal ensemble" if mode == "temporal" else "Latest prediction only"
        ax.plot(x, y, marker="o", linewidth=2.2, color=MODE_COLORS[mode], label=label)
        ax.fill_between(x, lower, upper, color=MODE_COLORS[mode], alpha=0.14)
    ax.axhline(80, color="#555555", linestyle="--", linewidth=1.1, label="80% gate")
    ax.set(xlabel="Target displacement (cm)", ylabel="Task success rate (%)", ylim=(55, 102))
    ax.set_xticks(range(9))
    ax.set_title("V3 Dynamic Target Displacement Robustness")
    ax.legend(loc="lower left", frameon=True)
    save_figure(fig, output_dir, "success_decay_curve")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for task_type, linestyle in (("pick", "-"), ("push", "--")):
        values = []
        for level_cm in range(9):
            group = [
                row
                for row in task_rows
                if row["ensemble_mode"] == "temporal"
                and row["level_cm"] == level_cm
                and row["task"].startswith(task_type)
            ]
            successes = sum(row["successes"] for row in group)
            episodes = sum(row["episodes"] for row in group)
            values.append(100.0 * successes / episodes)
        ax.plot(
            range(9),
            values,
            marker="o",
            linewidth=2.2,
            linestyle=linestyle,
            color="#16697A" if task_type == "pick" else "#C33C54",
            label=task_type.capitalize(),
        )
    ax.axhline(80, color="#555555", linestyle=":", linewidth=1.1)
    ax.set(xlabel="Target displacement (cm)", ylabel="Task success rate (%)", ylim=(45, 102))
    ax.set_xticks(range(9))
    ax.set_title("Temporal Ensemble: Pick vs Push Robustness")
    ax.legend(loc="lower left")
    save_figure(fig, output_dir, "task_family_decay_curve")

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    bottoms = [0.0] * 9
    palette = ("#D95F59", "#6B5B95", "#4C78A8", "#F2B134", "#59A14F", "#B07AA1", "#76B7B2", "#9C755F")
    for category, color in zip(FAILURE_ORDER, palette):
        values = []
        for level_cm in range(9):
            matches = [
                row
                for row in failure_rows
                if row["ensemble_mode"] == "temporal"
                and row["level_cm"] == level_cm
                and row["failure_category"] == category
            ]
            values.append(100.0 * matches[0]["rate_over_all_episodes"] if matches else 0.0)
        if any(values):
            ax.bar(
                range(9),
                values,
                bottom=bottoms,
                width=0.72,
                color=color,
                label=category.replace("_", " "),
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set(xlabel="Target displacement (cm)", ylabel="Failure rate over all episodes (%)")
    ax.set_xticks(range(9))
    ax.set_ylim(0, max(bottoms) * 1.15)
    ax.set_title("Temporal Ensemble Failure Taxonomy")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    save_figure(fig, output_dir, "failure_taxonomy_temporal")

    fig, (rate_ax, latency_ax) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for mode in MODES:
        group = [
            row
            for row in summary_rows
            if row["ensemble_mode"] == mode and row["level_cm"] > 0
        ]
        x = [row["level_cm"] for row in group]
        rate_ax.plot(
            x,
            [100.0 * row["reacquisition_rate_0_5cm"] for row in group],
            marker="o",
            linewidth=2,
            color=MODE_COLORS[mode],
            label=mode,
        )
        latency_ax.plot(
            x,
            [row["mean_successful_reacquisition_latency_steps_0_5cm"] / 20.0 for row in group],
            marker="o",
            linewidth=2,
            color=MODE_COLORS[mode],
            label=mode,
        )
    rate_ax.set(xlabel="Displacement (cm)", ylabel="Reacquisition rate (%)", ylim=(55, 102))
    latency_ax.set(xlabel="Displacement (cm)", ylabel="Successful reacquisition latency (s)")
    rate_ax.set_title("Within 0.5 cm")
    latency_ax.set_title("20 Hz control loop")
    rate_ax.legend(loc="lower left")
    fig.suptitle("Post-displacement Visual Reacquisition")
    fig.tight_layout()
    save_figure(fig, output_dir, "reacquisition_metrics")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, run_metadata = load_and_validate(args.input_dirs)
    summary_rows, task_rows, failure_rows, comparison_rows = summarize(rows)

    write_csv(output_dir / "combined_episodes.csv", rows)
    write_csv(output_dir / "success_summary.csv", summary_rows)
    write_csv(output_dir / "task_success_summary.csv", task_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)
    write_csv(output_dir / "paired_mode_comparison.csv", comparison_rows)
    make_plots(summary_rows, task_rows, failure_rows, output_dir)

    report = {
        "status": "complete",
        "input_runs": run_metadata,
        "total_episode_rows": len(rows),
        "paired_scenes_per_level": comparison_rows[0]["paired_episodes"],
        "protocol_validation_passed": True,
        "success_summary": summary_rows,
        "paired_mode_comparison": comparison_rows,
    }
    with (output_dir / "combined_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Combined {len(rows)} episode rows from {len(run_metadata)} seed schedules.")
    print(f"Analysis and paper figures saved to {output_dir}")


if __name__ == "__main__":
    main()
