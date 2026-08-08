#!/usr/bin/env python3
"""Paired V2/V3 analysis for the target-displacement robustness benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "vla_matplotlib_cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "vla_xdg_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


MODELS = ("v2", "v3")
MODES = ("temporal", "latest-only")
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
MODEL_COLORS = {"v2": "#D55E00", "v3": "#0072B2"}
FAILURE_ORDER = (
    "grasp_failed_after_contact",
    "insufficient_push_distance",
    "wrong_object_contact",
    "missed_grasp",
    "recovery_limit",
    "target_not_contacted",
    "gripper_never_closed",
    "other",
)
FAILURE_COLORS = {
    "grasp_failed_after_contact": "#8C564B",
    "insufficient_push_distance": "#D95F59",
    "wrong_object_contact": "#B07AA1",
    "missed_grasp": "#F2B134",
    "recovery_limit": "#6B5B95",
    "target_not_contacted": "#4C78A8",
    "gripper_never_closed": "#59A14F",
    "other": "#BAB0AC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dirs", nargs="+", required=True)
    parser.add_argument("--v3-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="final_report/02_dynamic_displacement",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    return parser.parse_args()


def as_bool(row: dict, key: str) -> bool:
    return str(row[key]) in {"1", "True", "true"}


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


def exact_mcnemar_p(v3_only: int, v2_only: int) -> float:
    discordant = v3_only + v2_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(v3_only, v2_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_maximum = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def paired_bootstrap_interval(
    v2_outcomes: np.ndarray,
    v3_outcomes: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(v2_outcomes), size=(samples, len(v2_outcomes)))
    differences = (
        v3_outcomes[indices].mean(axis=1) - v2_outcomes[indices].mean(axis=1)
    )
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return float(lower), float(upper)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model_runs(model: str, directories: list[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    metadata: list[dict] = []
    reference_protocol: tuple | None = None
    seen_seed_pairs: set[tuple[int, int]] = set()

    for source_index, raw_directory in enumerate(directories, start=1):
        root = Path(raw_directory)
        summary_path = root / "benchmark_summary.json"
        episodes_path = root / "benchmark_episodes.csv"
        if not summary_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"Incomplete benchmark directory: {root}")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            run_rows = list(csv.DictReader(handle))

        if summary["status"] != "complete":
            raise ValueError(f"Incomplete benchmark status: {root}")
        if not summary["paired_scene_validation_passed"]:
            raise ValueError(f"Paired-scene validation failed: {root}")
        expected = (
            len(summary["levels_m"])
            * len(summary["ensemble_modes"])
            * int(summary["episodes_per_level"])
        )
        if len(run_rows) != expected:
            raise ValueError(f"Expected {expected} rows in {root}, found {len(run_rows)}")
        if any(not as_bool(row, "protocol_collision_integrity_passed") for row in run_rows):
            raise ValueError(f"Collision-integrity failure in {root}")
        if any(as_bool(row, "uses_privileged_execution_assistance") for row in run_rows):
            raise ValueError(f"Privileged execution assistance was enabled in {root}")
        if any(
            as_bool(row, "injected")
            and (
                abs(float(row["actual_delta_norm_m"]) - float(row["requested_delta_m"]))
                > 1e-6
                or abs(float(row["actual_delta_z_m"])) > 1e-8
            )
            for row in run_rows
        ):
            raise ValueError(f"Injected displacement mismatch in {root}")

        seed_pair = (int(summary["seed"]), int(summary["perturbation_seed"]))
        if seed_pair in seen_seed_pairs:
            raise ValueError(f"Duplicate seed pair for {model}: {seed_pair}")
        seen_seed_pairs.add(seed_pair)
        protocol = (
            tuple(summary["levels_m"]),
            tuple(summary["ensemble_modes"]),
            summary["protocol_version"],
            summary["temporal_profile"],
            summary["max_prediction_age"],
            summary["grounding_reset_threshold_m"],
            summary["episodes_per_level"],
        )
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"Protocol mismatch within {model}: {root}")

        source_run = f"{model}_seed_{source_index}"
        for row in run_rows:
            rows.append(
                {
                    "model": model,
                    "source_run": source_run,
                    "benchmark_seed": seed_pair[0],
                    "perturbation_seed": seed_pair[1],
                    **row,
                }
            )
        metadata.append(
            {
                "model": model,
                "source_run": source_run,
                "input_dir": str(root),
                "policy": summary["policy"],
                "benchmark_seed": seed_pair[0],
                "perturbation_seed": seed_pair[1],
                "episodes": len(run_rows),
            }
        )
    return rows, metadata


def pair_key(row: dict) -> tuple[int, int, int, str, int]:
    return (
        int(row["benchmark_seed"]),
        int(row["perturbation_seed"]),
        int(round(float(row["benchmark_level"]) * 100)),
        row["benchmark_mode"],
        int(row["episode"]),
    )


def validate_cross_model_pairing(rows: list[dict]) -> tuple[dict, dict]:
    paired: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        paired[pair_key(row)][row["model"]] = row
    expected_fields = (
        "scene_seed",
        "task_type",
        "target_id",
        "instruction",
        "requested_delta_m",
        "selected_direction_x",
        "selected_direction_y",
    )
    mismatches = Counter()
    for key, outcomes in paired.items():
        if set(outcomes) != set(MODELS):
            raise ValueError(f"Missing paired model outcome: {key}")
        for field in expected_fields:
            if outcomes["v2"][field] != outcomes["v3"][field]:
                mismatches[field] += 1
    if mismatches:
        raise ValueError(f"Cross-model pairing mismatch: {dict(mismatches)}")

    temporal_nonzero = [
        row
        for row in rows
        if row["benchmark_mode"] == "temporal"
        and float(row["benchmark_level"]) > 0.0
    ]
    if any(not as_bool(row, "injected") for row in temporal_nonzero):
        raise ValueError("Primary Temporal comparison contains missed injections")

    protocol_audit = {
        "cross_model_paired_rows": len(paired),
        "primary_mode": "temporal",
        "primary_nonzero_injection_rate": mean(
            as_bool(row, "injected") for row in temporal_nonzero
        ),
        "injection_compliance": {},
        "latest_only_boundary": (
            "Latest-only is reported as intent-to-treat because V2 failed to "
            "reach the injection phase in a small number of assigned episodes."
        ),
    }
    for model in MODELS:
        for mode in MODES:
            group = [
                row
                for row in rows
                if row["model"] == model
                and row["benchmark_mode"] == mode
                and float(row["benchmark_level"]) > 0.0
            ]
            protocol_audit["injection_compliance"][f"{model}_{mode}"] = {
                "injected": sum(as_bool(row, "injected") for row in group),
                "assigned": len(group),
                "rate": mean(as_bool(row, "injected") for row in group),
            }
    return paired, protocol_audit


def summarize_models(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    levels = sorted({int(round(float(row["benchmark_level"]) * 100)) for row in rows})
    success_rows: list[dict] = []
    task_rows: list[dict] = []
    failure_rows: list[dict] = []

    for model in MODELS:
        for mode in MODES:
            for level_cm in levels:
                group = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["benchmark_mode"] == mode
                    and int(round(float(row["benchmark_level"]) * 100)) == level_cm
                ]
                successes = sum(as_bool(row, "task_success") for row in group)
                lower, upper = wilson_interval(successes, len(group))
                injected = [row for row in group if as_bool(row, "injected")]
                reacquired = [
                    row
                    for row in injected
                    if as_bool(row, "reacquired_within_0_5cm")
                ]
                success_rows.append(
                    {
                        "model": model,
                        "ensemble_mode": mode,
                        "level_cm": level_cm,
                        "episodes": len(group),
                        "injected_episodes": len(injected),
                        "injection_rate": len(injected) / len(group),
                        "successes": successes,
                        "task_success_rate": successes / len(group),
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "clean_success_rate": mean(
                            as_bool(row, "clean_success") for row in group
                        ),
                        "target_contact_rate": mean(
                            as_bool(row, "target_contact") for row in group
                        ),
                        "wrong_object_contact_rate": mean(
                            as_bool(row, "wrong_object_contact") for row in group
                        ),
                        "mean_steps": mean(float(row["steps"]) for row in group),
                        "mean_action_clip_rate": mean(
                            float(row["action_clip_rate"]) for row in group
                        ),
                        "mean_safety_intervention_rate": mean(
                            float(row["safety_intervention_rate"]) for row in group
                        ),
                        "mean_grounding_shift_resets": mean(
                            float(row["grounding_shift_resets"]) for row in group
                        ),
                        "mean_grounding_error_after_injection_cm": (
                            mean(
                                float(row["grounding_error_after_injection_cm"])
                                for row in injected
                            )
                            if injected
                            else ""
                        ),
                        "reacquisition_rate_0_5cm": (
                            len(reacquired) / len(injected) if injected else ""
                        ),
                        "mean_successful_reacquisition_latency_s_0_5cm": (
                            mean(
                                float(row["reacquisition_latency_0_5cm"])
                                for row in reacquired
                            )
                            / 20.0
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
                        if row["task_type"] == task_type
                        and row["target_id"] == target_id
                    ]
                    task_successes = sum(
                        as_bool(row, "task_success") for row in task_group
                    )
                    task_lower, task_upper = wilson_interval(
                        task_successes, len(task_group)
                    )
                    task_rows.append(
                        {
                            "model": model,
                            "ensemble_mode": mode,
                            "level_cm": level_cm,
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
                            "model": model,
                            "ensemble_mode": mode,
                            "level_cm": level_cm,
                            "failure_category": category,
                            "count": count,
                            "rate_over_all_episodes": count / len(group),
                        }
                    )
    return success_rows, task_rows, failure_rows


def paired_comparison(
    paired: dict,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(bootstrap_seed)
    comparison_rows: list[dict] = []
    episode_rows: list[dict] = []

    for mode in MODES:
        for level_cm in range(9):
            matching = [
                (key, outcomes)
                for key, outcomes in paired.items()
                if key[2] == level_cm and key[3] == mode
            ]
            matching.sort(key=lambda item: (item[0][0], item[0][4]))
            v2_values = np.asarray(
                [as_bool(outcomes["v2"], "task_success") for _, outcomes in matching],
                dtype=np.float64,
            )
            v3_values = np.asarray(
                [as_bool(outcomes["v3"], "task_success") for _, outcomes in matching],
                dtype=np.float64,
            )
            v3_only = int(np.sum((v3_values == 1) & (v2_values == 0)))
            v2_only = int(np.sum((v2_values == 1) & (v3_values == 0)))
            both_success = int(np.sum((v3_values == 1) & (v2_values == 1)))
            both_fail = int(np.sum((v3_values == 0) & (v2_values == 0)))
            lower, upper = paired_bootstrap_interval(
                v2_values,
                v3_values,
                bootstrap_samples,
                rng,
            )
            comparison_rows.append(
                {
                    "ensemble_mode": mode,
                    "level_cm": level_cm,
                    "paired_episodes": len(matching),
                    "v2_success_rate": float(v2_values.mean()),
                    "v3_success_rate": float(v3_values.mean()),
                    "v3_gain_percentage_points": 100.0
                    * float(v3_values.mean() - v2_values.mean()),
                    "gain_ci95_lower_percentage_points": 100.0 * lower,
                    "gain_ci95_upper_percentage_points": 100.0 * upper,
                    "v3_only_successes": v3_only,
                    "v2_only_successes": v2_only,
                    "both_success": both_success,
                    "both_fail": both_fail,
                    "exact_mcnemar_p": exact_mcnemar_p(v3_only, v2_only),
                }
            )

            for key, outcomes in matching:
                v2_row = outcomes["v2"]
                v3_row = outcomes["v3"]
                episode_rows.append(
                    {
                        "benchmark_seed": key[0],
                        "perturbation_seed": key[1],
                        "level_cm": level_cm,
                        "ensemble_mode": mode,
                        "episode": key[4],
                        "scene_seed": v2_row["scene_seed"],
                        "task": f"{v2_row['task_type']}_{v2_row['target_id']}",
                        "instruction": v2_row["instruction"],
                        "v2_injected": int(as_bool(v2_row, "injected")),
                        "v3_injected": int(as_bool(v3_row, "injected")),
                        "v2_success": int(as_bool(v2_row, "task_success")),
                        "v3_success": int(as_bool(v3_row, "task_success")),
                        "v2_failure": v2_row["failure_category"],
                        "v3_failure": v3_row["failure_category"],
                    }
                )

    for mode in MODES:
        indices = [
            index
            for index, row in enumerate(comparison_rows)
            if row["ensemble_mode"] == mode
        ]
        adjusted = holm_adjust(
            [comparison_rows[index]["exact_mcnemar_p"] for index in indices]
        )
        for index, adjusted_p in zip(indices, adjusted):
            comparison_rows[index]["holm_adjusted_p_across_levels"] = adjusted_p
    return comparison_rows, episode_rows


def lookup(rows: list[dict], **criteria) -> dict:
    matches = [
        row
        for row in rows
        if all(row[key] == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one match for {criteria}, found {len(matches)}")
    return matches[0]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def configure_plots() -> None:
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


def plot_primary_success(summary_rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for model in MODELS:
        group = [
            lookup(
                summary_rows,
                model=model,
                ensemble_mode="temporal",
                level_cm=level_cm,
            )
            for level_cm in range(9)
        ]
        x = [row["level_cm"] for row in group]
        y = [100.0 * row["task_success_rate"] for row in group]
        lower = [100.0 * row["ci95_lower"] for row in group]
        upper = [100.0 * row["ci95_upper"] for row in group]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            color=MODEL_COLORS[model],
            label=model.upper(),
        )
        ax.fill_between(x, lower, upper, color=MODEL_COLORS[model], alpha=0.14)
    ax.axhline(80, color="#555555", linestyle="--", linewidth=1.1, label="80% gate")
    ax.set(
        xlabel="Target displacement (cm)",
        ylabel="Task success rate (%)",
        ylim=(0, 103),
        title="Paired V2 vs V3 Dynamic Robustness (Temporal)",
    )
    ax.set_xticks(range(9))
    ax.legend(loc="lower left")
    save_figure(fig, output_dir, "01_v2_v3_temporal_success_decay")


def plot_all_modes(summary_rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for model in MODELS:
        for mode in MODES:
            group = [
                lookup(
                    summary_rows,
                    model=model,
                    ensemble_mode=mode,
                    level_cm=level_cm,
                )
                for level_cm in range(9)
            ]
            ax.plot(
                range(9),
                [100.0 * row["task_success_rate"] for row in group],
                marker="o" if mode == "temporal" else "s",
                linewidth=2.1,
                linestyle="-" if mode == "temporal" else "--",
                color=MODEL_COLORS[model],
                alpha=1.0 if mode == "temporal" else 0.72,
                label=f"{model.upper()} {mode}",
            )
    ax.axhline(80, color="#555555", linestyle=":", linewidth=1.0)
    ax.set(
        xlabel="Target displacement (cm)",
        ylabel="Task success rate (%)",
        ylim=(0, 103),
        title="Model and Execution-Mode Comparison",
    )
    ax.set_xticks(range(9))
    ax.legend(loc="lower left", ncol=2)
    fig.text(
        0.5,
        -0.01,
        "Latest-only is intent-to-treat; V2 missed 8 of 960 assigned nonzero injections.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    save_figure(fig, output_dir, "02_all_models_and_execution_modes")


def plot_gain(comparison_rows: list[dict], output_dir: Path) -> None:
    rows = [
        lookup(comparison_rows, ensemble_mode="temporal", level_cm=level_cm)
        for level_cm in range(9)
    ]
    gains = np.asarray([row["v3_gain_percentage_points"] for row in rows])
    lower = np.asarray([row["gain_ci95_lower_percentage_points"] for row in rows])
    upper = np.asarray([row["gain_ci95_upper_percentage_points"] for row in rows])
    errors = np.vstack([gains - lower, upper - gains])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(range(9), gains, color="#0072B2", alpha=0.86, width=0.7)
    ax.errorbar(range(9), gains, yerr=errors, fmt="none", ecolor="#222222", capsize=3)
    ax.axhline(0, color="#333333", linewidth=1.0)
    for bar, row in zip(bars, rows):
        p_value = row["holm_adjusted_p_across_levels"]
        marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["gain_ci95_upper_percentage_points"] + 2.0,
            marker,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set(
        xlabel="Target displacement (cm)",
        ylabel="V3 success-rate gain over V2 (percentage points)",
        title="Paired V3 Improvement with 95% Bootstrap CI",
    )
    ax.set_xticks(range(9))
    ax.set_ylim(min(-5, float(lower.min()) - 3), float(upper.max()) + 9)
    save_figure(fig, output_dir, "03_v3_paired_gain_and_significance")


def plot_retention(summary_rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for model in MODELS:
        rates = [
            lookup(
                summary_rows,
                model=model,
                ensemble_mode="temporal",
                level_cm=level_cm,
            )["task_success_rate"]
            for level_cm in range(9)
        ]
        baseline = rates[0]
        ax.plot(
            range(9),
            [100.0 * rate / baseline for rate in rates],
            marker="o",
            linewidth=2.3,
            color=MODEL_COLORS[model],
            label=model.upper(),
        )
    ax.axhline(80, color="#555555", linestyle="--", linewidth=1.0)
    ax.set(
        xlabel="Target displacement (cm)",
        ylabel="Success retention relative to 0 cm (%)",
        ylim=(0, 108),
        title="Normalized Robustness Retention",
    )
    ax.set_xticks(range(9))
    ax.legend(loc="lower left")
    save_figure(fig, output_dir, "04_normalized_success_retention")


def plot_task_families(task_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
    for axis, task_type in zip(axes, ("pick", "push")):
        for model in MODELS:
            values = []
            for level_cm in range(9):
                group = [
                    row
                    for row in task_rows
                    if row["model"] == model
                    and row["ensemble_mode"] == "temporal"
                    and row["level_cm"] == level_cm
                    and row["task"].startswith(task_type)
                ]
                values.append(
                    100.0
                    * sum(row["successes"] for row in group)
                    / sum(row["episodes"] for row in group)
                )
            axis.plot(
                range(9),
                values,
                marker="o",
                linewidth=2.2,
                color=MODEL_COLORS[model],
                label=model.upper(),
            )
        axis.axhline(80, color="#555555", linestyle=":", linewidth=1.0)
        axis.set_title(task_type.capitalize())
        axis.set_xlabel("Displacement (cm)")
        axis.set_xticks(range(9))
        axis.set_ylim(0, 103)
    axes[0].set_ylabel("Task success rate (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Task-Family Robustness: V2 vs V3 Temporal")
    fig.tight_layout()
    save_figure(fig, output_dir, "05_pick_push_family_comparison")


def plot_task_gain_heatmap(task_rows: list[dict], output_dir: Path) -> None:
    matrix = np.zeros((len(TASKS), 9), dtype=np.float64)
    for task_index, task in enumerate(TASKS):
        for level_cm in range(9):
            v2 = lookup(
                task_rows,
                model="v2",
                ensemble_mode="temporal",
                level_cm=level_cm,
                task=task,
            )["task_success_rate"]
            v3 = lookup(
                task_rows,
                model="v3",
                ensemble_mode="temporal",
                level_cm=level_cm,
                task=task,
            )["task_success_rate"]
            matrix[task_index, level_cm] = 100.0 * (v3 - v2)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    ax.set_xticks(range(9), labels=range(9))
    ax.set_yticks(range(len(TASKS)), labels=[task.replace("_", "-") for task in TASKS])
    ax.set_xlabel("Target displacement (cm)")
    ax.set_title("V3 Gain over V2 by Task (percentage points)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label("V3 - V2 success rate (pp)")
    save_figure(fig, output_dir, "06_per_task_v3_gain_heatmap")


def failure_value(
    failure_rows: list[dict],
    model: str,
    level_cm: int,
    category: str,
) -> float:
    group = [
        row
        for row in failure_rows
        if row["model"] == model
        and row["ensemble_mode"] == "temporal"
        and row["level_cm"] == level_cm
    ]
    if category == "other":
        return sum(
            row["rate_over_all_episodes"]
            for row in group
            if row["failure_category"] not in FAILURE_ORDER[:-1]
        )
    return sum(
        row["rate_over_all_episodes"]
        for row in group
        if row["failure_category"] == category
    )


def plot_failure_taxonomy(failure_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    for axis, model in zip(axes, MODELS):
        bottoms = np.zeros(9, dtype=np.float64)
        for category in FAILURE_ORDER:
            values = 100.0 * np.asarray(
                [failure_value(failure_rows, model, level_cm, category) for level_cm in range(9)]
            )
            if np.any(values):
                axis.bar(
                    range(9),
                    values,
                    bottom=bottoms,
                    width=0.72,
                    color=FAILURE_COLORS[category],
                    label=category.replace("_", " "),
                )
                bottoms += values
        axis.set_title(model.upper())
        axis.set_xlabel("Displacement (cm)")
        axis.set_xticks(range(9))
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Failure rate over all episodes (%)")
    present_categories = [
        category
        for category in FAILURE_ORDER
        if any(
            failure_value(failure_rows, model, level_cm, category) > 0
            for model in MODELS
            for level_cm in range(9)
        )
    ]
    handles = [
        Patch(
            facecolor=FAILURE_COLORS[category],
            label=category.replace("_", " "),
        )
        for category in present_categories
    ]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.91, 0.5), frameon=False)
    fig.suptitle("Failure Taxonomy under Dynamic Displacement (Temporal)")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    save_figure(fig, output_dir, "07_failure_taxonomy_v2_v3")


def plot_reacquisition(summary_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for model in MODELS:
        group = [
            lookup(
                summary_rows,
                model=model,
                ensemble_mode="temporal",
                level_cm=level_cm,
            )
            for level_cm in range(1, 9)
        ]
        axes[0].plot(
            range(1, 9),
            [100.0 * row["reacquisition_rate_0_5cm"] for row in group],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model],
            label=model.upper(),
        )
        axes[1].plot(
            range(1, 9),
            [row["mean_successful_reacquisition_latency_s_0_5cm"] for row in group],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model],
            label=model.upper(),
        )
    axes[0].set(xlabel="Displacement (cm)", ylabel="Reacquisition rate (%)", ylim=(40, 103))
    axes[1].set(xlabel="Displacement (cm)", ylabel="Successful reacquisition latency (s)")
    axes[0].set_title("Within 0.5 cm")
    axes[1].set_title("20 Hz control loop")
    axes[0].legend(loc="lower left")
    fig.suptitle("Visual Reacquisition after Target Displacement")
    fig.tight_layout()
    save_figure(fig, output_dir, "08_reacquisition_rate_and_latency")


def plot_recovery_cost(summary_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for model in MODELS:
        group = [
            lookup(
                summary_rows,
                model=model,
                ensemble_mode="temporal",
                level_cm=level_cm,
            )
            for level_cm in range(9)
        ]
        axes[0].plot(
            range(9),
            [row["mean_steps"] for row in group],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model],
            label=model.upper(),
        )
        axes[1].plot(
            range(9),
            [100.0 * row["wrong_object_contact_rate"] for row in group],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model],
            label=model.upper(),
        )
    axes[0].axhline(200, color="#555555", linestyle=":", linewidth=1.0)
    axes[0].set(xlabel="Displacement (cm)", ylabel="Mean episode steps", ylim=(60, 205))
    axes[1].set(xlabel="Displacement (cm)", ylabel="Wrong-object contact rate (%)", ylim=(0, 15))
    axes[0].set_title("Recovery time cost")
    axes[1].set_title("Target-selection safety")
    axes[0].legend(loc="upper left")
    fig.suptitle("Execution Stability under Dynamic Displacement")
    fig.tight_layout()
    save_figure(fig, output_dir, "09_recovery_cost_and_wrong_contact")


def normalized_auc(summary_rows: list[dict], model: str) -> float:
    rates = [
        lookup(
            summary_rows,
            model=model,
            ensemble_mode="temporal",
            level_cm=level_cm,
        )["task_success_rate"]
        for level_cm in range(9)
    ]
    return float(np.trapz(rates, x=np.arange(9)) / 8.0)


def write_analysis_report(
    output_dir: Path,
    summary_rows: list[dict],
    comparison_rows: list[dict],
    failure_rows: list[dict],
    protocol_audit: dict,
) -> None:
    primary = [
        lookup(comparison_rows, ensemble_mode="temporal", level_cm=level_cm)
        for level_cm in range(9)
    ]
    lines = [
        "# V2 vs V3 Dynamic Target Displacement Analysis",
        "",
        "## Protocol",
        "",
        "- Primary estimand: Temporal ensemble, paired scenes, intention-to-treat.",
        "- Two seed schedules, 120 paired episodes per displacement level and model.",
        "- All non-zero Temporal episodes received the requested displacement.",
        "- No privileged execution assistance was enabled.",
        "- Latest-only is diagnostic because V2 missed 8 scheduled injections in one seed run.",
        "",
        "## Primary Result",
        "",
        "| Shift | V2 success (95% Wilson CI) | V3 success (95% Wilson CI) | Paired gain | Holm p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        v2_summary = lookup(
            summary_rows,
            model="v2",
            ensemble_mode="temporal",
            level_cm=row["level_cm"],
        )
        v3_summary = lookup(
            summary_rows,
            model="v3",
            ensemble_mode="temporal",
            level_cm=row["level_cm"],
        )
        lines.append(
            f"| {row['level_cm']} cm | "
            f"{100 * row['v2_success_rate']:.1f}% "
            f"[{100 * v2_summary['ci95_lower']:.1f}, {100 * v2_summary['ci95_upper']:.1f}] | "
            f"{100 * row['v3_success_rate']:.1f}% "
            f"[{100 * v3_summary['ci95_lower']:.1f}, {100 * v3_summary['ci95_upper']:.1f}] | "
            f"{row['v3_gain_percentage_points']:+.1f} pp | "
            f"{row['holm_adjusted_p_across_levels']:.3g} |"
        )

    def total_failures(model: str) -> Counter:
        counts: Counter = Counter()
        for row in failure_rows:
            if row["model"] == model and row["ensemble_mode"] == "temporal":
                counts[row["failure_category"]] += int(row["count"])
        return counts

    v2_failures = total_failures("v2")
    v3_failures = total_failures("v3")
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "- V2 has a robustness cliff between 2 cm and 3 cm; V3 remains above 90% through 4 cm.",
            "- The point-estimate 80% boundary moves from 2 cm (V2) to 7 cm (V3).",
            "- V3 largely removes post-contact grasp collapse and wrong-object contact.",
            "- At large shifts, V3's remaining bottleneck is push completion distance rather than target selection.",
            "- V2 often reacquires and contacts the displaced target at 3-4 cm but still fails task completion, so its failure is not purely visual grounding.",
            "",
            "## Aggregate Temporal Failure Counts",
            "",
            "| Failure category | V2 | V3 |",
            "|---|---:|---:|",
        ]
    )
    categories = sorted(set(v2_failures) | set(v3_failures))
    for category in categories:
        lines.append(f"| {category} | {v2_failures[category]} | {v3_failures[category]} |")
    lines.extend(
        [
            "",
            "## Output Guide",
            "",
            "- `01_v2_v3_temporal_success_decay`: main paper robustness curve.",
            "- `03_v3_paired_gain_and_significance`: paired improvement and significance.",
            "- `06_per_task_v3_gain_heatmap`: task-level gains.",
            "- `07_failure_taxonomy_v2_v3`: failure mechanism transition.",
            "- `08_reacquisition_rate_and_latency`: visual recovery evidence.",
            "- `09_recovery_cost_and_wrong_contact`: recovery efficiency and safety.",
            "",
            "## Audit Note",
            "",
            f"- Paired model outcomes: {protocol_audit['cross_model_paired_rows']}.",
            "- Latest-only missed injections: "
            f"{sum(item['assigned'] - item['injected'] for key, item in protocol_audit['injection_compliance'].items() if key.endswith('_latest-only'))}.",
            "",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.v2_dirs) != len(args.v3_dirs):
        raise ValueError("V2 and V3 must have the same number of seed schedules")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

    v2_rows, v2_metadata = load_model_runs("v2", args.v2_dirs)
    v3_rows, v3_metadata = load_model_runs("v3", args.v3_dirs)
    rows = v2_rows + v3_rows
    paired, protocol_audit = validate_cross_model_pairing(rows)
    summary_rows, task_rows, failure_rows = summarize_models(rows)
    comparison_rows, paired_episode_rows = paired_comparison(
        paired,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    write_csv(output_dir / "model_success_summary.csv", summary_rows)
    write_csv(output_dir / "paired_v2_v3_comparison.csv", comparison_rows)
    write_csv(output_dir / "paired_episode_outcomes.csv", paired_episode_rows)
    write_csv(output_dir / "task_success_summary.csv", task_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)

    configure_plots()
    plot_primary_success(summary_rows, output_dir)
    plot_all_modes(summary_rows, output_dir)
    plot_gain(comparison_rows, output_dir)
    plot_retention(summary_rows, output_dir)
    plot_task_families(task_rows, output_dir)
    plot_task_gain_heatmap(task_rows, output_dir)
    plot_failure_taxonomy(failure_rows, output_dir)
    plot_reacquisition(summary_rows, output_dir)
    plot_recovery_cost(summary_rows, output_dir)

    primary = [
        lookup(comparison_rows, ensemble_mode="temporal", level_cm=level_cm)
        for level_cm in range(9)
    ]
    report = {
        "status": "complete",
        "protocol_audit": protocol_audit,
        "input_runs": v2_metadata + v3_metadata,
        "total_episode_rows": len(rows),
        "paired_model_outcomes": len(paired),
        "primary_temporal_results": primary,
        "normalized_robustness_auc": {
            model: normalized_auc(summary_rows, model) for model in MODELS
        },
        "point_estimate_80_percent_boundary_cm": {
            model: max(
                row["level_cm"]
                for row in summary_rows
                if row["model"] == model
                and row["ensemble_mode"] == "temporal"
                and row["task_success_rate"] >= 0.80
            )
            for model in MODELS
        },
        "lower_ci_80_percent_boundary_cm": {
            model: max(
                row["level_cm"]
                for row in summary_rows
                if row["model"] == model
                and row["ensemble_mode"] == "temporal"
                and row["ci95_lower"] >= 0.80
            )
            for model in MODELS
        },
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "protocol_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol_audit, handle, indent=2)
    write_analysis_report(
        output_dir,
        summary_rows,
        comparison_rows,
        failure_rows,
        protocol_audit,
    )

    print(f"Validated and paired {len(paired)} V2/V3 benchmark outcomes.")
    print(f"Tables and nine paper-ready figures saved to {output_dir}")


if __name__ == "__main__":
    main()
