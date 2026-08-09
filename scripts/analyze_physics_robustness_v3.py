#!/usr/bin/env python3
"""Audit and plot paired MiniVLA V3 mass and friction drift sweeps."""

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


CURVES = ("target-mass", "target-friction")
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
TITLES = {
    "target-mass": "Target mass and inertia",
    "target-friction": "Target contact friction",
}
COLORS = {"target-mass": "#0072B2", "target-friction": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="final_report/06_physics_drift",
    )
    return parser.parse_args()


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true"}


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


def exact_mcnemar_p(clean_only: int, drift_only: int) -> float:
    discordant = clean_only + drift_only
    if discordant == 0:
        return 1.0
    smaller = min(clean_only, drift_only)
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_for_curve(rows: list[dict], curve: str, multiplier: float) -> list[dict]:
    if np.isclose(multiplier, 1.0):
        return [row for row in rows if row["benchmark_parameter"] == "clean"]
    return [
        row
        for row in rows
        if row["benchmark_parameter"] == curve
        and np.isclose(float(row["benchmark_multiplier"]), multiplier)
    ]


def load_and_audit(input_dirs: list[str]) -> tuple[list[dict], dict, list[dict]]:
    all_rows: list[dict] = []
    sources: list[dict] = []
    reference_protocol: tuple | None = None
    reference_drifts: dict | None = None
    for raw_dir in input_dirs:
        root = Path(raw_dir)
        with (root / "benchmark_summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with (root / "benchmark_episodes.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
        if summary.get("status") != "complete":
            raise ValueError(f"Benchmark is not complete: {root}")
        expected = int(summary["episodes_per_level"]) * int(
            summary["total_conditions"]
        )
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} rows in {root}, found {len(rows)}")
        if not summary.get("paired_scene_validation_passed"):
            raise ValueError(f"Paired-scene audit failed: {root}")
        if any(as_bool(row, "uses_privileged_execution_assistance") for row in rows):
            raise ValueError(f"Privileged execution assistance detected: {root}")
        if any(
            not as_bool(row, "physics_application_verified")
            or not as_bool(row, "physics_restoration_verified")
            or row["physics_target_id"] != row["target_id"]
            for row in rows
        ):
            raise ValueError(f"Physics application/restoration audit failed: {root}")
        clean = [row for row in rows if row["benchmark_parameter"] == "clean"]
        if any(as_bool(row, "injected") for row in clean):
            raise ValueError(f"Clean physics condition was marked injected: {root}")
        drifted = [row for row in rows if row["benchmark_parameter"] != "clean"]
        if any(not as_bool(row, "injected") for row in drifted):
            raise ValueError(f"A physics drift was not injected: {root}")
        for row in drifted:
            multiplier = float(row["benchmark_multiplier"])
            if row["benchmark_parameter"] == "target-mass":
                ratio = float(row["applied_body_mass"]) / float(
                    row["original_body_mass"]
                )
            else:
                ratio = float(row["applied_mean_sliding_friction"]) / float(
                    row["original_mean_sliding_friction"]
                )
            if not np.isclose(ratio, multiplier, rtol=1e-6, atol=1e-8):
                raise ValueError(f"Applied multiplier mismatch in {root}: {row}")

        clean_signature = [
            (row["episode"], row["task_type"], row["target_id"], row["scene_seed"])
            for row in clean
        ]
        for condition in sorted({row["benchmark_condition"] for row in drifted}):
            group = [row for row in drifted if row["benchmark_condition"] == condition]
            signature = [
                (row["episode"], row["task_type"], row["target_id"], row["scene_seed"])
                for row in group
            ]
            if signature != clean_signature:
                raise ValueError(f"Paired scene mismatch for {condition} in {root}")

        protocol = (
            summary["protocol_version"],
            summary["policy"],
            summary["episodes_per_level"],
            summary["ensemble_mode"],
            summary["temporal_profile"],
        )
        if reference_protocol is None:
            reference_protocol = protocol
            reference_drifts = summary["drifts"]
        elif protocol != reference_protocol or summary["drifts"] != reference_drifts:
            raise ValueError(f"Protocol mismatch in {root}")
        source = f"seed_{summary['seed']}"
        all_rows.extend({"source_run": source, **row} for row in rows)
        sources.append(
            {
                "source_run": source,
                "input_dir": str(root),
                "scene_seed": summary["seed"],
                "episode_rows": len(rows),
            }
        )
    audit = {
        "status": "PASS",
        "protocol_version": reference_protocol[0],
        "policy": reference_protocol[1],
        "seeds": len(input_dirs),
        "episode_rows": len(all_rows),
        "paired_scene_validation_passed": True,
        "application_verification_rate": 1.0,
        "restoration_verification_rate": 1.0,
        "target_only_scope_verified": True,
        "drifts": reference_drifts,
    }
    return all_rows, audit, sources


def summarize(rows: list[dict], drifts: dict) -> tuple[list[dict], list[dict], list[dict]]:
    summary_rows: list[dict] = []
    task_rows: list[dict] = []
    failure_rows: list[dict] = []
    for curve in CURVES:
        for level_index, multiplier in enumerate(drifts[curve]):
            group = rows_for_curve(rows, curve, float(multiplier))
            successes = sum(as_bool(row, "task_success") for row in group)
            lower, upper = wilson_interval(successes, len(group))
            summary_rows.append(
                {
                    "physics_parameter": curve,
                    "level_index": level_index,
                    "multiplier": multiplier,
                    "episodes": len(group),
                    "successes": successes,
                    "success_rate": successes / len(group),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "pick_success_rate": np.mean(
                        [
                            as_bool(row, "task_success")
                            for row in group
                            if row["task_type"] == "pick"
                        ]
                    ),
                    "push_success_rate": np.mean(
                        [
                            as_bool(row, "task_success")
                            for row in group
                            if row["task_type"] == "push"
                        ]
                    ),
                    "wrong_object_contact_rate": np.mean(
                        [as_bool(row, "wrong_object_contact") for row in group]
                    ),
                    "target_contact_rate": np.mean(
                        [as_bool(row, "target_contact") for row in group]
                    ),
                    "mean_steps": np.mean([float(row["steps"]) for row in group]),
                }
            )
            for task in TASKS:
                task_type, target_id = task.split("_")
                task_group = [
                    row
                    for row in group
                    if row["task_type"] == task_type and row["target_id"] == target_id
                ]
                task_rows.append(
                    {
                        "physics_parameter": curve,
                        "level_index": level_index,
                        "multiplier": multiplier,
                        "task": task,
                        "episodes": len(task_group),
                        "success_rate": np.mean(
                            [as_bool(row, "task_success") for row in task_group]
                        ),
                    }
                )
            failures = Counter(
                row["failure_category"]
                for row in group
                if not as_bool(row, "task_success")
            )
            for category, count in sorted(failures.items()):
                failure_rows.append(
                    {
                        "physics_parameter": curve,
                        "level_index": level_index,
                        "multiplier": multiplier,
                        "failure_category": category,
                        "count": count,
                        "rate_over_all_episodes": count / len(group),
                    }
                )
    return summary_rows, task_rows, failure_rows


def paired_tests(rows: list[dict], drifts: dict) -> list[dict]:
    clean = {
        (row["source_run"], row["episode"]): as_bool(row, "task_success")
        for row in rows
        if row["benchmark_parameter"] == "clean"
    }
    tests: list[dict] = []
    for curve in CURVES:
        curve_tests = []
        for level_index, multiplier in enumerate(drifts[curve]):
            if np.isclose(float(multiplier), 1.0):
                continue
            drifted_rows = rows_for_curve(rows, curve, float(multiplier))
            drifted = {
                (row["source_run"], row["episode"]): as_bool(row, "task_success")
                for row in drifted_rows
            }
            if clean.keys() != drifted.keys():
                raise ValueError(f"Pairing mismatch for {curve} {multiplier}")
            clean_only = sum(clean[key] and not drifted[key] for key in clean)
            drift_only = sum(not clean[key] and drifted[key] for key in clean)
            curve_tests.append(
                {
                    "physics_parameter": curve,
                    "level_index": level_index,
                    "multiplier": multiplier,
                    "clean_only_successes": clean_only,
                    "drift_only_successes": drift_only,
                    "success_difference_percentage_points": 100.0
                    * (np.mean(list(drifted.values())) - np.mean(list(clean.values()))),
                    "mcnemar_p": exact_mcnemar_p(clean_only, drift_only),
                }
            )
        adjusted = holm_adjust([row["mcnemar_p"] for row in curve_tests])
        for row, value in zip(curve_tests, adjusted):
            row["holm_p_within_curve"] = value
            tests.append(row)
    return tests


def lookup(rows: list[dict], **conditions) -> dict:
    matches = [
        row
        for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one match for {conditions}, found {len(matches)}")
    return matches[0]


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_success(summary: list[dict], drifts: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    for axis, curve in zip(axes, CURVES):
        group = [lookup(summary, physics_parameter=curve, level_index=index) for index in range(len(drifts[curve]))]
        rates = 100.0 * np.asarray([row["success_rate"] for row in group])
        lower = np.maximum(0.0, rates - 100.0 * np.asarray([row["ci95_lower"] for row in group]))
        upper = np.maximum(0.0, 100.0 * np.asarray([row["ci95_upper"] for row in group]) - rates)
        x = np.arange(len(group))
        axis.errorbar(x, rates, yerr=np.vstack([lower, upper]), marker="o", linewidth=2.2, capsize=4, color=COLORS[curve])
        for index, rate in enumerate(rates):
            axis.annotate(f"{rate:.1f}", (index, rate), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(x, labels=[f"{float(value):g}x" for value in drifts[curve]])
        axis.set_xlabel("Baseline parameter multiplier")
        axis.set_title(TITLES[curve])
        axis.set_ylim(0, 105)
    axes[0].set_ylabel("Task success rate (%)")
    fig.suptitle("MiniVLA V3 Physics-Parameter Robustness")
    fig.tight_layout()
    save(fig, output_dir, "01_physics_success_decay")


def plot_task_family(summary: list[dict], drifts: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    for axis, curve in zip(axes, CURVES):
        group = [lookup(summary, physics_parameter=curve, level_index=index) for index in range(len(drifts[curve]))]
        x = np.arange(len(group))
        axis.plot(x, 100.0 * np.asarray([row["pick_success_rate"] for row in group]), marker="o", linewidth=2.0, label="Pick")
        axis.plot(x, 100.0 * np.asarray([row["push_success_rate"] for row in group]), marker="s", linewidth=2.0, label="Push")
        axis.set_xticks(x, labels=[f"{float(value):g}x" for value in drifts[curve]])
        axis.set_xlabel("Baseline parameter multiplier")
        axis.set_title(TITLES[curve])
        axis.set_ylim(0, 105)
    axes[0].set_ylabel("Task-family success rate (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Pick and Push Sensitivity to Physics Drift")
    fig.tight_layout()
    save(fig, output_dir, "02_pick_push_physics_sensitivity")


def plot_extreme_heatmap(task_rows: list[dict], drifts: dict, output_dir: Path) -> None:
    columns: list[tuple[str, int, str]] = []
    for curve in CURVES:
        columns.append((curve, 0, f"{TITLES[curve]}\n{float(drifts[curve][0]):g}x"))
        columns.append((curve, len(drifts[curve]) - 1, f"{TITLES[curve]}\n{float(drifts[curve][-1]):g}x"))
    matrix = np.zeros((len(TASKS), len(columns)), dtype=np.float64)
    for column_index, (curve, level_index, _) in enumerate(columns):
        for task_index, task in enumerate(TASKS):
            matrix[task_index, column_index] = 100.0 * lookup(
                task_rows,
                physics_parameter=curve,
                level_index=level_index,
                task=task,
            )["success_rate"]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center")
    ax.set_xticks(range(len(columns)), labels=[entry[2] for entry in columns])
    ax.set_yticks(range(len(TASKS)), labels=[task.replace("_", "-") for task in TASKS])
    ax.set_title("Per-Task Success at Low and High Physics Multipliers")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Task success rate (%)")
    fig.tight_layout()
    save(fig, output_dir, "03_extreme_per_task_heatmap")


def plot_failure_taxonomy(
    failure_rows: list[dict],
    drifts: dict,
    output_dir: Path,
) -> None:
    conditions = [
        (curve, level_index, f"{('M' if curve == 'target-mass' else 'F')} {float(multiplier):g}x")
        for curve in CURVES
        for level_index, multiplier in enumerate(drifts[curve])
    ]
    categories = sorted({row["failure_category"] for row in failure_rows})
    x = np.arange(len(conditions))
    bottoms = np.zeros(len(conditions), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    palette = plt.get_cmap("tab10")
    for category_index, category in enumerate(categories):
        values = np.asarray(
            [
                100.0
                * sum(
                    row["rate_over_all_episodes"]
                    for row in failure_rows
                    if row["physics_parameter"] == curve
                    and row["level_index"] == level_index
                    and row["failure_category"] == category
                )
                for curve, level_index, _ in conditions
            ]
        )
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=palette(category_index % 10),
            label=category.replace("_", " "),
        )
        bottoms += values
    ax.axvline(4.5, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xticks(x, labels=[label for _, _, label in conditions])
    ax.set_ylabel("Failure rate over all episodes (%)")
    ax.set_xlabel("M = target mass/inertia; F = target contact friction")
    ax.set_title("Failure Taxonomy Across Physics-Parameter Drift")
    if categories:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    save(fig, output_dir, "04_physics_failure_taxonomy")


def write_report(output_dir: Path, audit: dict, summary: list[dict], tests: list[dict]) -> None:
    test_map = {(row["physics_parameter"], row["level_index"]): row for row in tests}
    lines = [
        "# MiniVLA V3 Physics-Parameter Robustness",
        "",
        "## Protocol Audit",
        "",
        f"- Status: {audit['status']}.",
        f"- Raw episode rows: {audit['episode_rows']}.",
        f"- Seeds: {audit['seeds']}.",
        "- Target-only parameter application and per-episode restoration both verified at 100%.",
        "- Mass drift scales mass and inertia together; friction drift scales all target contact-friction components.",
        "- Policy observations, scoring, and execution receive no privileged assistance.",
        "",
        "## Success Rates",
        "",
        "| Parameter | Multiplier | Episodes | Success | 95% Wilson CI | Holm p vs 1x |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        test = test_map.get((row["physics_parameter"], row["level_index"]))
        p_text = "reference" if test is None else f"{test['holm_p_within_curve']:.4g}"
        lines.append(
            f"| {row['physics_parameter']} | {float(row['multiplier']):g}x | {row['episodes']} | "
            f"{100.0 * row['success_rate']:.2f}% | [{100.0 * row['ci95_lower']:.1f}, "
            f"{100.0 * row['ci95_upper']:.1f}] | {p_text} |"
        )
    mass_1 = lookup(summary, physics_parameter="target-mass", level_index=2)
    mass_4 = lookup(summary, physics_parameter="target-mass", level_index=4)
    friction_1 = lookup(summary, physics_parameter="target-friction", level_index=2)
    friction_2 = lookup(summary, physics_parameter="target-friction", level_index=3)
    friction_4 = lookup(summary, physics_parameter="target-friction", level_index=4)
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- Mass/inertia is stable through 2x, then falls from {100.0 * mass_1['success_rate']:.2f}% at 1x to {100.0 * mass_4['success_rate']:.2f}% at 4x.",
            f"- At 4x mass, Pick retains {100.0 * mass_4['pick_success_rate']:.2f}% while Push retains {100.0 * mass_4['push_success_rate']:.2f}%.",
            f"- Friction is stable through 1x, but overall success drops to {100.0 * friction_2['success_rate']:.2f}% at 2x and {100.0 * friction_4['success_rate']:.2f}% at 4x.",
            f"- High friction is task-selective: Pick remains {100.0 * friction_4['pick_success_rate']:.2f}% at 4x while Push falls to {100.0 * friction_4['push_success_rate']:.2f}%.",
            "",
            "## Interpretation Checklist",
            "",
            "- Separate Pick and Push because their sensitivity mechanisms differ.",
            "- Attribute changes to the tested simulator parameters, not general sim-to-real robustness.",
            "- Use failure taxonomy to distinguish grasp loss, push distance, and contact failures.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, audit, sources = load_and_audit(args.input_dirs)
    drifts = audit["drifts"]
    summary, task_rows, failure_rows = summarize(rows, drifts)
    tests = paired_tests(rows, drifts)
    write_csv(output_dir / "success_summary.csv", summary)
    write_csv(output_dir / "task_success_summary.csv", task_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)
    write_csv(output_dir / "paired_clean_tests.csv", tests)
    write_csv(output_dir / "source_runs.csv", sources)
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_success(summary, drifts, output_dir)
    plot_task_family(summary, drifts, output_dir)
    plot_extreme_heatmap(task_rows, drifts, output_dir)
    plot_failure_taxonomy(failure_rows, drifts, output_dir)
    write_report(output_dir, audit, summary, tests)
    print(f"Physics protocol audit: PASS ({len(rows)} episode rows)", flush=True)
    print(f"Analysis saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
