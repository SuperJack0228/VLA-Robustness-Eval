#!/usr/bin/env python3
"""Audit and plot completed paired MiniVLA V3 visual robustness sweeps."""

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


CURVES = ("gaussian-noise", "gaussian-blur", "brightness")
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
CURVE_TITLES = {
    "gaussian-noise": "Gaussian noise",
    "gaussian-blur": "Gaussian blur",
    "brightness": "Brightness reduction",
}
CURVE_XLABELS = {
    "gaussian-noise": "Noise standard deviation (0-255 RGB)",
    "gaussian-blur": "Blur sigma (pixels)",
    "brightness": "RGB gain",
}
CURVE_COLORS = {
    "gaussian-noise": "#0072B2",
    "gaussian-blur": "#009E73",
    "brightness": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Completed visual benchmark directories, one per seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="final_report/05_visual_degradation",
    )
    return parser.parse_args()


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true"}


def optional_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


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


def exact_mcnemar_p(clean_only: int, degraded_only: int) -> float:
    discordant = clean_only + degraded_only
    if discordant == 0:
        return 1.0
    smaller = min(clean_only, degraded_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity_level(curve: str) -> float:
    return 1.0 if curve == "brightness" else 0.0


def rows_for_curve(rows: list[dict], curve: str, level: float) -> list[dict]:
    if np.isclose(level, identity_level(curve)):
        return [row for row in rows if row["benchmark_corruption"] == "clean"]
    return [
        row
        for row in rows
        if row["benchmark_corruption"] == curve
        and np.isclose(float(row["benchmark_level"]), level)
    ]


def load_and_audit(input_dirs: list[str]) -> tuple[list[dict], dict, list[dict]]:
    all_rows: list[dict] = []
    run_metadata: list[dict] = []
    reference_protocol: tuple | None = None
    reference_curves: dict | None = None

    for source_index, raw_dir in enumerate(input_dirs, start=1):
        root = Path(raw_dir)
        summary_path = root / "benchmark_summary.json"
        episodes_path = root / "benchmark_episodes.csv"
        if not summary_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"Incomplete benchmark directory: {root}")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if summary.get("status") != "complete":
            raise ValueError(f"Benchmark is not complete: {root}")
        if not summary.get("paired_scene_validation_passed"):
            raise ValueError(f"Paired-scene validation failed: {root}")
        expected = int(summary["episodes_per_level"]) * int(
            summary["total_conditions"]
        )
        if len(rows) != expected:
            raise ValueError(
                f"Expected {expected} rows in {root}, found {len(rows)}"
            )
        if any(as_bool(row, "uses_privileged_execution_assistance") for row in rows):
            raise ValueError(f"Privileged execution assistance detected: {root}")

        clean = [row for row in rows if row["benchmark_corruption"] == "clean"]
        if len(clean) != int(summary["episodes_per_level"]):
            raise ValueError(f"Invalid shared Clean baseline in {root}")
        if any(
            as_bool(row, "injected")
            or float(row["mean_absolute_pixel_delta"]) != 0.0
            for row in clean
        ):
            raise ValueError(f"Clean image identity audit failed: {root}")
        degraded = [row for row in rows if row["benchmark_corruption"] != "clean"]
        if any(
            not as_bool(row, "injected")
            or int(row["frames_changed"]) <= 0
            or float(row["mean_absolute_pixel_delta"]) <= 0.0
            for row in degraded
        ):
            raise ValueError(f"A visual degradation was not applied: {root}")

        clean_signature = [
            (row["episode"], row["task_type"], row["target_id"], row["scene_seed"])
            for row in clean
        ]
        for condition in sorted({row["benchmark_condition"] for row in degraded}):
            group = [row for row in degraded if row["benchmark_condition"] == condition]
            signature = [
                (row["episode"], row["task_type"], row["target_id"], row["scene_seed"])
                for row in group
            ]
            if signature != clean_signature:
                raise ValueError(
                    f"Paired scene mismatch for {condition} in {root}"
                )

        protocol = (
            summary["protocol_version"],
            summary["policy"],
            summary["episodes_per_level"],
            summary["ensemble_mode"],
            summary["temporal_profile"],
            tuple(summary["affected_cameras"]),
        )
        if reference_protocol is None:
            reference_protocol = protocol
            reference_curves = summary["curves"]
        elif protocol != reference_protocol or summary["curves"] != reference_curves:
            raise ValueError(f"Protocol mismatch in {root}")

        source = f"seed_{summary['seed']}"
        for row in rows:
            all_rows.append({"source_run": source, **row})
        run_metadata.append(
            {
                "source_run": source,
                "input_dir": str(root),
                "scene_seed": summary["seed"],
                "perturbation_seed": summary["perturbation_seed"],
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
        "clean_pixel_identity_passed": True,
        "nonzero_injection_compliance": 1.0,
        "affected_cameras": list(reference_protocol[-1]),
        "curves": reference_curves,
    }
    return all_rows, audit, run_metadata


def summarize(rows: list[dict], curves: dict) -> tuple[list[dict], list[dict], list[dict]]:
    summary_rows: list[dict] = []
    task_rows: list[dict] = []
    failure_rows: list[dict] = []
    for curve in CURVES:
        for severity_index, level in enumerate(curves[curve]):
            group = rows_for_curve(rows, curve, float(level))
            successes = sum(as_bool(row, "task_success") for row in group)
            lower, upper = wilson_interval(successes, len(group))
            psnr_values = [
                value
                for row in group
                if (value := optional_float(row, "input_psnr_db")) is not None
            ]
            summary_rows.append(
                {
                    "corruption": curve,
                    "severity_index": severity_index,
                    "level": level,
                    "episodes": len(group),
                    "successes": successes,
                    "success_rate": successes / len(group),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "wrong_object_contact_rate": np.mean(
                        [as_bool(row, "wrong_object_contact") for row in group]
                    ),
                    "target_contact_rate": np.mean(
                        [as_bool(row, "target_contact") for row in group]
                    ),
                    "mean_steps": np.mean([float(row["steps"]) for row in group]),
                    "mean_grounding_error_cm": np.mean(
                        [float(row["mean_grounding_error_cm"]) for row in group]
                    ),
                    "mean_target_class_accuracy": np.mean(
                        [float(row["mean_target_class_accuracy"]) for row in group]
                    ),
                    "mean_absolute_pixel_delta": np.mean(
                        [float(row["mean_absolute_pixel_delta"]) for row in group]
                    ),
                    "mean_input_psnr_db": (
                        np.mean(psnr_values) if psnr_values else ""
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
                task_rows.append(
                    {
                        "corruption": curve,
                        "severity_index": severity_index,
                        "level": level,
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
                        "corruption": curve,
                        "severity_index": severity_index,
                        "level": level,
                        "failure_category": category,
                        "count": count,
                        "rate_over_all_episodes": count / len(group),
                    }
                )
    return summary_rows, task_rows, failure_rows


def paired_tests(rows: list[dict], curves: dict) -> list[dict]:
    tests: list[dict] = []
    clean = {
        (row["source_run"], row["episode"]): as_bool(row, "task_success")
        for row in rows
        if row["benchmark_corruption"] == "clean"
    }
    for curve in CURVES:
        curve_tests = []
        for severity_index, level in enumerate(curves[curve]):
            if np.isclose(float(level), identity_level(curve)):
                continue
            group = rows_for_curve(rows, curve, float(level))
            degraded = {
                (row["source_run"], row["episode"]): as_bool(row, "task_success")
                for row in group
            }
            if clean.keys() != degraded.keys():
                raise ValueError(f"Pairing mismatch for {curve} level {level}")
            clean_only = sum(clean[key] and not degraded[key] for key in clean)
            degraded_only = sum(not clean[key] and degraded[key] for key in clean)
            curve_tests.append(
                {
                    "corruption": curve,
                    "severity_index": severity_index,
                    "level": level,
                    "clean_only_successes": clean_only,
                    "degraded_only_successes": degraded_only,
                    "success_difference_percentage_points": 100.0
                    * (
                        np.mean(list(degraded.values()))
                        - np.mean(list(clean.values()))
                    ),
                    "mcnemar_p": exact_mcnemar_p(clean_only, degraded_only),
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


def plot_success(summary: list[dict], curves: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for axis, curve in zip(axes, CURVES):
        group = [lookup(summary, corruption=curve, severity_index=index) for index in range(len(curves[curve]))]
        rates = 100.0 * np.asarray([row["success_rate"] for row in group])
        lower = np.maximum(
            0.0,
            rates - 100.0 * np.asarray([row["ci95_lower"] for row in group]),
        )
        upper = np.maximum(
            0.0,
            100.0 * np.asarray([row["ci95_upper"] for row in group]) - rates,
        )
        x = np.arange(len(group))
        axis.errorbar(
            x,
            rates,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=CURVE_COLORS[curve],
        )
        for index, rate in enumerate(rates):
            axis.annotate(
                f"{rate:.1f}",
                (index, rate),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_xticks(x, labels=[f"{float(level):g}" for level in curves[curve]])
        axis.set_xlabel(CURVE_XLABELS[curve])
        axis.set_title(CURVE_TITLES[curve])
        axis.set_ylim(0, 105)
    axes[0].set_ylabel("Task success rate (%)")
    fig.suptitle("MiniVLA V3 Visual-Degradation Robustness")
    fig.tight_layout()
    save(fig, output_dir, "01_visual_success_decay")


def plot_diagnostics(summary: list[dict], curves: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
    for axis, curve in zip(axes, CURVES):
        group = [lookup(summary, corruption=curve, severity_index=index) for index in range(len(curves[curve]))]
        x = np.arange(len(group))
        grounding_line = axis.plot(
            x,
            np.asarray([row["mean_grounding_error_cm"] for row in group]),
            marker="o",
            linewidth=2.2,
            color="#0072B2",
            label="Grounding error",
        )[0]
        rate_axis = axis.twinx()
        contact_line = rate_axis.plot(
            x,
            100.0 * np.asarray([row["target_contact_rate"] for row in group]),
            marker="s",
            color="#009E73",
            label="Target contact rate",
        )[0]
        class_line = rate_axis.plot(
            x,
            100.0 * np.asarray([row["mean_target_class_accuracy"] for row in group]),
            linestyle="--",
            color="#CC79A7",
            label="Target class accuracy",
        )[0]
        axis.set_xticks(x, labels=[f"{float(level):g}" for level in curves[curve]])
        axis.set_xlabel(CURVE_XLABELS[curve])
        axis.set_title(CURVE_TITLES[curve])
        grounding_values = np.asarray(
            [row["mean_grounding_error_cm"] for row in group]
        )
        diagnostic_rates = np.asarray(
            [
                row[metric]
                for row in group
                for metric in ("target_contact_rate", "mean_target_class_accuracy")
            ]
        )
        axis.set_ylim(0.0, max(1.35, 1.15 * float(np.max(grounding_values))))
        rate_axis.set_ylim(
            max(0.0, 100.0 * float(np.min(diagnostic_rates)) - 5.0),
            101.0,
        )
        if axis is not axes[-1]:
            rate_axis.set_yticklabels([])
        else:
            rate_axis.set_ylabel("Classification / contact rate (%)")
    axes[0].set_ylabel("Mean grounding error (cm)")
    axes[0].legend(
        handles=[grounding_line, contact_line, class_line],
        loc="upper left",
        fontsize=8,
    )
    fig.suptitle("Visual Grounding Degradation Before Task Failure")
    fig.tight_layout()
    save(fig, output_dir, "02_grounding_and_contact_survival")


def plot_severe_task_heatmap(task_rows: list[dict], curves: dict, output_dir: Path) -> None:
    matrix = np.zeros((len(TASKS), len(CURVES)), dtype=np.float64)
    labels = []
    for column, curve in enumerate(CURVES):
        severe_index = len(curves[curve]) - 1
        labels.append(f"{CURVE_TITLES[curve]}\n{float(curves[curve][-1]):g}")
        for row_index, task in enumerate(TASKS):
            matrix[row_index, column] = 100.0 * lookup(
                task_rows,
                corruption=curve,
                severity_index=severe_index,
                task=task,
            )["success_rate"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center")
    ax.set_xticks(range(len(CURVES)), labels=labels)
    ax.set_yticks(range(len(TASKS)), labels=[task.replace("_", "-") for task in TASKS])
    ax.set_title("Per-Task Success at Maximum Tested Visual Severity")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Task success rate (%)")
    fig.tight_layout()
    save(fig, output_dir, "03_severe_per_task_heatmap")


def plot_failure_taxonomy(
    failure_rows: list[dict],
    curves: dict,
    output_dir: Path,
) -> None:
    conditions: list[tuple[str, int, str]] = [("gaussian-noise", 0, "Clean")]
    prefixes = {
        "gaussian-noise": "N",
        "gaussian-blur": "B",
        "brightness": "L",
    }
    for curve in CURVES:
        for severity_index, level in enumerate(curves[curve]):
            if severity_index == 0:
                continue
            conditions.append(
                (curve, severity_index, f"{prefixes[curve]} {float(level):g}")
            )
    categories = sorted({row["failure_category"] for row in failure_rows})
    palette = plt.get_cmap("tab10")
    fig_width = max(11.2, 0.62 * len(conditions) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bottoms = np.zeros(len(conditions), dtype=np.float64)
    for category_index, category in enumerate(categories):
        rates = []
        for curve, severity_index, _ in conditions:
            rate = sum(
                100.0 * row["rate_over_all_episodes"]
                for row in failure_rows
                if row["corruption"] == curve
                and row["severity_index"] == severity_index
                and row["failure_category"] == category
            )
            rates.append(rate)
        values = np.asarray(rates)
        ax.bar(
            np.arange(len(conditions)),
            values,
            bottom=bottoms,
            color=palette(category_index % 10),
            label=category.replace("_", " "),
        )
        bottoms += values
    ax.set_xticks(
        np.arange(len(conditions)),
        labels=[label for _, _, label in conditions],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Failure rate over all episodes (%)")
    ax.set_title("Failure Taxonomy Across Visual-Degradation Severity")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    save(fig, output_dir, "04_failure_taxonomy_by_severity")


def write_report(
    output_dir: Path,
    audit: dict,
    summary: list[dict],
    tests: list[dict],
) -> None:
    lines = [
        "# MiniVLA V3 Visual-Degradation Robustness",
        "",
        "## Protocol Audit",
        "",
        f"- Status: {audit['status']}.",
        f"- Raw episode rows: {audit['episode_rows']}.",
        f"- Seeds: {audit['seeds']}.",
        "- Shared paired Clean scenes across every visual condition.",
        "- Both policy cameras are degraded; simulator physics and scoring remain unchanged.",
        "- No privileged execution assistance or perturbation oracle.",
        "",
        "## Success Rates",
        "",
        "| Corruption | Level | Episodes | Success | 95% Wilson CI | Holm p vs Clean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    test_lookup = {
        (row["corruption"], row["severity_index"]): row for row in tests
    }
    for row in summary:
        test = test_lookup.get((row["corruption"], row["severity_index"]))
        p_text = "reference" if test is None else f"{test['holm_p_within_curve']:.4g}"
        lines.append(
            f"| {row['corruption']} | {float(row['level']):g} | {row['episodes']} | "
            f"{100.0 * row['success_rate']:.2f}% | "
            f"[{100.0 * row['ci95_lower']:.1f}, {100.0 * row['ci95_upper']:.1f}] | {p_text} |"
        )
    clean = lookup(summary, corruption="gaussian-noise", severity_index=0)
    severe = {
        curve: max(
            (row for row in summary if row["corruption"] == curve),
            key=lambda row: row["severity_index"],
        )
        for curve in CURVES
    }
    significant = [row for row in tests if row["holm_p_within_curve"] < 0.05]

    def threshold_text(curve: str, threshold: float) -> str:
        matches = [
            row
            for row in summary
            if row["corruption"] == curve
            and row["severity_index"] > 0
            and row["success_rate"] < threshold
        ]
        if not matches:
            return f"not reached below {100.0 * threshold:.0f}%"
        first = min(matches, key=lambda row: row["severity_index"])
        return (
            f"level {float(first['level']):g} "
            f"({100.0 * first['success_rate']:.2f}% success)"
        )

    severe_class_accuracy = {
        curve: 100.0 * severe[curve]["mean_target_class_accuracy"]
        for curve in CURVES
    }
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- Shared Clean success is {100.0 * clean['success_rate']:.2f}%.",
            (
                "- No tested visual condition produces a statistically significant "
                "success change after within-curve Holm correction."
                if not significant
                else f"- {len(significant)} degraded condition(s) differ significantly "
                "from paired Clean after within-curve Holm correction."
            ),
            f"- At maximum severity, success remains {100.0 * severe['gaussian-noise']['success_rate']:.2f}% for noise, {100.0 * severe['gaussian-blur']['success_rate']:.2f}% for blur, and {100.0 * severe['brightness']['success_rate']:.2f}% for brightness reduction.",
            f"- Mean grounding error rises from {clean['mean_grounding_error_cm']:.3f} cm Clean to {severe['gaussian-noise']['mean_grounding_error_cm']:.3f}, {severe['gaussian-blur']['mean_grounding_error_cm']:.3f}, and {severe['brightness']['mean_grounding_error_cm']:.3f} cm at the three maximum severities.",
            f"- First sub-80% point: noise {threshold_text('gaussian-noise', 0.8)}; blur {threshold_text('gaussian-blur', 0.8)}; brightness {threshold_text('brightness', 0.8)}.",
            f"- First sub-50% point: noise {threshold_text('gaussian-noise', 0.5)}; blur {threshold_text('gaussian-blur', 0.5)}; brightness {threshold_text('brightness', 0.5)}.",
            f"- Target-class accuracy at maximum severity is {severe_class_accuracy['gaussian-noise']:.2f}% for noise, {severe_class_accuracy['gaussian-blur']:.2f}% for blur, and {severe_class_accuracy['brightness']:.2f}% for brightness reduction.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Report degradation curves and confidence intervals, not a single pass/fail threshold.",
            "- Use target-class/contact diagnostics to separate perception failure from manipulation failure.",
            "- Do not interpret visual simulation robustness as sim-to-real evidence.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, audit, run_metadata = load_and_audit(args.input_dirs)
    curves = audit["curves"]
    summary, task_rows, failure_rows = summarize(rows, curves)
    tests = paired_tests(rows, curves)

    write_csv(output_dir / "success_summary.csv", summary)
    write_csv(output_dir / "task_success_summary.csv", task_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)
    write_csv(output_dir / "paired_clean_tests.csv", tests)
    write_csv(output_dir / "source_runs.csv", run_metadata)
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_success(summary, curves, output_dir)
    plot_diagnostics(summary, curves, output_dir)
    plot_severe_task_heatmap(task_rows, curves, output_dir)
    plot_failure_taxonomy(failure_rows, curves, output_dir)
    write_report(output_dir, audit, summary, tests)
    print(
        f"Visual protocol audit: PASS ({len(rows)} episode rows)",
        flush=True,
    )
    print(f"Analysis saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
