#!/usr/bin/env python3
"""Audit and plot paired MiniVLA V3 camera-extrinsic drift results."""

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
COLOR = "#7A5195"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="final_report/07_camera_extrinsics",
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


def exact_mcnemar_p(level_zero_only: int, shifted_only: int) -> float:
    discordant = level_zero_only + shifted_only
    if discordant == 0:
        return 1.0
    smaller = min(level_zero_only, shifted_only)
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


def level_rows(rows: list[dict], level: int) -> list[dict]:
    return [row for row in rows if int(row["benchmark_level"]) == level]


def direction(row: dict, prefix: str) -> np.ndarray:
    return np.asarray(
        [
            float(row[f"{prefix}_x"]),
            float(row[f"{prefix}_y"]),
            float(row[f"{prefix}_z"]),
        ],
        dtype=np.float64,
    )


def load_and_audit(input_dirs: list[str]) -> tuple[list[dict], dict, list[dict]]:
    all_rows: list[dict] = []
    sources: list[dict] = []
    reference_protocol: tuple | None = None
    reference_levels: list[dict] | None = None

    for raw_dir in input_dirs:
        root = Path(raw_dir)
        summary_path = root / "benchmark_summary.json"
        episodes_path = root / "benchmark_episodes.csv"
        if not summary_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"Incomplete camera benchmark: {root}")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if summary.get("status") != "complete":
            raise ValueError(f"Camera benchmark is not complete: {root}")
        expected = int(summary["episodes_per_level"]) * int(
            summary["total_conditions"]
        )
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} rows in {root}, found {len(rows)}")
        if not summary.get("paired_scene_validation_passed"):
            raise ValueError(f"Paired-scene validation failed: {root}")
        if summary.get("affected_camera") != "agentview":
            raise ValueError(f"Unexpected affected camera in {root}")
        if summary.get("unaffected_camera") != "robot0_eye_in_hand":
            raise ValueError(f"Unexpected unaffected camera in {root}")
        if any(as_bool(row, "uses_privileged_execution_assistance") for row in rows):
            raise ValueError(f"Privileged execution assistance detected: {root}")
        if any(
            not as_bool(row, "camera_application_verified")
            or not as_bool(row, "camera_restoration_verified")
            or row["camera_name"] != "agentview"
            for row in rows
        ):
            raise ValueError(f"Camera application/restoration audit failed: {root}")

        levels = summary["levels"]
        zero_rows = level_rows(rows, 0)
        if len(zero_rows) != int(summary["episodes_per_level"]):
            raise ValueError(f"Invalid Level 0 baseline in {root}")
        if any(
            as_bool(row, "injected")
            or not np.isclose(
                float(row["initial_frame_mean_absolute_pixel_delta"]),
                0.0,
            )
            for row in zero_rows
        ):
            raise ValueError(f"Level 0 camera identity audit failed: {root}")

        zero_by_episode = {row["episode"]: row for row in zero_rows}
        zero_signature = [
            (row["episode"], row["task_type"], row["target_id"], row["scene_seed"])
            for row in zero_rows
        ]
        for specification in levels:
            level = int(specification["level"])
            group = level_rows(rows, level)
            if len(group) != int(summary["episodes_per_level"]):
                raise ValueError(f"Invalid Level {level} count in {root}")
            signature = [
                (row["episode"], row["task_type"], row["target_id"], row["scene_seed"])
                for row in group
            ]
            if signature != zero_signature:
                raise ValueError(f"Paired scene mismatch at Level {level}: {root}")
            for row in group:
                if not np.isclose(
                    float(row["actual_translation_mm"]),
                    float(specification["translation_mm"]),
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    raise ValueError(f"Translation mismatch at Level {level}: {root}")
                if not np.isclose(
                    float(row["actual_rotation_deg"]),
                    float(specification["rotation_deg"]),
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    raise ValueError(f"Rotation mismatch at Level {level}: {root}")
                reference = zero_by_episode[row["episode"]]
                if not np.allclose(
                    direction(row, "translation_direction"),
                    direction(reference, "translation_direction"),
                    rtol=0.0,
                    atol=1e-12,
                ) or not np.allclose(
                    direction(row, "rotation_axis"),
                    direction(reference, "rotation_axis"),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Unpaired camera direction at Level {level}: {root}"
                    )
                if level > 0 and (
                    not as_bool(row, "injected")
                    or float(row["initial_frame_mean_absolute_pixel_delta"])
                    <= 0.0
                ):
                    raise ValueError(
                        f"Camera shift did not change the first frame: {root}"
                    )

        protocol = (
            summary["protocol_version"],
            summary["policy"],
            summary["episodes_per_level"],
            summary["ensemble_mode"],
            summary["temporal_profile"],
            summary["affected_camera"],
            summary["unaffected_camera"],
        )
        if reference_protocol is None:
            reference_protocol = protocol
            reference_levels = levels
        elif protocol != reference_protocol or levels != reference_levels:
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

    audit = {
        "status": "PASS",
        "protocol_version": reference_protocol[0],
        "policy": reference_protocol[1],
        "seeds": len(input_dirs),
        "episode_rows": len(all_rows),
        "paired_scene_validation_passed": True,
        "paired_direction_validation_passed": True,
        "application_verification_rate": 1.0,
        "restoration_verification_rate": 1.0,
        "level_zero_image_identity_passed": True,
        "nonzero_first_frame_injection_rate": 1.0,
        "affected_camera": "agentview",
        "unaffected_camera": "robot0_eye_in_hand",
        "levels": reference_levels,
    }
    return all_rows, audit, sources


def summarize(
    rows: list[dict],
    levels: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    summary_rows: list[dict] = []
    task_rows: list[dict] = []
    failure_rows: list[dict] = []
    for specification in levels:
        level = int(specification["level"])
        group = level_rows(rows, level)
        successes = sum(as_bool(row, "task_success") for row in group)
        lower, upper = wilson_interval(successes, len(group))
        psnr_values = [
            value
            for row in group
            if (value := optional_float(row, "initial_frame_psnr_db")) is not None
        ]
        summary_rows.append(
            {
                "level": level,
                "translation_mm": specification["translation_mm"],
                "rotation_deg": specification["rotation_deg"],
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
                "mean_grounding_error_cm": np.mean(
                    [float(row["mean_grounding_error_cm"]) for row in group]
                ),
                "mean_target_class_accuracy": np.mean(
                    [float(row["mean_target_class_accuracy"]) for row in group]
                ),
                "mean_initial_frame_pixel_delta": np.mean(
                    [
                        float(row["initial_frame_mean_absolute_pixel_delta"])
                        for row in group
                    ]
                ),
                "mean_initial_frame_psnr_db": (
                    np.mean(psnr_values) if psnr_values else ""
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
                    "level": level,
                    "translation_mm": specification["translation_mm"],
                    "rotation_deg": specification["rotation_deg"],
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
                    "level": level,
                    "translation_mm": specification["translation_mm"],
                    "rotation_deg": specification["rotation_deg"],
                    "failure_category": category,
                    "count": count,
                    "rate_over_all_episodes": count / len(group),
                }
            )
    return summary_rows, task_rows, failure_rows


def paired_tests(rows: list[dict], levels: list[dict]) -> list[dict]:
    baseline = {
        (row["source_run"], row["episode"]): as_bool(row, "task_success")
        for row in level_rows(rows, 0)
    }
    tests: list[dict] = []
    for specification in levels:
        level = int(specification["level"])
        if level == 0:
            continue
        shifted = {
            (row["source_run"], row["episode"]): as_bool(row, "task_success")
            for row in level_rows(rows, level)
        }
        if baseline.keys() != shifted.keys():
            raise ValueError(f"Pairing mismatch for camera Level {level}")
        baseline_only = sum(
            baseline[key] and not shifted[key] for key in baseline
        )
        shifted_only = sum(
            not baseline[key] and shifted[key] for key in baseline
        )
        tests.append(
            {
                "level": level,
                "translation_mm": specification["translation_mm"],
                "rotation_deg": specification["rotation_deg"],
                "level_zero_only_successes": baseline_only,
                "shifted_only_successes": shifted_only,
                "success_difference_percentage_points": 100.0
                * (
                    np.mean(list(shifted.values()))
                    - np.mean(list(baseline.values()))
                ),
                "mcnemar_p": exact_mcnemar_p(baseline_only, shifted_only),
            }
        )
    adjusted = holm_adjust([row["mcnemar_p"] for row in tests])
    for row, value in zip(tests, adjusted):
        row["holm_p"] = value
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


def level_label(row: dict) -> str:
    return (
        f"L{int(row['level'])}\n"
        f"{float(row['translation_mm']):g} mm / "
        f"{float(row['rotation_deg']):g} deg"
    )


def plot_success(summary: list[dict], output_dir: Path) -> None:
    x = np.arange(len(summary))
    rates = 100.0 * np.asarray([row["success_rate"] for row in summary])
    lower = np.maximum(
        0.0,
        rates - 100.0 * np.asarray([row["ci95_lower"] for row in summary]),
    )
    upper = np.maximum(
        0.0,
        100.0 * np.asarray([row["ci95_upper"] for row in summary]) - rates,
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.errorbar(
        x,
        rates,
        yerr=np.vstack([lower, upper]),
        marker="o",
        linewidth=2.4,
        capsize=5,
        color=COLOR,
    )
    for index, rate in enumerate(rates):
        ax.annotate(
            f"{rate:.1f}",
            (index, rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xticks(x, labels=[level_label(row) for row in summary])
    ax.set_ylabel("Task success rate (%)")
    ax.set_xlabel("Camera extrinsic severity")
    ax.set_ylim(0, 112)
    ax.set_title("MiniVLA V3 Camera-Extrinsic Robustness", pad=12)
    fig.tight_layout()
    save(fig, output_dir, "01_camera_extrinsic_success_decay")


def plot_diagnostics(summary: list[dict], output_dir: Path) -> None:
    x = np.arange(len(summary))
    labels = [f"L{int(row['level'])}" for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    axes[0].plot(
        x,
        [row["mean_grounding_error_cm"] for row in summary],
        marker="o",
        linewidth=2.2,
        color="#0072B2",
        label="Grounding error",
    )
    contact_axis = axes[0].twinx()
    contact_axis.plot(
        x,
        100.0 * np.asarray([row["target_contact_rate"] for row in summary]),
        marker="s",
        color="#009E73",
        label="Target contact",
    )
    axes[0].set_xticks(x, labels=labels)
    axes[0].set_xlabel("Camera extrinsic level")
    axes[0].set_ylabel("Mean grounding error (cm)")
    contact_axis.set_ylabel("Target contact rate (%)")
    contact_axis.set_ylim(0, 105)
    axes[0].set_title("Policy grounding and contact")

    axes[1].plot(
        x,
        [row["mean_initial_frame_pixel_delta"] for row in summary],
        marker="o",
        linewidth=2.2,
        color="#D55E00",
    )
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_xlabel("Camera extrinsic level")
    axes[1].set_ylabel("Initial-frame mean absolute pixel delta")
    axes[1].set_title("Rendered image change from pose drift")

    fig.suptitle("Camera Calibration Drift Diagnostics")
    fig.tight_layout()
    save(fig, output_dir, "02_camera_extrinsic_diagnostics")


def plot_task_heatmap(
    task_rows: list[dict],
    summary: list[dict],
    output_dir: Path,
) -> None:
    matrix = np.zeros((len(TASKS), len(summary)), dtype=np.float64)
    for column, level_summary in enumerate(summary):
        for row_index, task in enumerate(TASKS):
            matrix[row_index, column] = 100.0 * lookup(
                task_rows,
                level=level_summary["level"],
                task=task,
            )["success_rate"]
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center")
    ax.set_xticks(
        range(len(summary)),
        labels=[f"L{int(row['level'])}" for row in summary],
    )
    ax.set_yticks(
        range(len(TASKS)),
        labels=[task.replace("_", "-") for task in TASKS],
    )
    ax.set_title("Per-Task Success Across Camera-Extrinsic Levels")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Task success rate (%)")
    fig.tight_layout()
    save(fig, output_dir, "03_camera_extrinsic_task_heatmap")


def plot_failure_taxonomy(
    failure_rows: list[dict],
    summary: list[dict],
    output_dir: Path,
) -> None:
    categories = sorted({row["failure_category"] for row in failure_rows})
    x = np.arange(len(summary))
    bottoms = np.zeros(len(summary), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    palette = plt.get_cmap("tab10")
    for category_index, category in enumerate(categories):
        values = np.asarray(
            [
                100.0
                * sum(
                    row["rate_over_all_episodes"]
                    for row in failure_rows
                    if row["level"] == level_summary["level"]
                    and row["failure_category"] == category
                )
                for level_summary in summary
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
    ax.set_xticks(x, labels=[f"L{int(row['level'])}" for row in summary])
    ax.set_ylabel("Failure rate over all episodes (%)")
    ax.set_title("Failure Taxonomy Under Camera-Extrinsic Drift")
    if categories:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    save(fig, output_dir, "04_camera_extrinsic_failure_taxonomy")


def write_report(
    output_dir: Path,
    audit: dict,
    summary: list[dict],
    tests: list[dict],
) -> None:
    test_map = {row["level"]: row for row in tests}
    lines = [
        "# MiniVLA V3 Camera-Extrinsic Robustness",
        "",
        "## Protocol Audit",
        "",
        f"- Status: {audit['status']}.",
        f"- Raw episode rows: {audit['episode_rows']}.",
        f"- Seeds: {audit['seeds']}.",
        "- MuJoCo agentview position and quaternion are modified directly.",
        "- Scene-paired translation directions and rotation axes verified.",
        "- Application, first-frame refresh, and per-episode restoration verified.",
        "- Wrist camera remains unchanged; no privileged execution assistance.",
        "",
        "## Success Rates",
        "",
        "| Level | Translation | Rotation | Episodes | Success | 95% Wilson CI | Holm p vs L0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        test = test_map.get(row["level"])
        p_text = "reference" if test is None else f"{test['holm_p']:.4g}"
        lines.append(
            f"| {row['level']} | {float(row['translation_mm']):g} mm | "
            f"{float(row['rotation_deg']):g} deg | {row['episodes']} | "
            f"{100.0 * row['success_rate']:.2f}% | "
            f"[{100.0 * row['ci95_lower']:.1f}, "
            f"{100.0 * row['ci95_upper']:.1f}] | {p_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Treat translation and rotation as a combined calibration severity, not separable causal effects.",
            "- Compare paired episodes to distinguish calibration sensitivity from scene difficulty.",
            "- Do not interpret simulated camera drift as complete real-camera calibration evidence.",
        ]
    )
    (output_dir / "analysis_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, audit, sources = load_and_audit(args.input_dirs)
    levels = audit["levels"]
    summary, task_rows, failure_rows = summarize(rows, levels)
    tests = paired_tests(rows, levels)
    write_csv(output_dir / "success_summary.csv", summary)
    write_csv(output_dir / "task_success_summary.csv", task_rows)
    write_csv(
        output_dir / "failure_taxonomy.csv",
        failure_rows,
        fieldnames=[
            "level",
            "translation_mm",
            "rotation_deg",
            "failure_category",
            "count",
            "rate_over_all_episodes",
        ],
    )
    write_csv(output_dir / "paired_level_zero_tests.csv", tests)
    write_csv(output_dir / "source_runs.csv", sources)
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_success(summary, output_dir)
    plot_diagnostics(summary, output_dir)
    plot_task_heatmap(task_rows, summary, output_dir)
    plot_failure_taxonomy(failure_rows, summary, output_dir)
    write_report(output_dir, audit, summary, tests)
    print(
        f"Camera protocol audit: PASS ({len(rows)} episode rows)",
        flush=True,
    )
    print(f"Analysis saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
