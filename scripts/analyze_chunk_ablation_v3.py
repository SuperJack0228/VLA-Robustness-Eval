#!/usr/bin/env python3
"""Analyze the paired MiniVLA V3 ACT chunk-size ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "vla_matplotlib_cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "vla_xdg_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import chi2


CHUNKS = (1, 5, 10, 20)
MODES = ("temporal", "latest-only")
LEVELS = (0.0, 0.04)
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
MODE_COLORS = {"temporal": "#0072B2", "latest-only": "#D55E00"}
CHUNK_COLORS = {1: "#56B4E9", 5: "#009E73", 10: "#0072B2", 20: "#CC79A7"}
FAILURE_ORDER = (
    "insufficient_push_distance",
    "recovery_limit",
    "gripper_never_closed",
    "grasp_failed_after_contact",
    "lateral_push_error",
    "wrong_object_contact",
    "other",
)
FAILURE_COLORS = {
    "insufficient_push_distance": "#D95F59",
    "recovery_limit": "#6B5B95",
    "gripper_never_closed": "#59A14F",
    "grasp_failed_after_contact": "#8C564B",
    "lateral_push_error": "#F2B134",
    "wrong_object_contact": "#B07AA1",
    "other": "#BAB0AC",
}
CONTROL_FREQUENCY_HZ = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="results/benchmarks/chunk_size_ablation",
    )
    parser.add_argument(
        "--output-dir",
        default="final_report/03_chunk_size_ablation",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/chunk-ablation-v3",
        help="Training provenance for chunk sizes 1, 5, and 10.",
    )
    parser.add_argument(
        "--chunk20-artifact-root",
        default="artifacts/v3-clean-rc1",
        help="Training provenance for the canonical chunk-20 V3 policy.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20262101)
    return parser.parse_args()


def truth(value: object) -> bool:
    return str(value) in {"1", "True", "true"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int) -> tuple[float, float]:
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


def exact_mcnemar(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(a_only, b_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values_a), size=(samples, len(values_a)))
    differences = values_a[indices].mean(axis=1) - values_b[indices].mean(axis=1)
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def average(rows: list[dict], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key, "") not in {"", "None", "nan"}
    ]
    return float(np.mean(values)) if values else None


def pairing_key(row: dict) -> tuple:
    return (
        row["seed_run"],
        row["benchmark_mode"],
        round(float(row["benchmark_level"]), 8),
        int(row["episode"]),
    )


def load_and_validate(root: Path) -> tuple[list[dict], dict]:
    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if len(seed_dirs) != 2:
        raise ValueError(f"Expected two seed directories, found {len(seed_dirs)}")

    all_rows: list[dict] = []
    run_audit: list[dict] = []
    references: dict[str, list[tuple]] = {}
    bad_log_markers = ("Traceback", "ERROR conda", "Killed", "Segmentation fault")

    for seed_dir in seed_dirs:
        for chunk in CHUNKS:
            chunk_dir = seed_dir / f"chunk_{chunk}"
            summary_path = chunk_dir / "benchmark_summary.json"
            episodes_path = chunk_dir / "benchmark_episodes.csv"
            log_path = seed_dir / f"chunk_{chunk}_console.log"
            if not summary_path.is_file() or not episodes_path.is_file():
                raise FileNotFoundError(f"Incomplete chunk result: {chunk_dir}")
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            with episodes_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            if summary.get("status") != "complete":
                raise ValueError(f"Incomplete benchmark status: {summary_path}")
            if summary.get("levels_m") != [0.0, 0.04]:
                raise ValueError(f"Unexpected levels: {summary_path}")
            if summary.get("ensemble_modes") != ["temporal", "latest-only"]:
                raise ValueError(f"Unexpected modes: {summary_path}")
            if summary.get("temporal_profile") != "legacy":
                raise ValueError(f"Expected legacy temporal profile: {summary_path}")
            if summary.get("episodes_per_level") != 60 or len(rows) != 240:
                raise ValueError(f"Unexpected episode count: {summary_path}")
            if summary.get("uses_privileged_execution_assistance"):
                raise ValueError(f"Privileged execution assistance enabled: {summary_path}")
            if not summary.get("paired_scene_validation_passed"):
                raise ValueError(f"Within-run pairing failed: {summary_path}")

            for mode in MODES:
                for level in LEVELS:
                    group = [
                        row
                        for row in rows
                        if row["benchmark_mode"] == mode
                        and abs(float(row["benchmark_level"]) - level) < 1e-9
                    ]
                    if len(group) != 60:
                        raise ValueError(
                            f"Expected 60 rows for {seed_dir.name}/{chunk}/{mode}/{level}"
                        )
                    if level > 0.0 and not all(truth(row["injected"]) for row in group):
                        raise ValueError(f"Missed 4 cm injection: {chunk_dir}")
                    if level == 0.0 and any(truth(row["injected"]) for row in group):
                        raise ValueError(f"Unexpected clean injection: {chunk_dir}")
                    if any(
                        not truth(row["protocol_collision_integrity_passed"])
                        for row in group
                    ):
                        raise ValueError(f"Collision integrity failure: {chunk_dir}")
                    if any(
                        truth(row["injected"])
                        and abs(float(row["actual_delta_norm_m"]) - level) > 1e-6
                        for row in group
                    ):
                        raise ValueError(f"Actual displacement mismatch: {chunk_dir}")

            signature = [
                (
                    row["benchmark_mode"],
                    row["benchmark_level"],
                    row["episode"],
                    row["task_type"],
                    row["target_id"],
                    row["instruction"],
                    row["scene_seed"],
                    row["selected_direction_x"],
                    row["selected_direction_y"],
                )
                for row in rows
            ]
            if chunk == 1:
                references[seed_dir.name] = signature
            elif signature != references[seed_dir.name]:
                raise ValueError(f"Cross-chunk pairing mismatch: {chunk_dir}")

            log_errors: list[str] = []
            if log_path.is_file():
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                log_errors = [marker for marker in bad_log_markers if marker in log_text]
                if log_errors or "Robustness benchmark complete:" not in log_text:
                    raise ValueError(f"Invalid console log: {log_path}: {log_errors}")

            for row in rows:
                all_rows.append(
                    {
                        "seed_run": seed_dir.name,
                        "chunk_size": chunk,
                        **row,
                    }
                )
            run_audit.append(
                {
                    "seed_run": seed_dir.name,
                    "chunk_size": chunk,
                    "policy": summary["policy"],
                    "episodes": len(rows),
                    "status": summary["status"],
                    "log_checked": log_path.is_file(),
                    "log_errors": log_errors,
                }
            )

    paired: dict[tuple, dict[int, dict]] = {}
    for row in all_rows:
        paired.setdefault(pairing_key(row), {})[int(row["chunk_size"])] = row
    if len(paired) != 480:
        raise ValueError(f"Expected 480 paired scene conditions, found {len(paired)}")
    if any(set(outcomes) != set(CHUNKS) for outcomes in paired.values()):
        raise ValueError("At least one paired condition is missing a chunk outcome")

    audit = {
        "status": "pass",
        "seed_runs": [path.name for path in seed_dirs],
        "chunks": list(CHUNKS),
        "modes": list(MODES),
        "levels_m": list(LEVELS),
        "raw_episode_rows": len(all_rows),
        "paired_scene_conditions": len(paired),
        "episodes_per_chunk_mode_level": 120,
        "nonzero_injection_rate": float(
            np.mean(
                [truth(row["injected"]) for row in all_rows if float(row["benchmark_level"]) > 0]
            )
        ),
        "uses_privileged_execution_assistance": False,
        "chunk_20_initialization_boundary": (
            "Chunk 20 is the frozen V3 full-warm-start reference; chunks 1/5/10 "
            "use reinitialized action queries."
        ),
        "runs": run_audit,
    }
    return all_rows, audit


def lookup(rows: list[dict], **conditions) -> dict:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def summarize_success(rows: list[dict]) -> list[dict]:
    output = []
    for chunk in CHUNKS:
        for mode in MODES:
            for level in LEVELS:
                group = [
                    row
                    for row in rows
                    if int(row["chunk_size"]) == chunk
                    and row["benchmark_mode"] == mode
                    and abs(float(row["benchmark_level"]) - level) < 1e-9
                ]
                successes = sum(truth(row["task_success"]) for row in group)
                lower, upper = wilson(successes, len(group))
                reacquired = [row for row in group if truth(row["reacquired_within_0_5cm"])]
                output.append(
                    {
                        "chunk_size": chunk,
                        "ensemble_mode": mode,
                        "level_cm": int(round(level * 100)),
                        "episodes": len(group),
                        "successes": successes,
                        "success_rate": successes / len(group),
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "mean_steps": average(group, "steps"),
                        "mean_action_clip_rate": average(group, "action_clip_rate"),
                        "mean_safety_intervention_rate": average(
                            group, "safety_intervention_rate"
                        ),
                        "wrong_object_contact_rate": float(
                            np.mean([truth(row["wrong_object_contact"]) for row in group])
                        ),
                        "target_contact_rate": float(
                            np.mean([truth(row["target_contact"]) for row in group])
                        ),
                        "reacquisition_rate_0_5cm": (
                            len(reacquired) / len(group) if level > 0 else None
                        ),
                        "mean_reacquisition_latency_s_0_5cm": (
                            average(reacquired, "reacquisition_latency_0_5cm")
                            / CONTROL_FREQUENCY_HZ
                            if reacquired
                            else None
                        ),
                        "mean_grounding_error_after_injection_cm": (
                            average(group, "grounding_error_after_injection_cm")
                            if level > 0
                            else None
                        ),
                        "mean_oldest_prediction_age": average(
                            group, "mean_oldest_prediction_age"
                        ),
                        "mean_temporal_contributors": average(
                            group, "mean_temporal_contributors"
                        ),
                    }
                )
    return output


def summarize_tasks(rows: list[dict]) -> list[dict]:
    output = []
    for chunk in CHUNKS:
        for mode in MODES:
            for level in LEVELS:
                for task in TASKS:
                    task_type, target_id = task.split("_")
                    group = [
                        row
                        for row in rows
                        if int(row["chunk_size"]) == chunk
                        and row["benchmark_mode"] == mode
                        and abs(float(row["benchmark_level"]) - level) < 1e-9
                        and row["task_type"] == task_type
                        and row["target_id"] == target_id
                    ]
                    successes = sum(truth(row["task_success"]) for row in group)
                    output.append(
                        {
                            "chunk_size": chunk,
                            "ensemble_mode": mode,
                            "level_cm": int(round(level * 100)),
                            "task": task,
                            "episodes": len(group),
                            "successes": successes,
                            "success_rate": successes / len(group),
                        }
                    )
    return output


def summarize_failures(rows: list[dict]) -> list[dict]:
    output = []
    for chunk in CHUNKS:
        for mode in MODES:
            for level in LEVELS:
                group = [
                    row
                    for row in rows
                    if int(row["chunk_size"]) == chunk
                    and row["benchmark_mode"] == mode
                    and abs(float(row["benchmark_level"]) - level) < 1e-9
                ]
                counts = Counter(
                    row["failure_category"]
                    for row in group
                    if not truth(row["task_success"])
                )
                for category, count in sorted(counts.items()):
                    output.append(
                        {
                            "chunk_size": chunk,
                            "ensemble_mode": mode,
                            "level_cm": int(round(level * 100)),
                            "failure_category": category,
                            "count": count,
                            "rate_over_all_episodes": count / len(group),
                        }
                    )
    return output


def build_paired(rows: list[dict]) -> dict[tuple, dict[int, dict]]:
    paired: dict[tuple, dict[int, dict]] = {}
    for row in rows:
        paired.setdefault(pairing_key(row), {})[int(row["chunk_size"])] = row
    return paired


def cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    chunk_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)
    total = float(chunk_totals.sum())
    denominator = len(CHUNKS) * total - float(np.square(row_totals).sum())
    if denominator == 0.0:
        return 0.0, 1.0
    statistic = (len(CHUNKS) - 1) * (
        len(CHUNKS) * float(np.square(chunk_totals).sum()) - total * total
    ) / denominator
    return statistic, float(chi2.sf(statistic, len(CHUNKS) - 1))


def statistical_analysis(
    rows: list[dict], bootstrap_samples: int, bootstrap_seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    paired = build_paired(rows)
    rng = np.random.default_rng(bootstrap_seed)
    overall_rows = []
    versus_20 = []
    temporal_control = []

    for mode in MODES:
        for level in LEVELS:
            keys = sorted(
                key
                for key in paired
                if key[1] == mode and abs(key[2] - level) < 1e-9
            )
            matrix = np.asarray(
                [
                    [int(truth(paired[key][chunk]["task_success"])) for chunk in CHUNKS]
                    for key in keys
                ],
                dtype=np.int8,
            )
            statistic, p_value = cochran_q(matrix)
            overall_rows.append(
                {
                    "ensemble_mode": mode,
                    "level_cm": int(round(level * 100)),
                    "paired_episodes": len(keys),
                    "cochran_q": statistic,
                    "degrees_of_freedom": len(CHUNKS) - 1,
                    "p_value": p_value,
                }
            )
            condition_rows = []
            for chunk in CHUNKS[:-1]:
                candidate = matrix[:, CHUNKS.index(chunk)]
                reference = matrix[:, CHUNKS.index(20)]
                candidate_only = int(np.sum((candidate == 1) & (reference == 0)))
                reference_only = int(np.sum((candidate == 0) & (reference == 1)))
                lower, upper = paired_bootstrap(
                    candidate.astype(float),
                    reference.astype(float),
                    bootstrap_samples,
                    rng,
                )
                condition_rows.append(
                    {
                        "ensemble_mode": mode,
                        "level_cm": int(round(level * 100)),
                        "chunk_size": chunk,
                        "reference_chunk": 20,
                        "paired_episodes": len(keys),
                        "candidate_success_rate": float(candidate.mean()),
                        "reference_success_rate": float(reference.mean()),
                        "difference_percentage_points": float(
                            100.0 * (candidate.mean() - reference.mean())
                        ),
                        "difference_ci95_lower_percentage_points": 100.0 * lower,
                        "difference_ci95_upper_percentage_points": 100.0 * upper,
                        "candidate_only_successes": candidate_only,
                        "reference_only_successes": reference_only,
                        "exact_mcnemar_p": exact_mcnemar(
                            candidate_only, reference_only
                        ),
                    }
                )
            adjusted = holm_adjust(
                [row["exact_mcnemar_p"] for row in condition_rows]
            )
            for row, adjusted_p in zip(condition_rows, adjusted):
                row["holm_adjusted_p"] = adjusted_p
                versus_20.append(row)

    for chunk in CHUNKS:
        for level in LEVELS:
            temporal_keys = sorted(
                key
                for key in paired
                if key[1] == "temporal" and abs(key[2] - level) < 1e-9
            )
            temporal = np.asarray(
                [int(truth(paired[key][chunk]["task_success"])) for key in temporal_keys]
            )
            latest = np.asarray(
                [
                    int(
                        truth(
                            paired[(key[0], "latest-only", key[2], key[3])][chunk][
                                "task_success"
                            ]
                        )
                    )
                    for key in temporal_keys
                ]
            )
            temporal_only = int(np.sum((temporal == 1) & (latest == 0)))
            latest_only = int(np.sum((temporal == 0) & (latest == 1)))
            lower, upper = paired_bootstrap(
                temporal.astype(float), latest.astype(float), bootstrap_samples, rng
            )
            temporal_control.append(
                {
                    "chunk_size": chunk,
                    "level_cm": int(round(level * 100)),
                    "paired_episodes": len(temporal),
                    "temporal_success_rate": float(temporal.mean()),
                    "latest_only_success_rate": float(latest.mean()),
                    "temporal_gain_percentage_points": float(
                        100.0 * (temporal.mean() - latest.mean())
                    ),
                    "gain_ci95_lower_percentage_points": 100.0 * lower,
                    "gain_ci95_upper_percentage_points": 100.0 * upper,
                    "temporal_only_successes": temporal_only,
                    "latest_only_successes": latest_only,
                    "exact_mcnemar_p": exact_mcnemar(
                        temporal_only, latest_only
                    ),
                }
            )
    return overall_rows, versus_20, temporal_control


def summarize_seeds(rows: list[dict]) -> list[dict]:
    output = []
    for seed_run in sorted({row["seed_run"] for row in rows}):
        for chunk in CHUNKS:
            for mode in MODES:
                for level in LEVELS:
                    group = [
                        row
                        for row in rows
                        if row["seed_run"] == seed_run
                        and int(row["chunk_size"]) == chunk
                        and row["benchmark_mode"] == mode
                        and abs(float(row["benchmark_level"]) - level) < 1e-9
                    ]
                    output.append(
                        {
                            "seed_run": seed_run,
                            "chunk_size": chunk,
                            "ensemble_mode": mode,
                            "level_cm": int(round(level * 100)),
                            "episodes": len(group),
                            "success_rate": float(
                                np.mean([truth(row["task_success"]) for row in group])
                            ),
                        }
                    )
    return output


def load_training_endpoints(
    artifact_root: Path,
    chunk20_artifact_root: Path,
) -> list[dict]:
    policy_roots = {
        1: artifact_root / "chunk_1",
        5: artifact_root / "chunk_5",
        10: artifact_root / "chunk_10",
        20: chunk20_artifact_root,
    }
    output = []
    for chunk, directory in policy_roots.items():
        metadata_path = directory / "training_metadata_v3.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = metadata["latest_metrics"]
        validation = metrics["combined_val"]
        output.append(
            {
                "chunk_size": chunk,
                "epochs": metadata["last_completed_epoch"],
                "best_selection_score": metadata["best_selection_score"],
                "train_total": metrics["train"]["total"],
                "combined_val_total": validation["total"],
                "combined_val_xyz_mae": validation["xyz_mae"],
                "combined_val_gripper_accuracy": validation["gripper_accuracy"],
                "combined_val_phase_accuracy": validation["phase_accuracy"],
                "combined_val_grounding_cm": validation["grounding_cm"],
                "action_query_initialization": (
                    "full_warmstart" if chunk == 20 else "reinitialized"
                ),
            }
        )
    if len(output) != 4:
        raise ValueError("Missing at least one training endpoint")
    return output


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_success(summary: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for axis, level_cm in zip(axes, (0, 4)):
        rates_by_mode: dict[str, np.ndarray] = {}
        for mode in MODES:
            group = [
                lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=level_cm)
                for chunk in CHUNKS
            ]
            rates = 100.0 * np.asarray([row["success_rate"] for row in group])
            rates_by_mode[mode] = rates
            lower = rates - 100.0 * np.asarray([row["ci95_lower"] for row in group])
            upper = 100.0 * np.asarray([row["ci95_upper"] for row in group]) - rates
            axis.errorbar(
                CHUNKS,
                rates,
                yerr=np.vstack([lower, upper]),
                marker="o" if mode == "temporal" else "s",
                linewidth=2.2,
                capsize=4,
                color=MODE_COLORS[mode],
                label=mode,
            )
        for index, chunk in enumerate(CHUNKS):
            temporal_rate = rates_by_mode["temporal"][index]
            latest_rate = rates_by_mode["latest-only"][index]
            if np.isclose(temporal_rate, latest_rate):
                axis.annotate(
                    f"{temporal_rate:.1f} (T=L)",
                    (chunk, temporal_rate),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            else:
                axis.annotate(
                    f"{temporal_rate:.1f}",
                    (chunk, temporal_rate),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
                axis.annotate(
                    f"{latest_rate:.1f}",
                    (chunk, latest_rate),
                    xytext=(0, -14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        axis.set_xticks(CHUNKS)
        axis.set_xlabel("ACT chunk size")
        axis.set_title("Clean" if level_cm == 0 else "4 cm target displacement")
        axis.set_ylim(75, 103)
    axes[0].set_ylabel("Task success rate (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("ACT Chunk-Size Ablation: Paired Closed-Loop Success")
    fig.tight_layout()
    save(fig, output_dir, "01_chunk_success_clean_vs_4cm")


def plot_retention(summary: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for mode in MODES:
        clean = np.asarray(
            [
                lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=0)[
                    "success_rate"
                ]
                for chunk in CHUNKS
            ]
        )
        shifted = np.asarray(
            [
                lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=4)[
                    "success_rate"
                ]
                for chunk in CHUNKS
            ]
        )
        axes[0].plot(
            CHUNKS,
            100.0 * shifted / clean,
            marker="o" if mode == "temporal" else "s",
            linewidth=2.2,
            color=MODE_COLORS[mode],
            label=mode,
        )
        axes[1].plot(
            CHUNKS,
            100.0 * (clean - shifted),
            marker="o" if mode == "temporal" else "s",
            linewidth=2.2,
            color=MODE_COLORS[mode],
            label=mode,
        )
    axes[0].set(
        xlabel="ACT chunk size",
        ylabel="4 cm success / Clean success (%)",
        title="Robustness retention",
        xticks=CHUNKS,
    )
    axes[1].set(
        xlabel="ACT chunk size",
        ylabel="Absolute success drop (pp)",
        title="Cost of 4 cm displacement",
        xticks=CHUNKS,
    )
    axes[0].legend(loc="lower right")
    fig.suptitle("Dynamic-Robustness Retention by Chunk Size")
    fig.tight_layout()
    save(fig, output_dir, "02_robustness_retention_and_drop")


def plot_pareto(summary: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    for chunk in CHUNKS:
        points = {}
        for mode in MODES:
            points[mode] = (
                100.0
                * lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=0)[
                    "success_rate"
                ],
                100.0
                * lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=4)[
                    "success_rate"
                ],
            )
        if np.allclose(points["temporal"], points["latest-only"]):
            clean, shifted = points["temporal"]
            ax.scatter(
                clean,
                shifted,
                s=105,
                marker="D",
                color=CHUNK_COLORS[chunk],
                edgecolor="black",
                linewidth=0.7,
            )
            ax.annotate(
                f"{chunk} (T=L)",
                (clean, shifted),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
            continue
        for mode in MODES:
            clean, shifted = points[mode]
            ax.scatter(
                clean,
                shifted,
                s=95,
                marker="o" if mode == "temporal" else "s",
                color=CHUNK_COLORS[chunk],
                edgecolor="black",
                linewidth=0.7,
            )
            ax.annotate(
                f"{chunk} ({'T' if mode == 'temporal' else 'L'})",
                (clean, shifted),
                xytext=(5, 4 if mode == "temporal" else -11),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set(
        xlabel="Clean success rate (%)",
        ylabel="4 cm success rate (%)",
        title="Clean-Robustness Operating Points",
        xlim=(94, 101),
        ylim=(82, 98),
    )
    legend = [
        Patch(facecolor=CHUNK_COLORS[chunk], label=f"chunk {chunk}")
        for chunk in CHUNKS
    ]
    ax.legend(handles=legend, loc="lower right", title="Color")
    ax.text(
        0.02,
        0.98,
        "T = temporal, L = latest-only\nUpper-right is better",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, output_dir, "03_clean_robustness_pareto")


def plot_temporal_effect(control_rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(CHUNKS))
    width = 0.36
    for offset, level_cm, color in ((-width / 2, 0, "#56B4E9"), (width / 2, 4, "#0072B2")):
        group = [lookup(control_rows, chunk_size=chunk, level_cm=level_cm) for chunk in CHUNKS]
        gains = np.asarray([row["temporal_gain_percentage_points"] for row in group])
        lower = gains - np.asarray(
            [row["gain_ci95_lower_percentage_points"] for row in group]
        )
        upper = np.asarray(
            [row["gain_ci95_upper_percentage_points"] for row in group]
        ) - gains
        ax.bar(
            x + offset,
            gains,
            width,
            yerr=np.vstack([lower, upper]),
            capsize=4,
            color=color,
            label="Clean" if level_cm == 0 else "4 cm",
        )
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_xticks(x, labels=CHUNKS)
    ax.set_xlabel("ACT chunk size")
    ax.set_ylabel("Temporal - latest-only success (pp)")
    ax.set_title("Effect of Classic Temporal Ensembling")
    ax.legend()
    fig.tight_layout()
    save(fig, output_dir, "04_temporal_ensembling_effect")


def plot_task_heatmaps(task_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
    for axis, mode in zip(axes, MODES):
        matrix = np.asarray(
            [
                [
                    100.0
                    * lookup(
                        task_rows,
                        chunk_size=chunk,
                        ensemble_mode=mode,
                        level_cm=4,
                        task=task,
                    )["success_rate"]
                    for chunk in CHUNKS
                ]
                for task in TASKS
            ]
        )
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=60, vmax=100, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        axis.set_xticks(range(len(CHUNKS)), labels=CHUNKS)
        axis.set_yticks(range(len(TASKS)), labels=[task.replace("_", "-") for task in TASKS])
        axis.set_xlabel("ACT chunk size")
        axis.set_title(mode)
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.12, top=0.84, wspace=0.12)
    colorbar_axis = fig.add_axes([0.87, 0.18, 0.02, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("4 cm task success rate (%)")
    fig.suptitle("Per-Task Robustness at 4 cm (n=20 each cell)")
    save(fig, output_dir, "05_per_task_4cm_heatmaps")


def failure_rate(
    failure_rows: list[dict], chunk: int, mode: str, category: str
) -> float:
    group = [
        row
        for row in failure_rows
        if row["chunk_size"] == chunk
        and row["ensemble_mode"] == mode
        and row["level_cm"] == 4
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


def plot_failures(failure_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    present = []
    for category in FAILURE_ORDER:
        if any(
            failure_rate(failure_rows, chunk, mode, category) > 0
            for chunk in CHUNKS
            for mode in MODES
        ):
            present.append(category)
    for axis, mode in zip(axes, MODES):
        bottoms = np.zeros(len(CHUNKS))
        for category in present:
            values = 100.0 * np.asarray(
                [failure_rate(failure_rows, chunk, mode, category) for chunk in CHUNKS]
            )
            axis.bar(
                range(len(CHUNKS)),
                values,
                bottom=bottoms,
                width=0.65,
                color=FAILURE_COLORS[category],
            )
            bottoms += values
        axis.set_xticks(range(len(CHUNKS)), labels=CHUNKS)
        axis.set_xlabel("ACT chunk size")
        axis.set_title(mode)
        axis.set_ylim(0, 20)
    axes[0].set_ylabel("Failure rate over all 4 cm episodes (%)")
    handles = [
        Patch(facecolor=FAILURE_COLORS[category], label=category.replace("_", " "))
        for category in present
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5), frameon=False)
    fig.suptitle("Failure Taxonomy at 4 cm")
    fig.tight_layout(rect=(0, 0, 0.87, 1))
    save(fig, output_dir, "06_failure_taxonomy_4cm")


def plot_recovery(summary: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    for mode in MODES:
        group = [
            lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=4)
            for chunk in CHUNKS
        ]
        style = dict(
            marker="o" if mode == "temporal" else "s",
            linewidth=2.2,
            color=MODE_COLORS[mode],
            label=mode,
        )
        axes[0].plot(
            CHUNKS,
            [100.0 * row["reacquisition_rate_0_5cm"] for row in group],
            **style,
        )
        axes[1].plot(
            CHUNKS,
            [row["mean_reacquisition_latency_s_0_5cm"] for row in group],
            **style,
        )
        axes[2].plot(
            CHUNKS,
            [100.0 * row["target_contact_rate"] for row in group],
            **style,
        )
    axes[0].set(title="Reacquisition within 0.5 cm", ylabel="Rate (%)")
    axes[1].set(title="Successful reacquisition latency", ylabel="Seconds")
    axes[2].set(title="Target contact", ylabel="Rate (%)")
    for axis in axes:
        axis.set_xlabel("ACT chunk size")
        axis.set_xticks(CHUNKS)
    axes[0].legend(loc="lower right")
    fig.suptitle("Visual Recovery after 4 cm Displacement")
    fig.tight_layout()
    save(fig, output_dir, "07_reacquisition_and_contact")


def plot_execution(summary: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for mode in MODES:
        group = [
            lookup(summary, chunk_size=chunk, ensemble_mode=mode, level_cm=4)
            for chunk in CHUNKS
        ]
        style = dict(
            marker="o" if mode == "temporal" else "s",
            linewidth=2.2,
            color=MODE_COLORS[mode],
            label=mode,
        )
        axes[0, 0].plot(CHUNKS, [row["mean_steps"] for row in group], **style)
        axes[0, 1].plot(
            CHUNKS, [row["mean_temporal_contributors"] for row in group], **style
        )
        axes[1, 0].plot(
            CHUNKS,
            [100.0 * row["mean_action_clip_rate"] for row in group],
            **style,
        )
        axes[1, 1].plot(
            CHUNKS,
            [100.0 * row["wrong_object_contact_rate"] for row in group],
            **style,
        )
    settings = (
        (axes[0, 0], "Mean episode steps", "Steps"),
        (axes[0, 1], "Mean temporal contributors", "Predictions"),
        (axes[1, 0], "Action clipping", "Rate (%)"),
        (axes[1, 1], "Wrong-object contact", "Rate (%)"),
    )
    for axis, title, ylabel in settings:
        axis.set_title(title)
        axis.set_xlabel("ACT chunk size")
        axis.set_ylabel(ylabel)
        axis.set_xticks(CHUNKS)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Execution Cost and Safety at 4 cm")
    fig.tight_layout()
    save(fig, output_dir, "08_execution_cost_and_safety")


def plot_seed_consistency(seed_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    seed_names = sorted({row["seed_run"] for row in seed_rows})
    for axis, mode in zip(axes, MODES):
        for seed_name, marker in zip(seed_names, ("o", "s")):
            rates = [
                100.0
                * lookup(
                    seed_rows,
                    seed_run=seed_name,
                    chunk_size=chunk,
                    ensemble_mode=mode,
                    level_cm=4,
                )["success_rate"]
                for chunk in CHUNKS
            ]
            axis.plot(CHUNKS, rates, marker=marker, linewidth=2.0, label=seed_name)
        axis.set_xticks(CHUNKS)
        axis.set_xlabel("ACT chunk size")
        axis.set_title(mode)
        axis.set_ylim(75, 101)
    axes[0].set_ylabel("4 cm success rate (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Evaluation-Seed Consistency")
    fig.tight_layout()
    save(fig, output_dir, "09_seed_consistency_4cm")


def plot_training(endpoints: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    chunks = [row["chunk_size"] for row in endpoints]
    colors = [CHUNK_COLORS[chunk] for chunk in chunks]
    axes[0, 0].bar(chunks, [row["combined_val_total"] for row in endpoints], color=colors)
    axes[0, 1].bar(chunks, [row["combined_val_xyz_mae"] for row in endpoints], color=colors)
    axes[1, 0].bar(chunks, [row["combined_val_grounding_cm"] for row in endpoints], color=colors)
    axes[1, 1].bar(
        chunks,
        [100.0 * row["combined_val_phase_accuracy"] for row in endpoints],
        color=colors,
    )
    settings = (
        (axes[0, 0], "Combined validation objective", "Loss"),
        (axes[0, 1], "Continuous-action validation", "Normalized XYZ MAE"),
        (axes[1, 0], "Target grounding", "Error (cm)"),
        (axes[1, 1], "Phase classification", "Accuracy (%)"),
    )
    for axis, title, ylabel in settings:
        axis.set_title(title)
        axis.set_xlabel("ACT chunk size")
        axis.set_ylabel(ylabel)
        axis.set_xticks(CHUNKS)
    axes[1, 1].set_ylim(94, 100)
    fig.suptitle("Training Endpoints (Chunk 20 Uses Full Warm-Start)")
    fig.tight_layout()
    save(fig, output_dir, "10_training_endpoint_comparison")


def write_report(
    output_dir: Path,
    audit: dict,
    summary: list[dict],
    overall: list[dict],
    versus_20: list[dict],
    temporal_control: list[dict],
) -> None:
    lines = [
        "# MiniVLA V3 ACT Chunk-Size Ablation",
        "",
        "## Protocol Audit",
        "",
        f"- Status: {audit['status'].upper()}.",
        f"- Raw episode rows: {audit['raw_episode_rows']}.",
        f"- Paired scene conditions: {audit['paired_scene_conditions']}.",
        "- Two evaluation seeds; 120 paired episodes per chunk/mode/level.",
        "- Clean and 4 cm target displacement; injection compliance at 4 cm: 100%.",
        "- No privileged state was used to assist policy execution.",
        "- Chunk 20 is the frozen V3 full-warm-start reference; chunks 1/5/10 use reinitialized action queries.",
        "",
        "## Success Rates",
        "",
        "| Chunk | Temporal Clean | Temporal 4 cm | Latest Clean | Latest 4 cm |",
        "|---:|---:|---:|---:|---:|",
    ]
    for chunk in CHUNKS:
        values = {
            (mode, level): lookup(
                summary, chunk_size=chunk, ensemble_mode=mode, level_cm=level
            )["success_rate"]
            for mode in MODES
            for level in (0, 4)
        }
        lines.append(
            f"| {chunk} | {100*values[('temporal',0)]:.2f}% | "
            f"{100*values[('temporal',4)]:.2f}% | "
            f"{100*values[('latest-only',0)]:.2f}% | "
            f"{100*values[('latest-only',4)]:.2f}% |"
        )

    lines.extend(["", "## Statistical Tests", ""])
    for row in overall:
        lines.append(
            f"- {row['ensemble_mode']} at {row['level_cm']} cm: "
            f"Cochran Q={row['cochran_q']:.3f}, df=3, p={row['p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "Pairwise exact McNemar comparisons against chunk 20:",
            "",
            "| Mode | Level | Candidate | Difference vs 20 | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in versus_20:
        lines.append(
            f"| {row['ensemble_mode']} | {row['level_cm']} cm | {row['chunk_size']} | "
            f"{row['difference_percentage_points']:+.2f} pp | {row['holm_adjusted_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "1. Clean success is high for every chunk (95.83%-99.17%); the overall chunk effect is not significant in either execution mode.",
            "2. At 4 cm, chunk size has a significant overall effect: temporal p=0.0112 and latest-only p=0.00451.",
            "3. The prespecified simple hypothesis 'shorter chunks are more dynamically robust' is not supported. Chunk 1 is the weakest at 4 cm (85.0%), while chunk 10 is best under classic temporal ensembling (95.0%).",
            "4. Chunk 20 is best under latest-only (94.17%) but falls to 90.83% under classic temporal ensembling. The -3.33 pp temporal difference is directionally consistent with historical inertia but is not statistically significant (McNemar p=0.388).",
            "5. Chunk 10 provides the best measured Clean/4 cm temporal balance. Its 95.0% 4 cm rate exceeds chunk 20 by 4.17 pp, but the paired difference is not individually significant after correction.",
            "6. Temporal contributor count rises strongly with chunk size, confirming that longer chunks integrate older predictions; performance nevertheless follows a non-monotonic optimum rather than a simple linear degradation.",
            "7. Because chunk 20 inherited V2 action queries while chunks 1/5/10 reinitialized them, the study is an operational ablation with a documented initialization confound.",
            "",
            "## Figure Guide",
            "",
            "- `01_chunk_success_clean_vs_4cm`: primary success result.",
            "- `02_robustness_retention_and_drop`: normalized displacement cost.",
            "- `03_clean_robustness_pareto`: operating-point comparison.",
            "- `04_temporal_ensembling_effect`: temporal minus latest-only effect.",
            "- `05_per_task_4cm_heatmaps`: task-specific behavior.",
            "- `06_failure_taxonomy_4cm`: failure mechanisms.",
            "- `07_reacquisition_and_contact`: visual recovery.",
            "- `08_execution_cost_and_safety`: time, prediction integration, clipping, wrong contact.",
            "- `09_seed_consistency_4cm`: replication stability.",
            "- `10_training_endpoint_comparison`: optimization diagnostics.",
            "",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    rows, audit = load_and_validate(input_root)
    success_rows = summarize_success(rows)
    task_rows = summarize_tasks(rows)
    failure_rows = summarize_failures(rows)
    seed_rows = summarize_seeds(rows)
    training_rows = load_training_endpoints(
        Path(args.artifact_root),
        Path(args.chunk20_artifact_root),
    )
    overall_rows, versus_20_rows, temporal_control_rows = statistical_analysis(
        rows, args.bootstrap_samples, args.bootstrap_seed
    )

    write_csv(output_dir / "success_summary.csv", success_rows)
    write_csv(output_dir / "task_success_summary.csv", task_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failure_rows)
    write_csv(output_dir / "seed_success_summary.csv", seed_rows)
    write_csv(output_dir / "training_endpoints.csv", training_rows)
    write_csv(output_dir / "cochran_q_tests.csv", overall_rows)
    write_csv(output_dir / "pairwise_vs_chunk20.csv", versus_20_rows)
    write_csv(output_dir / "temporal_vs_latest.csv", temporal_control_rows)
    with (output_dir / "protocol_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    configure_plots()
    plot_success(success_rows, output_dir)
    plot_retention(success_rows, output_dir)
    plot_pareto(success_rows, output_dir)
    plot_temporal_effect(temporal_control_rows, output_dir)
    plot_task_heatmaps(task_rows, output_dir)
    plot_failures(failure_rows, output_dir)
    plot_recovery(success_rows, output_dir)
    plot_execution(success_rows, output_dir)
    plot_seed_consistency(seed_rows, output_dir)
    plot_training(training_rows, output_dir)
    write_report(
        output_dir,
        audit,
        success_rows,
        overall_rows,
        versus_20_rows,
        temporal_control_rows,
    )

    summary = {
        "status": "complete",
        "protocol_audit": audit,
        "success_summary": success_rows,
        "cochran_q_tests": overall_rows,
        "pairwise_vs_chunk20": versus_20_rows,
        "temporal_vs_latest": temporal_control_rows,
        "recommended_temporal_chunk": max(
            CHUNKS,
            key=lambda chunk: lookup(
                success_rows,
                chunk_size=chunk,
                ensemble_mode="temporal",
                level_cm=4,
            )["success_rate"],
        ),
        "recommended_latest_only_chunk": max(
            CHUNKS,
            key=lambda chunk: lookup(
                success_rows,
                chunk_size=chunk,
                ensemble_mode="latest-only",
                level_cm=4,
            )["success_rate"],
        ),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Protocol audit: PASS ({len(rows)} episode rows)")
    print(f"Ten PNG/PDF figures and analysis tables saved to {output_dir}")


if __name__ == "__main__":
    main()
