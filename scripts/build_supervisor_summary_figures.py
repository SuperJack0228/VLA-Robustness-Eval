#!/usr/bin/env python3
"""Build compact Clean and training figures for the supervisor report."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
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


ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = ROOT / "artifacts" / "v3-clean-rc1" / "clean_comparison"
TRAINING_LOG = ROOT / "artifacts" / "v3-clean-rc1" / "training_log_v3.csv"
CLEAN_OUTPUT_DIR = ROOT / "final_report" / "01_clean_baseline"
TRAINING_OUTPUT_DIR = ROOT / "final_report" / "04_training_diagnostics"
TASKS = ("pick_A", "pick_B", "pick_C", "push_A", "push_B", "push_C")
COLORS = {"v2": "#D55E00", "v3": "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def exact_mcnemar_p(v3_only: int, v2_only: int) -> float:
    discordant = v3_only + v2_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(v3_only, v2_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
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


def load_clean() -> dict[str, list[dict[str, str]]]:
    return {
        model: read_csv(CLEAN_ROOT / f"{model}_20260714.csv")
        + read_csv(CLEAN_ROOT / f"{model}_20261017.csv")
        for model in ("v2", "v3")
    }


def plot_clean(clean: dict[str, list[dict[str, str]]]) -> dict:
    aggregate = {}
    task_rates: dict[str, dict[str, float]] = defaultdict(dict)
    for model, rows in clean.items():
        successes = sum(int(row["task_success"]) for row in rows)
        lower, upper = wilson(successes, len(rows))
        aggregate[model] = {
            "successes": successes,
            "episodes": len(rows),
            "rate": successes / len(rows),
            "ci95_lower": lower,
            "ci95_upper": upper,
        }
        for task in TASKS:
            group = [
                row
                for row in rows
                if f"{row['task_type']}_{row['target_id']}" == task
            ]
            task_rates[model][task] = sum(
                int(row["task_success"]) for row in group
            ) / len(group)

    paired = {}
    for model, rows in clean.items():
        paired[model] = {
            (row["scene_seed"], row["task_type"], row["target_id"]): int(
                row["task_success"]
            )
            for row in rows
        }
    if set(paired["v2"]) != set(paired["v3"]):
        raise ValueError("Clean V2/V3 scene schedules are not paired")
    v3_only = sum(
        paired["v3"][key] == 1 and paired["v2"][key] == 0 for key in paired["v2"]
    )
    v2_only = sum(
        paired["v2"][key] == 1 and paired["v3"][key] == 0 for key in paired["v2"]
    )
    p_value = exact_mcnemar_p(v3_only, v2_only)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    models = ("v2", "v3")
    rates = np.asarray([100 * aggregate[model]["rate"] for model in models])
    lower_errors = rates - np.asarray(
        [100 * aggregate[model]["ci95_lower"] for model in models]
    )
    upper_errors = np.asarray(
        [100 * aggregate[model]["ci95_upper"] for model in models]
    ) - rates
    bars = axes[0].bar(
        ["V2", "V3"],
        rates,
        color=[COLORS[model] for model in models],
        width=0.58,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=5,
    )
    for bar, model in zip(bars, models):
        item = aggregate[model]
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            item["rate"] * 100 + 1.4,
            f"{item['successes']}/{item['episodes']}\n{100 * item['rate']:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_ylim(80, 102)
    axes[0].set_ylabel("Task success rate (%)")
    axes[0].set_title(f"Paired Clean Benchmark\nMcNemar p={p_value:.4f}")

    x = np.arange(len(TASKS))
    width = 0.36
    for offset, model in zip((-width / 2, width / 2), models):
        axes[1].bar(
            x + offset,
            [100 * task_rates[model][task] for task in TASKS],
            width,
            color=COLORS[model],
            label=model.upper(),
        )
    axes[1].set_xticks(x, labels=[task.replace("_", "-") for task in TASKS])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(70, 102)
    axes[1].set_ylabel("Task success rate (%)")
    axes[1].set_title("Clean Success by Task (n=40 each)")
    axes[1].legend(loc="lower left")
    fig.suptitle("MiniVLA V2 to V3 Clean Improvement")
    fig.tight_layout()
    save(fig, CLEAN_OUTPUT_DIR, "paired_clean_v2_vs_v3")

    return {
        "aggregate": aggregate,
        "task_success_rates": task_rates,
        "v3_only_successes": v3_only,
        "v2_only_successes": v2_only,
        "exact_mcnemar_p": p_value,
    }


def plot_training() -> dict:
    rows = read_csv(TRAINING_LOG)
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    best_index = int(np.argmin([float(row["selection_score"]) for row in rows]))
    best_epoch = int(epochs[best_index])

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    axes[0, 0].plot(epochs, [float(row["train_total"]) for row in rows], label="Train")
    axes[0, 0].plot(
        epochs,
        [float(row["combined_val_total"]) for row in rows],
        label="Combined validation",
    )
    axes[0, 0].set(title="Objective", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs,
        [float(row["combined_val_xyz_mae"]) for row in rows],
        color="#009E73",
    )
    axes[0, 1].set(
        title="Continuous-Action Validation",
        xlabel="Epoch",
        ylabel="Normalized XYZ MAE",
    )

    axes[1, 0].plot(
        epochs,
        [float(row["combined_val_grounding_cm"]) for row in rows],
        color="#CC79A7",
    )
    axes[1, 0].set(
        title="Target Grounding",
        xlabel="Epoch",
        ylabel="Grounding error (cm)",
    )

    axes[1, 1].plot(
        epochs,
        [100 * float(row["combined_val_gripper_accuracy"]) for row in rows],
        label="Gripper",
    )
    axes[1, 1].plot(
        epochs,
        [100 * float(row["combined_val_phase_accuracy"]) for row in rows],
        label="Phase",
    )
    axes[1, 1].set(
        title="Discrete Validation Heads",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        ylim=(94, 100.2),
    )
    axes[1, 1].legend(loc="lower right")
    for axis in axes.flat:
        axis.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.0)
    fig.suptitle(f"MiniVLA V3 Training Diagnostics (best selection epoch {best_epoch})")
    fig.tight_layout()
    save(fig, TRAINING_OUTPUT_DIR, "v3_training_diagnostics")
    return {"epochs": len(rows), "best_selection_epoch": best_epoch}


def main() -> None:
    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    configure_plots()
    metrics = {"clean": plot_clean(load_clean()), "training": plot_training()}
    with (CLEAN_OUTPUT_DIR / "clean_summary_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)
    print(
        f"Supervisor figures saved to {CLEAN_OUTPUT_DIR} and "
        f"{TRAINING_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
