"""Strict gate for V3 recovery data and language-conditioned fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.collect_data_v3 import verify_dataset as verify_recovery_dataset
from scripts.preflight_v2 import GateReport, audit_archives, require
from scripts.train_v2 import (
    EPISODES_PER_BATCH,
    compute_objective,
    forward_batch,
    move_batch,
)
from scripts.train_v3 import (
    CHECKPOINT_FORMAT_VERSION,
    CHUNK_SIZE,
    DATASET_VERSION,
    PHASE_FAMILY_WEIGHT,
    SUPPORTED_CHUNK_SIZES,
    build_model_from_checkpoint,
)
from utils.language_augmentation_v3 import (
    DEFAULT_LANGUAGE_CATALOG,
    LanguageAugmentationCatalog,
)
from utils.training_dataset_v2 import (
    ActionChunkDatasetV2,
    InterleavedTaskBatchSampler,
    NormalizationStats,
    V2EpisodeStore,
)
from utils.v2_schema import TASK_BUCKETS


DEFAULT_REPORT = "results/training/v3/preflight_report_v3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-data-dir",
        default="data/dataset_v2_clean",
    )
    parser.add_argument(
        "--recovery-data-dir",
        default="data/dataset_v3_recovery",
    )
    parser.add_argument("--expected-base-episodes", type=int, default=1200)
    parser.add_argument("--expected-recovery-episodes", type=int, default=600)
    parser.add_argument(
        "--init-policy",
        default="artifacts/v2-clean-rc1/mini_vla_v2_clean_policy.pth",
    )
    parser.add_argument(
        "--language-catalog",
        default=DEFAULT_LANGUAGE_CATALOG,
    )
    parser.add_argument("--output", default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=int,
        choices=SUPPORTED_CHUNK_SIZES,
        default=CHUNK_SIZE,
    )
    parser.add_argument("--reinitialize-action-queries", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--skip-archive-audit", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def language_audit(catalog: LanguageAugmentationCatalog) -> dict:
    summary = catalog.summary()
    all_train = set()
    all_eval = set()
    for task_type, target_id in TASK_BUCKETS:
        train = catalog.expressions(task_type, target_id, "train")
        evaluation = catalog.expressions(task_type, target_id, "eval")
        require(len(train) >= 50, f"{task_type}_{target_id} lacks 50 train phrases")
        require(
            set(train).isdisjoint(evaluation),
            f"{task_type}_{target_id} leaks train phrases into evaluation",
        )
        all_train.update(train)
        all_eval.update(evaluation)
    require(
        all_train.isdisjoint(all_eval),
        "Language train and held-out evaluation vocabularies overlap",
    )
    summary["global_train_expressions"] = len(all_train)
    summary["global_eval_expressions"] = len(all_eval)
    return summary


def mixed_loader_audit(
    data_dirs: list[str],
    stats: NormalizationStats,
    catalog: LanguageAugmentationCatalog,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
) -> tuple[dict, dict]:
    require(
        batch_size % EPISODES_PER_BATCH == 0,
        "batch-size must be divisible by 16",
    )
    train_store = V2EpisodeStore(data_dirs, "train", cache_size=8)
    train_dataset = ActionChunkDatasetV2(
        train_store,
        stats,
        chunk_size=chunk_size,
        samples_per_episode=64,
        language_catalog=catalog,
    )
    sampler = InterleavedTaskBatchSampler(
        train_dataset,
        batch_size=batch_size,
        episodes_per_batch=EPISODES_PER_BATCH,
        shuffle=True,
        seed=20261201,
    )
    store_sources = {os.path.dirname(path) for path in train_store.paths}
    require(
        len(store_sources) == len(data_dirs),
        "Combined store does not contain both V2 Clean and V3 recovery data",
    )
    checked_batches = 0
    for indices in sampler:
        episode_indices = {
            train_dataset.sample_index[index][0] for index in indices
        }
        buckets = {train_dataset.sample_buckets[index] for index in indices}
        require(
            len(episode_indices) == EPISODES_PER_BATCH,
            "A batch does not contain 16 distinct episodes",
        )
        require(
            buckets == set(TASK_BUCKETS),
            "A batch does not contain all six task buckets",
        )
        checked_batches += 1
        if checked_batches >= 12:
            break
    require(checked_batches > 0, "Interleaved sampler emitted no batches")
    loader_kwargs = {
        "dataset": train_dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_batch = next(iter(DataLoader(**loader_kwargs)))
    require(
        set(train_batch["language_split"]) == {"train"},
        "Training loader emitted a non-training language variant",
    )
    require(
        len(set(train_batch["instruction"])) >= 8,
        "A training batch contains too little language variation",
    )

    val_store = V2EpisodeStore(data_dirs, "val", cache_size=2)
    val_dataset = ActionChunkDatasetV2(
        val_store,
        stats,
        chunk_size=chunk_size,
        samples_per_episode=1,
        initial_only=True,
        history_dropout_probability=0.0,
        state_noise_std=0.0,
        language_catalog=catalog,
    )
    val_batch = next(
        iter(
            DataLoader(
                val_dataset,
                batch_size=min(batch_size, len(val_dataset)),
                shuffle=False,
                num_workers=0,
            )
        )
    )
    require(
        set(val_batch["language_split"]) == {"eval"},
        "Validation loader leaked training language variants",
    )
    return {
        "train_episodes": len(train_store),
        "train_windows": len(train_dataset),
        "checked_batches": checked_batches,
        "episodes_per_batch": EPISODES_PER_BATCH,
        "first_batch_unique_instructions": len(
            set(train_batch["instruction"])
        ),
        "chunk_size": chunk_size,
        "pose_target_shape": list(train_batch["pose_target"].shape),
    }, train_batch


def model_audit(
    checkpoint: dict,
    raw_batch: dict,
    device: torch.device,
    local_files_only: bool,
    chunk_size: int,
    reinitialize_action_queries: bool,
) -> dict:
    require(
        checkpoint.get("format_version") == 7,
        "init-policy must use final V2 Clean checkpoint format 7",
    )
    require(
        checkpoint.get("dataset_version") == "v2.clean",
        "init-policy is not the frozen V2 Clean policy",
    )
    stats = NormalizationStats.from_checkpoint(checkpoint["normalization"])
    model = build_model_from_checkpoint(
        checkpoint,
        local_files_only=local_files_only,
        chunk_size=chunk_size,
        reinitialize_action_queries=reinitialize_action_queries,
    ).to(device)
    batch_size = min(2, int(raw_batch["state_history"].shape[0]))
    compact = {}
    for key, value in raw_batch.items():
        if torch.is_tensor(value):
            compact[key] = value[:batch_size]
        elif isinstance(value, (list, tuple)):
            compact[key] = list(value[:batch_size])
        else:
            compact[key] = value
    batch = move_batch(compact, device)
    model.train()
    output = forward_batch(model, batch)
    metrics = compute_objective(
        output,
        batch,
        stats,
        phase_family_weight=PHASE_FAMILY_WEIGHT,
    )
    require(torch.isfinite(metrics["total"]), "V3 objective is not finite")
    require(
        tuple(output["pose"].shape[1:]) == (chunk_size, 6),
        "Model pose output does not match the requested chunk size",
    )
    metrics["total"].backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    require(finite_gradients, "V3 backward pass produced non-finite gradients")
    return {
        "device": str(device),
        "input_batch_size": batch_size,
        "pose_shape": list(output["pose"].shape),
        "phase_shape": list(output["phase_logits"].shape),
        "chunk_size": chunk_size,
        "reinitialized_action_queries": reinitialize_action_queries,
        "loss": float(metrics["total"].detach().cpu()),
        "phase_family_accuracy": float(
            metrics["phase_family_accuracy"].detach().cpu()
        ),
        "v3_checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "v3_dataset_version": DATASET_VERSION,
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size != CHUNK_SIZE and not args.reinitialize_action_queries:
        raise ValueError(
            "Non-default chunk sizes require --reinitialize-action-queries"
        )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report = GateReport()
    details = {}

    try:
        catalog = LanguageAugmentationCatalog(args.language_catalog)
        details["language"] = language_audit(catalog)
        report.pass_check("language_catalog_and_split", details["language"])
    except Exception as error:
        report.fail_check("language_catalog_and_split", error)
        catalog = None

    if not args.skip_archive_audit:
        try:
            details["base_archives"] = audit_archives(
                args.base_data_dir,
                args.expected_base_episodes,
            )
            report.pass_check("frozen_v2_clean_archives", details["base_archives"])
        except Exception as error:
            report.fail_check("frozen_v2_clean_archives", error)
        try:
            verified = verify_recovery_dataset(
                args.recovery_data_dir,
                args.expected_recovery_episodes,
            )
            details["recovery_archives"] = audit_archives(
                args.recovery_data_dir,
                verified,
            )
            report.pass_check(
                "balanced_v3_recovery_archives",
                details["recovery_archives"],
            )
        except Exception as error:
            report.fail_check("balanced_v3_recovery_archives", error)

    raw_batch = None
    checkpoint = None
    if catalog is not None:
        try:
            checkpoint = torch.load(args.init_policy, map_location="cpu")
            stats = NormalizationStats.from_checkpoint(
                checkpoint["normalization"]
            )
            details["mixed_loader"], raw_batch = mixed_loader_audit(
                [args.base_data_dir, args.recovery_data_dir],
                stats,
                catalog,
                args.batch_size,
                args.num_workers,
                args.chunk_size,
            )
            report.pass_check(
                "interleaved_multitask_language_loader",
                details["mixed_loader"],
            )
        except Exception as error:
            report.fail_check("interleaved_multitask_language_loader", error)

    if not args.skip_model and checkpoint is not None and raw_batch is not None:
        check_name = (
            "v3_chunk_ablation_warmstart"
            if args.reinitialize_action_queries
            else "v2_to_v3_full_warmstart"
        )
        try:
            details["model"] = model_audit(
                checkpoint,
                raw_batch,
                get_device(args.device),
                args.local_files_only,
                args.chunk_size,
                args.reinitialize_action_queries,
            )
            report.pass_check(check_name, details["model"])
        except Exception as error:
            report.fail_check(check_name, error)

    payload = report.payload()
    payload["details"] = details
    temporary = f"{args.output}.tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.output)
    print(f"Preflight report: {args.output}", flush=True)
    if not report.passed:
        print("V3 PREFLIGHT: FAIL", flush=True)
        raise SystemExit(1)
    print("V3 PREFLIGHT: PASS", flush=True)


if __name__ == "__main__":
    main()
