"""Safety-shielded closed-loop evaluation for MiniVLA V2 policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
import torch

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

import robosuite


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mini_vla_v2 import MiniVLAV2
from scripts.collect_data_v2 import (
    CONTROL_FREQ,
    ORACLE_ACTION_LIMIT,
    PUSH_MAX_LATERAL_ERROR,
    PUSH_SUCCESS_DISTANCE,
    ProprioceptionTracker,
    build_osc_pose_controller_config_v2,
    reseed_environment,
    schedule_task_v2,
)
from utils.training_dataset_v2 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NormalizationStats,
    TARGET_ID_TO_INDEX,
)
from utils.v2_schema import (
    ACTION_DIM,
    DATASET_VERSION,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_LABELS,
    TASK_BUCKETS,
    instruction_for,
)


DEFAULT_POLICY_PATH = "results/v2_clean/mini_vla_v2_clean_policy.pth"
DEFAULT_OUTPUT_PREFIX = "results/v2_clean/evaluation_v2_clean"
DEFAULT_NUM_EPISODES = 120
MAX_STEPS = 200
REPLAN_INTERVAL = 1
ENSEMBLE_DECAY = 0.25
GRIPPER_CLOSE_THRESHOLD = 0.60
GRIPPER_OPEN_THRESHOLD = 0.40
PICK_TASK_HEIGHT = 0.04
PICK_CLEAN_HEIGHT = 0.08
SUCCESS_HOLD_STEPS = 5
SUPPORTED_CHECKPOINT_FORMATS = {4, 6, 7}
OBJECT_MOVEMENT_CONTACT_THRESHOLD = 5e-4
WRONG_OBJECT_MOVEMENT_THRESHOLD = 8e-3
WORKSPACE_X_LIMIT = 0.27
WORKSPACE_Y_LIMIT = 0.28
WORKSPACE_Z_MAX = 1.07
WORKSPACE_XY_GUARD_BAND = 0.01
WORKSPACE_Z_FLOOR_OFFSET = 0.008
WORKSPACE_Z_GUARD_BAND = 0.002
PUSH_EEF_FLOOR_OFFSET = {"A": 0.016, "B": 0.018, "C": 0.016}
SPHERE_PENETRATION_TOLERANCE = 0.006
CYLINDER_CLEAN_UPRIGHT_THRESHOLD = 0.75
CYLINDER_TOPPLE_THRESHOLD = 0.65
PICK_CONTACT_PHASES = frozenset({2, 3})
PUSH_CONTACT_PHASES = frozenset({6, 7})
MAX_PICK_RECOVERY_CYCLES = 3
MAX_PUSH_RECOVERY_CYCLES = 3
MAX_SAFETY_INTERVENTION_STREAK = 25
VISUAL_PERTURBATIONS = (
    "clean",
    "bright",
    "dark",
    "gaussian_noise",
    "camera_shift",
    "center_occlusion",
)
EXECUTION_MODES = ("raw",)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--replan-interval", type=int, default=REPLAN_INTERVAL)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enforce-80", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="raw",
        help=(
            "Only raw policy execution is supported for clean evaluation."
        ),
    )
    parser.add_argument(
        "--visual-perturbation",
        choices=VISUAL_PERTURBATIONS,
        default="clean",
    )
    return parser.parse_args()


def load_policy(
    path: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[MiniVLAV2, NormalizationStats]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format_version") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise RuntimeError("Checkpoint is not a supported MiniVLA V2 policy")
    checkpoint_dataset = checkpoint.get("dataset_version")
    if checkpoint_dataset != DATASET_VERSION:
        print(
            f"WARNING: policy dataset version is {checkpoint_dataset!r}; "
            f"the current environment is {DATASET_VERSION}.",
            flush=True,
        )
    config = dict(checkpoint["model_config"])
    config["local_files_only"] = local_files_only
    model = MiniVLAV2(**config)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    stats = NormalizationStats.from_checkpoint(checkpoint["normalization"])
    return model, stats


def make_environment(render: bool, horizon: int = MAX_STEPS):
    camera_names = ["agentview", "robot0_eye_in_hand"]
    camera_heights = [IMAGE_HEIGHT, IMAGE_HEIGHT]
    camera_widths = [IMAGE_WIDTH, IMAGE_WIDTH]
    if render:
        camera_names.append("frontview")
        camera_heights.append(512)
        camera_widths.append(512)
    env = robosuite.make(
        env_name="MultiObjectVLAEnvV2",
        robots="Panda",
        controller_configs=build_osc_pose_controller_config_v2(),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=camera_names,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        horizon=horizon,
        ignore_done=True,
        control_freq=CONTROL_FREQ,
        hard_reset=False,
    )
    low, high = env.action_spec
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        env.close()
        raise RuntimeError(f"Expected 7D OSC_POSE action space, got {low.shape}")
    return env


def parse_instruction(instruction: str) -> tuple[str, str]:
    normalized = " ".join(instruction.strip().lower().split())
    for task_type, target_id in TASK_BUCKETS:
        canonical = instruction_for(task_type, target_id)
        if normalized == canonical.lower():
            return task_type, target_id
    valid = [instruction_for(*bucket) for bucket in TASK_BUCKETS]
    raise ValueError(f"Unsupported instruction. Choose one of: {valid}")


def balanced_schedule(
    num_episodes: int,
    seed: int,
    fixed_instruction: str | None,
) -> list[tuple[str, str]]:
    if num_episodes <= 0:
        raise ValueError("num-episodes must be positive")
    if fixed_instruction:
        return [parse_instruction(fixed_instruction)] * num_episodes
    repeated = [
        TASK_BUCKETS[index % len(TASK_BUCKETS)]
        for index in range(num_episodes)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    return repeated


def capture_model_image(
    obs: dict,
    camera_name: str,
    perturbation: str = "clean",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    image = np.ascontiguousarray(np.flipud(obs[f"{camera_name}_image"]))
    image = apply_visual_perturbation(image, perturbation, rng)
    return image


def preprocess_image(
    frame: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    image = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    image = np.ascontiguousarray((image - mean) / std)
    return torch.from_numpy(image).unsqueeze(0).to(device)


def apply_visual_perturbation(
    image: np.ndarray,
    perturbation: str,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if perturbation == "clean":
        return np.ascontiguousarray(image)
    working = image.astype(np.float32)
    if perturbation == "bright":
        working *= 1.25
    elif perturbation == "dark":
        working *= 0.65
    elif perturbation == "gaussian_noise":
        if rng is None:
            raise ValueError("gaussian_noise requires a seeded RNG")
        working += rng.normal(0.0, 12.0, size=working.shape)
    elif perturbation == "camera_shift":
        height, width = image.shape[:2]
        transform = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, -4.0]])
        return cv2.warpAffine(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    elif perturbation == "center_occlusion":
        height, width = image.shape[:2]
        half_size = 10
        center_y, center_x = height // 2, width // 2
        working[
            center_y - half_size : center_y + half_size,
            center_x - half_size : center_x + half_size,
        ] = working.mean(axis=(0, 1), keepdims=True)
    else:
        raise ValueError(f"Unsupported visual perturbation: {perturbation}")
    return np.ascontiguousarray(np.clip(working, 0, 255).astype(np.uint8))


def preprocess_history(
    state_queue: deque[np.ndarray],
    stats: NormalizationStats,
    device: torch.device,
) -> torch.Tensor:
    states = np.stack(state_queue).astype(np.float32)
    states = (states - stats.state_mean) / stats.state_std
    return torch.from_numpy(np.ascontiguousarray(states)).unsqueeze(0).to(device)


class TemporalActionEnsembler:
    """Blend all action chunks that predict the current absolute timestep."""

    def __init__(self, decay: float = ENSEMBLE_DECAY) -> None:
        self.decay = decay
        self.predictions: dict[int, list[tuple[int, np.ndarray, float, int]]] = (
            defaultdict(list)
        )

    def add(
        self,
        start_step: int,
        pose_chunk: np.ndarray,
        gripper_probability: np.ndarray,
        phase_chunk: np.ndarray,
    ) -> None:
        for offset in range(len(pose_chunk)):
            self.predictions[start_step + offset].append(
                (
                    start_step,
                    pose_chunk[offset].copy(),
                    float(gripper_probability[offset]),
                    int(phase_chunk[offset]),
                )
            )

    def clear(self) -> None:
        self.predictions.clear()

    def has_step(self, step: int) -> bool:
        return bool(self.predictions.get(step))

    def action(
        self,
        step: int,
        latest_only_phases: frozenset[int] = frozenset(),
    ) -> tuple[np.ndarray, float, int, bool]:
        entries = self.predictions.pop(step, None)
        if not entries:
            raise RuntimeError(f"No chunk prediction available for step {step}")

        newest_created = max(entry[0] for entry in entries)
        newest_entries = [entry for entry in entries if entry[0] == newest_created]
        newest_phase = newest_entries[-1][3]
        used_latest_only = newest_phase in latest_only_phases
        if used_latest_only:
            entries = newest_entries

        ages = np.asarray([step - created for created, _, _, _ in entries])
        weights = np.exp(-self.decay * ages)
        weights /= weights.sum()
        pose = np.sum(
            np.stack([entry[1] for entry in entries]) * weights[:, None],
            axis=0,
        )
        gripper = float(
            np.sum(np.asarray([entry[2] for entry in entries]) * weights)
        )
        phase_votes: dict[int, float] = defaultdict(float)
        for weight, entry in zip(weights, entries):
            phase_votes[entry[3]] += float(weight)
        phase = max(phase_votes, key=phase_votes.get)
        stale_steps = [target for target in self.predictions if target < step]
        for target in stale_steps:
            del self.predictions[target]
        return pose.astype(np.float32), gripper, phase, used_latest_only


class GripperHysteresis:
    def __init__(self) -> None:
        self.action = -1.0

    def update(self, probability: float) -> float:
        if self.action < 0.0 and probability >= GRIPPER_CLOSE_THRESHOLD:
            self.action = 1.0
        elif self.action > 0.0 and probability <= GRIPPER_OPEN_THRESHOLD:
            self.action = -1.0
        return self.action


def apply_safety_shield(
    action: np.ndarray,
    obs: dict,
    task,
    table_height: float,
) -> tuple[np.ndarray, bool, bool, bool, bool]:
    """Keep OSC commands inside the demonstrated Cartesian workspace."""
    corrected = action.copy()
    eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    lower = np.asarray(
        [
            -WORKSPACE_X_LIMIT,
            -WORKSPACE_Y_LIMIT,
            table_height + WORKSPACE_Z_FLOOR_OFFSET,
        ],
        dtype=np.float32,
    )
    upper = np.asarray(
        [WORKSPACE_X_LIMIT, WORKSPACE_Y_LIMIT, WORKSPACE_Z_MAX],
        dtype=np.float32,
    )
    outside_workspace = bool(
        np.any(eef_position < lower) or np.any(eef_position > upper)
    )
    workspace_intervened = False
    workspace_xy_intervened = False
    for axis in range(3):
        guard_band = (
            WORKSPACE_XY_GUARD_BAND if axis < 2 else WORKSPACE_Z_GUARD_BAND
        )
        if (
            eef_position[axis] <= lower[axis] + guard_band
            and corrected[axis] < 0.0
        ):
            corrected[axis] = 0.0
            workspace_intervened = True
            workspace_xy_intervened |= axis < 2
        if (
            eef_position[axis] >= upper[axis] - guard_band
            and corrected[axis] > 0.0
        ):
            corrected[axis] = 0.0
            workspace_intervened = True
            workspace_xy_intervened |= axis < 2
        if eef_position[axis] < lower[axis]:
            corrected[axis] = max(
                corrected[axis],
                min(0.4, float((lower[axis] - eef_position[axis]) * 8.0)),
            )
            workspace_intervened = True
            workspace_xy_intervened |= axis < 2
        elif eef_position[axis] > upper[axis]:
            corrected[axis] = min(
                corrected[axis],
                max(-0.4, float((upper[axis] - eef_position[axis]) * 8.0)),
            )
            workspace_intervened = True
            workspace_xy_intervened |= axis < 2

    if task.task_type == "push":
        push_floor = table_height + PUSH_EEF_FLOOR_OFFSET[task.target_id]
        if eef_position[2] <= push_floor + 0.003 and corrected[2] < 0.0:
            corrected[2] = 0.0
    intervened = not np.allclose(corrected, action, atol=1e-7)
    return corrected, intervened, workspace_intervened, workspace_xy_intervened, outside_workspace


def write_results(prefix: str, rows: list[dict], report: dict) -> None:
    directory = os.path.dirname(prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)
    csv_path = f"{prefix}.csv"
    json_path = f"{prefix}.json"
    temporary_csv = f"{csv_path}.tmp"
    temporary_json = f"{json_path}.tmp"
    with open(temporary_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())
    with open(temporary_json, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.flush()
        os.fsync(json_file.fileno())
    os.replace(temporary_csv, csv_path)
    os.replace(temporary_json, json_path)


def evaluate_success_conditions(
    task_type: str,
    target_id: str,
    current_position: np.ndarray,
    initial_position: np.ndarray,
    push_direction: np.ndarray,
    grasping_target: bool,
    uprightness: float,
    wrong_object_contact: bool,
    table_height: float,
) -> tuple[bool, bool, float, float]:
    """Return task/clean success conditions and planar push diagnostics."""
    displacement = current_position[:2] - initial_position[:2]
    push_forward = float(np.dot(displacement, push_direction))
    push_lateral = float(
        np.linalg.norm(displacement - push_forward * push_direction)
    )
    if task_type == "pick":
        task_condition = bool(
            grasping_target
            and current_position[2] >= table_height + PICK_TASK_HEIGHT
        )
        clean_condition = bool(
            grasping_target
            and current_position[2] >= table_height + PICK_CLEAN_HEIGHT
            and not wrong_object_contact
        )
        return task_condition, clean_condition, push_forward, push_lateral

    remains_on_table = bool(current_position[2] > table_height - 0.02)
    task_condition = bool(
        push_forward >= PUSH_SUCCESS_DISTANCE
        and push_lateral <= PUSH_MAX_LATERAL_ERROR
        and remains_on_table
    )
    cylinder_clean = bool(
        target_id != "C"
        or uprightness >= CYLINDER_CLEAN_UPRIGHT_THRESHOLD
    )
    clean_condition = bool(
        task_condition and cylinder_clean and not wrong_object_contact
    )
    return task_condition, clean_condition, push_forward, push_lateral


def classify_failure(
    task_type: str,
    task_success: bool,
    wrong_object_contact: bool,
    termination: str,
    min_eef_target_distance: float,
    gripper_close_step: int,
    target_contact: bool,
    grasped_once: bool,
    final_grasp: bool,
    max_target_height: float,
    final_target_height: float,
    table_height: float,
    push_forward: float,
    push_lateral: float,
) -> str:
    if task_success and termination == "success" and not wrong_object_contact:
        return "success"
    if termination == "user_stop":
        return "user_stop"
    if termination == "object_penetration":
        return "object_penetration"
    if termination == "target_toppled":
        return "target_toppled"
    if termination == "recovery_limit":
        return "recovery_limit"
    if termination == "workspace_stall":
        return "workspace_stall"
    if wrong_object_contact:
        return "wrong_object_contact"
    if task_success:
        return "task_success_only"
    if task_type == "pick":
        if min_eef_target_distance > 0.08:
            return "target_not_reached"
        if gripper_close_step == 0:
            return "gripper_never_closed"
        if grasped_once and not final_grasp:
            return "object_dropped"
        if final_grasp or max_target_height > table_height + 0.03:
            return "insufficient_lift"
        return "missed_grasp"
    if final_target_height < table_height - 0.02:
        return "object_left_table"
    if not target_contact:
        return "target_not_contacted"
    if push_forward >= PUSH_SUCCESS_DISTANCE and push_lateral > PUSH_MAX_LATERAL_ERROR:
        return "lateral_push_error"
    return "insufficient_push_distance"


def summarize(rows: list[dict], perturbation: str = "clean") -> dict:
    report = {
        "episodes": len(rows),
        "visual_perturbation": perturbation,
        "execution_mode": "raw",
        "uses_privileged_execution_assistance": False,
        "score_scope": "policy_with_workspace_safety_only",
        "buckets": {},
    }
    for task_type, target_id in TASK_BUCKETS:
        key = f"{task_type}_{target_id}"
        selected = [
            row
            for row in rows
            if row["task_type"] == task_type and row["target_id"] == target_id
        ]
        if not selected:
            continue
        report["buckets"][key] = {
            "episodes": len(selected),
            "task_success_rate": float(
                np.mean([row["task_success"] for row in selected])
            ),
            "clean_success_rate": float(
                np.mean([row["clean_success"] for row in selected])
            ),
            "wrong_contact_rate": float(
                np.mean([row["wrong_object_contact"] for row in selected])
            ),
            "mean_steps": float(np.mean([row["steps"] for row in selected])),
            "mean_grounding_error_cm": float(
                np.mean([row["mean_grounding_error_cm"] for row in selected])
            ),
            "mean_target_class_accuracy": float(
                np.mean([row["mean_target_class_accuracy"] for row in selected])
            ),
            "mean_action_clip_rate": float(
                np.mean([row["action_clip_rate"] for row in selected])
            ),
            "mean_safety_intervention_rate": float(
                np.mean(
                    [row["safety_intervention_rate"] for row in selected]
                )
            ),
            "mean_latest_only_rate": float(
                np.mean([row["latest_only_rate"] for row in selected])
            ),
            "minimum_target_uprightness": float(
                np.min([row["min_target_uprightness"] for row in selected])
            ),
            "failure_counts": dict(
                Counter(row["failure_category"] for row in selected)
            ),
        }
    report["overall"] = {
        "task_success_rate": float(
            np.mean([row["task_success"] for row in rows])
        ),
        "clean_success_rate": float(
            np.mean([row["clean_success"] for row in rows])
        ),
        "wrong_contact_rate": float(
            np.mean([row["wrong_object_contact"] for row in rows])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_action_clip_rate": float(
            np.mean([row["action_clip_rate"] for row in rows])
        ),
        "mean_target_class_accuracy": float(
            np.mean([row["mean_target_class_accuracy"] for row in rows])
        ),
        "mean_safety_intervention_rate": float(
            np.mean([row["safety_intervention_rate"] for row in rows])
        ),
        "mean_latest_only_rate": float(
            np.mean([row["latest_only_rate"] for row in rows])
        ),
        "failure_counts": dict(Counter(row["failure_category"] for row in rows)),
    }
    task_gate = bool(
        report["overall"]["task_success_rate"] >= 0.80
        and all(
            bucket["task_success_rate"] >= 0.70
            for bucket in report["buckets"].values()
        )
    )
    clean_gate = bool(
        report["overall"]["clean_success_rate"] >= 0.80
        and all(
            bucket["clean_success_rate"] >= 0.70
            for bucket in report["buckets"].values()
        )
    )
    report["overall"]["passed_80_percent_task_gate"] = task_gate
    report["overall"]["passed_80_percent_clean_gate"] = clean_gate
    # Preserve the historical strict meaning used by --enforce-80.
    report["overall"]["passed_80_percent_gate"] = clean_gate
    return report


def main() -> None:
    args = parse_args()
    if args.replan_interval <= 0:
        raise ValueError("replan-interval must be positive")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(
        "Execution mode: raw | privileged assistance: False",
        flush=True,
    )
    model, stats = load_policy(args.policy, device, args.local_files_only)
    env = make_environment(args.render, horizon=args.max_steps)
    low, high = env.action_spec
    schedule = balanced_schedule(
        args.num_episodes,
        args.seed,
        args.instruction,
    )
    scene_rng = np.random.default_rng(args.seed)
    visual_rng = np.random.default_rng(args.seed + 1_000_003)
    rows: list[dict] = []
    stop_requested = False
    try:
        for episode_id, bucket in enumerate(schedule, start=1):
            while True:
                scene_seed = int(scene_rng.integers(0, np.iinfo(np.int32).max))
                reseed_environment(env, scene_seed)
                obs = env.reset()
                try:
                    task = schedule_task_v2(env, obs, bucket[0], bucket[1])
                    break
                except RuntimeError as error:
                    print(
                        f"Scene rejected for {bucket}: {error}",
                        flush=True,
                    )
            env.set_task(task)
            print(
                f"Episode {episode_id}/{args.num_episodes}: {task.instruction}",
                flush=True,
            )

            tracker = ProprioceptionTracker()
            initial_state = tracker.extract(obs)
            state_queue: deque[np.ndarray] = deque(
                [initial_state.copy() for _ in range(model.history_length)],
                maxlen=model.history_length,
            )
            agent_frame = capture_model_image(
                obs,
                "agentview",
                args.visual_perturbation,
                visual_rng,
            )
            wrist_frame = capture_model_image(
                obs,
                "robot0_eye_in_hand",
                args.visual_perturbation,
                visual_rng,
            )
            ensembler = TemporalActionEnsembler()
            gripper = GripperHysteresis()
            wrong_object_contact = False
            termination = "horizon"
            task_success = False
            clean_success = False
            task_success_streak = 0
            clean_success_streak = 0
            task_success_step = 0
            clean_success_step = 0
            grounding_errors = []
            target_class_correctness = []
            predicted_target_id = "?"
            target_index = TARGET_ID_TO_INDEX[task.target_id]
            initial_target_position = env.get_object_position(task.target_id).copy()
            initial_object_positions = np.stack(
                [env.get_object_position(object_id) for object_id in ("A", "B", "C")]
            )
            previous_object_positions = initial_object_positions.copy()
            table_height = float(env.table_offset[2])
            min_eef_target_distance = float("inf")
            target_contact = False
            grasped_once = False
            gripper_close_step = 0
            max_target_height = float(initial_target_position[2])
            min_target_height = float(initial_target_position[2])
            min_target_uprightness = env.object_uprightness(task.target_id)
            target_toppled = False
            action_clip_steps = 0
            safety_intervention_steps = 0
            safety_intervention_streak = 0
            workspace_violation_steps = 0
            recovery_cycles = 0
            previous_executed_phase = None
            pose_action_norms = []
            latest_only_steps = 0

            for zero_based_step in range(args.max_steps):
                step = zero_based_step + 1
                if (
                    zero_based_step % args.replan_interval == 0
                    or not ensembler.has_step(zero_based_step)
                ):
                    state_history = preprocess_history(
                        state_queue,
                        stats,
                        device,
                    )
                    with torch.no_grad():
                        output = model(
                            image_agentview=preprocess_image(agent_frame, device),
                            image_wrist=preprocess_image(wrist_frame, device),
                            state_history=state_history,
                            instructions=[task.instruction],
                        )
                    pose_normalized = output["pose"][0].cpu().numpy()
                    pose_chunk = (
                        pose_normalized * stats.action_pose_std
                        + stats.action_pose_mean
                    )
                    gripper_probability = torch.sigmoid(
                        output["gripper_logits"][0]
                    ).cpu().numpy()
                    phase_chunk = output["phase_logits"][0].argmax(dim=-1).cpu().numpy()
                    target_prediction = (
                        output["target_position"][0].cpu().numpy()
                        * stats.target_position_std
                        + stats.target_position_mean
                    )
                    ensembler.add(
                        zero_based_step,
                        pose_chunk,
                        gripper_probability,
                        phase_chunk,
                    )
                    true_target = env.get_object_position(task.target_id)
                    grounding_errors.append(
                        float(np.linalg.norm(target_prediction - true_target) * 100.0)
                    )
                    predicted_target_index = int(
                        output["target_class_logits"][0].argmax().item()
                    )
                    predicted_target_id = tuple(OBJECT_LABELS)[
                        predicted_target_index
                    ]
                    target_class_correctness.append(
                        float(predicted_target_index == target_index)
                    )

                reactive_phases = (
                    PICK_CONTACT_PHASES
                    if task.task_type == "pick"
                    else PUSH_CONTACT_PHASES
                )
                (
                    pose_action,
                    gripper_probability,
                    predicted_phase,
                    used_latest_only,
                ) = ensembler.action(zero_based_step, reactive_phases)
                latest_only_steps += int(used_latest_only)

                executed_phase = predicted_phase

                unclipped_pose_action = pose_action.copy()
                pose_action = np.clip(
                    pose_action,
                    -ORACLE_ACTION_LIMIT,
                    ORACLE_ACTION_LIMIT,
                )
                action = np.concatenate(
                    [
                        pose_action,
                        np.asarray([gripper.update(gripper_probability)]),
                    ]
                ).astype(np.float32)
                action = np.clip(action, low, high)
                (
                    action,
                    safety_intervened,
                    workspace_intervened,
                    workspace_xy_intervened,
                    outside_workspace,
                ) = apply_safety_shield(
                    action,
                    obs,
                    task,
                    table_height,
                )
                action = np.clip(action, low, high)
                safety_intervention_steps += int(safety_intervened)
                safety_intervention_streak = (
                    safety_intervention_streak + 1
                    if workspace_xy_intervened
                    else 0
                )
                workspace_violation_steps += int(outside_workspace)
                recovery_phase = 4 if task.task_type == "pick" else 9
                if (
                    executed_phase == recovery_phase
                    and previous_executed_phase != recovery_phase
                ):
                    recovery_cycles += 1
                previous_executed_phase = executed_phase
                action_clip_steps += int(
                    not np.allclose(unclipped_pose_action, pose_action, atol=1e-7)
                )
                pose_action_norms.append(float(np.linalg.norm(action[:6])))
                if action[6] > 0 and gripper_close_step == 0:
                    gripper_close_step = step
                obs, _, _, _ = env.step(action)
                state_queue.append(tracker.extract(obs))
                agent_frame = capture_model_image(
                    obs,
                    "agentview",
                    args.visual_perturbation,
                    visual_rng,
                )
                wrist_frame = capture_model_image(
                    obs,
                    "robot0_eye_in_hand",
                    args.visual_perturbation,
                    visual_rng,
                )

                contact_flags = env.object_contact_flags().astype(bool)
                grasp_flags = env.object_grasp_flags().astype(bool)
                current_object_positions = np.stack(
                    [
                        env.get_object_position(object_id)
                        for object_id in ("A", "B", "C")
                    ]
                )
                step_movement = np.linalg.norm(
                    current_object_positions - previous_object_positions,
                    axis=1,
                )
                total_movement = np.linalg.norm(
                    current_object_positions - initial_object_positions,
                    axis=1,
                )
                target_contact |= bool(
                    contact_flags[target_index]
                    or grasp_flags[target_index]
                    or step_movement[target_index]
                    > OBJECT_MOVEMENT_CONTACT_THRESHOLD
                )
                grasped_once |= bool(grasp_flags[target_index])
                wrong_object_contact |= bool(
                    np.any(np.delete(contact_flags, target_index))
                    or np.any(
                        np.delete(total_movement, target_index)
                        > WRONG_OBJECT_MOVEMENT_THRESHOLD
                    )
                )
                previous_object_positions = current_object_positions
                current_target_position = env.get_object_position(task.target_id)
                current_uprightness = env.object_uprightness(task.target_id)
                max_target_height = max(
                    max_target_height,
                    float(current_target_position[2]),
                )
                min_target_height = min(
                    min_target_height,
                    float(current_target_position[2]),
                )
                min_target_uprightness = min(
                    min_target_uprightness,
                    current_uprightness,
                )
                current_eef_target_distance = float(
                    np.linalg.norm(
                        np.asarray(obs["robot0_eef_pos"])
                        - current_target_position
                    )
                )
                min_eef_target_distance = min(
                    min_eef_target_distance,
                    current_eef_target_distance,
                )

                if (
                    task.task_type == "push"
                    and task.target_id == "C"
                    and current_uprightness < CYLINDER_TOPPLE_THRESHOLD
                    and not target_toppled
                ):
                    target_toppled = True
                    print(
                        f"Episode {episode_id}: green cylinder toppled at step "
                        f"{step}; continuing raw policy execution.",
                        flush=True,
                    )

                (
                    task_condition,
                    clean_condition,
                    push_forward,
                    push_lateral,
                ) = evaluate_success_conditions(
                    task_type=task.task_type,
                    target_id=task.target_id,
                    current_position=current_target_position,
                    initial_position=initial_target_position,
                    push_direction=task.push_direction,
                    grasping_target=bool(grasp_flags[target_index]),
                    uprightness=current_uprightness,
                    wrong_object_contact=wrong_object_contact,
                    table_height=table_height,
                )

                task_success_streak = (
                    task_success_streak + 1 if task_condition else 0
                )
                clean_success_streak = (
                    clean_success_streak + 1 if clean_condition else 0
                )
                new_task_success = bool(
                    not task_success
                    and task_success_streak >= SUCCESS_HOLD_STEPS
                )
                new_clean_success = bool(
                    not clean_success
                    and clean_success_streak >= SUCCESS_HOLD_STEPS
                )
                if new_task_success:
                    task_success = True
                    task_success_step = step
                if new_clean_success:
                    clean_success = True
                    clean_success_step = step

                if args.log_every and (
                    step == 1
                    or step % args.log_every == 0
                    or new_task_success
                    or new_clean_success
                ):
                    print(
                        f"Ep {episode_id} | Step {step}/{args.max_steps} | "
                        f"XYZ {action[:3].round(3)} | "
                        f"RPY {action[3:6].round(3)} | "
                        f"Grip {action[6]:.0f} ({gripper_probability:.2f}) | "
                        f"Phase {predicted_phase}->{executed_phase} | "
                        f"Ground {grounding_errors[-1]:.1f}cm | "
                        f"Target {predicted_target_id} | "
                        f"EEF {np.asarray(obs['robot0_eef_pos']).round(3)} | "
                        f"ObjZ {current_target_position[2]:.3f} | "
                        f"Up {current_uprightness:.2f} | "
                        f"Hold {task_success_streak}/{clean_success_streak} | "
                        f"Shield {safety_intervention_steps}",
                        flush=True,
                    )

                if args.render:
                    frame = np.ascontiguousarray(np.flipud(obs["frontview_image"]))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Live Demo - MiniVLA V2 Safety", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        termination = "user_stop"
                        stop_requested = True
                        break

                if task.task_type == "push" and task.target_id == "B":
                    sphere_floor = (
                        table_height
                        + float(env.objects_by_id["B"].horizontal_radius)
                        - SPHERE_PENETRATION_TOLERANCE
                    )
                    if current_target_position[2] < sphere_floor:
                        termination = "object_penetration"
                        print(
                            f"Episode {episode_id}: stopped at step {step}; "
                            f"blue ball center z={current_target_position[2]:.4f} "
                            f"fell below safety floor {sphere_floor:.4f}.",
                            flush=True,
                        )
                        break
                if (
                    not task_success
                    and target_contact
                    and recovery_cycles
                    >= (
                        MAX_PICK_RECOVERY_CYCLES
                        if task.task_type == "pick"
                        else MAX_PUSH_RECOVERY_CYCLES
                    )
                ):
                    termination = "recovery_limit"
                    print(
                        f"Episode {episode_id}: stopped at step {step} after "
                        f"{recovery_cycles} recovery cycles.",
                        flush=True,
                    )
                    break
                if (
                    not task_success
                    and target_contact
                    and safety_intervention_streak
                    >= MAX_SAFETY_INTERVENTION_STREAK
                ):
                    termination = "workspace_stall"
                    print(
                        f"Episode {episode_id}: stopped at step {step} after "
                        f"{safety_intervention_streak} consecutive safety "
                        "interventions.",
                        flush=True,
                    )
                    break
                if clean_success:
                    termination = "success"
                    break
                if task_success and (
                    wrong_object_contact
                    or (
                        task.task_type == "push"
                        and task.target_id == "C"
                        and current_uprightness < CYLINDER_TOPPLE_THRESHOLD
                    )
                ):
                    termination = "task_success_only"
                    break

            if termination in {
                "object_penetration",
                "recovery_limit",
                "workspace_stall",
            }:
                task_success = False
                clean_success = False
            elif termination == "horizon" and task_success:
                termination = "task_success_only"
            final_target_position = env.get_object_position(task.target_id)
            displacement = final_target_position[:2] - initial_target_position[:2]
            push_forward = float(np.dot(displacement, task.push_direction))
            push_lateral = float(
                np.linalg.norm(displacement - push_forward * task.push_direction)
            )
            final_grasp = bool(env.is_grasping(task.target_id))
            failure_category = classify_failure(
                task_type=task.task_type,
                task_success=task_success,
                wrong_object_contact=wrong_object_contact,
                termination=termination,
                min_eef_target_distance=min_eef_target_distance,
                gripper_close_step=gripper_close_step,
                target_contact=target_contact,
                grasped_once=grasped_once,
                final_grasp=final_grasp,
                max_target_height=max_target_height,
                final_target_height=float(final_target_position[2]),
                table_height=table_height,
                push_forward=push_forward,
                push_lateral=push_lateral,
            )
            row = {
                "episode": episode_id,
                "task_type": task.task_type,
                "target_id": task.target_id,
                "instruction": task.instruction,
                "visual_perturbation": args.visual_perturbation,
                "execution_mode": "raw",
                "uses_privileged_execution_assistance": 0,
                "scene_seed": scene_seed,
                "task_success": int(task_success),
                "clean_success": int(clean_success),
                "task_success_step": task_success_step,
                "clean_success_step": clean_success_step,
                "wrong_object_contact": int(wrong_object_contact),
                "steps": step,
                "termination": termination,
                "failure_category": failure_category,
                "mean_grounding_error_cm": float(np.mean(grounding_errors)),
                "mean_target_class_accuracy": float(
                    np.mean(target_class_correctness)
                ),
                "min_eef_target_distance": min_eef_target_distance,
                "gripper_close_step": gripper_close_step,
                "target_contact": int(target_contact),
                "grasped_once": int(grasped_once),
                "max_target_height": max_target_height,
                "min_target_height": min_target_height,
                "min_target_uprightness": min_target_uprightness,
                "target_toppled": int(target_toppled),
                "final_target_height": float(final_target_position[2]),
                "push_forward_displacement": push_forward,
                "push_lateral_displacement": push_lateral,
                "action_clip_rate": action_clip_steps / step,
                "safety_intervention_rate": (
                    safety_intervention_steps / step
                ),
                "workspace_violation_rate": (
                    workspace_violation_steps / step
                ),
                "recovery_cycles": recovery_cycles,
                "mean_pose_action_norm": float(np.mean(pose_action_norms)),
                "latest_only_rate": latest_only_steps / step,
            }
            rows.append(row)
            write_results(
                args.output_prefix,
                rows,
                summarize(rows, perturbation=args.visual_perturbation),
            )
            print(
                f"Episode {episode_id}: "
                f"{('Clean Success' if clean_success else 'Task Success' if task_success else 'Fail')} | "
                f"failure={failure_category} | termination={termination}",
                flush=True,
            )
            if stop_requested:
                break
    finally:
        env.close()
        cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("Evaluation ended before any episode completed")
    report = summarize(rows, perturbation=args.visual_perturbation)
    write_results(args.output_prefix, rows, report)
    print(json.dumps(report, indent=2), flush=True)
    print(
        f"Results: {args.output_prefix}.csv and {args.output_prefix}.json",
        flush=True,
    )
    if args.enforce_80 and not report["overall"]["passed_80_percent_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
