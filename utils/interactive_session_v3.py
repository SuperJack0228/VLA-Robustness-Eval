"""Persistent, non-logging runtime used by the MiniVLA desktop demo."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from queue import Empty, Full
from typing import Callable

import numpy as np
import torch

from scripts.collect_data_v2 import (
    CONTROL_FREQ,
    reseed_environment,
    schedule_task_v2,
)
from utils.evaluation_core_v2 import (
    EvaluationConfig,
    EvaluationCore,
    get_device,
    load_policy,
    make_environment,
)
from utils.instruction_resolver_v3 import ResolvedInstruction
from utils.perturbations_v2 import (
    DynamicTargetDisplacement,
    PerturbationContext,
    PerturbationManager,
    PerturbationSceneRejected,
)
from utils.v2_schema import TASK_BUCKETS


DEFAULT_DEMO_POLICY = "artifacts/v3-clean-rc1/mini_vla_v3_policy.pth"
DEFAULT_DEMO_SEED = 20260808
DEFAULT_NUDGE_DISTANCE_M = 0.04
MAX_SCENE_ATTEMPTS = 120


class InteractiveControl:
    """Thread-safe cooperative controls for a blocking evaluation episode."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def begin_episode(self) -> None:
        self._stop_event.clear()

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()


class InteractiveTargetNudge(DynamicTargetDisplacement):
    """Queue one safe displacement and fire it in the next reactive phase."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._request_event = threading.Event()
        self._available = False
        self._unavailable_reason = "Scene has not been prepared"

    def on_episode_start(self, context: PerturbationContext) -> None:
        self._request_event.clear()
        try:
            super().on_episode_start(context)
        except PerturbationSceneRejected as error:
            self._available = False
            self._unavailable_reason = str(error)
            return
        self._available = True
        self._unavailable_reason = ""

    def request(self) -> bool:
        if not self._available or self._fired:
            return False
        self._request_event.set()
        return True

    def before_step(self, context: PerturbationContext) -> dict | None:
        if not self._request_event.is_set():
            return None
        event = super().before_step(context)
        if event is not None:
            self._request_event.clear()
        return event

    @property
    def available(self) -> bool:
        return self._available and not self._fired

    @property
    def queued(self) -> bool:
        return self._request_event.is_set() and not self._fired

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason


def _display_frames(obs: dict) -> dict[str, np.ndarray]:
    return {
        "frontview": np.ascontiguousarray(np.flipud(obs["frontview_image"])),
        "agentview": np.ascontiguousarray(np.flipud(obs["agentview_image"])),
        "wrist": np.ascontiguousarray(
            np.flipud(obs["robot0_eye_in_hand_image"])
        ),
    }


class InteractiveDemoSession:
    """Own one policy and one MuJoCo environment across multiple demo runs."""

    def __init__(
        self,
        policy_path: str = DEFAULT_DEMO_POLICY,
        seed: int = DEFAULT_DEMO_SEED,
        max_steps: int = 200,
        local_files_only: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.policy_path = policy_path
        self.max_steps = int(max_steps)
        self.local_files_only = bool(local_files_only)
        self.device = device or get_device()
        self.rng = np.random.default_rng(seed)
        self.visual_rng = np.random.default_rng(seed + 1_000_003)
        self.control = InteractiveControl()
        self.model = None
        self.stats = None
        self.env = None
        self.obs: dict | None = None
        self.scene_seed = 0
        self.scene_rejections = 0
        self.episode_id = 0
        self.available_tasks: dict[tuple[str, str], object] = {}
        self.scene_consumed = False
        self.active_nudge: InteractiveTargetNudge | None = None
        self._nudge_lock = threading.Lock()

    def initialize(self) -> dict:
        if self.env is not None:
            raise RuntimeError("Interactive demo session is already initialized")
        self.model, self.stats = load_policy(
            self.policy_path,
            self.device,
            self.local_files_only,
        )
        self.env = make_environment(render=True, horizon=self.max_steps)
        return self.reset_scene()

    def _sample_scene(
        self,
        required_task: tuple[str, str] | None,
    ) -> tuple[dict, int, int, dict[tuple[str, str], object]]:
        if self.env is None:
            raise RuntimeError("Interactive demo session is not initialized")
        best: tuple[dict, int, int, dict[tuple[str, str], object]] | None = None
        for rejection_count in range(MAX_SCENE_ATTEMPTS):
            scene_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
            reseed_environment(self.env, scene_seed)
            obs = self.env.reset()
            tasks = {}
            for bucket in TASK_BUCKETS:
                try:
                    tasks[bucket] = schedule_task_v2(
                        self.env,
                        obs,
                        bucket[0],
                        bucket[1],
                    )
                except RuntimeError:
                    continue
            candidate = (obs, scene_seed, rejection_count, tasks)
            if best is None or len(tasks) > len(best[3]):
                best = candidate
            if required_task is not None and required_task in tasks:
                return candidate
            if required_task is None and len(tasks) == len(TASK_BUCKETS):
                return candidate
        if best is not None and (required_task is None or required_task in best[3]):
            return best
        requested = "all tasks" if required_task is None else str(required_task)
        raise RuntimeError(
            f"Could not generate a collision-safe scene for {requested} after "
            f"{MAX_SCENE_ATTEMPTS} attempts"
        )

    def reset_scene(
        self,
        required_task: tuple[str, str] | None = None,
    ) -> dict:
        if self.env is None:
            raise RuntimeError("Interactive demo session is not initialized")
        obs, scene_seed, rejections, tasks = self._sample_scene(required_task)
        self.obs = obs
        self.scene_seed = scene_seed
        self.scene_rejections = rejections
        self.available_tasks = tasks
        self.scene_consumed = False
        self.env.active_task = None
        self.env.initial_target_position = None
        with self._nudge_lock:
            self.active_nudge = None
        payload = _display_frames(obs)
        payload.update(
            {
                "scene_seed": scene_seed,
                "scene_rejections": rejections,
                "available_tasks": tuple(sorted(tasks)),
            }
        )
        return payload

    def ensure_task_scene(self, task: tuple[str, str]) -> dict | None:
        if self.scene_consumed or task not in self.available_tasks:
            return self.reset_scene(required_task=task)
        return None

    def run_task(
        self,
        resolved: ResolvedInstruction,
        step_callback: Callable[[dict], None],
        episode_ready_callback: Callable[[dict], None] | None = None,
        external_stop_requested: Callable[[], bool] | None = None,
    ) -> dict:
        if self.env is None or self.model is None or self.stats is None:
            raise RuntimeError("Interactive demo session is not initialized")
        if self.obs is None:
            raise RuntimeError("No prepared scene is available")
        task_key = (resolved.task_type, resolved.target_id)
        if task_key not in self.available_tasks:
            raise RuntimeError(f"Prepared scene does not support {task_key}")
        self.scene_consumed = True
        self.episode_id += 1
        self.control.begin_episode()
        task = replace(
            self.available_tasks[task_key],
            instruction=resolved.model_instruction,
        )
        self.env.set_task(task)
        nudge = InteractiveTargetNudge(
            distance_m=DEFAULT_NUDGE_DISTANCE_M,
            base_seed=self.scene_seed + 31,
        )
        manager = PerturbationManager([nudge])
        manager.on_episode_start(
            self.episode_id,
            self.scene_seed,
            self.env,
            task,
            self.obs,
        )
        with self._nudge_lock:
            self.active_nudge = nudge
        if episode_ready_callback is not None:
            episode_ready_callback(
                {
                    "episode": self.episode_id,
                    "task": resolved.display_name,
                    "model_instruction": resolved.model_instruction,
                    "nudge_available": nudge.available,
                    "nudge_unavailable_reason": nudge.unavailable_reason,
                }
            )
        config = EvaluationConfig(
            policy_path=self.policy_path,
            num_episodes=1,
            max_steps=self.max_steps,
            seed=self.scene_seed,
            instruction=resolved.model_instruction,
            task_type=resolved.task_type,
            target_id=resolved.target_id,
            replan_interval=1,
            render=False,
            log_every=0,
            local_files_only=self.local_files_only,
            visual_perturbation="clean",
            ensemble_mode="temporal",
            temporal_profile="robust",
            perturbation_label="interactive_demo",
            diagnostic_trace=False,
            write_outputs=False,
        )
        core = EvaluationCore(
            self.model,
            self.stats,
            self.env,
            self.device,
            config,
            manager,
            step_callback=self._paced_callback(step_callback),
            stop_requested=lambda: self.control.stop_requested()
            or (
                external_stop_requested is not None
                and external_stop_requested()
            ),
        )
        low, high = self.env.action_spec
        try:
            row, _ = core.run_episode(
                self.episode_id,
                self.scene_seed,
                task,
                self.obs,
                self.visual_rng,
                low,
                high,
                self.scene_rejections,
            )
            return row
        finally:
            with self._nudge_lock:
                self.active_nudge = None

    @staticmethod
    def _paced_callback(
        callback: Callable[[dict], None],
    ) -> Callable[[dict], None]:
        period = 1.0 / float(CONTROL_FREQ)
        previous_tick = time.perf_counter()

        def emit(payload: dict) -> None:
            nonlocal previous_tick
            now = time.perf_counter()
            remaining = period - (now - previous_tick)
            if remaining > 0.0:
                time.sleep(remaining)
            previous_tick = time.perf_counter()
            callback(payload)

        return emit

    def request_stop(self) -> None:
        self.control.request_stop()

    def request_nudge(self) -> bool:
        with self._nudge_lock:
            if self.active_nudge is None:
                return False
            return self.active_nudge.request()

    def close(self) -> None:
        self.request_stop()
        if self.env is not None:
            self.env.close()
            self.env = None


def _put_latest(queue, payload: dict) -> None:
    """Keep video real-time by dropping stale frames when the UI falls behind."""
    try:
        queue.put_nowait(payload)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        pass
    try:
        queue.put_nowait(payload)
    except Full:
        pass


def demo_process_main(
    policy_path: str,
    seed: int,
    max_steps: int,
    local_files_only: bool,
    command_queue,
    status_queue,
    frame_queue,
    stop_event,
    nudge_event,
) -> None:
    """Run MuJoCo in a child process so its GL context stays on that main thread."""
    session: InteractiveDemoSession | None = None
    try:
        session = InteractiveDemoSession(
            policy_path=policy_path,
            seed=seed,
            max_steps=max_steps,
            local_files_only=local_files_only,
        )
        scene = session.initialize()
        _put_latest(frame_queue, {"kind": "scene", "payload": scene})
        status_queue.put({"kind": "initialized", "device": str(session.device)})
        while True:
            command = command_queue.get()
            command_type = command.get("type")
            if command_type == "shutdown":
                break
            if command_type == "reset":
                stop_event.clear()
                nudge_event.clear()
                scene = session.reset_scene()
                _put_latest(frame_queue, {"kind": "scene", "payload": scene})
                continue
            if command_type != "start":
                status_queue.put(
                    {
                        "kind": "error",
                        "message": f"Unsupported demo command: {command_type}",
                    }
                )
                continue

            resolved = command["resolved"]
            stop_event.clear()
            nudge_event.clear()
            replacement = session.ensure_task_scene(
                (resolved.task_type, resolved.target_id)
            )
            if replacement is not None:
                _put_latest(
                    frame_queue,
                    {"kind": "scene", "payload": replacement},
                )

            def episode_ready(payload: dict) -> None:
                status_queue.put({"kind": "episode_ready", "payload": payload})

            def step_ready(payload: dict) -> None:
                if nudge_event.is_set():
                    accepted = session.request_nudge()
                    nudge_event.clear()
                    status_queue.put(
                        {
                            "kind": "nudge_status",
                            "accepted": accepted,
                        }
                    )
                _put_latest(
                    frame_queue,
                    {"kind": "step", "payload": payload},
                )

            row = session.run_task(
                resolved,
                step_callback=step_ready,
                episode_ready_callback=episode_ready,
                external_stop_requested=stop_event.is_set,
            )
            status_queue.put({"kind": "episode_finished", "payload": row})
    except Exception as error:
        status_queue.put(
            {
                "kind": "error",
                "message": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        if session is not None:
            session.close()
        status_queue.put({"kind": "shutdown_complete"})
