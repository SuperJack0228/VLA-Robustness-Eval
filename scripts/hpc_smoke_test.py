"""End-to-end CUDA and MuJoCo EGL smoke test for Cognition HPC."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "robosuite_numba_cache"),
)

import cv2
import mujoco
import robosuite
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.evaluation_core_v2 import EvaluationConfig, EvaluationCore, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Host: {platform.node()}", flush=True)
    print(
        f"PyTorch: {torch.__version__} | runtime CUDA: {torch.version.cuda}",
        flush=True,
    )
    print(
        f"CUDA available: {torch.cuda.is_available()} | device: {device}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
            flush=True,
        )
    elif not args.allow_cpu:
        raise RuntimeError("A Slurm GPU allocation is required for this smoke test")

    print(
        f"MuJoCo: {mujoco.__version__} | robosuite: {robosuite.__version__} | "
        f"OpenCV: {cv2.__version__} | MUJOCO_GL={os.environ['MUJOCO_GL']}",
        flush=True,
    )

    config = EvaluationConfig(
        policy_path=args.policy,
        num_episodes=1,
        max_steps=2,
        seed=args.seed,
        instruction="Pick up the red cube",
        render=False,
        log_every=1,
        local_files_only=True,
        write_outputs=False,
    )
    core = EvaluationCore.from_policy(config, device=device)
    result = core.run()
    if len(result.rows) != 1:
        raise RuntimeError("The end-to-end evaluator did not return one row")
    print("HPC CUDA + MuJoCo EGL + MiniVLA smoke test: PASS", flush=True)


if __name__ == "__main__":
    main()
