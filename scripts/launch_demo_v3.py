"""Launch the local MiniVLA V3 interactive desktop application."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robosuite_numba_cache")

from utils.interactive_session_v3 import (
    DEFAULT_DEMO_POLICY,
    DEFAULT_DEMO_SEED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_DEMO_POLICY)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help="Allow Hugging Face downloads instead of requiring local files.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-command",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-nudge",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from ui.minivla_demo_window import MiniVLADemoWindow

    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("MiniVLA Control Deck")
    window = MiniVLADemoWindow(
        policy_path=args.policy,
        seed=args.seed,
        max_steps=args.max_steps,
        local_files_only=not args.allow_model_downloads,
        close_when_ready=args.smoke_test,
        smoke_command=args.smoke_command,
        smoke_nudge=args.smoke_nudge,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
