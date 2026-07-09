"""Minimal robosuite Panda Lift smoke test for macOS."""

from __future__ import annotations

import numpy as np
import robosuite as suite


def main() -> None:
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
    )

    env.reset()

    for _ in range(200):
        low, high = env.action_spec
        action = np.random.uniform(low, high)
        env.step(action)
        env.render()

    env.close()

    print("Robosuite Panda Lift environment ran successfully.")


if __name__ == "__main__":
    main()
