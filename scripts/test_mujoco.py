"""Minimal official MuJoCo baseline for macOS / Python 3.10.

This script builds a tiny MJCF scene directly from an XML string, advances the
simulation for 100 steps, and prints the cube qpos. Use ``--viewer`` to open the
interactive MuJoCo viewer via ``mjpython`` on macOS.
"""

from __future__ import annotations

import argparse
import time

import mujoco


MJCF_XML = """
<mujoco model="minimal_blue_cube">
  <compiler angle="degree" />

  <option timestep="0.01" gravity="0 0 -9.81" />

  <visual>
    <global azimuth="120" elevation="-20" />
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.20 0.22 0.24" rgb2="0.32 0.34 0.36" />
    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.2" />
    <material name="blue_mat" rgba="0.05 0.25 0.95 1" />
  </asset>

  <worldbody>
    <light name="key_light" pos="0 -3 4" dir="0 0 -1" diffuse="0.8 0.8 0.8" />
    <geom name="floor" type="plane" size="2 2 0.05" material="floor_mat" />

    <body name="blue_cube" pos="0 0 0.25">
      <freejoint name="cube_freejoint" />
      <geom name="blue_cube_geom" type="box" size="0.1 0.1 0.1"
            material="blue_mat" mass="0.1" />
    </body>
  </worldbody>
</mujoco>
"""


def run_simulation(model: mujoco.MjModel, data: mujoco.MjData, num_steps: int) -> None:
    """Advance the simulation by a fixed number of MuJoCo steps."""
    for _ in range(num_steps):
        mujoco.mj_step(model, data)


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, num_steps: int) -> None:
    """Open the native MuJoCo viewer and step the same minimal scene."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(num_steps):
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal official MuJoCo scene.")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer.")
    parser.add_argument("--steps", type=int, default=100, help="Number of mj_step calls.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(MJCF_XML)
    data = mujoco.MjData(model)

    if args.viewer:
        run_viewer(model, data, args.steps)
    else:
        run_simulation(model, data, args.steps)

    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
    cube_qpos_address = model.jnt_qposadr[model.body_jntadr[cube_body_id]]
    cube_qpos = data.qpos[cube_qpos_address : cube_qpos_address + 7]

    print(f"blue_cube qpos: {cube_qpos}")
    print("MuJoCo simulation ran successfully.")


if __name__ == "__main__":
    main()
