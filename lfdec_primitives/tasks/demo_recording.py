# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Record scripted demonstrations, store individual EC segments and compile their task YAML."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import lfdec_primitives as lp

from .envs.cabinet_ec_cfg import PANDA_FINGER_OPEN

__all__ = ["home", "record_approach", "write_task"]


def home(robot, term, sim_dt: float) -> np.ndarray:
    """Open the hand, let the spawn settle, home the wrench bias and let the filter catch up;
    returns the bias. The arm spawns at rest and out of contact, which is the condition homing
    requires."""
    robot.set_gripper(closed=False)
    robot.run(1.0, 0.0, robot.ee_pose()[1])
    term.controller.start_homing(num_ticks=round(0.5 / sim_dt))
    robot.run(0.8, 0.0, robot.ee_pose()[1])
    return term.controller.wrench_bias[0].cpu().numpy()


def record_approach(robot, cam, R_grasp: np.ndarray, p_grasp: np.ndarray, step: float = 0.02,
                    verbose: bool = True) -> lp.Demo:
    """The approach segment: the camera's view from here, a straight line to the grasp, the hand
    closing at its end. Recorded, then driven, so the next segment is recorded from a hand that
    actually holds.

    The view is taken at ``poses[0]`` because that is the pose the servo has to recover:
    ``Demo.poses[0]`` and ``Demo.view`` describe the same instant.
    """
    R0, p0 = robot.ee_pose()
    view = cam.capture()
    T0, T_grasp = lp.make_transform(R0, p0), lp.make_transform(R_grasp, p_grasp)
    commanded = [T0] + lp.line_trajectory(T0, T_grasp, max_step=step)
    poses = [T0]
    for T in commanded[1:]:
        robot.set_goal(T[:3, 3], T[:3, :3])
        robot.run(0.35, 0.0, robot.ee_pose()[1])
        poses.append(lp.make_transform(*robot.ee_pose()))
    robot.run(0.6, 0.0, robot.ee_pose()[1])
    R_end, p_end = robot.ee_pose()
    # The final sample commands closure at the reached grasp pose. Closure can
    # displace the hand; that measured motion belongs to the subsequent segment,
    # rather than being replayed with open fingers before the grasp command.
    poses.append(lp.make_transform(R_end, p_end))
    robot.set_gripper(closed=True)
    robot.run(1.2, 0.0, p_end)
    demo = lp.Demo.from_poses(poses, close_at_end=True, view=view)
    gap = robot.gripper_gap()
    if gap > 0.9 * PANDA_FINGER_OPEN:
        raise RuntimeError(
            f"the gripper did not close (gap {1e3 * gap:.1f} mm of an open {1e3 * PANDA_FINGER_OPEN:.0f} mm): "
            "the demonstration would record a hand sliding off rather than holding")
    if verbose:
        print(f"   approach: {len(poses)} waypoints over {1e3 * float(np.linalg.norm(p_grasp - p0)):.0f} mm, "
              f"reference view {view.rgb.shape[1]}x{view.rgb.shape[0]}; drove it to "
              f"{1e3 * float(np.linalg.norm(p_end - p_grasp)):.1f} mm of the grasp and closed to a "
              f"{1e3 * gap:.1f} mm finger gap")
    return demo


def write_task(path: Path, task: str, start: str, segments: dict, edges: list[dict], writer: str,
               execution: dict | None = None) -> None:
    """Save the provided segmentation and compile its recorded trajectories through the learner."""
    import yaml

    doc = {"schema_version": 1, "task": task, "start": start, "segments": segments, "edges": edges}
    if execution is not None:
        doc["execution"] = execution
    manifest = path.with_suffix(".demo.yaml")
    manifest.write_text(f"# Recorded by {writer}; EC labels and sample ranges are provided annotations.\n"
                        + yaml.safe_dump(doc, sort_keys=False))
    lp.compile_demonstration(manifest, path)
