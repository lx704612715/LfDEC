# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Record scripted block grasp and lift segments and compile their task YAML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import _repo_path  # noqa: F401
from _launcher import add_launcher_args
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

from lfdec_primitives.tasks import resolve_ec_task

TASK = "LfDEC-Grasp-Free-Franka"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("tasks"))
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    cfg = resolve_ec_task(TASK)
    with launch_simulation(cfg, args):
        return record(args, cfg)


def record(args, cfg):
    import gymnasium as gym
    import numpy as np
    import lfdec_primitives as lp
    from lfdec_primitives.tasks.demo_recording import home, record_approach, write_task
    from lfdec_primitives.tasks.robots.task_robots import BlockRobot
    from lfdec_primitives.tasks.envs.grasp_free_cfg import GRASP_R
    from lfdec_primitives.tasks.sensors.camera import mount_transform
    from lfdec_primitives.tasks.sensors.camera import IsaacCamera

    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        term = env.action_manager.get_term("arm_action")
        robot = BlockRobot(env, term, cfg.sim.dt * cfg.decimation)
        cam = IsaacCamera(robot, "wrist_cam", mount_transform(robot.finger_axis_ee()))
        home(robot, term, cfg.sim.dt)
        R0, p0 = robot.root_pose()
        block = R0.T @ (env.scene["block"].data.root_pos_w.torch[0].cpu().numpy() - p0)
        robot.set_goal(block + [0, 0, .15], GRASP_R)
        robot.run(3., 0., robot.ee_pose()[1])
        initial = robot.extra()["block_z"]
        grasp = record_approach(robot, cam, GRASP_R, block, step=.01)
        R, p = robot.ee_pose()
        poses = [lp.make_transform(R, p)]
        for step in np.linspace(0., .08, 17)[1:]:
            robot.set_goal(p + [0., 0., step], R)
            robot.run(.35, 0., robot.ee_pose()[1])
            poses.append(lp.make_transform(*robot.ee_pose()))
        robot.run(1., 0., robot.ee_pose()[1])
        poses.append(lp.make_transform(*robot.ee_pose()))
        lifted = robot.extra()["block_z"] - initial
        if lifted < .04:
            raise RuntimeError(f"The teacher did not retain a physical block grasp: lift={lifted}")
        lift = lp.Demo.from_poses(poses, grip_closed=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        grasp.save(args.out_dir / "block_grasp.npz")
        lift.save(args.out_dir / "block_lift.npz")
        output = args.out_dir / "block_grasp_lift.yaml"
        write_task(output, TASK, "grasp", {
            "grasp": dict(state="free", demo="block_grasp.npz", cfg=dict(
                max_approach=0., waypoint_tol=.001, waypoint_rot_tol=.01, grip_settle=1.2),
                servo=dict(tol=.001, rot_tol=.005)),
            "lift": dict(state="free", demo="block_lift.npz", cfg=dict(
                max_approach=0., waypoint_tol=.008, waypoint_rot_tol=.02)),
        }, [dict(node="grasp", successor="lift", event="NonC")], writer="scripts/record_grasp_ec_demo.py",
            execution=dict(max_iters=200, hand_dx=.02, hand_dy=-.02, hand_dz=.03, hand_dyaw_deg=5.))
        report = dict(success=True, lifted=lifted, finger_gap=robot.gripper_gap())
        output.with_suffix(".teacher.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
