# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Record scripted drawer approach and pull segments and compile their task YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _repo_path  # puts the repo root on sys.path; REPO_ROOT used below
from _launcher import add_launcher_args  # Isaac Lab's flags, viewer on by default
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

from lfdec_primitives.tasks import resolve_ec_task  # registers LfDEC-* on import

TASK = "LfDEC-Drawer-EC-Franka"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launcher_args(p)
    p.add_argument("--task", default=TASK)
    p.add_argument("--pull", type=float, default=0.12,
                   help="m of drawer travel to demonstrate. Only long enough to fit the axis "
                        "and show the intent; the execution decides how far to actually open it")
    p.add_argument("--approach-step", type=float, default=0.02,
                   help="m between recorded approach waypoints")
    p.add_argument("--open-target", type=float, default=0.20,
                   help="m of drawer travel the execution should aim for, written into the "
                        "translation node. Not 0.25: from this posture the arm has ~28.9 cm of "
                        "pull and the primitive stops 3 cm short of its joint limits, so 0.25 "
                        "races the reach check and the run ends JointLimit instead of Done")
    p.add_argument("--out-dir", default=None, help="default: <repo>/tasks")
    p.add_argument("--name", default="drawer_grasp_pull", help="stem of the YAML that is written")
    return p


def main() -> int:
    args, hydra_args = setup_preset_cli(build_parser())
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg = resolve_ec_task(args.task)
    with launch_simulation(env_cfg, args):
        return _record(args, env_cfg)


def _record(args, env_cfg) -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    import lfdec_primitives as lp
    from lfdec_primitives.tasks.demo_recording import home, record_approach, write_task
    from lfdec_primitives.tasks.envs.drawer_ec_cfg import (
        PROGRESS_JOINT,
        STANDOFF,
        grasp_frame,
    )
    from lfdec_primitives.tasks.envs.drawer_translation_cfg import drawer_pull_axis
    from lfdec_primitives.tasks.sensors.camera import mount_transform
    from lfdec_primitives.tasks.sensors.camera import IsaacCamera
    from lfdec_primitives.tasks.robots.isaac_robot import IsaacECRobot

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    term = env.action_manager.get_term("arm_action")
    dt_step = env_cfg.sim.dt * env_cfg.decimation
    env.reset()
    robot = IsaacECRobot(env, term, dt_step)
    cam = IsaacCamera(robot, "wrist_cam", mount_transform(robot.finger_axis_ee()))
    bias = home(robot, term, env_cfg.sim.dt)
    print(f"\n   wrench bias = F {bias[:3].round(2)} N  T {bias[3:].round(3)} N.m")

    R_handle, p_handle = grasp_frame(env)
    u_truth = drawer_pull_axis(env.scene["cabinet_frame"].data.target_quat_w.torch[:, 0],
                               torch.as_tensor(p_handle, device=env.device).unsqueeze(0),
                               env.scene["robot"].data.root_quat_w.torch)
    _, p_now = robot.ee_pose()
    print(f"   handle {p_handle.round(4)}, TCP {p_now.round(4)}, "
          f"{1e3 * float(np.linalg.norm(p_now - p_handle)):.0f} mm apart (STANDOFF {1e3 * STANDOFF:.0f} mm)")

    # ---------------------------------------------------------------- segment 1: the approach
    approach = record_approach(robot, cam, R_handle, p_handle, step=args.approach_step)

    # ---------------------------------------------------------------- segment 2: the pull
    cabinet = env.scene["cabinet"]
    j = cabinet.find_joints("drawer_top_joint")[0][0]
    R_p, p_p = robot.ee_pose()
    pull_poses, drawer_q = [lp.make_transform(R_p, p_p)], [float(cabinet.data.joint_pos.torch[0, j])]
    steps = max(2, round(args.pull / 0.01))
    for i in range(1, steps + 1):
        robot.set_goal(p_p + (i * args.pull / steps) * u_truth, R_p)
        robot.run(0.30, 0.0, robot.ee_pose()[1])
        R_k, p_k = robot.ee_pose()
        pull_poses.append(lp.make_transform(R_k, p_k))
        drawer_q.append(float(cabinet.data.joint_pos.torch[0, j]))
    pull = lp.Demo.from_poses(pull_poses, grip_closed=True)
    opened = drawer_q[-1] - drawer_q[0]
    if opened < 0.5 * args.pull:
        raise RuntimeError(
            f"the drawer only moved {1e3 * opened:.0f} mm while the hand travelled "
            f"{1e3 * float(np.linalg.norm(pull_poses[-1][:3, 3] - pull_poses[0][:3, 3])):.0f} mm: "
            "the grasp slipped, so this is not a demonstration of a prismatic EC")

    # The EC parameter, fitted to what was recorded rather than read from the scene.
    path = np.stack([T[:3, 3] for T in pull_poses])
    u_fit = lp.unit(lp.fit_line_direction(path))   # already signed first point -> last
    print(f"   pull: drawer {1e3 * drawer_q[0]:.0f} -> {1e3 * drawer_q[-1]:.0f} mm, TCP travelled "
          f"{1e3 * float(np.linalg.norm(path[-1] - path[0])):.0f} mm over {len(pull_poses)} samples")
    print(f"   EC parameter (prismatic axis) fitted from the demonstration: {u_fit.round(4)}")
    print(f"       scene truth {u_truth.round(4)}, fit error {np.degrees(lp.angle_between(u_fit, u_truth)):.2f} deg")

    # ---------------------------------------------------------------- write it out
    out_dir = Path(args.out_dir) if args.out_dir else Path(_repo_path.REPO_ROOT) / "tasks"
    out_dir.mkdir(parents=True, exist_ok=True)
    approach.save(out_dir / "drawer_approach.npz")
    pull.save(out_dir / "drawer_pull.npz")
    yaml_path = out_dir / f"{args.name}.yaml"
    write_task(yaml_path, args.task, "grasp", {
        "grasp": {
            "state": "free",
            "demo": "drawer_approach.npz",
            # The approach ends holding the bar, so there is no next constraint to seek: the
            # grasp itself is the contact-changing event.
            "cfg": {"max_approach": 0.0},
        },
        "pull": {
            "state": "prismatic",
            "demo": "drawer_pull.npz",
            "cfg": {"delta_p": 0.10, "terminate_extra": [PROGRESS_JOINT, round(float(args.open_target), 3)]},
        },
    }, [{"node": "grasp", "successor": "pull", "event": "NonC"}], writer="scripts/record_drawer_ec_demo.py")
    print(f"\n   wrote {yaml_path.name}, drawer_approach.npz, drawer_pull.npz in {out_dir}")
    print(f"   run it:  uv run python scripts/run_ec_task.py --task-file tasks/{yaml_path.name} physics=newton_mjwarp")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
