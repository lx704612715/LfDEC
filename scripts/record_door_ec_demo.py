# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Record scripted door approach and opening segments and compile their task YAML."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import _repo_path  # puts the repo root on sys.path; REPO_ROOT used below
from _launcher import add_launcher_args  # Isaac Lab's flags, viewer on by default
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

from lfdec_primitives.tasks import resolve_ec_task  # registers LfDEC-* on import

TASK = "LfDEC-Door-EC-Franka"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launcher_args(p)
    p.add_argument("--task", default=TASK)
    p.add_argument("--turn", type=float, default=25.0,
                   help="deg of door opening to demonstrate: enough arc to fit the hinge, well "
                        "inside the ~38 deg the wrist can follow")
    p.add_argument("--approach-step", type=float, default=0.02, help="m between recorded approach waypoints")
    p.add_argument("--open-target", type=float, default=25.0,
                   help="deg the execution should open the door to, written into the rotation node. "
                        "Inside the arm's range: from the standoff posture the wrist runs out at ~30 deg "
                        "and the primitive stops 5.7 deg before that, so 30 ends JointLimit and 25 Done")
    p.add_argument("--out-dir", default=None, help="default: <repo>/tasks")
    p.add_argument("--name", default="door_grasp_open", help="stem of the YAML that is written")
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

    import lfdec_primitives as lp
    from lfdec_primitives.tasks.demo_recording import home, record_approach, write_task
    from lfdec_primitives.tasks.envs.door_ec_cfg import PROGRESS_JOINT, STANDOFF, grasp_frame
    from lfdec_primitives.tasks.envs.door_rotation_cfg import DOOR_JOINT, door_hinge
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

    cabinet, arm = env.scene["cabinet"], env.scene["robot"]
    R_knob, p_knob = grasp_frame(env)
    axis_true, hinge_true = door_hinge(cabinet, arm)
    _, p_now = robot.ee_pose()
    print(f"   knob {p_knob.round(4)}, TCP {p_now.round(4)}, "
          f"{1e3 * float(np.linalg.norm(p_now - p_knob)):.0f} mm apart (STANDOFF {1e3 * STANDOFF:.0f} mm)")

    # ---------------------------------------------------------------- segment 1: the approach
    approach = record_approach(robot, cam, R_knob, p_knob, step=args.approach_step)

    # ---------------------------------------------------------------- segment 2: the turn
    # The door opens along -door_left_joint (its range is [-1.57, 0]), so the hand is carried
    # round -axis. The hand turns with the door: that is what a person's does on a knob.
    j = cabinet.find_joints(DOOR_JOINT)[0][0]
    R0, p0 = robot.ee_pose()
    T0 = lp.make_transform(R0, p0)
    turn_poses, door_q = [T0], [float(cabinet.data.joint_pos.torch[0, j])]
    steps = max(2, math.ceil(args.turn / 2.5))
    for i in range(1, steps + 1):
        T = lp.screw_transform(-axis_true, hinge_true, math.radians(args.turn) * i / steps) @ T0
        robot.set_goal(T[:3, 3], T[:3, :3])
        robot.run(0.35, 0.0, robot.ee_pose()[1])
        R_k, p_k = robot.ee_pose()
        turn_poses.append(lp.make_transform(R_k, p_k))
        door_q.append(float(cabinet.data.joint_pos.torch[0, j]))
    turn = lp.Demo.from_poses(turn_poses, grip_closed=True)
    opened = math.degrees(door_q[0] - door_q[-1])
    if opened < 0.5 * args.turn:
        raise RuntimeError(f"the door only opened {opened:.1f} deg of the {args.turn:g} demonstrated: the grasp "
                           "slipped off the knob, so this is not a demonstration of a revolute EC")

    # The EC parameters, fitted to what was recorded rather than read from the scene.
    axis_fit, point_fit = lp.fit_revolute_axis(np.stack(turn_poses))
    print(f"   turn: door 0 -> {opened:.1f} deg over {len(turn_poses)} samples, knob on a "
          f"{1e2 * lp.point_line_distance(p0, axis_fit, point_fit):.1f} cm radius")
    print(f"   EC parameters (hinge) fitted from the demonstration: axis {axis_fit.round(4)} "
          f"through {point_fit.round(4)}")
    print(f"       scene truth axis {(-axis_true).round(4)} through {hinge_true.round(4)}: fit error "
          f"{math.degrees(lp.angle_between(axis_fit, -axis_true)):.2f} deg / "
          f"{1e3 * lp.point_line_distance(point_fit, axis_true, hinge_true):.1f} mm")

    # ---------------------------------------------------------------- write it out
    out_dir = Path(args.out_dir) if args.out_dir else Path(_repo_path.REPO_ROOT) / "tasks"
    out_dir.mkdir(parents=True, exist_ok=True)
    approach.save(out_dir / "door_approach.npz")
    turn.save(out_dir / "door_turn.npz")
    yaml_path = out_dir / f"{args.name}.yaml"
    write_task(yaml_path, args.task, "grasp", {
        "grasp": {
            "state": "free",
            "demo": "door_approach.npz",
            "cfg": {"max_approach": 0.0},   # the grasp itself is the contact-changing event
        },
        "open": {
            "state": "revolute",
            "demo": "door_turn.npz",
            # The joint runs negative as the door opens; the primitive reads the side off the
            # start, so the threshold is just the joint's value at the target opening.
            "cfg": {"terminate_extra": [PROGRESS_JOINT, -round(math.radians(args.open_target), 3)]},
        },
    }, [{"node": "grasp", "successor": "open", "event": "NonC"}], writer="scripts/record_door_ec_demo.py")
    print(f"\n   wrote {yaml_path.name}, door_approach.npz, door_turn.npz in {out_dir}")
    print(f"   run it:  uv run python scripts/run_ec_task.py --task-file tasks/{yaml_path.name} physics=newton_mjwarp")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
