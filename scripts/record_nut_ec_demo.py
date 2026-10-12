# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Record scripted nut turning segments and compile their task YAML."""
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
from lfdec_primitives.tasks.envs.factory_nut_cfg import ENTRY_TASK, TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entry-yaw-deg", type=float, default=0., help="teacher's free-space nut orientation before entry")
    parser.add_argument("--turn-deg", type=float, default=90., help="teacher's demonstrated turn")
    parser.add_argument("--step-time", type=float, default=1., help="seconds allowed per teacher waypoint")
    parser.add_argument("--preload", type=float, default=.005, help="teacher's axial spring preload in metres")
    parser.add_argument("--teacher-mode", choices=("primitive", "waypoints"), default="primitive")
    parser.add_argument("--pre-engaged", action="store_true", default=True,
                        help="record the qualified screw-contact starting condition (default)")
    parser.add_argument("--thread-entry", action="store_false", dest="pre_engaged",
                        help="experimental: start clear of the bolt; recordings must pass the same quality checks")
    args, hydra = setup_preset_cli(parser)
    if args.out_dir is None:
        args.out_dir = Path("tasks" if args.pre_engaged else "outputs/nut-entry")
    if args.pre_engaged and args.entry_yaw_deg:
        raise ValueError("--entry-yaw-deg requires the experimental --thread-entry mode")
    sys.argv = [sys.argv[0], *hydra]
    cfg = resolve_ec_task(TASK if args.pre_engaged else ENTRY_TASK)
    cfg.seed = args.seed
    with launch_simulation(cfg, args):
        return record(args, cfg)


def record(args, cfg):
    import gymnasium as gym
    import numpy as np
    import lfdec_primitives as lp
    from lfdec_primitives.tasks.demo_recording import write_task
    from lfdec_primitives.tasks.envs.factory_nut_cfg import BOLT_POSITION, ENTRY_BOLT_POSITION, NUT_OFFSET, THREAD_PITCH
    from lfdec_primitives.tasks.sensors.camera import mount_transform
    from lfdec_primitives.tasks.sensors.camera import IsaacCamera
    from lfdec_primitives.tasks.robots.isaac_robot import IsaacECRobot

    task = TASK if args.pre_engaged else ENTRY_TASK
    bolt_position = BOLT_POSITION if args.pre_engaged else ENTRY_BOLT_POSITION
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        env.reset()
        term = env.action_manager.get_term("arm_action")
        robot = IsaacECRobot(env, term, cfg.sim.dt * cfg.decimation)
        cam = None if args.pre_engaged else IsaacCamera(robot, "wrist_cam", mount_transform(robot.finger_axis_ee()))
        robot.set_gripper(closed=True)
        robot.run(1., 0., robot.ee_pose()[1])
        term.controller.start_homing(num_ticks=round(.5 / cfg.sim.dt))
        robot.run(.5, 0., robot.ee_pose()[1])
        R0, p0 = robot.ee_pose()
        initial_bottom = (p0 + R0 @ NUT_OFFSET)[2] + .01
        bolt_top = bolt_position[2] + .035
        if not args.pre_engaged and initial_bottom - bolt_top < .003:
            raise RuntimeError("The demonstration must start with the nut clear of the thread")
        reference = None if cam is None else cam.capture()
        poses = [lp.make_transform(R0, p0)]
        for yaw in np.linspace(0., np.deg2rad(args.entry_yaw_deg), max(2, int(abs(args.entry_yaw_deg) / 5) + 1))[1:]:
            robot.set_goal(p0, lp.rz(yaw) @ R0)
            robot.run(.6, 0., robot.ee_pose()[1])
            poses.append(lp.make_transform(*robot.ee_pose()))
        R0 = robot.ee_pose()[0]
        for i in ([] if args.pre_engaged else range(1, 20)):
            robot.set_goal(p0 - [0., 0., .001 * i], R0)
            robot.run(.35, 0., robot.ee_pose()[1])
            poses.append(lp.make_transform(*robot.ee_pose()))
            F = robot.ee_pose()[0] @ robot.wrench()[:3]
            print(f"approach {i}: axial force {F[2]:.3f} N", flush=True)
            if F[2] > .2:
                break
        else:
            if not args.pre_engaged:
                raise RuntimeError("The demonstration did not make thread-entry contact")
        # Continue loading the compliant spring at the entrance before turning;
        # a light touch alone can skate over the entrance chamfer.
        R_touch, p_touch = robot.ee_pose()
        robot.set_goal(p_touch - [0., 0., args.preload], R_touch)
        robot.run(1., 0., p_touch)
        poses.append(lp.make_transform(*robot.ee_pose()))
        approach = lp.Demo.from_poses(poses, grip_closed=True, view=reference)
        T_touch = lp.make_transform(*robot.ee_pose())
        screw_poses = [T_touch]
        axis = np.array([0., 0., -1.])
        if args.teacher_mode == "primitive":
            run_robot = robot.run

            def measured_run(*run_args):
                run_robot(*run_args)
                screw_poses.append(lp.make_transform(*robot.ee_pose()))

            robot.run = measured_run
            ekf = lp.ScrewEKF(axis, bolt_position, lp.ScrewCfg(
                pitch=THREAD_PITCH / (2 * np.pi), init_axis=1e-4, init_point=1e-8,
                q_axis=1e-7, q_point=1e-9, r_trans=1e-8, r_rot=1e-5), np.random.default_rng(args.seed))
            teacher = lp.RotationPrimitive(robot, ekf, axis, lp.RotCfg(
                period=.25, delta_theta=.05, min_rot=.001, min_tran=.00001,
                terminate_theta=float(np.deg2rad(args.turn_deg)), early_exit_disp=0.))
            event = teacher.exploitation(max_iters=160)
            robot.run = run_robot
            if event != "Done":
                raise RuntimeError(f"The thread teacher failed: {event}")
        else:
            steps = max(2, round(args.turn_deg / 2.8125))
            for i, theta in enumerate(np.linspace(0., np.deg2rad(args.turn_deg), steps + 1)[1:], 1):
                T = lp.screw_transform(axis, bolt_position, theta, THREAD_PITCH / (2 * np.pi)) @ T_touch
                robot.set_goal(T[:3, 3] - [0., 0., args.preload], T[:3, :3])
                robot.run(args.step_time, 0., robot.ee_pose()[1])
                screw_poses.append(lp.make_transform(*robot.ee_pose()))
                if i % 4 == 0:
                    print(f"thread {i}/{steps}: TCP {robot.ee_pose()[1]}", flush=True)
        screw = lp.Demo.from_poses(screw_poses, grip_closed=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        learned = lp.learn_ec(screw, "screw")
        descent = float(screw.poses[0, 2, 3] - screw.poses[-1, 2, 3])
        expected = learned.diagnostics["turn_rad"] * THREAD_PITCH / (2 * np.pi)
        if (descent < .0002 or not .0001 < learned.parameters["pitch"] < .0005
                or abs(descent - expected) > .0002):
            rejected = args.out_dir / "rejected"
            rejected.mkdir(parents=True, exist_ok=True)
            approach.save(rejected / "nut_approach.npz")
            screw.save(rejected / "nut_thread.npz")
            raise RuntimeError(f"The recorded motion did not demonstrate thread advance: {learned}, descent={descent}")
        if not args.pre_engaged:
            approach.save(args.out_dir / "nut_approach.npz")
        screw.save(args.out_dir / "nut_thread.npz")
        output = args.out_dir / ("nut_turn.yaml" if args.pre_engaged else "nut_thread.yaml")
        segments = {
            "approach": dict(state="free", demo="nut_approach.npz",
                             cfg=dict(approach_step=.001, touch_force=.7, max_approach=.02,
                                      period=.2, waypoint_tol=.0005, waypoint_rot_tol=.003, early_exit_disp=.0005),
                             servo=dict(z_range=[.05, .3], tol=.0005, rot_tol=.003, max_time=30.)),
            "thread": dict(state="screw", demo="nut_thread.npz",
                           estimator=dict(init_axis=1e-4, init_point=1e-8, q_axis=1e-7, q_point=1e-9,
                                          r_trans=1e-8, r_rot=1e-5),
                           cfg=dict(period=.25, delta_theta=.05, min_rot=.001, min_tran=.00001,
                                    terminate_theta=float(np.deg2rad(args.turn_deg)), early_exit_disp=0.)),
        }
        if args.pre_engaged:
            segments.pop("approach")
        write_task(output, task, "thread" if args.pre_engaged else "approach", segments,
            [] if args.pre_engaged else [dict(node="approach", successor="thread", event="MC")],
            writer="scripts/record_nut_ec_demo.py", execution=dict(
                max_iters=250, hand_dx=0., hand_dy=0., hand_dz=0., hand_dyaw_deg=0.))
        report = dict(success=True, starting_condition="pre-engaged" if args.pre_engaged else "clear of bolt",
                      start_clearance=initial_bottom - bolt_top,
                      descent=descent, learned=learned.parameters, diagnostics=learned.diagnostics)
        output.with_suffix(".teacher.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
