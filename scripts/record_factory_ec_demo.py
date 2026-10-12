# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Record scripted Factory peg approach, plane slide and insertion segments and compile their task YAML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import _repo_path  # noqa: F401
from _launcher import add_launcher_args
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

import lfdec_primitives as lp
from lfdec_primitives.tasks import resolve_ec_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--task-file", type=Path, default=Path("outputs/recordings/factory_peg_insert.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    args, hydra = setup_preset_cli(parser)
    if not any(a.startswith(("physics=", "env.sim.physics=")) for a in hydra):
        hydra.append("physics=newton_mjwarp")
    sys.argv = [sys.argv[0], *hydra]
    from lfdec_primitives.tasks.envs.factory_peg_ec_cfg import VISUAL_TASK

    cfg = resolve_ec_task(VISUAL_TASK)
    cfg.seed = args.seed
    with launch_simulation(cfg, args):
        return record(args, cfg)


def record(args, cfg):
    import gymnasium as gym
    from lfdec_primitives.tasks.envs.factory_peg_ec_cfg import GRASP_R, VISUAL_TASK
    from lfdec_primitives.tasks.sensors.camera import mount_transform
    from lfdec_primitives.tasks.robots.task_robots import FactoryPegRobot
    from lfdec_primitives.tasks.sensors.camera import IsaacCamera

    class RecordingRobot(FactoryPegRobot):
        samples = None

        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            if self.samples is not None:
                self.samples.append(lp.make_transform(*self.ee_pose()))
            return result

    env = gym.make(VISUAL_TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        term = env.action_manager.get_term("arm_action")
        robot = RecordingRobot(env, term, cfg.sim.dt * cfg.decimation)
        camera = IsaacCamera(robot, "wrist_cam", mount_transform(robot.finger_axis_ee()))
        robot.set_gripper(closed=True)
        robot.run(1., 0., robot.ee_pose()[1])
        robot.set_goal(robot.ee_pose()[1], GRASP_R)
        robot.run(2., 0., robot.ee_pose()[1])
        term.controller.start_homing(num_ticks=round(.5 / cfg.sim.dt))
        robot.run(.5, 0., robot.ee_pose()[1])
        return teach(args, robot, camera)
    finally:
        env.close()


def teach(args, rb, cam):
    import numpy as np
    from lfdec_primitives.tasks.demo_recording import write_task
    from lfdec_primitives.tasks.envs.factory_peg_ec_cfg import (
        GRASP_R, SOCKET_POSITION, SOCKET_HEIGHT, TIP_OFFSET, STANDOFF, VISUAL_TASK, insertion_succeeded,
    )

    def pose():
        return lp.make_transform(*rb.ee_pose())

    def drive(p, seconds=0.5):
        rb.set_goal(p, GRASP_R)
        rb.run(seconds, 0.0, rb.ee_pose()[1])

    up = np.array([0., 0., 1.])
    view = cam.capture()
    features = lp.FeatureMatcher(lp.ServoCfg(z_range=(0.25, 0.50))).extract(view)
    print("reference features", len(features.xyz), flush=True)
    if len(features.xyz) < 10:
        raise RuntimeError("Too few fixture features to teach visual alignment")
    start = pose()
    metrics = rb.extra()
    if abs(np.hypot(metrics["insertion_depth"], metrics["hole_error"]) - STANDOFF) > 0.01:
        raise RuntimeError("The teacher must start 30 cm from the opening")
    rb.samples = [start]
    end = SOCKET_POSITION + np.array([-0.001, -0.007, SOCKET_HEIGHT + TIP_OFFSET[2] + 0.003])
    count = int(np.ceil(np.linalg.norm(start[:3, 3] - end) / 0.01))
    for index, s in enumerate(np.linspace(0.0, 1.0, count + 1)[1:], 1):
        target = start[:3, 3] + s * (end - start[:3, 3])
        drive(target, float(np.linalg.norm(target - rb.ee_pose()[1])) / rb.v_max_lin + 0.3)
        if index % 5 == 0 or index == count:
            print(f"teacher approach {index}/{count}: TCP {rb.ee_pose()[1].round(4)}", flush=True)
    approach = lp.Demo.from_poses(rb.samples, grip_closed=True, view=view)
    rb.samples = None
    free_cfg = dict(approach_step=0.001, touch_force=0.3, max_approach=0.04,
                    period=0.2, early_exit_disp=0.003, waypoint_tol=0.0005)
    seek = lp.FreeSpacePrimitive(rb, lp.Demo.from_poses([pose()], grip_closed=True),
                                 lp.FreeCfg(approach_dir=-up, **free_cfg))
    rb.set_active_primitive("teacher: contact", seek)
    if seek.exploitation() != "MakingContact":
        raise RuntimeError(f"Teacher missed the plane: {seek.event}")
    contact = pose()
    # Measured dogleg under constant preload, before going toward the opening. This excites
    # two independent surface directions, so the normal is identifiable from the recording.
    target = contact[:3, 3].copy()
    target[2] -= 0.002
    drive(target, 0.8)
    rb.samples = [pose()]
    for dx, dy in [(0, 0.0015), (0, 0.003), (0.0015, 0.003), (0.003, 0.003), (0.003, 0.0015), (0.003, 0)]:
        drive(target + [dx, dy, 0], 0.8)
    surface = list(rb.samples)
    normal, _ = lp.fit_plane(np.array([T[:3, 3] for T in surface]))
    if normal @ up < 0:
        normal = -normal
    print("measured plane normal", normal, flush=True)
    plane_cfg = dict(press_depth=0.002, contact_force=0.2, bce_depth=0.001,
                     period=0.2, stall_period=0.2, settle=0.1, waypoint_tol=0.001,
                     waypoint_attempts=2, spiral_radius=0.003, spiral_pitch=0.00075,
                     spiral_step=0.0005, min_tran=0.00002, mce_attempts=100,
                     mce_force=3.0, early_exit_disp=0.0002,
                     servo_z_range=(0.09, 0.23), servo_max_disp=0.003, servo_time=4.0)
    current = pose()
    end = current[:3, 3].copy()
    end[:2] = SOCKET_POSITION[:2] - [0.001, 0]
    scripted = lp.Demo.from_poses([current] + lp.line_trajectory(
        current, lp.make_transform(GRASP_R, end), max_step=0.0005), grip_closed=True)
    search = lp.PlanePrimitive(rb, up, scripted, lp.PlaneCfg(**plane_cfg))
    rb.set_active_primitive("teacher: surface", search)
    if search.exploitation(max_iters=250) != "BreakingContact":
        raise RuntimeError(f"Teacher did not enter the bore: {search.event}")
    slide = lp.Demo.from_poses([contact] + rb.samples, grip_closed=True)
    rb.samples = [pose()]
    hole_view = cam.capture()
    trans_cfg = dict(delta_p=0.002, period=0.2, min_tran=0.00002, early_exit_disp=0.0005,
                     terminate_extra=("insertion_depth", 0.024))
    pf_cfg = dict(init_std=0.001, pred_std=0.0001)
    insert = lp.TranslationPrimitive(rb, lp.PrismaticPF(-up, lp.PFCfg(**pf_cfg),
                                     np.random.default_rng(args.seed)), -up, lp.TransCfg(**trans_cfg))
    rb.set_active_primitive("teacher: insert", insert)
    insert.exploitation(max_iters=250)
    if not insertion_succeeded(rb.extra()):
        raise RuntimeError(f"Teacher did not seat the peg: {rb.extra()}")
    descent = lp.Demo.from_poses(rb.samples, grip_closed=True, view=hole_view)
    axis = lp.fit_line_direction(np.array([T[:3, 3] for T in rb.samples]))
    rb.samples = None
    directory = args.task_file.parent
    directory.mkdir(parents=True, exist_ok=True)
    names = {key: f"{args.task_file.stem}_{key}.npz" for key in ("approach", "search", "insert")}
    for key, demo in zip(names, (approach, slide, descent)):
        demo.save(directory / names[key])
    write_task(args.task_file, VISUAL_TASK, "approach", {
        "approach": dict(state="free", demo=names["approach"], cfg=free_cfg,
                         servo=dict(z_range=[0.25, 0.50], tol=0.0005, rot_tol=0.003,
                                    max_time=30.0, error_window=5)),
        "search": dict(state="plane", demo=names["search"], fit_samples=[1, 8], cfg=plane_cfg),
        "insert": dict(state="prismatic", demo=names["insert"], estimator=pf_cfg, cfg=trans_cfg),
    }, [dict(node="approach", successor="search", event="MC"),
        dict(node="search", successor="insert", event="BC")], writer="scripts/record_factory_ec_demo.py",
        execution=dict(max_iters=250, hand_dy=0.006, hand_dz=0.006, hand_dyaw_deg=-3.0))
    report = dict(success=True, metrics=rb.extra(), normal=normal.tolist(), axis=axis.tolist(),
                  initial_distance_to_hole=float(np.hypot(metrics["insertion_depth"], metrics["hole_error"])),
                  reference_features=len(features.xyz),
                  events=[seek.event, search.event, insert.event],
                  samples=dict(approach=len(approach.poses), search=len(slide.poses), insert=len(descent.poses)))
    args.task_file.with_suffix(".teacher.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
