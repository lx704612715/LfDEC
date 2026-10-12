# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Record scripted peg approach, plane slide and insertion segments and compile their task YAML."""

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

TASK = "LfDEC-Peg-Insert-Franka"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launcher_args(p)
    p.add_argument("--task", default=TASK)
    p.add_argument("--approach-step", type=float, default=0.02, help="m between recorded approach waypoints")
    p.add_argument("--clearance", type=float, default=0.02,
                   help="m above the nominal touch the recorded approach ends; the seek does the rest")
    p.add_argument("--touch-force", type=float, default=3.0, help="N along the normal that counts as touching")
    p.add_argument("--across", type=float, default=0.02, help="m of the slide's first leg, across the way to the opening")
    p.add_argument("--slide-step", type=float, default=0.015, help="m between recorded samples of the slide")
    p.add_argument("--press", type=float, default=0.015, help="m the equilibrium pose is held below the surface while sliding")
    p.add_argument("--out-dir", default=None, help="default: <repo>/tasks")
    p.add_argument("--name", default="peg_insert", help="stem of the YAML that is written")
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
    from lfdec_primitives.tasks.demo_recording import home, write_task
    from lfdec_primitives.tasks.sensors.camera import mount_transform
    from lfdec_primitives.tasks.sensors.camera import IsaacCamera
    from lfdec_primitives.tasks.robots.isaac_robot import IsaacECRobot
    from lfdec_primitives.tasks.envs.peg_insert_ec_cfg import (
        PEG_PROTRUDE,
        STANDOFF,
        grasp_frame,
        hole_axis_point,
    )

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    term = env.action_manager.get_term("arm_action")
    dt_step = env_cfg.sim.dt * env_cfg.decimation
    env.reset()
    robot = IsaacECRobot(env, term, dt_step)
    cam = IsaacCamera(robot, "wrist_cam", mount_transform(robot.finger_axis_ee()))
    robot.set_gripper(closed=True)          # the peg is in hand from the start
    bias = home(robot, term, env_cfg.sim.dt)
    robot.set_gripper(closed=True)          # home() opens the hand; here it stays on the peg
    robot.run(0.5, 0.0, robot.ee_pose()[1])
    print(f"\n   wrench bias = F {bias[:3].round(2)} N  T {bias[3:].round(3)} N.m")

    R_hand, p_touch_nominal = grasp_frame(env)
    n_true = -R_hand[:, 2]                  # the plate's outward normal, from the approach axis
    hole, axis_true = hole_axis_point()
    _, p0 = robot.ee_pose()
    print(f"   nominal touch {p_touch_nominal.round(4)}, TCP {p0.round(4)}, "
          f"{1e3 * float(np.linalg.norm(p0 - p_touch_nominal)):.0f} mm above it (STANDOFF {1e3 * STANDOFF:.0f} mm)")

    def normal_force() -> float:
        R, _ = robot.ee_pose()
        return float(np.dot(R @ robot.wrench()[:3], n_true))

    def drive(p, seconds=0.35):
        robot.set_goal(p, R_hand)
        robot.run(seconds, 0.0, robot.ee_pose()[1])
        return robot.ee_pose()[1]

    # ---------------------------------------------------------------- segment 1: the approach
    # Recorded to a point in the air; the free-space node's own seek (and the recorder, below)
    # closes the last centimetres on force, which is the paper's contact-changing event.
    view = cam.capture()
    T0 = lp.make_transform(*robot.ee_pose())
    T_end = lp.make_transform(R_hand, p_touch_nominal + args.clearance * n_true)
    approach_poses = [T0] + lp.line_trajectory(T0, T_end, max_step=args.approach_step)
    measured_approach = [T0]
    for T in approach_poses[1:]:
        drive(T[:3, 3])
        measured_approach.append(lp.make_transform(*robot.ee_pose()))
    drive(T_end[:3, 3], 0.6)
    measured_approach.append(lp.make_transform(*robot.ee_pose()))
    approach = lp.Demo.from_poses(measured_approach, grip_closed=True, view=view)
    # Seek from the *equilibrium* pose, not the measured one: a press is sized against F_m / K_p,
    # and a goal re-anchored 5 mm under wherever the hand is can never load the spring past
    # 0.75 N at K_p = 150 N/m (PRIMITIVES.md section 6). The lag grows until the force says contact.
    p_start, p_eq = robot.ee_pose()[1], robot.desired_pose()[1]
    k = 0
    while normal_force() < args.touch_force:
        k += 1
        drive(p_eq - 0.005 * k * n_true, 0.3)
        if float(np.dot(p_start - robot.ee_pose()[1], n_true)) > 0.06 or k * 0.005 > 0.10:
            raise RuntimeError(f"no contact: the hand went {1e3 * float(np.dot(p_start - robot.ee_pose()[1], n_true)):.0f} mm "
                               f"below the recorded approach with the goal {1e3 * 0.005 * k:.0f} mm under it, "
                               f"F.n {normal_force():.1f} N")
    p_touch = robot.ee_pose()[1].copy()
    print(f"   approach: {len(approach_poses)} waypoints, reference view {view.rgb.shape[1]}x{view.rgb.shape[0]}; "
          f"touched at {p_touch.round(4)} with F.n {normal_force():.1f} N, "
          f"{1e3 * float(np.dot(p_touch - p_touch_nominal, n_true)):+.1f} mm from the nominal touch")

    # ---------------------------------------------------------------- segment 2: the slide
    # Across first, then straight at the opening: the samples taken before the drop fix the
    # plane, and the peg drops in where a person's would. The equilibrium pose is held ``press``
    # inside the plate so the tip stays down; what is recorded is where the tip went, which
    # trails the command by the stiction dead band (mu * press, 9 mm here).
    def slide_to(target, max_wait=2.0):
        """Hold the equilibrium ``press`` under ``target`` until the tip has stopped creeping. A
        part under stiction closes on its command at millimetres per second once the spring is
        near the friction cone, and a sample taken after a fixed half second records a lag that
        is determined by the clock rather than surface motion."""
        robot.set_goal(target - args.press * n_true, R_hand)
        for _ in range(int(max_wait / 0.2)):
            p_before = robot.ee_pose()[1]
            robot.run(0.2, 0.0, p_before)
            if float(np.linalg.norm(robot.ee_pose()[1] - p_before)) < 0.0005:
                break
        return robot.ee_pose()[1]

    to_hole = lp.project_to_plane(hole - p_touch, n_true)
    e_hole = to_hole / float(np.linalg.norm(to_hole))
    corner = p_touch + args.across * np.cross(n_true, e_hole)
    over_hole = p_touch + to_hole
    n_toward = max(2, math.ceil(float(np.linalg.norm(over_hole - corner)) / args.slide_step))
    legs = ([p_touch + s * (corner - p_touch) for s in np.linspace(0.0, 1.0, 3)[1:]]
            + [corner + s * (over_hole - corner) for s in np.linspace(0.0, 1.0, n_toward + 1)[1:]])
    slide_poses = [lp.make_transform(R_hand, p_touch)]
    dropped_at, trail = None, []             # first sample taken inside the opening; lag per sample
    for target in legs:
        p_now = slide_to(target)
        slide_poses.append(lp.make_transform(*robot.ee_pose()))
        trail.append(1e3 * float(np.linalg.norm(lp.project_to_plane(target - p_now, n_true))))
        if dropped_at is None and float(np.dot(p_touch - p_now, n_true)) >= 0.5 * args.press:
            dropped_at = len(slide_poses) - 1
    print(f"   slide: the tip settled {np.round(trail, 1)} mm behind its commands")
    if dropped_at is None:
        raise RuntimeError(f"the peg did not drop into the opening: the slide ended {p_now.round(4)}, "
                           f"{1e3 * float(np.linalg.norm(lp.project_to_plane(hole - p_now, n_true))):.0f} mm from its centre")
    if dropped_at < 4:
        raise RuntimeError(f"only {dropped_at} samples on the surface before the drop, too few to fit the plane")
    slide = lp.Demo.from_poses(slide_poses, grip_closed=True)
    P = np.stack([T[:3, 3] for T in slide_poses])
    n_fit, _ = lp.fit_plane(P[:dropped_at])
    if float(np.dot(n_fit, n_true)) < 0.0:   # out of the surface, toward the hand
        n_fit = -n_fit
    def off_centre(p):
        return 1e3 * float(np.linalg.norm(lp.project_to_plane(hole - p, n_true)))

    print(f"   slide: {len(slide_poses)} samples, {dropped_at} on the surface over {1e2 * args.across:.1f} cm across and "
          f"{1e2 * float(np.linalg.norm(over_hole - corner)):.1f} cm toward the opening; dropped in "
          f"{off_centre(P[dropped_at]):.0f} mm from its centre, seated {off_centre(P[-1]):.0f} mm from it, "
          f"{1e3 * float(np.dot(p_touch - P[-1], n_true)):.0f} mm in")
    print(f"   EC parameter (plane normal) fitted from the demonstration: {n_fit.round(4)}")
    print(f"       scene truth {n_true.round(4)}, fit error {math.degrees(lp.angle_between(n_fit, n_true)):.2f} deg")

    # ---------------------------------------------------------------- segment 3: the insertion
    # The view from here is the plane node's servo reference: the surface with the peg in the
    # opening, the start of the segment that follows the slide. Then push in until the
    # fingertips land on the plate.
    view_hole = cam.capture()
    feats = lp.FeatureMatcher(lp.ServoCfg(z_range=lp.PlaneCfg().servo_z_range)).extract(view_hole)
    print(f"   surface view for the plane node's servo: {feats.xyz.shape[0]} features inside the plane servo's depth "
          f"range" + (f", median depth {float(np.median(feats.xyz[:, 2])):.3f} m" if feats.xyz.shape[0] else ""))
    p_seat = robot.ee_pose()[1].copy()
    descent_poses = [lp.make_transform(*robot.ee_pose())]
    p_prev = p_seat
    for k in range(1, 13):
        p_now = drive(p_seat - 0.01 * k * n_true, 0.4)
        descent_poses.append(lp.make_transform(*robot.ee_pose()))
        if float(np.linalg.norm(p_now - p_prev)) < 0.002 and k > 2:
            break                            # stopped: the fingertips are on the plate
        p_prev = p_now
    descent = lp.Demo.from_poses(descent_poses, grip_closed=True, view=view_hole)
    D = np.stack([T[:3, 3] for T in descent_poses])
    axis_fit = lp.fit_line_direction(D)
    inserted = float(np.dot(p_touch - D[-1], n_true))
    if inserted < 0.5 * PEG_PROTRUDE:
        raise RuntimeError(f"the peg went only {1e3 * inserted:.0f} mm below the touch height: it did not enter the opening")
    print(f"   insert: descended to {1e3 * inserted:.0f} mm below the touch height over {len(descent_poses)} samples "
          f"(peg protrudes {1e3 * PEG_PROTRUDE:.0f} mm); F.n at the end {normal_force():.1f} N")
    print(f"   EC parameter (opening axis) fitted from the demonstration: {axis_fit.round(4)}")
    print(f"       scene truth {axis_true.round(4)}, fit error {math.degrees(lp.angle_between(axis_fit, axis_true)):.2f} deg")

    # ---------------------------------------------------------------- write it out
    out_dir = Path(args.out_dir) if args.out_dir else Path(_repo_path.REPO_ROOT) / "tasks"
    out_dir.mkdir(parents=True, exist_ok=True)
    approach.save(out_dir / "insert_approach.npz")
    slide.save(out_dir / "insert_slide.npz")
    descent.save(out_dir / "insert_descent.npz")
    yaml_path = out_dir / f"{args.name}.yaml"
    write_task(yaml_path, args.task, "approach", {
        "approach": {
            "state": "free",
            "demo": "insert_approach.npz",
            # Seek along the plane's normal (its constraint direction) for up to this far.
            "cfg": {"max_approach": round(args.clearance + 0.04, 3)},
        },
        "search": {
            "state": "plane",
            "demo": "insert_slide.npz",
            "fit_samples": [0, dropped_at],
            # waypoint_tol above the dead band mu * press (9 mm), bce_depth at press / 2.5, and
            # the servo's bound at two dead bands, since it moves the peg in jumps of about a
            # step once the spring crosses the band: the couplings of PRIMITIVES.md section 5.
            # The spiral is the paper's 2 cm: the replay lands within a dead band of the
            # opening and the servo takes up the rest.
            "cfg": {"press_depth": args.press, "spiral_radius": 0.02, "waypoint_tol": 0.012,
                    "bce_depth": 0.006, "mce_force": 10.0, "servo_max_disp": 0.03, "servo_time": 6.0},
            "servo": {"servo_mode": "pbvs"},
        },
        "insert": {
            "state": "prismatic",
            "demo": "insert_descent.npz",    # its start view is what the plane node servoes toward
            "cfg": {"delta_p": 0.03},
        },
    }, [{"node": "approach", "successor": "search", "event": "MC"},
        {"node": "search", "successor": "insert", "event": "BC"}], writer="scripts/record_insertion_ec_demo.py")
    print(f"\n   wrote {yaml_path.name}, insert_approach.npz, insert_slide.npz, insert_descent.npz in {out_dir}")
    print(f"   run it:  uv run python scripts/run_ec_task.py --task-file tasks/{yaml_path.name} physics=newton_mjwarp")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
