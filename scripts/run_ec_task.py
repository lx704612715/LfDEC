# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Learn and execute an EC example, recording the simulation and manipulation monitor.

    uv run python scripts/run_ec_task.py --example drawer_grasp_pull
    uv run python scripts/run_ec_task.py --example peg_insert --show

Each --example invocation compiles supplied annotated demonstrations into its own execution
bundle before running ECPlanner. --task-file executes an already compiled YAML instead.
Default runs open no windows and save a combined Newton view / visual alignment / force / EC
video. --show displays both windows while still recording. --no-record disables recording;
--viz none deliberately disables rendering and recording for numerical qualification.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import sys
from pathlib import Path

import _repo_path  # puts the repo root on sys.path; REPO_ROOT used below
from _run_options import EXAMPLES, configure_display, configure_outputs

configure_display(sys.argv[1:])
from _launcher import add_launcher_args  # Isaac Lab's flags, viewer on by default
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

from lfdec_primitives.vis import ECMonitor, RunVideo, plot_forces
from lfdec_primitives.vis.simulation import activate_primitive, viewer_frames
from lfdec_primitives.tasks import resolve_ec_task  # registers LfDEC-* on import

DEFAULT_TASK_FILE = "tasks/drawer_grasp_pull.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launcher_args(p)
    task = p.add_mutually_exclusive_group()
    task.add_argument("--task-file", default=None, help=f"execute an existing YAML; default: {DEFAULT_TASK_FILE}")
    task.add_argument("--example", choices=EXAMPLES, help="learn and compile the supplied demonstration, then execute it")
    p.add_argument("--show", action="store_true", help="show the simulation and live monitor while also recording")
    p.add_argument("--no-record", action="store_true", help="explicitly disable automatic video recording")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=None, help="iterations allowed per node (YAML execution settings)")
    p.add_argument("--no-servo", action="store_true",
                   help="ignore the demonstrations' reference views and replay from where the "
                        "hand is; separates a servo failure from a replay failure")
    p.add_argument("--servo-max-time", type=float, default=30.0)
    p.add_argument("--no-monitor", action="store_true", help="omit the monitor from the video and live display")
    p.add_argument("--record", default=None,
                   help="override the automatic <run-directory>/run.mp4 path")
    p.add_argument("--record-speed", type=float, default=1.0,
                   help="playback speed of --record relative to sim time: 1 plays the run at its own pace however "
                        "slowly it simulated")
    p.add_argument("--no-markers", action="store_true", help="skip the EC-parameter arrows")
    p.add_argument("--marker-length", type=float, default=0.45, help="m, arrow length")
    p.add_argument("--hand-dx", type=float, default=None,
                   help="m the hand starts off the demonstrated standoff (root frame); the "
                        "free-space node has to servo this back out. The defaults go sideways "
                        "and up, off the approach axis, so they suit the drawer and the door alike")
    p.add_argument("--hand-dy", type=float, default=None)
    p.add_argument("--hand-dz", type=float, default=None)
    p.add_argument("--hand-dyaw-deg", type=float, default=None,
                   help="deg of roll about the hand's own approach axis, likewise")
    p.add_argument("--cabinet-dx", type=float, default=0.0, help="m to move the cabinet along x")
    p.add_argument("--cabinet-dy", type=float, default=0.0, help="m to move the cabinet along y")
    p.add_argument("--verify-transitions", action="store_true",
                   help="after each event, ask the primitive to confirm the successor's constraint")
    p.add_argument("--out-dir", default=None, help="default: outputs/<example>/<unique-run-id>")
    p.add_argument("--tag", default=None)
    return p


def main() -> int:
    args, hydra_args = setup_preset_cli(build_parser())
    configure_outputs(args, Path(_repo_path.REPO_ROOT))
    if not any(a.startswith(("physics=", "env.sim.physics=")) for a in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    sys.argv = [sys.argv[0], *hydra_args]

    import lfdec_primitives as lp

    if args.example:
        args.task_file = str(lp.compile_demonstration(
            Path(_repo_path.REPO_ROOT) / "tasks" / f"{args.example}.demo.yaml",
            Path(args.out_dir) / "prepared" / f"{args.example}.yaml"))
        print(f"Prepared {args.task_file}")
    args.task_file = args.task_file or DEFAULT_TASK_FILE
    task_file = Path(args.task_file)
    if not task_file.is_absolute():
        task_file = Path(_repo_path.REPO_ROOT) / task_file
    spec = lp.load_task_spec(task_file)
    from lfdec_primitives.primitives.graph import validate_configuration

    validate_configuration(spec)
    for node in spec.nodes.values():
        if node.demo is not None:
            lp.Demo.load(node.demo)
    defaults = dict(max_iters=200, hand_dx=0., hand_dy=.04, hand_dz=.04, hand_dyaw_deg=8.)
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, spec.execution.get(key, fallback))
    if args.num_envs != 1:
        raise ValueError("EC sequence execution supports one environment; batched planners are not implemented")
    print(f"\n   {task_file.name}: {len(spec.nodes)} nodes, {len(spec.edges)} edges, "
          f"start {spec.start!r}, scene {spec.task}")
    for name, node in spec.nodes.items():
        detail = f"demo {node.demo.name}" if node.demo else f"ec {node.ec}"
        print(f"       {name:<10} {node.type:<12} {detail}")
    for e in spec.edges:
        print(f"       {e.src} --{e.event}--> {e.dst}")

    env_cfg = resolve_ec_task(spec.task, args.num_envs)
    env_cfg.seed = args.seed
    if args.cabinet_dx or args.cabinet_dy:
        pos = list(env_cfg.scene.cabinet.init_state.pos)
        pos[0] += args.cabinet_dx
        pos[1] += args.cabinet_dy
        env_cfg.scene.cabinet.init_state.pos = tuple(pos)
        print(f"   cabinet moved by ({args.cabinet_dx:+.3f}, {args.cabinet_dy:+.3f}) m since the "
              f"demonstration")

    with launch_simulation(env_cfg, args):
        return _run(args, env_cfg, spec)


def _run(args, env_cfg, spec) -> int:
    import gymnasium as gym

    env = gym.make(spec.task, cfg=env_cfg).unwrapped
    try:
        with ExitStack() as resources:
            return _execute(args, env_cfg, spec, env, resources)
    finally:
        env.close()


def _execute(args, env_cfg, spec, env, resources) -> int:
    import numpy as np

    import lfdec_primitives as lp
    from lfdec_primitives.tasks.sensors.camera import mount_transform
    from lfdec_primitives.tasks.sensors.camera import IsaacCamera
    from lfdec_primitives.tasks.execution import make_robot, evaluate_outcome

    term = env.action_manager.get_term("arm_action")
    dt_step = env_cfg.sim.dt * env_cfg.decimation
    env.reset()

    f_max = term.controller.gains["f_max"][0].cpu().numpy()
    f_max_lin, f_max_rot = float(np.min(f_max[:3])), float(np.min(f_max[3:]))
    markers = None
    if not args.no_markers:
        from lfdec_primitives.vis.ec_markers import ECAxisMarkers

        # A task file carries the demonstration's prior and nothing else, so there is no truth
        # arrow to draw: what the estimate is judged against here is where it started.
        markers = ECAxisMarkers(length=args.marker_length, keys=("initial", "estimate"))
    monitor = None if args.no_monitor else ECMonitor(
        f_max_lin, f_max_rot, title=f"EC execution: {Path(args.task_file).stem}")
    if monitor is not None:
        resources.callback(monitor.close)
    # The joints the task file watches (``terminate_extra``) are what the adapter reports, so
    # nothing here has to know which joint the scene's constraint is.
    watched = sorted({node.cfg["terminate_extra"][0] for node in spec.nodes.values() if node.cfg.get("terminate_extra")})
    video = None if args.record is None else RunVideo(args.record, viewer_frames(env), monitor, speed=args.record_speed)
    if video is not None:
        resources.enter_context(video)
    robot = make_robot(spec.task, env, term, dt_step, markers=markers, monitor=monitor,
                       watch=tuple(key for key in watched if "/" in key), video=video)
    from lfdec_primitives.tasks.robots.factory_grasp import FactoryGraspRobot

    if isinstance(robot, FactoryGraspRobot):
        robot.position_peg_in_gripper()
    # Apply the demonstration's initial gripper command before stepping: cabinet/block
    # approaches start open, while the fixed-grasp peg and nut examples start closed.
    first_demo = spec.nodes[spec.start].demo
    initial_grip = False if first_demo is None else bool(lp.Demo.load(first_demo).grip_closed[0])
    robot.set_gripper(closed=initial_grip)
    robot.run(1.0, 0.0, robot.ee_pose()[1])
    # Home the wrench bias with the fingers where they will be for the approach: homing after a
    # grasp bakes the grip preload into the bias.
    term.controller.start_homing(num_ticks=int(round(0.5 / env_cfg.sim.dt)))
    robot.run(0.8, 0.0, robot.ee_pose()[1])
    bias = term.controller.wrench_bias[0].cpu().numpy()
    print(f"\n   wrench bias = F {bias[:3].round(2)} N  T {bias[3:].round(3)} N.m")

    # Start off the demonstrated standoff. The arm spawns at the pose the demonstration began
    # at, so without this the servo converges on its first observation and the free-space node
    # has nothing to show. The roll is about the hand's own approach axis -- the offset composes
    # on the right -- which turns the camera in its image plane.
    R_demo, p_demo = robot.ee_pose()
    offset = np.array([args.hand_dx, args.hand_dy, args.hand_dz])
    if offset.any() or args.hand_dyaw_deg:
        robot.set_goal(p_demo + offset, R_demo @ lp.rz(math.radians(args.hand_dyaw_deg)))
        robot.run(2.5, 0.0, p_demo)
        R_start, p_start = robot.ee_pose()
        rot = lp.rotvec_from_matrix(R_start @ R_demo.T)
        print(f"   hand started {1e3 * float(np.linalg.norm(p_start - p_demo)):.0f} mm / "
              f"{math.degrees(float(np.linalg.norm(rot))):.1f} deg off the demonstrated standoff; "
              f"the free-space node has to servo that back out")

    needs_camera = not args.no_servo and any(
        node.demo is not None and lp.Demo.load(node.demo).view is not None for node in spec.nodes.values())
    cam = IsaacCamera(robot, "wrist_cam", mount_transform(robot.finger_axis_ee())) if needs_camera else None
    servos: list = []

    def servo_factory(view, **overrides):
        settings = {"max_time": args.servo_max_time} | overrides
        if args.servo_max_time != 30.:
            settings["max_time"] = args.servo_max_time
        s = lp.VisualServo(cam, view, lp.ServoCfg(**settings))
        servos.append(s)
        return s

    on_node = robot.set_active_primitive
    if isinstance(robot, FactoryGraspRobot) or spec.task.startswith("LfDEC-Factory-Peg-Insert-"):
        from functools import partial

        on_node = partial(activate_primitive, robot, env)
    initial_pose = lp.make_transform(*robot.ee_pose())
    ctx = lp.BuildContext(servo_factory=None if cam is None else servo_factory,
                          rng=np.random.default_rng(args.seed))
    plan = lp.build_planner(spec, robot, ctx, verify_transitions=args.verify_transitions,
                            on_node=on_node)

    print(f"\n== executing from {spec.start!r} ==")
    before = robot.extra()
    status = plan.execution(spec.start, max_iters=args.max_iters, verbose=True)
    robot.set_goal(robot.ee_pose()[1], robot.ee_pose()[0])
    robot.run(1., 0., robot.ee_pose()[1])
    from lfdec_primitives.tasks.robots.task_robots import FactoryNutRobot

    if status == "Finished" and isinstance(robot, FactoryNutRobot):
        robot.probe_engagement()
    after = robot.extra()
    outcome = evaluate_outcome(robot, spec, plan, before)

    print(f"\n   status = {status}")
    print(f"   physical task = {'SUCCESS' if outcome['success'] else 'FAILED'}; "
          f"metrics = {outcome.get('metrics', {})}")
    for step in plan.trace:
        print(f"       {step['node']:<10} event {step['event']:<28} "
              f"contact {step['contact_event']:<5} -> {step['next']}")
    for key in watched:
        print(f"   {key}: {before[key]:+.3f} -> {after[key]:+.3f}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(_repo_path.REPO_ROOT) / "scripts" / "results_isaac"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"ec_task_{args.tag or Path(args.task_file).stem}"
    summary = dict(task_file=str(args.task_file), scene=spec.task, device=str(env.device), status=status,
                   **outcome, learning=spec.learning,
                   trace=plan.trace, progress={k: [before[k], after[k]] for k in watched},
                   cabinet_dx=args.cabinet_dx, cabinet_dy=args.cabinet_dy,
                   hand_dx=args.hand_dx, hand_dy=args.hand_dy, hand_dz=args.hand_dz,
                   hand_dyaw_deg=args.hand_dyaw_deg,
                   servo=not args.no_servo, gripper_gap=robot.gripper_gap(),
                   servo_events=[getattr(s, "event", None) for s in servos],
                   primitives={name: primitive.summary() for name, primitive in plan.primitives.items()},
                   initial_pose=initial_pose.tolist(),
                   reference_pose=None if first_demo is None else lp.Demo.load(first_demo).T_start.tolist(),
                   aligned_pose=plan.primitives[spec.start].init_base_ht_ee.tolist(),
                   initial_distance_to_hole=(float(np.hypot(before["insertion_depth"], before["hole_error"]))
                                             if "hole_error" in before else None))
    summary["video"] = None if video is None else dict(path=str(video.path), frames=video.frames,
                                                      fps=video.fps, speed=args.record_speed)
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    lp.write_rows(stem.with_name(stem.name + "_ticks.csv"), robot.ticks)
    for name, prim in plan.primitives.items():   # each node's own iteration log, as the scripted runs write
        if prim.rows:
            prim.write_csv(stem.with_name(f"{stem.name}_{name}.csv"))
        if prim.servo is not None and prim.servo.rows:
            lp.write_rows(stem.with_name(f"{stem.name}_{name}_servo.csv"), prim.servo.rows)
    written = ".json/_ticks.csv/_<node>.csv/_<node>_servo.csv"
    if video is not None:
        video.close()
        print(f"   wrote {args.record}: {video.frames} frames at {args.record_speed:g}x")
    if monitor is not None:
        lp.write_rows(stem.with_name(stem.name + "_ec.csv"), monitor.ec_rows)
        monitor.close(stem.with_name(stem.name + "_monitor.png"))
        written += "/_ec.csv/_monitor.png"
    plot_forces(robot.ticks, stem.with_name(stem.name + "_forces.png"),
                   f"{spec.task} — {status}", f_max_lin, f_max_rot,
                   **({"progress_key": watched[0], "progress_label": watched[0]} if watched else {}))
    print(f"   wrote {stem.name}{written}/_forces.png in {out_dir}")

    return 0 if outcome["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
