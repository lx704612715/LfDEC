# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Isaac adapters and physical outcome checks for the demonstration examples.

The planner and learned geometry stay simulator-independent. These adapters expose measured
task progress and evaluate the outcome; scene truth is never used to steer an EC direction.
"""
from __future__ import annotations

import numpy as np

import lfdec_primitives as lp
from .robots.isaac_robot import IsaacECRobot
from .robots.task_robots import BlockRobot, FactoryNutRobot, FactoryPegRobot, PlatePegRobot
from .robots.factory_grasp import FactoryGraspRobot


def make_robot(task, env, term, dt_step, **kwargs):
    """Select the task's measured-progress adapter; joint-watch keys are shared by all adapters."""
    types = {
        "LfDEC-Factory-Peg-Insert-Franka": FactoryPegRobot,
        "LfDEC-Factory-Peg-Insert-Visual-Franka": FactoryPegRobot,
        "LfDEC-Factory-Peg-Grasp-Franka": FactoryGraspRobot,
        "LfDEC-Factory-Peg-Grasp-Visual-Franka": FactoryGraspRobot,
        "LfDEC-Peg-Insert-Franka": PlatePegRobot,
        "LfDEC-Grasp-Free-Franka": BlockRobot,
        "LfDEC-Factory-Nut-Thread-EC-Franka": FactoryNutRobot,
        "LfDEC-Factory-Nut-Thread-Franka": FactoryNutRobot,
    }
    robot = types.get(task, IsaacECRobot)(env, term, dt_step, **kwargs)
    if isinstance(robot, FactoryNutRobot):
        from .envs.factory_nut_cfg import BOLT_POSITION, ENTRY_BOLT_POSITION

        robot.bolt_position = ENTRY_BOLT_POSITION if task.endswith("-EC-Franka") else BOLT_POSITION
    return robot


def evaluate_outcome(robot, spec, plan, before):
    """Return physical success in addition to planner completion, with reviewable measurements."""
    metrics = robot.extra()
    if plan.status != "Finished":
        return dict(success=False, failure=plan.status, metrics=metrics)
    success, reason = True, None
    if isinstance(robot, (FactoryPegRobot, FactoryGraspRobot)):
        from .envs.factory_peg_ec_cfg import RADIAL_CLEARANCE, insertion_succeeded

        success = insertion_succeeded(metrics)
        metrics["radial_clearance"] = RADIAL_CLEARANCE
        reason = "peg did not seat inside the original socket clearance"
    elif isinstance(robot, PlatePegRobot):
        from .envs.plate_plane_cfg import PLATE_CONTACT_TOLERANCE

        # The large-opening plate example must enter past the rim, rather than stop on it.
        metrics["plate_contact_tolerance"] = PLATE_CONTACT_TOLERANCE
        success = metrics["insertion_depth"] > .025 and metrics["plate_clearance"] >= -PLATE_CONTACT_TOLERANCE
        reason = "peg stopped on the rim or missed the plate opening"
    elif isinstance(robot, BlockRobot):
        metrics["block_lift"] = metrics["block_z"] - before["block_z"]
        success = metrics["block_lift"] > .04 and robot.gripper_gap() > .005
        reason = "block did not remain grasped during the lift"
    elif isinstance(robot, FactoryNutRobot):
        from .envs.factory_nut_cfg import THREAD_PITCH

        if robot.thread_start is None:
            return dict(success=False, failure="no screw primitive was executed", metrics=metrics)
        R0, start = robot.thread_start
        R_end, metrics = robot.thread_end if robot.thread_end is not None else (robot.ee_pose()[0], metrics)
        metrics = dict(metrics)
        turn = -float(lp.rotvec_from_matrix(R_end @ R0.T)[2])
        descent = start["nut_z"] - metrics["nut_z"]
        metrics.update(turn=turn, descent=descent, expected_descent=turn * THREAD_PITCH / (2 * np.pi),
                       start_clearance=-before["nut_overlap"])
        starts_correctly = (metrics["start_clearance"] > .003 if spec.task.endswith("-EC-Franka")
                            else before["nut_overlap"] > .001)
        probe = robot.thread_probe
        metrics["axial_probe"] = probe
        success = (starts_correctly and 1.4 < turn < 1.8
                   and descent > .0002 and abs(descent - metrics["expected_descent"]) < .0002
                   and metrics["nut_axis_error"] < .0005 and metrics["nut_overlap"] > .0002
                   and probe is not None and abs(probe["displacement"]) < .0015 and probe["resistance"] > .2)
        reason = "nut did not retain thread engagement and advance along the bolt thread"
    else:
        for node in spec.nodes.values():
            stop = node.cfg.get("terminate_extra")
            if stop is None:
                continue
            key, target = stop
            start, end = before[key], metrics[key]
            direction = 1. if target >= start else -1.
            success &= bool(direction * (end - target) >= 0)
        if any("cabinet/" in key for key in before):
            success &= bool(.002 < robot.gripper_gap() < .03)
        reason = "physical joint target or retained grasp check failed"
    return dict(success=bool(success), failure=None if success else reason, metrics=metrics)
