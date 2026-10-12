# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Door approach and opening scene with a wrist camera and knob texture."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .door_rotation_cfg import (
    KNOB_GRASP_YAW,
    PANDA_FINGER_OPEN,
    FrankaDoorRotationECCfg,
    knob_grasp_pose,
)
from ..sensors.camera import wrist_camera_cfg
from ...vis.sticker import STICKER_THICK, sticker_usd

__all__ = [
    "DOOR_FACE",
    "FRANKA_DOOR_STANDOFF_JOINT_POS",
    "PROBE_OPEN",
    "PROGRESS_JOINT",
    "STANDOFF",
    "STICKER_POS",
    "STICKER_SIZE",
    "FrankaDoorECCfg",
    "grasp_frame",
]

STANDOFF = 0.10
"""Distance back from the knob along the hand's approach axis where the approach starts [m].

Shorter than the drawer's 0.15: the knob is only 0.645 m out at z = 0.45, and backing off toward
the base folds the elbow -- at 0.15 the controller settled with ``panda_joint4`` on its limit
(-3.0718 rad), which the spawn then refuses. The sticker is closer as well, so the servo has more
to see."""

PROGRESS_JOINT = "cabinet/door_left_joint"
"""What a task file's ``terminate_extra`` names as this scene's progress: the hinge joint [rad].
It runs from 0 to -1.57 as the door opens, so a target is negative."""

PROBE_OPEN = -0.35
"""[rad] the probe opens the door by to check that the sticker rides with it."""

DOOR_FACE = 0.0098
"""The door's front face in its own link frame [m]."""

STICKER_SIZE = (0.22, 0.10)
"""Width x height of the sticker [m], the logo's own aspect inside its margin. Smaller than the
drawer's: it has to fit between the hinge and the knob and stay inside the camera's field from
the standoff, where a label this far off the optical axis is already near the frame's edge."""

STICKER_POS = (DOOR_FACE + 0.5 * STICKER_THICK, 0.17, 0.185)
"""Sticker centre in the door's link frame [m]: flush on the face, at the knob's height, between
the hinge and the knob with 6 cm to the knob."""

FRANKA_DOOR_STANDOFF_JOINT_POS = {
    "panda_joint1": -0.639669,
    "panda_joint2": -1.649561,
    "panda_joint3": 1.158729,
    "panda_joint4": -3.000508,
    "panda_joint5": 2.473865,
    "panda_joint6": 2.683935,
    "panda_joint7": 0.561498,
}
"""Arm joint configuration back from the knob along the approach axis."""


@configclass
class FrankaDoorECCfg(FrankaDoorRotationECCfg):
    """Grasp then open, on the left door. See the module docstring."""

    motion_stiffness = (500.0, 500.0, 500.0, 30.0, 30.0, 30.0)
    """Task-space stiffness for grasping and opening the door."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_DOOR_STANDOFF_JOINT_POS,
            "panda_finger_joint.*": PANDA_FINGER_OPEN,
        }
        self.scene.wrist_cam = wrist_camera_cfg(focal_length=8.0)
        # Under the door's link prim: a visual shape *of the door*, posed in its frame.
        self.scene.sticker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/door_left_link/Sticker",
            spawn=sim_utils.UsdFileCfg(usd_path=str(sticker_usd(STICKER_SIZE))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=STICKER_POS),
        )


def grasp_frame(env):
    """``(R, p)`` of the grasp on the knob in the robot root frame, from the live scene; the
    hand's approach axis is its z, so the standoff is ``p - STANDOFF * R[:, 2]``."""
    return knob_grasp_pose(env.scene["cabinet"], env.scene["robot"], KNOB_GRASP_YAW)
