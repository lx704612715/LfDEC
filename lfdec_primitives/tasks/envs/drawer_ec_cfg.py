# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Drawer approach and pull scene with a wrist camera and handle texture."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .drawer_translation_cfg import (  # noqa: F401  shared scene constants
    FRANKA_DRAWER_GRASP_JOINT_POS,
    PANDA_FINGER_OPEN,
    FrankaDrawerTranslationECCfg,
    drawer_pull_axis,
)
from ..sensors.camera import wrist_camera_cfg
from ...vis.sticker import STICKER_THICK, sticker_usd

__all__ = [
    "DRAWER_FRONT",
    "FRANKA_DRAWER_STANDOFF_JOINT_POS",
    "PROBE_OPEN",
    "PROGRESS_JOINT",
    "STANDOFF",
    "STICKER_POS",
    "STICKER_SIZE",
    "FrankaDrawerECCfg",
    "grasp_frame",
]

STANDOFF = 0.15
"""Distance back from the handle along the hand's approach axis where the approach starts [m].

Far enough to be a real free-space segment and to put the drawer front in frame, near enough
that the posture stays inside the arm's range: the handle already sits at 0.852 m of the Panda's
~0.855 m reach at the stock ``CABINET_X``, and the standoff is *toward* the base."""

PROGRESS_JOINT = "cabinet/drawer_top_joint"
"""What a task file's ``terminate_extra`` names as this scene's progress: the drawer's joint [m]."""

PROBE_OPEN = 0.20
"""[m] the probe opens the drawer by to check that the sticker rides with it."""

DRAWER_FRONT = 0.2732
"""The top drawer's front face in its own link frame [m]."""

STICKER_SIZE = (0.24, 0.11)
"""Width x height of the sticker [m]: the drawer front's 0.124 m less a rim, and the width the
logo's own 2.35:1 gives inside :data:`~.sticker.STICKER_MARGIN`, so nothing is stretched."""

STICKER_POS = (DRAWER_FRONT + 0.5 * STICKER_THICK, -0.23, 0.0)
"""Sticker centre in the drawer link frame [m], flush beside the handle and
visible to the wrist camera from the standoff."""

FRANKA_DRAWER_STANDOFF_JOINT_POS = {
    "panda_joint1": 0.179798,
    "panda_joint2": -1.302916,
    "panda_joint3": -0.171703,
    "panda_joint4": -2.673534,
    "panda_joint5": -0.722649,
    "panda_joint6": 2.927981,
    "panda_joint7": -0.240671,
}
"""Arm joint configuration back from the drawer handle along the approach axis."""


@configclass
class FrankaDrawerECCfg(FrankaDrawerTranslationECCfg):
    """Grasp then pull, on the top drawer. See the module docstring."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_DRAWER_STANDOFF_JOINT_POS,
            "panda_finger_joint.*": PANDA_FINGER_OPEN,
        }
        self.scene.wrist_cam = wrist_camera_cfg(focal_length=8.0)
        # Under the drawer's link prim: a visual shape *of the drawer*, posed in its frame.
        self.scene.sticker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/drawer_top/Sticker",
            spawn=sim_utils.UsdFileCfg(usd_path=str(sticker_usd(STICKER_SIZE))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=STICKER_POS),
        )


def grasp_frame(env):
    """``(R, p)`` of the grasp on the top drawer handle in the robot root frame, from the live
    scene: the stock ``cabinet_frame`` handle target, whose frame *is* the grasp orientation
    The hand's approach axis is its
    z, so the standoff is ``p - STANDOFF * R[:, 2]``."""
    import isaaclab.utils.math as math_utils

    rd = env.scene["robot"].data
    h = env.scene["cabinet_frame"].data
    pos, quat = math_utils.subtract_frame_transforms(
        rd.root_pos_w.torch, rd.root_quat_w.torch, h.target_pos_w.torch[:, 0], h.target_quat_w.torch[:, 0])
    return math_utils.matrix_from_quat(quat)[0].cpu().numpy(), pos[0].cpu().numpy()
