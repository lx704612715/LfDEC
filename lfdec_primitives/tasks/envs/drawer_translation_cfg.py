# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Drawer environment with a grasped handle and prismatic-motion configuration."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .cabinet_ec_cfg import (  # noqa: F401  shared scene constants
    CABINET_X,
    PANDA_FINGER_OPEN,
    PANDA_TCP_OFFSET,
    FrankaCabinetECCfg,
    furniture_joint,
)

__all__ = [
    "DRAWER_SLIDE_DAMPING",
    "FRANKA_DRAWER_GRASP_JOINT_POS",
    "FrankaDrawerTranslationECCfg",
    "drawer_pull_axis",
]

FRANKA_DRAWER_GRASP_JOINT_POS = {
    "panda_joint1": 0.004574,
    "panda_joint2": -0.747810,
    "panda_joint3": -0.147921,
    "panda_joint4": -2.421058,
    "panda_joint5": -0.819076,
    "panda_joint6": 3.283792,
    "panda_joint7": -0.072321,
}
"""Arm joint configuration with the TCP on the top drawer handle."""

DRAWER_SLIDE_DAMPING = 25.0
"""Drawer slide viscous damping [N.s/m]."""


@configclass
class FrankaDrawerTranslationECCfg(FrankaCabinetECCfg):
    """Prismatic EC: the Panda spawns on the top drawer handle."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_DRAWER_GRASP_JOINT_POS,
            "panda_finger_joint.*": PANDA_FINGER_OPEN,
        }
        furniture_joint(self.scene.cabinet, "drawer", DRAWER_SLIDE_DAMPING)
        self.drawer_joint_cfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])




def drawer_pull_axis(handle_quat_w, handle_pos_b, root_quat_w):
    """Unit vector along which the top drawer slides *out*, in the robot root frame.

    The drawer's prismatic axis is the cabinet's local +x, which the stock ``cabinet_frame``
    offset maps onto the handle frame's -z. The sign is then chosen to point back toward the
    robot base, i.e. the direction the arm has to pull.

    Args:
        handle_quat_w: (N, 4) handle-frame orientation in world, w-first.
        handle_pos_b: (N, 3) handle position in the root frame.
        root_quat_w: (N, 4) robot root orientation in world, w-first.

    Returns:
        (3,) numpy unit vector for env 0.
    """
    import isaaclab.utils.math as math_utils
    import numpy as np
    import torch

    z_minus = torch.tensor([[0.0, 0.0, -1.0]], device=handle_quat_w.device).expand(handle_quat_w.shape[0], 3)
    axis_b = math_utils.quat_apply(math_utils.quat_inv(root_quat_w), math_utils.quat_apply(handle_quat_w, z_minus))
    u = axis_b[0].cpu().numpy()
    u = u / np.linalg.norm(u)
    return -u if u @ handle_pos_b[0].cpu().numpy() > 0 else u
