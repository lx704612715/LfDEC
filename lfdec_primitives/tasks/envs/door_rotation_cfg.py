# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Door environment with a grasped knob and revolute-motion configuration."""

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
    "DOOR_HINGE_DAMPING",
    "DOOR_JOINT",
    "DOOR_LINK",
    "DOOR_SIDE",
    "FRANKA_DOOR_GRASP_JOINT_POS",
    "KNOB_BODY",
    "KNOB_GRASP_YAW",
    "FrankaDoorRotationECCfg",
    "door_hinge",
    "knob_grasp_pose",
    "knob_pos",
    "opening_tangent",
]

DOOR_SIDE = "left"
"""Cabinet door used by this task."""

DOOR_JOINT = f"door_{DOOR_SIDE}_joint"
DOOR_LINK = f"door_{DOOR_SIDE}_link"
KNOB_BODY = f"door_{DOOR_SIDE}_nob_link"

KNOB_GRASP_YAW = 2.618
"""Finger-closing orientation about the knob approach axis [rad]."""

FRANKA_DOOR_GRASP_JOINT_POS = {
    "panda_joint1": -1.059874,
    "panda_joint2": -1.528563,
    "panda_joint3": 1.219023,
    "panda_joint4": -2.751590,
    "panda_joint5": 2.672025,
    "panda_joint6": 2.512377,
    "panda_joint7": 0.418462,
}
"""Arm joint configuration with the TCP on the door knob."""

DOOR_HINGE_DAMPING = 10.0
"""Viscous friction [N.m.s/rad] on the door hinge, standing in for a real hinge.

The stock cabinet's return spring has to go for the same reason it does on the drawer (see
:func:`~.cabinet_ec_cfg.furniture_joint`), and its 1.0 N.m.s/rad of damping leaves a door that
coasts. At 10 the primitive pulls against ~0.9 N.m -- 2.5 N at the 0.349 m radius, about what the
drawer's slides resist -- so the wrench gate has something to work against and the door stays
where it is let go."""


@configclass
class FrankaDoorRotationECCfg(FrankaCabinetECCfg):
    """Revolute EC: the Panda spawns holding the left door's knob."""

    viewer_eye = (-0.4, 1.7, 1.0)
    viewer_lookat = (0.45, -0.1, 0.5)
    """Over the robot's left shoulder, the hinge side: the door swings toward the robot and its
    face, with the sticker and the knob, turns toward this camera as it opens."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_DOOR_GRASP_JOINT_POS,
            "panda_finger_joint.*": PANDA_FINGER_OPEN,
        }
        furniture_joint(self.scene.cabinet, "door", DOOR_HINGE_DAMPING)
        self.door_joint_cfg = SceneEntityCfg("cabinet", joint_names=[DOOR_JOINT])


def door_hinge(cabinet, robot):
    """``(axis, point)`` of the door hinge in the robot root frame, from the live scene.

    Both are read off the door link rather than baked: the hinge is that link's own origin, and
    the axis is its local z. Verified against the knob's measured trajectory -- driving
    ``door_left_joint`` through its range traces a circle of radius 0.349 m about exactly this
    line, to four decimals.

    Returns:
        ``(axis, point)``, each a (3,) numpy array in the robot root frame; ``axis`` is a unit
        vector and points along +``door_left_joint``.
    """
    import isaaclab.utils.math as math_utils
    import numpy as np
    import torch

    i = list(cabinet.body_names).index(DOOR_LINK)
    q_w = cabinet.data.body_link_quat_w.torch[:, i]
    p_w = cabinet.data.body_link_pos_w.torch[:, i]
    z = torch.tensor([[0.0, 0.0, 1.0]], device=q_w.device).expand(q_w.shape[0], 3)
    axis_w = math_utils.quat_apply(q_w, z)
    root_p, root_q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
    point_b, _ = math_utils.subtract_frame_transforms(root_p, root_q, p_w, q_w)
    axis_b = math_utils.quat_apply(math_utils.quat_inv(root_q), axis_w)
    u = axis_b[0].cpu().numpy()
    return u / np.linalg.norm(u), point_b[0].cpu().numpy()


def knob_grasp_pose(cabinet, robot, yaw: float = 0.0):
    """Grasp orientation and position on the door knob in the robot root frame."""
    import numpy as np

    axis, hinge = door_hinge(cabinet, robot)
    p = knob_pos(cabinet, robot)[0].cpu().numpy()
    z_hand = -opening_tangent(axis, hinge, p)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    y_hand = c * axis + s * np.cross(z_hand, axis)
    y_hand -= float(y_hand @ z_hand) * z_hand
    y_hand /= np.linalg.norm(y_hand)
    return np.stack([np.cross(y_hand, z_hand), y_hand, z_hand], axis=1), p




def opening_tangent(axis, hinge, p):
    """Unit direction the point ``p`` travels as the door *opens*, in the root frame.

    The door opens along -``door_left_joint`` (its range is ``[-1.57, 0]``), so this is the
    tangent for a rotation about ``-axis``.
    """
    import numpy as np

    v = np.cross(-np.asarray(axis, dtype=float), np.asarray(p, dtype=float) - np.asarray(hinge, dtype=float))
    return v / np.linalg.norm(v)


def knob_pos(cabinet, robot):
    """Knob centre in the robot root frame, (1, 3).

    The knob is its own link but its *link frame* sits at the hinge, 35 cm away -- the graspable
    part is the link's centre of mass. Verified: driving the door through its range moves this
    point on a clean circle, while the link origin does not move at all.
    """
    import isaaclab.utils.math as math_utils

    i = list(cabinet.body_names).index(KNOB_BODY)
    pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
        cabinet.data.body_com_pos_w.torch[:, i], cabinet.data.body_com_quat_w.torch[:, i],
    )
    return pos_b
