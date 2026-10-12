# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Panda above a tilted plate with an opening, for contact and plane-motion execution."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

import lfdec_primitives as lp

from .cabinet_ec_cfg import (  # noqa: F401  shared scene constants
    PANDA_FINGER_OPEN,
    PANDA_TCP_OFFSET,
    FrankaCabinetECCfg,
)

__all__ = [
    "FRANKA_PLATE_START_JOINT_POS",
    "GRASP_R",
    "HOLE_SIDE",
    "HOLE_XY",
    "PLATE_CENTER",
    "PLATE_CONTACT_TOLERANCE",
    "PLATE_HALF",
    "PLATE_PEDESTAL",
    "PLATE_THICK",
    "PLATE_TILT_DEG",
    "PLATE_YAW_DEG",
    "PROBE_FINGER_GAP",
    "FrankaPlatePlaneECCfg",
    "hole_point",
    "plate_normal",
    "plate_rotation",
    "plate_slabs",
    "plate_top_point",
    "static_box",
]

PLATE_CENTER = (0.55, 0.0, 0.42)
"""Plate top-face center in the robot root frame [m]."""

PLATE_HALF = 0.12
PLATE_THICK = 0.02
PLATE_CONTACT_TOLERANCE = 0.0005
"""Maximum accepted contact-solver penetration at the opening's walls [m]."""
PLATE_TILT_DEG = 8.0
"""Tilt of the true surface off horizontal. Large enough that a primitive which simply assumed
+z would ride 2.5 cm off the surface over the search, small enough that the closed gripper keeps
a single contact corner and the ridden surface stays planar."""

PLATE_YAW_DEG = 35.0
"""Direction of the tilt. Deliberately not aligned with any scene axis, so neither the estimator
nor the tangent basis can be right for the wrong reason."""

PROBE_FINGER_GAP = 0.015
"""Finger joint position [m] that keeps the pads separated during probing."""

PLATE_PEDESTAL = False
"""Stand the plate on a leg. Off: it is cosmetic, and a 12 cm column directly under the search
is one more thing for the arm to find."""

HOLE_SIDE = 0.05
HOLE_XY = (0.06, 0.04)
"""Centre of the opening on the plate's top face, in its own axes -- 7.2 cm from the nominal
landing point. The approach lands on solid plate, then the demonstrated slide brings the probe
within the spiral's search radius. Account for the probe's footprint as well as its centre
when checking whether a landing or a slide is clear of the opening's rim."""

FRANKA_PLATE_START_JOINT_POS = {
    "panda_joint1": 1.706169,
    "panda_joint2": -0.541962,
    "panda_joint3": -1.364396,
    "panda_joint4": -1.674913,
    "panda_joint5": -0.529331,
    "panda_joint6": 1.554727,
    "panda_joint7": -0.419734,
}
"""Arm configuration with the TCP above the plate and the hand facing downward."""


GRASP_R = np.array([[0.0, 1.0, 0.0],    # x_hand = +y_root
                    [1.0, 0.0, 0.0],    # y_hand = +x_root  (the fingers close along root +x)
                    [0.0, 0.0, -1.0]])  # z_hand = -z_root  (hand points straight down)
"""Hand orientation held for the whole task.

Vertical rather than square to the tilted surface: the contact is then a single fingertip corner,
and the surface the control point rides is still a plane parallel to the plate -- the plane
the primitive transfers from the demonstration. Squaring the hand to the plate would make the two pads land
together and the standoff depend on which one touched first."""


def plate_rotation() -> np.ndarray:
    """World orientation of the plate: its local z is the outward normal."""
    return lp.rz(math.radians(PLATE_YAW_DEG)) @ lp.rx(math.radians(PLATE_TILT_DEG))


def plate_normal() -> np.ndarray:
    """Outward normal of the plate's top face -- the true planar EC, robot root frame."""
    return plate_rotation() @ np.array([0.0, 0.0, 1.0])


def plate_top_point() -> np.ndarray:
    return np.array(PLATE_CENTER, dtype=float)


def hole_point() -> np.ndarray:
    """Centre of the opening, on the top face, robot root frame."""
    R = plate_rotation()
    return plate_top_point() + HOLE_XY[0] * R[:, 0] + HOLE_XY[1] * R[:, 1]


def plate_slabs() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """``(name, size, centre)`` of the four slabs that tile the face around the opening.

    A box cannot have a hole and a mesh would put the geometry out of reach of this config, so
    the plate is four boxes. Sizes and centres are in the plate's own frame with the
    top face at local z = 0.
    """
    S, t = PLATE_HALF, PLATE_THICK
    u0, v0 = HOLE_XY
    r = 0.5 * HOLE_SIDE
    if min(S - abs(u0) - r, S - abs(v0) - r) <= 0.0:
        raise ValueError(f"a {HOLE_SIDE} m opening at {HOLE_XY} does not fit a {2 * S} m plate")
    spans = [
        (0.5 * (-S + u0 - r), 0.0, 0.5 * (S + u0 - r), S),
        (0.5 * (S + u0 + r), 0.0, 0.5 * (S - u0 - r), S),
        (u0, 0.5 * (-S + v0 - r), r, 0.5 * (S + v0 - r)),
        (u0, 0.5 * (S + v0 + r), r, 0.5 * (S - v0 - r)),
    ]
    return [(f"plate_{i}", (2 * hu, 2 * hv, t), (cu, cv, -0.5 * t))
            for i, (cu, cv, hu, hv) in enumerate(spans)]


def static_box(prim: str, size, pos, quat=(0.0, 0.0, 0.0, 1.0), rgb=(0.45, 0.45, 0.52)) -> AssetBaseCfg:
    """A collidable, immovable box. ``quat`` is Isaac's **(x, y, z, w)**, as ``init_state.rot`` is.

    Explicitly ``kinematic_enabled``. A collider with no rigid-body API at all is static on
    PhysX, but leaving it implicit is not worth the ambiguity: the surface the whole task is
    about must not be able to move.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(x) for x in size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, solver_position_iteration_count=64,
                solver_velocity_iteration_count=1),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            # Contact offset detects fingertip contact before deep penetration.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            # mu = 0.6 sets the tangential stiction dead band to about 0.6 * press_depth.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(x) for x in pos), rot=tuple(float(x) for x in quat)
        ),
    )


@configclass
class FrankaPlatePlaneECCfg(FrankaCabinetECCfg):
    """Tilted-plate scene with compliant control, separated finger pads and an opening."""

    viewer_eye = (0.1, -0.9, 0.8)
    viewer_lookat = (0.55, 0.02, 0.44)
    """Half the cabinet tasks' distance, from the robot's right: the plate with its opening and
    sticker fills the lower half of the frame, the hand and the peg the upper."""

    motion_stiffness = (150.0, 150.0, 150.0, 30.0, 30.0, 30.0)
    """Task-space stiffness for contact with the plate."""

    def __post_init__(self):
        super().__post_init__()
        # Filter collision transients before applying the wrench gate.
        self.actions.arm_action.wrench_lpf_tau = 0.05
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_PLATE_START_JOINT_POS,
            "panda_finger_joint.*": PROBE_FINGER_GAP,
        }
        # "Closed" means the probe gap here, not pad-on-pad -- see PROBE_FINGER_GAP.
        self.actions.gripper_action.close_command_expr = {"panda_finger_.*": PROBE_FINGER_GAP}

        self.drop_cabinet()

        R = plate_rotation()
        # Convert the wxyz pose utility result to Isaac Lab's xyzw scene convention.
        qw, qx, qy, qz = lp.quat_from_matrix(R)
        quat = (qx, qy, qz, qw)
        origin = plate_top_point()
        for name, size, centre in plate_slabs():
            setattr(self.scene, name, static_box(name.title().replace("_", ""), size,
                                                  origin + R @ np.asarray(centre), quat))
        if PLATE_PEDESTAL:
            leg = float(origin[2] - PLATE_THICK)
            self.scene.plate_pedestal = static_box(
                "PlatePedestal", (0.12, 0.12, leg), (origin[0], origin[1], 0.5 * leg),
                rgb=(0.3, 0.3, 0.33))

    def play_mode(self):
        super().play_mode()
        self.scene.env_spacing = 2.5
