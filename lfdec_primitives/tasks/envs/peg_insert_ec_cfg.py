# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Fixed-grasp peg insertion scene: approach, plane search and constrained insertion."""

from __future__ import annotations

import math
from collections.abc import Callable

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

import lfdec_primitives as lp

from .cabinet_ec_cfg import PANDA_TCP_OFFSET
from ..sensors.camera import wrist_camera_cfg
from .plate_plane_cfg import (
    FrankaPlatePlaneECCfg,
    hole_point,
    plate_normal,
    plate_rotation,
    plate_slabs,
    plate_top_point,
)
from ...vis.sticker import STICKER_THICK, sticker_usd

__all__ = [
    "FRANKA_PEG_STANDOFF_JOINT_POS",
    "PEG_MASS",
    "PEG_PROTRUDE",
    "PEG_SIZE",
    "PROBE_OPEN",
    "PROGRESS_JOINT",
    "STANDOFF",
    "STICKER_SIZE",
    "FrankaPegInsertECCfg",
    "grasp_frame",
    "hand_rotation",
    "hole_axis_point",
    "peg_yaw",
    "spawn_peg",
]

STANDOFF = 0.10
"""Approach distance above the touch pose [m], leaving arm travel for insertion."""

PROGRESS_JOINT = None
"""Nothing in this scene articulates: the task ends on a contact-changing event, not a joint."""
PROBE_OPEN = None

PEG_SIZE = (0.028, 0.028, 0.08)
PEG_POS = (0.0, 0.0, 0.12)
PEG_MASS = 0.06
PEG_FINGER_GAP = 0.025
"""A 28 mm square peg, 8 cm long, in the ``panda_hand`` frame: from the fingertip pads' upper
edge to 5.7 cm past them, with the fingers held 50 mm apart around it -- turned to the opening's
axes its corners reach 19.5 mm from its axis along the finger travel. It is fixed to the hand,
so the fingers need not touch it. The opening is 50 mm, so the peg enters with 11 mm to spare
on each side and needs no alignment finer than the servo and the spiral provide."""

PEG_PROTRUDE = PEG_POS[2] + 0.5 * PEG_SIZE[2] - PANDA_TCP_OFFSET[2]
"""How far the peg's tip is past the control point along the hand's z [m]. The surface the plane
primitive rides is the plate pushed out by this."""

STICKER_SIZE = (0.20, 0.09)
STICKER_UV = (-0.06, 0.0)
"""The sticker's size [m] and centre on the plate's top face in the plate's own axes: on the slab
opposite the opening (:data:`~.plate_plane_cfg.HOLE_XY` is at +u, +v), clear of the landing
point at the centre and of the spiral around it."""

FRANKA_PEG_STANDOFF_JOINT_POS = {
    "panda_joint1": -0.433118,
    "panda_joint2": 0.182206,
    "panda_joint3": 0.423354,
    "panda_joint4": -1.194241,
    "panda_joint5": 0.040438,
    "panda_joint6": 1.279112,
    "panda_joint7": -0.783637,
}
"""Arm joint configuration with the fixed peg above the plate contact pose."""


def spawn_peg(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a separate peg body with a fixed joint to the configured hand link.
    The attachment position is expressed in that link frame."""
    from pxr import Gf, Sdf, UsdPhysics

    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    for path in find_matching_prim_paths(prim_path):
        joint = UsdPhysics.FixedJoint.Define(stage, f"{path}/Joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{path.rsplit('/', 1)[0]}/{cfg.attach_to}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*cfg.attach_pos))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(*cfg.attach_rot))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return prim


@configclass
class PegCfg(sim_utils.CuboidCfg):
    """A cuboid spawned beside a robot's links that becomes one of them, fixed to ``attach_to``."""

    func: Callable = spawn_peg
    attach_to: str = "panda_hand"
    attach_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attach_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """The peg's orientation in ``attach_to``'s frame, ``(w, x, y, z)``."""


def hand_rotation() -> np.ndarray:
    """Hand orientation held through the task: z (the approach) along the plate's inward normal,
    so the peg is coaxial with the opening, and the fingers along root +y as in the plate task's
    solved posture -- tilting that posture by the plate's 8 deg rather than yawing the wrist to
    the plate's own axes, which the arm could not follow from where it spawns."""
    n = plate_normal()
    x = lp.unit(lp.project_to_plane(np.array([0.0, 1.0, 0.0]), n))
    return np.stack([x, np.cross(-n, x), -n], axis=1)


def peg_yaw() -> float:
    """Angle [rad] the peg is turned about the hand's approach axis so that its faces are
    parallel to the opening's: the plate's own axes seen from the hand, reduced to
    ``(-45, 45]`` deg since a square repeats every quarter turn."""
    u_hand = hand_rotation().T @ plate_rotation()[:, 0]      # the plate's u axis, in the hand frame
    psi = math.atan2(float(u_hand[1]), float(u_hand[0]))
    return (psi + math.pi / 4) % (math.pi / 2) - math.pi / 4


def grasp_frame(env):
    """``(R, p)`` of the nominal touch pose in the robot root frame: the control point
    :data:`PEG_PROTRUDE` above the plate's centre, hand square to the plate. What the approach is
    recorded toward; the recorder then feels for the actual touch."""
    del env   # the plate is static and its geometry is the config's
    return hand_rotation(), plate_top_point() + PEG_PROTRUDE * plate_normal()


def _sticker_pose():
    """Position and Isaac quaternion of the sticker in the robot root frame, lying on the plate.

    The sticker is a separate static visual shape so Newton also renders the
    plate's colliders, which provide its visible surface.
    """
    name, size, centre = plate_slabs()[0]        # the -u slab: full plate height, opening-free
    u, v = STICKER_UV
    hu, hv = 0.5 * size[0], 0.5 * size[1]
    if not (abs(u - centre[0]) + 0.5 * STICKER_SIZE[1] <= hu and abs(v - centre[1]) + 0.5 * STICKER_SIZE[0] <= hv):
        raise ValueError(f"a {STICKER_SIZE} sticker at {STICKER_UV} does not fit on {name}")
    R_plate = plate_rotation()
    pos = plate_top_point() + R_plate @ np.array([u, v, 0.5 * STICKER_THICK])
    # The slab's textured face is its +/-x; lay it flat with +x up, its width along the plate's v.
    R = R_plate @ np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    qw, qx, qy, qz = lp.quat_from_matrix(R)
    return tuple(float(x) for x in pos), (qx, qy, qz, qw)


@configclass
class FrankaPegInsertECCfg(FrankaPlatePlaneECCfg):
    """Peg in hand above the plate with the opening. See the module docstring."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_PEG_STANDOFF_JOINT_POS,
            "panda_finger_joint.*": PEG_FINGER_GAP,
        }
        self.actions.gripper_action.close_command_expr = {"panda_finger_.*": PEG_FINGER_GAP}
        self.scene.wrist_cam = wrist_camera_cfg(focal_length=8.0)
        # A link of the robot, fixed to the hand (see spawn_peg).
        self.scene.peg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Peg",
            spawn=PegCfg(
                size=PEG_SIZE, attach_to="panda_hand", attach_pos=PEG_POS,
                attach_rot=tuple(float(v) for v in lp.quat_from_matrix(lp.rz(peg_yaw()))),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=PEG_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.6, 0.25)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.6, restitution=0.0),
            ),
        )
        pos, rot = _sticker_pose()
        self.scene.sticker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Sticker",
            spawn=sim_utils.UsdFileCfg(usd_path=str(sticker_usd(STICKER_SIZE))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        )




def hole_axis_point():
    """Centre of the opening on the plate's top face and the inward axis of the hole -- the
    truth the recorder drives by and the insertion is judged against."""
    return hole_point(), -plate_normal()
