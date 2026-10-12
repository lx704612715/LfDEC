# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Factory's unscaled 8 mm peg/socket with the Newton EC controller, not the RL environment.

The peg is rigidly attached to the hand, as in the large-opening plane example. The socket is
fixed, its authored bottom retained, and both meshes use Newton SDF contact. No Kit/FactoryEnv
import is needed. A known horizontal normal and scripted slide are the initial baseline;
neither a learned demonstration nor visual localization is claimed here.
"""
from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_newton.physics import NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from .cabinet_ec_cfg import PANDA_TCP_OFFSET, FrankaCabinetECCfg
from .plate_plane_cfg import FRANKA_PLATE_START_JOINT_POS, GRASP_R

TASK = "LfDEC-Factory-Peg-Insert-Franka"
VISUAL_TASK = "LfDEC-Factory-Peg-Insert-Visual-Franka"
CONTACT_VIEW_EYE = (0.60, -0.32, 0.60)
CONTACT_VIEW_LOOKAT = (0.556, 0.0, 0.44)
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
PEG_DIAMETER = 0.007986
HOLE_DIAMETER = 0.009
# The Factory task metadata says 8.1 mm; the shipped mesh's cylindrical wall is at r=4.5 mm.
PEG_LENGTH = 0.05
SOCKET_HEIGHT = 0.025
SOCKET_POSITION = np.array([0.556, 0.0, 0.40])
TIP_OFFSET = np.array([0.0, 0.0, 0.04])  # tip relative to TCP, in the hand frame [m]
RADIAL_CLEARANCE = (HOLE_DIAMETER - PEG_DIAMETER) / 2.0
STANDOFF = 0.30  # peg-tip distance from the opening, not TCP height
STANDOFF_DIRECTION = np.array([-0.75, 0.0, np.sqrt(1.0 - 0.75**2)])
FRANKA_FACTORY_STANDOFF_JOINT_POS = {
    "panda_joint1": 1.320588,
    "panda_joint2": -0.778656,
    "panda_joint3": -0.591382,
    "panda_joint4": -1.634457,
    "panda_joint5": -0.482417,
    "panda_joint6": 1.004439,
    "panda_joint7": -0.184741,
}


def peg_tip(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Peg's bottom centre in the robot root frame [m], for the fixed grasp."""
    return p + R @ TIP_OFFSET


def insertion_metrics(R: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Geometric diagnostics only; these do not steer the plane search."""
    return insertion_metrics_at_tip(peg_tip(R, p), -R[:, 2])


def insertion_metrics_at_tip(tip: np.ndarray, axis: np.ndarray) -> dict[str, float]:
    """Seating from the peg's bottom centre and unit axis pointing toward its top.

    Loose-peg tasks must use the measured body pose, not infer it from the hand.
    """
    depth = SOCKET_POSITION[2] + SOCKET_HEIGHT - float(tip[2])
    mouth = tip + depth / axis[2] * axis if axis[2] > 0 else np.full(3, np.inf)
    return dict(insertion_depth=depth,
                hole_error=float(np.linalg.norm(tip[:2] - SOCKET_POSITION[:2])),
                mouth_error=float(np.linalg.norm(mouth[:2] - SOCKET_POSITION[:2])),
                peg_tilt=float(np.arccos(np.clip(axis[2], -1.0, 1.0))))


def insertion_succeeded(metrics: dict[str, float]) -> bool:
    """Require seating within 1 mm and the shaft axis inside the bore at both ends.

    In particular, neither a rim jam nor penetrating the socket's bottom is success.
    """
    return bool(SOCKET_HEIGHT - 0.001 <= metrics["insertion_depth"] <= SOCKET_HEIGHT + 0.001
                and metrics["hole_error"] <= RADIAL_CLEARANCE
                and metrics["mouth_error"] <= RADIAL_CLEARANCE)


def spawn_held_peg(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Reference the original peg mesh and attach its bottom frame to the hand."""
    from pxr import Gf, Sdf, Usd, UsdPhysics

    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    for path in find_matching_prim_paths(prim_path):
        # Factory registers the loose peg as its own articulation. Once attached it is part
        # of the Panda; retaining this API gives the robot two articulation roots.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        body = f"{path}/forge_round_peg_8mm"
        if not stage.GetPrimAtPath(body).HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Factory peg rigid body missing: {body}")
        joint = UsdPhysics.FixedJoint.Define(stage, f"{path}/GraspJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{path.rsplit('/', 1)[0]}/panda_hand")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, PANDA_TCP_OFFSET[2] + TIP_OFFSET[2]))
        # Asset +z runs from its bottom to its top; the hand's +z points downward.
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(0, 1, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return root


@configclass
class HeldPegCfg(sim_utils.UsdFileCfg):
    func: Callable = spawn_held_peg


def _contact():
    return [UsdPhysicsRigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.75),
            NewtonMaterialCfg(contact_stiffness=2.56e6, contact_damping=3200.0)]


def _sdf():
    return sim_utils.NewtonSDFCollisionPropertiesCfg(
        rest_offset=0.0, contact_gap=0.001, sdf_max_resolution=512,
        sdf_narrow_band_inner=-0.001, sdf_narrow_band_outer=0.001,
        sdf_texture_format="float32",
    )


@configclass
class FrankaFactoryPegECCfg(FrankaCabinetECCfg):
    viewer_eye = CONTACT_VIEW_EYE
    viewer_lookat = CONTACT_VIEW_LOOKAT
    motion_stiffness = (300.0, 300.0, 300.0, 100.0, 100.0, 100.0)
    f_max = (8.0, 8.0, 8.0, 2.0, 2.0, 2.0)

    def __post_init__(self):
        super().__post_init__()
        self.drop_cabinet()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_PLATE_START_JOINT_POS, "panda_finger_joint.*": 0.008,
        }
        # The welded grasp isolates the insertion mechanics from gripper slippage.
        self.actions.gripper_action.close_command_expr = {"panda_finger_.*": 0.008}
        self.actions.arm_action.wrench_lpf_tau = 0.05
        self.actions.arm_action.v_max = [0.01] * 3 + [0.1] * 3
        self.scene.socket = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Socket",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSET_DIR}/factory_hole_8mm.usd", collision_props=_sdf(),
                physics_material=_contact(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(SOCKET_POSITION)),
        )
        self.scene.peg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FactoryPeg",
            spawn=HeldPegCfg(usd_path=f"{ASSET_DIR}/factory_peg_8mm.usd",
                            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
                            collision_props=_sdf(), physics_material=_contact()),
        )
        # SDF meshes require Newton collision, not MuJoCo's convex mesh contact path.
        variants = [self.sim] if hasattr(self.sim, "dt") else [
            v for v in vars(self.sim).values() if hasattr(v, "dt")]
        for variant in variants:
            physics = variant.physics
            solver = getattr(physics, "solver_cfg", None)
            if solver is not None and hasattr(solver, "use_mujoco_contacts"):
                solver.use_mujoco_contacts = False
                solver.njmax, solver.nconmax = 1500, 400
                physics.num_substeps = 4
                physics.collision_cfg = NewtonCollisionPipelineCfg(
                    broad_phase="sap", max_triangle_pairs=1_000_000, rigid_contact_max=10_000)


@configclass
class FrankaFactoryPegVisualECCfg(FrankaFactoryPegECCfg):
    """Same contact geometry, with RGB-D and a visual-only fixture label for replay."""

    viewer_eye = (0.85, -0.65, 0.85)
    viewer_lookat = (0.55, 0.0, 0.58)

    def __post_init__(self):
        super().__post_init__()
        from ..sensors.camera import wrist_camera_cfg
        from ...vis.sticker import sticker_usd

        self.scene.robot.init_state.joint_pos = {
            **FRANKA_FACTORY_STANDOFF_JOINT_POS, "panda_finger_joint.*": 0.008,
        }
        self.scene.wrist_cam = wrist_camera_cfg(focal_length=8.0)
        # Label lies beside (not across) the original socket. It has no collision API.
        self.scene.sticker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FixtureLabel",
            spawn=sim_utils.UsdFileCfg(usd_path=str(sticker_usd((0.16, 0.08)))),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.49, 0.025, 0.425), rot=(0.0, -2**-0.5, 0.0, 2**-0.5)),
        )
