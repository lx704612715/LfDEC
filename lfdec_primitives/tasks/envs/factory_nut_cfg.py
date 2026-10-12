# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Newton screw-contact diagnostic with Factory's original M16 nut and bolt meshes.

The nut starts engaged and is rigidly grasped. This isolates thread contact and the screw
primitive; it does not model picking up a nut or finding the thread entrance.
"""
from collections.abc import Callable
import math

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from .cabinet_ec_cfg import PANDA_TCP_OFFSET
from .factory_peg_ec_cfg import ASSET_DIR, FrankaFactoryPegECCfg, _contact, _sdf

TASK = "LfDEC-Factory-Nut-Thread-Franka"
THREAD_PITCH = 0.002
NUT_OFFSET = np.array([0., 0., 0.0165])  # asset bottom frame relative to hand TCP
# Measured FK of the shipped plate posture, with nut bottom 3 mm below the bolt tip.
BOLT_POSITION = np.array([0.554636, -0.006820, 0.440925])
ENTRY_TASK = "LfDEC-Factory-Nut-Thread-EC-Franka"
ENTRY_BOLT_POSITION = BOLT_POSITION - [0., 0., 0.01]


def spawn_held_nut(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, Sdf, Usd, UsdPhysics

    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    for path in find_matching_prim_paths(prim_path):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        body = f"{path}/factory_nut_loose"
        if not stage.GetPrimAtPath(body).HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Factory nut body missing: {body}")
        joint = UsdPhysics.FixedJoint.Define(stage, f"{path}/GraspJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{path.rsplit('/', 1)[0]}/panda_hand")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, PANDA_TCP_OFFSET[2] + NUT_OFFSET[2]))
        # Cancel the vertical grasp orientation so nut and bolt threads start at the same yaw.
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(0, math.sqrt(0.5), math.sqrt(0.5), 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return root


@configclass
class HeldNutCfg(sim_utils.UsdFileCfg):
    func: Callable = spawn_held_nut


@configclass
class FrankaFactoryNutCfg(FrankaFactoryPegECCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.peg = None
        self.scene.socket = None
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.02
        self.actions.gripper_action.close_command_expr = {"panda_finger_.*": 0.02}
        self.actions.arm_action.v_max = [0.01] * 3 + [0.2] * 3
        self.scene.bolt = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Bolt",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSET_DIR}/factory_bolt_m16.usd",
                                      collision_props=_sdf(), physics_material=_contact()),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(BOLT_POSITION)),
        )
        self.scene.nut = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FactoryNut",
            spawn=HeldNutCfg(usd_path=f"{ASSET_DIR}/factory_nut_m16.usd",
                            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                            collision_props=_sdf(), physics_material=_contact()),
        )


@configclass
class FrankaFactoryNutECCfg(FrankaFactoryNutCfg):
    """The held nut starts clear of the bolt; the demonstration must find thread contact."""

    def __post_init__(self):
        super().__post_init__()
        from ..sensors.camera import wrist_camera_cfg
        from ...vis.sticker import sticker_usd

        self.scene.bolt.init_state.pos = tuple(ENTRY_BOLT_POSITION)
        self.scene.wrist_cam = wrist_camera_cfg(focal_length=8.0)
        self.scene.sticker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FixtureLabel",
            spawn=sim_utils.UsdFileCfg(usd_path=str(sticker_usd((0.16, 0.08)))),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.49, 0.025, float(ENTRY_BOLT_POSITION[2] + 0.035)),
                rot=(0., -2**-0.5, 0., 2**-0.5)),
        )
