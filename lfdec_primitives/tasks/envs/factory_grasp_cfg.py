# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Factory peg held by finger contact, with an independently measured rigid-body pose.

The fixed-grasp insertion baseline remains in factory_peg_ec_cfg. This variant removes the
grasp joint entirely; its peg must be positioned between the fingers before the first step.
"""
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from .factory_peg_ec_cfg import ASSET_DIR, FrankaFactoryPegECCfg, FrankaFactoryPegVisualECCfg, _contact, _sdf

TASK = "LfDEC-Factory-Peg-Grasp-Franka"
VISUAL_TASK = "LfDEC-Factory-Peg-Grasp-Visual-Franka"
PEG_PROTRUSION = 0.035


def spawn_free_peg(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Sdf, Usd, UsdPhysics

    if cfg.rolling_friction is not None:
        cfg = cfg.copy()
        for material in cfg.physics_material:
            if isinstance(material, NewtonMaterialCfg):
                material.rolling_friction = cfg.rolling_friction
    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    for path in find_matching_prim_paths(prim_path):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(cfg.contact_dim)
    return root


@configclass
class FreePegCfg(sim_utils.UsdFileCfg):
    func: Callable = spawn_free_peg
    # Finite pad support: torsional and rolling resistance. The 5 mm rolling-friction
    # length is a simulation contact-patch assumption, not a measured material property.
    contact_dim: int = 6
    rolling_friction: float | None = 0.005


@configclass
class FrankaFactoryPegGraspCfg(FrankaFactoryPegECCfg):
    def __post_init__(self):
        super().__post_init__()
        _free_peg(self)


@configclass
class FrankaFactoryPegGraspVisualCfg(FrankaFactoryPegVisualECCfg):
    def __post_init__(self):
        super().__post_init__()
        _free_peg(self)


def _free_peg(cfg):
    cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.004
    cfg.actions.gripper_action.close_command_expr = {"panda_finger_.*": 0.0}
    cfg.scene.peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=FreePegCfg(usd_path=f"{ASSET_DIR}/factory_peg_8mm.usd",
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
                        collision_props=_sdf(), physics_material=_contact()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.55)),
    )
