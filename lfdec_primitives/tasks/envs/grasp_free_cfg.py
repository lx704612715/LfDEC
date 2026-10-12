# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Textured block, table and wrist camera for demonstration-driven grasping and lifting."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import lfdec_primitives as lp

from .cabinet_ec_cfg import PANDA_FINGER_OPEN, FrankaCabinetECCfg
from ..sensors.camera import CAM_WIDTH, CAM_HEIGHT, wrist_camera_cfg
from .plate_plane_cfg import GRASP_R, static_box

__all__ = [
    "BLOCK_MASS",
    "BLOCK_POS",
    "BLOCK_SIDE",
    "FRANKA_GRASP_START_JOINT_POS",
    "OBSERVE_HEIGHT",
    "TABLE_TOP",
    "FrankaGraspFreeECCfg",
]

TABLE_TOP = 0.40
"""Top face of the table, robot root frame. Elbow height, for the reason the plate sits at 0.42:
lower and the arm's upper links are inside the slab at every posture that reaches it."""
TABLE_SIZE = (0.50, 0.60, 0.02)
TABLE_CENTER = (0.55, 0.0)

BLOCK_SIDE = 0.05
"""Cube side length [m]."""
BLOCK_MASS = 0.1
BLOCK_POS = (0.55, 0.0, TABLE_TOP + 0.5 * BLOCK_SIDE)
"""Nominal block centre, where the demonstration is recorded. The run moves it from here."""

OBSERVE_HEIGHT = 0.20
"""Initial TCP height above the block center [m]."""

FRANKA_GRASP_START_JOINT_POS = {
    "panda_joint1": -0.993268,
    "panda_joint2": 0.627411,
    "panda_joint3": 1.741030,
    "panda_joint4": -1.070155,
    "panda_joint5": -0.717903,
    "panda_joint6": 1.074898,
    "panda_joint7": -0.392492,
}
"""Arm joint configuration for the block observation pose."""


def textured_cube_usd(side: float = BLOCK_SIDE, seed: int = 7) -> Path:
    """Create a cached cube mesh with per-face UVs and a deterministic feature texture."""
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    cache = Path(os.environ.get("LFDEC_CACHE", Path.home() / ".cache" / "lfdec")) / "assets"
    cache.mkdir(parents=True, exist_ok=True)
    usd_path = cache / f"textured_cube_shapes2_{1e3 * side:.0f}mm_s{seed}.usd"
    png_path = cache / f"feature_pattern_shapes2_256_s{seed}.png"
    if usd_path.exists():
        return usd_path
    if not png_path.exists():
        Image.fromarray(lp.feature_pattern(256, seed)).save(png_path)
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Block")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    h = 0.5 * float(side)
    quads = [  # the four corners of each face, counter-clockwise seen from outside
        [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)],
        [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)],
        [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)],
        [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)],
        [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],
        [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)],
    ]
    cube = UsdGeom.Mesh.Define(stage, "/Block/geometry")
    cube.GetPointsAttr().Set([Gf.Vec3f(*v) for quad in quads for v in quad])
    cube.GetFaceVertexCountsAttr().Set([4] * 6)
    cube.GetFaceVertexIndicesAttr().Set([4 * i + k for i in range(6) for k in range(4)])
    cube.GetSubdivisionSchemeAttr().Set("none")
    st = UsdGeom.PrimvarsAPI(cube.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                            UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)] * 6)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(cube.GetPrim()).GetApproximationAttr().Set("convexHull")
    material = UsdShade.Material.Define(stage, "/Block/Looks/pattern")
    surface = UsdShade.Shader.Define(stage, "/Block/Looks/pattern/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    tex = UsdShade.Shader.Define(stage, "/Block/Looks/pattern/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{png_path.name}")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.6)])
    stage.GetRootLayer().Save()
    return usd_path


@configclass
class FrankaGraspFreeECCfg(FrankaCabinetECCfg):
    """Free-space grasp: the Panda over a textured block, wrist camera, gripper open."""

    def __post_init__(self):
        super().__post_init__()
        self.drop_cabinet()
        self.scene.robot.init_state.joint_pos = {
            **FRANKA_GRASP_START_JOINT_POS,
            "panda_finger_joint.*": PANDA_FINGER_OPEN,
        }
        cx, cy = TABLE_CENTER
        self.scene.table = static_box("Table", TABLE_SIZE, (cx, cy, TABLE_TOP - 0.5 * TABLE_SIZE[2]),
                                      rgb=(0.55, 0.55, 0.58))
        self.scene.block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in BLOCK_POS)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(textured_cube_usd()),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=BLOCK_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=[
                    UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                    NewtonMaterialCfg(contact_stiffness=2.56e6, contact_damping=3200.),
                ],
            ),
        )
        # A free prim: IsaacCamera poses it from the TCP every step. The offset is irrelevant.
        self.scene.wrist_cam = wrist_camera_cfg(focal_length=15.0, width=CAM_WIDTH, height=CAM_HEIGHT)

    def play_mode(self):
        super().play_mode()
        self.scene.env_spacing = 2.5
