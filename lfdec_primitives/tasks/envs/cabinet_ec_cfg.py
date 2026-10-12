# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Shared Panda/cabinet scene, compliant action, wrench sensor and Newton contact settings."""

from __future__ import annotations

from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.cabinet.config.franka.ik_rel_env_cfg import (
    FrankaCabinetEnvCfg,
)

from ..actions.compliant import CompliantCartesianActionCfg
from ..robots.franka import PANDA_FINGER_OPEN, PANDA_TCP_OFFSET

__all__ = [
    "CABINET_X",
    "PANDA_FINGER_OPEN",
    "PANDA_TCP_OFFSET",
    "FrankaCabinetECCfg",
    "furniture_joint",
]

_CONTROLLER_DT = 1.0 / 500.0
_DECIMATION = 10

# Cabinet position along the robot root x axis [m].
CABINET_X = 0.8


def _raise_physics_rate(sim_cfg, dt: float, decimation: int) -> None:
    """Force ``dt`` on a resolved ``SimulationCfg`` or on each PhysX variant of a ``PresetCfg``.

    Presets are resolved by Hydra *after* ``__post_init__`` runs, so at this point ``sim`` is
    still the ``PresetCfg`` and every variant has to be visited. Variants already stepping at or
    above the requested rate (Newton's 1/600) are left untouched, as is ``default``, which is the
    same object as ``newton_mjwarp``.
    """
    if hasattr(sim_cfg, "dt"):  # already resolved
        if sim_cfg.dt > dt:
            sim_cfg.dt = dt
            sim_cfg.render_interval = decimation
        return
    for name, variant in vars(sim_cfg).items():
        if not name.startswith("_") and hasattr(variant, "dt") and variant.dt > dt:
            variant.dt = dt
            variant.render_interval = decimation


def _tune_newton_contacts(sim_cfg) -> None:
    """Configure Newton finger contact impedance and preload for a retained grasp."""
    variants = [sim_cfg] if hasattr(sim_cfg, "dt") else [
        v for name, v in vars(sim_cfg).items() if not name.startswith("_") and hasattr(v, "dt")
    ]
    for variant in variants:
        solver = getattr(getattr(variant, "physics", None), "solver_cfg", None)
        if solver is None or not hasattr(solver, "impratio"):  # PhysX variants
            continue
        solver.impratio = 10.0
        solver.cone = "elliptic"
        solver.njmax = 300
        solver.nconmax = 200
        variant.physics.num_substeps = 2


def _zero_arm_pd(robot_cfg) -> None:
    """Zero the arm actuators' stiffness and damping so the joints are pure torque sources.

    The finger actuators keep their PD -- the gripper stays position-controlled.

    Each actuator config is replaced with a copy first. ``ArticulationCfg.replace()`` copies the
    config but the ``actuators`` entries are shared with the module-level
    ``FRANKA_PANDA_HIGH_PD_CFG``, so mutating them in place would silently zero the gains of every
    other Franka task created in the same process.
    """
    robot_cfg.actuators = {
        name: (act if ("finger" in name or "hand" in name) else act.replace(stiffness=0.0, damping=0.0))
        for name, act in robot_cfg.actuators.items()
    }


def furniture_joint(cabinet_cfg, match: str, damping: float) -> None:
    """Set passive cabinet joint damping and disable the return spring."""
    cabinet_cfg.actuators = {
        name: (act.replace(stiffness=0.0, damping=damping) if match in name else act)
        for name, act in cabinet_cfg.actuators.items()
    }


@configclass
class FrankaCabinetECCfg(FrankaCabinetEnvCfg):
    """Compliant-control Sektion cabinet env. Subclass and add the constraint-specific parts."""

    motion_stiffness = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
    """Task-space ``K_p``. Subclasses override; the run scripts override again per phase."""

    viewer_eye = (-0.9, -1.3, 1.0)
    viewer_lookat = (0.35, 0.15, 0.45)
    viewer_focal_length = 24.0
    """Simulation camera focal length [mm]."""

    motion_damping = (10.0, 10.0, 10.0, 20.0, 20.0, 20.0)
    """Task-space damping coefficients for translation and rotation."""

    f_max = (25.0, 25.0, 25.0, 5.0, 5.0, 5.0)
    """Wrench gate limit ``F_m`` [N] and ``T_m`` [N.m]."""

    def __post_init__(self):
        super().__post_init__()

        # -- 1. compliant Cartesian action in place of differential IK
        self.actions.arm_action = CompliantCartesianActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            tcp_offset=PANDA_TCP_OFFSET,
            wrench_sensor_name="ee_wrench",
            # PhysX reports panda_hand; MuJoCo-Warp merges that fixed-joint child into its parent
            # and stops at panda_link7. First covered name wins.
            wrench_frame_body=["panda_hand", "panda_link7"],
            motion_stiffness=list(self.motion_stiffness),
            motion_damping=list(self.motion_damping),
            f_max=list(self.f_max),
            action_mode="hold",
            # Backend-dependent, so let the action term resolve it: PhysX honours the asset's
            # disable_gravity=True and must NOT get g(q); Newton ignores it and must. See the
            # CompliantCartesianActionCfg docstring for the measurements.
            gravity_compensation="auto",
        )

        # -- 2. torque-controllable arm
        _zero_arm_pd(self.scene.robot)

        # -- 3. wrench feedback. One sensor on the articulation root covers every link; the
        #       action term picks out panda_hand. Works on PhysX, OV-PhysX and Newton alike.
        self.scene.ee_wrench = JointWrenchSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            convention="incoming_joint_frame",
        )

        # -- 4. physics rate, contact fidelity, and an episode that outlives the primitive
        _raise_physics_rate(self.sim, _CONTROLLER_DT, _DECIMATION)
        _tune_newton_contacts(self.sim)
        self.decimation = _DECIMATION
        self.episode_length_s = 1.0e6
        self.terminations.time_out = None

        # Preserve the configured start pose and authored contact materials.
        self.events.reset_robot_joints = None
        self.events.robot_physics_material = None
        self.events.cabinet_physics_material = None

        # Cabinet reachability (see CABINET_X).
        pos = list(self.scene.cabinet.init_state.pos)
        pos[0] = CABINET_X
        self.scene.cabinet.init_state.pos = tuple(pos)

    def drop_cabinet(self):
        """Remove the cabinet and everything that referenced it.

        For the tasks that keep the robot, the compliant action, the wrench sensor and the physics
        settings and put something else in front of the arm. A scripted primitive episode uses
        neither observations nor rewards -- they are the RL task's -- and leaving cabinet-shaped
        terms behind would only break at reset.
        """
        self.scene.cabinet = None
        self.scene.cabinet_frame = None
        for name in ("cabinet_joint_pos", "cabinet_joint_vel", "rel_ee_drawer_distance"):
            setattr(self.observations.policy, name, None)
        for name in list(vars(self.rewards)):
            if not name.startswith("_"):
                setattr(self.rewards, name, None)

    def play_mode(self):
        super().play_mode()
        self.scene.env_spacing = 2.5
