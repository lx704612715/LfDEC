# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Isaac Lab action term and configuration for compliant Cartesian control.

The adapter reads TCP pose, Jacobian and wrench, converts frames and writes joint
torques. Absolute goals drive EC execution; delta actions are optional."""

from __future__ import annotations

import torch
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from ...utils import math as cm
from ...compliant.controller import CompliantCartesianController
from ..sensors.wrench import SensorWrench

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["CompliantCartesianAction", "CompliantCartesianActionCfg"]


class CompliantCartesianAction(ActionTerm):
    """Paper §3.6 compliant Cartesian control as a manager-based action term."""

    cfg: CompliantCartesianActionCfg
    _asset: Articulation

    def __init__(self, cfg: CompliantCartesianActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._env = env
        asset, device = self._asset, self.device

        self._joint_ids, self._joint_names = asset.find_joints(cfg.joint_names, preserve_order=True)
        self._num_dof = len(self._joint_ids)
        body_ids, body_names = asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"body_name '{cfg.body_name}' matched {len(body_ids)} bodies: {body_names}")
        self._ee_body_idx = body_ids[0]
        self._ee_body_name = body_names[0]
        # Jacobian/mass-matrix DoF axis includes the floating-base DoFs and, for a fixed base,
        # omits the root body -- see IsaacLab 3.0 OperationalSpaceControllerAction.
        self._jacobi_body_idx = self._ee_body_idx - 1 if asset.is_fixed_base else self._ee_body_idx
        self._jacobi_joint_ids = [j + asset.num_base_dofs for j in self._joint_ids]

        torque_limit = cfg.torque_limit
        if torque_limit is None:
            torque_limit = asset.data.joint_effort_limits.torch[0, self._joint_ids].tolist()

        self.controller = CompliantCartesianController(
            num_envs=self.num_envs,
            num_dof=self._num_dof,
            device=device,
            dt=float(env.cfg.sim.dt),  # env.physics_dt may not be resolved during __init__
            kp=cfg.motion_stiffness,
            kv=cfg.motion_damping,
            damping_ratio=cfg.motion_damping_ratio,
            v_max=cfg.v_max,
            f_max=cfg.f_max,
            torque_limit=torque_limit,
            coupling=cfg.coupling,
            wrench_lpf_tau=cfg.wrench_lpf_tau,
            ee_vel_filter_tau=cfg.ee_vel_filter_tau,
            nullspace_stiffness=cfg.nullspace_stiffness,
            nullspace_damping_ratio=cfg.nullspace_damping_ratio,
            nullspace_ref_tau=cfg.nullspace_ref_tau,
            joint_damping=cfg.joint_damping,
        )
        self._needs_mass_matrix = cfg.nullspace_stiffness > 0.0

        tcp = torch.as_tensor(cfg.tcp_offset, device=device, dtype=torch.float32)
        self._tcp_offset = tcp.unsqueeze(0).repeat(self.num_envs, 1) if bool(torch.any(tcp != 0.0)) else None
        self._wrench_sign = float(cfg.wrench_sign)
        # Resolved lazily with the source, from body poses, so they need a stepped sim.
        self._wrench_rot: torch.Tensor | None = None    # EE <- reporting joint frame
        self._wrench_lever: torch.Tensor | None = None  # joint anchor -> TCP, EE frame
        self._wrench_src = None

        self._scale = torch.as_tensor(cfg.scale, device=device, dtype=torch.float32)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=device)
        self._wrench_ee = torch.zeros(self.num_envs, 6, device=device)
        self._reseat_pending = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self._has_pending_reseat = True  # idle physics ticks need no device synchronization
        self._payload_mass = torch.zeros(self.num_envs, 1, device=device, dtype=torch.float32)
        self._payload_com = torch.zeros(self.num_envs, 3, device=device, dtype=torch.float32)
        self._payload_active = False
        self._gravity_w = torch.as_tensor(env.cfg.sim.gravity, device=device, dtype=torch.float32)
        self._gravity_w = self._gravity_w.unsqueeze(0).repeat(self.num_envs, 1)

        # Resolve "auto" gravity compensation now that the backend is known (see the cfg docs).
        if isinstance(cfg.gravity_compensation, str):
            backend = str(getattr(env.scene, "physics_backend", "")).lower()
            self._gravity_comp = "physx" not in backend
            print(
                f"[CompliantCartesianAction] gravity_compensation={self._gravity_comp} "
                f"(auto, backend='{backend or 'unknown'}')"
            )
        else:
            self._gravity_comp = bool(cfg.gravity_compensation)

    # ------------------------------------------------------------------ ActionTerm API

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # Isaac Lab uses xyzw quaternions; the controller uses wxyz.

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        scaled = actions * self._scale
        self._processed_actions[:] = scaled
        if self.cfg.action_mode == "delta":
            p_des, q_des = self.desired_pose
            pos, quat = math_utils.apply_delta_pose(p_des, q_des, scaled)
            self.set_goal_pose(pos, quat)

    def apply_actions(self):
        ee_pos, ee_quat, r_b = self._ee_pose_root_frame()
        if self._has_pending_reseat:
            ids = self._reseat_pending.nonzero(as_tuple=True)[0]
            self.controller.reset(
                ee_pos, math_utils.convert_quat(ee_quat, to="wxyz"), self._asset.data.joint_pos.torch[:, self._joint_ids],
                env_ids=ids,
            )
            self._reseat_pending[ids] = False
            self._has_pending_reseat = False
        jac = self._jacobian_root_frame(r_b)
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel.torch[:, self._joint_ids]
        self._wrench_ee = self._read_wrench()

        mass_matrix = None
        if self._needs_mass_matrix:
            M = self._asset.data.mass_matrix.torch
            mass_matrix = M[:, self._jacobi_joint_ids, :][:, :, self._jacobi_joint_ids]
        gravity_ff = None
        if self._gravity_comp:
            gravity_ff = self._asset.data.gravity_compensation_forces.torch[:, self._jacobi_joint_ids]
        if self._payload_active:
            gravity_b = math_utils.quat_apply_inverse(self._asset.data.root_quat_w.torch, self._gravity_w)
            com_b = math_utils.quat_apply(ee_quat, self._payload_com)
            payload_ff = cm.point_mass_gravity_torques(jac, gravity_b, self._payload_mass, com_b)
            # Addition allocates: never modify the simulator's gravity tensor in place.
            gravity_ff = payload_ff if gravity_ff is None else gravity_ff + payload_ff

        tau = self.controller.step(
            ee_pos, math_utils.convert_quat(ee_quat, to="wxyz"), jac, joint_pos, joint_vel, self._wrench_ee,
            mass_matrix=mass_matrix, gravity_ff=gravity_ff,
        )
        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._payload_mass[env_ids] = 0.0
        self._payload_com[env_ids] = 0.0
        self._payload_active = bool(torch.any(self._payload_mass != 0.0))
        self._raw_actions[env_ids] = 0.0
        ee_pos, ee_quat, _ = self._ee_pose_root_frame()
        self.controller.reset(ee_pos, math_utils.convert_quat(ee_quat, to="wxyz"), self._asset.data.joint_pos.torch[:, self._joint_ids], env_ids)
        # Event-manager reset may update articulation state after the action reset.
        # Re-seat the reference on the first physics tick using the new state.
        self._reseat_pending[env_ids] = True
        self._has_pending_reseat = True

    # ------------------------------------------------------------------ convenience

    def set_payload(self, mass: float = 0.0, com_tcp: Sequence[float] = (0., 0., 0.),
                    env_ids=None) -> None:
        """Calibrate an assumed held payload, absent from articulation gravity. COM is
        relative to TCP in EE coordinates. Clear on release; this is not grasp detection."""
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError("payload mass must be finite and nonnegative")
        com = torch.as_tensor(com_tcp, device=self.device, dtype=self._payload_com.dtype)
        if com.shape != (3,) or not torch.isfinite(com).all():
            raise ValueError("payload COM must be a finite (3,) vector")
        i = slice(None) if env_ids is None else env_ids
        self._payload_mass[i] = mass
        self._payload_com[i] = com
        self._payload_active = bool(torch.any(self._payload_mass != 0.0))

    def reseat_equilibrium_pose(self) -> None:
        """Seat the reference on the measured TCP pose while preserving sensor calibration."""
        p, q, _ = self._ee_pose_root_frame()
        self.controller.reseat_equilibrium_pose(p, math_utils.convert_quat(q, to="wxyz"))

    def set_goal_pose(self, pos: torch.Tensor, quat: torch.Tensor | None = None, env_ids=None) -> None:
        """Latch an absolute goal pose in the root frame, ``quat`` in Isaac's (x, y, z, w). This is
        how a primitive drives the arm."""
        self.controller.set_goal_pose(pos, None if quat is None else math_utils.convert_quat(quat, to="wxyz"), env_ids)

    @property
    def desired_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The equilibrium pose ``T*`` in the root frame, ``quat`` in Isaac's (x, y, z, w)."""
        p, q = self.controller.desired_pose
        return p, math_utils.convert_quat(q, to="xyzw")

    @property
    def wrench_ee(self) -> torch.Tensor:
        """Last wrench handed to the controller: F/T convention, EE frame, about the TCP."""
        return self._wrench_ee

    @property
    def ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Control-point pose (pos, quat) in the root frame."""
        return self._ee_pose_root_frame()[:2]

    # ------------------------------------------------------------------ Isaac reads

    def _ee_pose_root_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Control-point pose in the root frame, plus the TCP lever in the root frame."""
        data = self._asset.data
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            data.root_pos_w.torch,
            data.root_quat_w.torch,
            data.body_pos_w.torch[:, self._ee_body_idx],
            data.body_quat_w.torch[:, self._ee_body_idx],
        )
        if self._tcp_offset is None:
            return pos_b, quat_b, None
        r_b = math_utils.quat_apply(quat_b, self._tcp_offset)
        return pos_b + r_b, quat_b, r_b

    def _jacobian_root_frame(self, r_b: torch.Tensor | None) -> torch.Tensor:
        """Geometric Jacobian of the control point in the root frame, (N, 6, D)."""
        jac_w = self._asset.data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        rot = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w.torch))
        jac = torch.empty_like(jac_w)
        jac[:, :3, :] = torch.bmm(rot, jac_w[:, :3, :])
        jac[:, 3:, :] = torch.bmm(rot, jac_w[:, 3:, :])
        if r_b is not None:  # shift to the TCP: v_tcp = v − r × ω
            jac[:, :3] -= torch.cross(r_b.unsqueeze(-1).expand_as(jac[:, 3:]), jac[:, 3:], dim=1)
        return jac

    def _read_wrench(self) -> torch.Tensor:
        """Wrench as an F/T sensor would report it: environment on EE, EE frame, about the TCP."""
        if self._wrench_src is None:
            self._wrench_src = self._resolve_wrench_source()
        w = cm.rotate_wrench(self._wrench_rot, self._wrench_sign * self._wrench_src.read())
        return torch.cat([w[:, :3], w[:, 3:] - torch.cross(self._wrench_lever, w[:, :3], dim=-1)], dim=-1)

    def _resolve_wrench_source(self):
        """Resolve the reporting joint frame and its constant transform to the TCP.
        The reporting link is rigidly attached to the end-effector. Joint-frame
        rotation and anchor position determine force rotation and torque transport."""
        cfg = self.cfg
        want = cfg.wrench_frame_body or self._ee_body_name
        sensor = self._env.scene.sensors[cfg.wrench_sensor_name]
        src = SensorWrench(sensor, want)
        if src.fell_back:
            print(f"[CompliantCartesianAction] wrench read at '{src.body_name}'")

        data = self._asset.data
        ids, _ = self._asset.find_bodies(src.body_name)
        q_ee_w, p_ee_w = data.body_quat_w.torch[:, self._ee_body_idx], data.body_pos_w.torch[:, self._ee_body_idx]
        q_body_w, p_body_w = data.body_quat_w.torch[:, ids[0]], data.body_pos_w.torch[:, ids[0]]
        X_c = src.joint_frame()
        if X_c is None:
            # The backend does not say where the joint frame is: take the config's word for the
            # anchor and the link frame for the orientation.
            p_joint_l = torch.as_tensor(cfg.wrench_ref_offset, device=self.device, dtype=torch.float32).expand(
                self.num_envs, 3)
            q_joint_l = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, 4)
        else:
            p_joint_l, q_joint_l = X_c[:, :3], X_c[:, 3:]
        q_joint_w = math_utils.quat_mul(q_body_w, q_joint_l)
        p_joint_w = p_body_w + math_utils.quat_apply(q_body_w, p_joint_l)
        q_ee_w_inv = math_utils.quat_inv(q_ee_w)
        self._wrench_rot = math_utils.convert_quat(math_utils.quat_mul(q_ee_w_inv, q_joint_w), to="wxyz")
        p_anchor_ee = math_utils.quat_apply(q_ee_w_inv, p_joint_w - p_ee_w)
        tcp = torch.as_tensor(cfg.tcp_offset, device=self.device, dtype=torch.float32)
        self._wrench_lever = tcp.unsqueeze(0) - p_anchor_ee
        return src


@configclass
class CompliantCartesianActionCfg(ActionTermCfg):
    """Controller gains, TCP geometry, wrench feedback and action-mode settings."""

    class_type: type[ActionTerm] = CompliantCartesianAction

    joint_names: list[str] = ["panda_joint.*"]
    body_name: str = "panda_hand"

    tcp_offset: list[float] = [0.0, 0.0, 0.1034]
    """TCP offset in the controlled body frame [m]."""
    motion_stiffness: list[float] = [300.0, 300.0, 300.0, 30.0, 30.0, 30.0]
    motion_damping: list[float] | None = [10.0, 10.0, 10.0, 3.0, 3.0, 3.0]
    motion_damping_ratio: list[float] | None = None
    v_max: list[float] = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5]
    f_max: list[float] = [25.0, 25.0, 25.0, 5.0, 5.0, 5.0]
    torque_limit: list[float] | None = None
    """Joint torque clamp; ``None`` takes the asset's effort limits."""

    coupling: str = "norm"
    """Norm couples each force/torque block; elementwise gates each axis independently."""

    wrench_lpf_tau: float = 0.015
    """Wrench low-pass time constant [s]."""
    ee_vel_filter_tau: float = 0.008

    nullspace_stiffness: float = 5.0
    nullspace_damping_ratio: float = 1.0
    nullspace_ref_tau: float = 2.0
    joint_damping: float = 2.0
    """Viscous joint damping coefficient."""
    gravity_compensation: bool | str = "auto"
    """Add articulation gravity feed-forward. Auto disables it on PhysX and enables
    it on Newton; an explicit boolean overrides that selection."""

    wrench_sensor_name: str = "ee_wrench"
    """Name of the scene JointWrenchSensor used for feedback."""
    wrench_frame_body: str | list[str] | None = None
    """Body whose incoming joint wrench to read, or an ordered candidate list (most distal
    first); defaults to ``body_name``. A sensor only covers bodies that have an incoming joint,
    and that set is backend-dependent: PhysX reports ``panda_hand``, MuJoCo-Warp merges it into
    ``panda_link7``. Give both and the first covered one is used."""
    wrench_ref_offset: list[float] = [0.0, 0.0, 0.0]
    """Reporting joint anchor in the reporting body frame when the sensor cannot
    provide its joint-frame transform."""
    wrench_sign: float = -1.0
    """Convert joint reaction to the environment-on-end-effector wrench convention."""

    action_mode: str = "hold"
    """``"hold"`` ignores the action vector (scripted primitives drive ``set_goal_pose``);
    ``"delta"`` anchors a policy delta at the equilibrium pose."""
    scale: list[float] = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
