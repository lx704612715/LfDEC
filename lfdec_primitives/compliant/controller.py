# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Compliant Cartesian control: interpolate the equilibrium pose and track it with joint torques.

Inputs use the robot root frame and wxyz quaternions. Wrenches use the end-effector
frame at the TCP, with the environment-on-robot sign convention. Goals are absolute.
Calibrate the wrench bias with the robot at rest and out of contact before execution."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from ..utils import math as cm

__all__ = ["CompliantCartesianController"]

_Index = Sequence[int] | slice | torch.Tensor | None


def _idx(env_ids: _Index) -> Sequence[int] | slice | torch.Tensor:
    return slice(None) if env_ids is None else env_ids


class CompliantCartesianController:
    """Stateful compliant Cartesian controller for ``num_envs`` parallel arms.

    Args:
        num_envs: number of parallel environments.
        num_dof: controlled joints per environment.
        device: torch device.
        dt: physics timestep [s] -- the rate :meth:`step` is called at.
        kp: (6,) task-space stiffness [N/m, N·m/rad]. Zero rotational stiffness selects
            orientation-free mode: ``q_des`` follows the measured orientation and the
            rotational channel is neither stepped nor gated (paper Table 1 uses this).
        kv: (6,) task-space damping. Defaults to ``2 sqrt(kp) * damping_ratio``.
        v_max: (6,) twist limit [m/s, rad/s] (Eq. 1).
        f_max: (6,) wrench limit [N, N·m] (Eq. 2).
        torque_limit: (num_dof,) joint torque clamp [N·m].
        coupling: ``"norm"`` treats the linear and angular triples as blocks, so ``|F|`` never
            exceeds ``F_m``; ``"elementwise"`` is the paper verbatim and admits ``sqrt(3) F_m``
            on a diagonal constraint force.
        wrench_lpf_tau: wrench low-pass time constant [s]. Mandatory in sim: the raw joint
            reaction wrench carries several N of solver chatter, and an unfiltered signal makes
            the gate flip every tick.
        ee_vel_filter_tau: EE twist EMA time constant [s] for the ``Kv`` term (0 disables).
        nullspace_stiffness / nullspace_damping_ratio: dynamically consistent posture PD for
            redundant arms (no-op at <= 6 DoF).
        nullspace_ref_tau: posture reference tracking time constant [s]; 0 pins it at reset.
        joint_damping: viscous joint damping [N·m·s/rad], standing in for real joint friction.
    """

    def __init__(
        self,
        num_envs: int,
        num_dof: int,
        device: torch.device | str,
        dt: float,
        *,
        kp: Sequence[float],
        kv: Sequence[float] | None = None,
        damping_ratio: Sequence[float] | None = None,
        v_max: Sequence[float],
        f_max: Sequence[float],
        torque_limit: Sequence[float],
        coupling: str = "norm",
        wrench_lpf_tau: float = 0.015,
        ee_vel_filter_tau: float = 0.008,
        nullspace_stiffness: float = 0.0,
        nullspace_damping_ratio: float = 1.0,
        nullspace_ref_tau: float = 2.0,
        joint_damping: float = 0.0,
    ):
        self.num_envs, self.num_dof, self.device, self.dt = num_envs, num_dof, torch.device(device), float(dt)
        self.coupling = coupling
        self.wrench_lpf_tau = wrench_lpf_tau
        self.ee_vel_filter_tau = ee_vel_filter_tau
        self.nullspace_stiffness = nullspace_stiffness
        self.nullspace_damping = 2.0 * nullspace_stiffness**0.5 * nullspace_damping_ratio
        self.nullspace_ref_tau = nullspace_ref_tau
        self.joint_damping = joint_damping

        def vec(x: Sequence[float], n: int) -> torch.Tensor:
            t = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            if t.numel() != n:
                raise ValueError(f"expected {n} elements, got {t.numel()}")
            return t.unsqueeze(0).repeat(num_envs, 1)

        self._kp = vec(kp, 6)
        if kv is not None:
            self._kv = vec(kv, 6)
        else:
            ratio = vec(damping_ratio if damping_ratio is not None else [1.0] * 6, 6)
            self._kv = 2.0 * torch.sqrt(self._kp) * ratio
        self._v_max = vec(v_max, 6)
        self._f_max = vec(f_max, 6)
        self._torque_max = torch.as_tensor(torque_limit, device=self.device, dtype=torch.float32)

        z6 = lambda: torch.zeros(num_envs, 6, device=self.device)  # noqa: E731
        self._p_des, self._q_des = torch.zeros(num_envs, 3, device=self.device), torch.zeros(num_envs, 4, device=self.device)
        self._q_des[:, 0] = 1.0
        self._goal_pos, self._goal_quat = self._p_des.clone(), self._q_des.clone()
        self._F_filt, self._F_bias, self._slack = z6(), z6(), z6()
        self._ee_vel_filt = z6()
        self._q_ref = torch.zeros(num_envs, num_dof, device=self.device)

        self._homing_left = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._homing_acc = z6()
        self._homed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._homing_ticks = 1
        self._homing_pending = 0  # host-side countdown; idle ticks skip the branch without a device sync
        self._update_active_axes()

    # ------------------------------------------------------------------ state

    @property
    def desired_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The equilibrium pose ``T*`` the PD tracks, root frame."""
        return self._p_des, self._q_des

    @property
    def wrench(self) -> torch.Tensor:
        """Bias-corrected, low-passed wrench the gate sees (N, 6)."""
        return self._F_filt

    @property
    def wrench_bias(self) -> torch.Tensor:
        return self._F_bias

    @property
    def slack(self) -> torch.Tensor:
        """``F_m − |F|`` per axis at the last tick; negative means the escape branch is active."""
        return self._slack

    @property
    def homed(self) -> torch.Tensor:
        return self._homed

    @property
    def gate_armed(self) -> torch.Tensor:
        return self._homed & (self._homing_left == 0)

    @property
    def gains(self) -> dict[str, torch.Tensor]:
        return {"kp": self._kp, "kv": self._kv, "v_max": self._v_max, "f_max": self._f_max}

    # ------------------------------------------------------------------ commands

    def set_goal_pose(
        self, pos: torch.Tensor, quat: torch.Tensor | None = None, env_ids: _Index = None
    ) -> None:
        """Latch an absolute goal pose in the root frame (the paper's ``T_BG``)."""
        i = _idx(env_ids)
        self._goal_pos[i] = pos.to(self.device, torch.float32)
        if quat is not None:
            self._goal_quat[i] = quat.to(self.device, torch.float32)

    def set_gains(
        self,
        kp: torch.Tensor | Sequence[float] | None = None,
        kv: torch.Tensor | Sequence[float] | None = None,
        v_max: torch.Tensor | Sequence[float] | None = None,
        f_max: torch.Tensor | Sequence[float] | None = None,
        env_ids: _Index = None,
    ) -> None:
        """Override any of Kp / Kv / V_m / F_m (each (6,) or (n, 6)) for some or all envs."""
        i = _idx(env_ids)
        for buf, val in ((self._kp, kp), (self._kv, kv), (self._v_max, v_max), (self._f_max, f_max)):
            if val is not None:
                buf[i] = torch.as_tensor(val, device=self.device, dtype=torch.float32)
        if kp is not None:
            self._update_active_axes()

    def start_homing(self, num_ticks: int = 50, env_ids: _Index = None) -> None:
        """Average the raw wrench over the next ``num_ticks`` ticks into the bias.

        Hold the arm at rest and out of contact meanwhile. The gate is off during homing and
        armed when it completes. Re-home after a large re-orientation if the arm has gravity on,
        and home with the gripper in the pose it will hold, or the grasp preload enters the bias.
        """
        i = _idx(env_ids)
        self._homing_left[i] = num_ticks
        self._homing_acc[i] = 0.0
        self._homing_ticks = num_ticks
        self._homing_pending = max(self._homing_pending, num_ticks)

    def reseat_equilibrium_pose(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor,
                               env_ids: _Index = None) -> None:
        """Discard a previous primitive's pose lead without erasing wrench calibration,
        velocity filtering or posture stabilization. Quaternions use (w,x,y,z)."""
        i = _idx(env_ids)
        self._p_des[i] = ee_pos.to(self._p_des)
        self._q_des[i] = ee_quat.to(self._q_des)
        self._goal_pos[i] = self._p_des[i]
        self._goal_quat[i] = self._q_des[i]

    def reset(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, joint_pos: torch.Tensor, env_ids: _Index = None
    ) -> None:
        """Re-seat the equilibrium pose and goal on the measured pose and drop all filter state."""
        i = _idx(env_ids)
        self._p_des[i] = ee_pos[i].to(self._p_des)
        self._q_des[i] = ee_quat[i].to(self._q_des)
        self._goal_pos[i] = self._p_des[i]
        self._goal_quat[i] = self._q_des[i]
        self._q_ref[i] = joint_pos[i].to(self._q_ref)
        for buf in (self._F_filt, self._F_bias, self._homing_acc, self._ee_vel_filt, self._slack):
            buf[i] = 0.0
        self._homing_left[i] = 0
        self._homed[i] = False

    # ------------------------------------------------------------------ the tick

    def step(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        wrench: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        gravity_ff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One physics tick: advance ``T*`` (Eq. 1-3) and return the tracking torques (Eq. 4).

        Args:
            ee_pos / ee_quat: (N, 3) / (N, 4) control-point pose in the root frame.
            jacobian: (N, 6, D) geometric Jacobian of the control point, root frame.
            joint_pos / joint_vel: (N, D).
            wrench: (N, 6) measured wrench in **F/T-sensor convention** (environment on
                end-effector), expressed in the EE frame about the control point.
            mass_matrix: (N, D, D), required only when the null-space term is enabled.
            gravity_ff: (N, D) gravity compensation feed-forward, added if given.

        Returns:
            (N, D) joint torques, clamped to ``torque_limit``.
        """
        dt = self.dt

        # 1) wrench: homing average -> bias, then low-pass
        if self._homing_pending > 0:
            self._homing_pending -= 1
            homing = self._homing_left > 0
            self._homing_acc += torch.where(homing.unsqueeze(-1), wrench, 0.0)
            self._homing_left -= homing.long()
            done = (homing & (self._homing_left == 0)).unsqueeze(-1)
            self._F_bias = torch.where(done, self._homing_acc / self._homing_ticks, self._F_bias)
            self._F_filt.masked_fill_(done, 0.0)
            self._homed |= done.squeeze(-1)
        cm.lpf_update(self._F_filt, wrench - self._F_bias, dt, self.wrench_lpf_tau)

        # 2) Eq. 1-3 in the equilibrium-pose frame; orientation-free envs track the measured quat
        q_des = torch.where(self._rot_free.unsqueeze(-1), ee_quat, self._q_des)
        F_gate = cm.rotate_wrench(cm.quat_mul(cm.quat_inv(q_des), ee_quat), self._F_filt)
        p_new, q_new, _, self._slack = cm.compliant_interpolation_step(
            self._p_des, q_des, self._goal_pos, self._goal_quat, F_gate,
            self._v_max, self._f_max, dt, self.gate_armed, self.coupling, self._active,
        )
        p_new, q_new = self._clamp_pose_lag(p_new, q_new, ee_pos, ee_quat)
        self._p_des.copy_(p_new)
        self._q_des.copy_(q_new)

        # 3) Eq. 4: tau = J^T (Kp e − Kv v) + null-space + joint damping + gravity
        ee_vel = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)
        if self.ee_vel_filter_tau > 0.0:
            ee_vel = cm.lpf_update(self._ee_vel_filt, ee_vel, dt, self.ee_vel_filter_tau)
        pos_err, rot_err = cm.pose_error(ee_pos, ee_quat, self._p_des, self._q_des)
        task_force = self._kp * torch.cat([pos_err, rot_err], dim=-1) - self._kv * ee_vel
        tau = torch.bmm(jacobian.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)

        if self.nullspace_stiffness > 0.0:
            if mass_matrix is None:
                raise ValueError("null-space control needs the mass matrix")
            if self.nullspace_ref_tau > 0.0:
                cm.lpf_update(self._q_ref, joint_pos, dt, self.nullspace_ref_tau)
            tau = tau + cm.nullspace_torques(
                jacobian, joint_pos, joint_vel, self._q_ref,
                self.nullspace_stiffness, self.nullspace_damping, mass_matrix,
            )
        if self.joint_damping > 0.0:
            tau = tau + cm.joint_damping_torques(joint_vel, self.joint_damping)
        if gravity_ff is not None:
            tau = tau + gravity_ff
        return torch.clamp(tau, -self._torque_max, self._torque_max)

    # ------------------------------------------------------------------ internals

    def _clamp_pose_lag(
        self, p_des: torch.Tensor, q_des: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Limit equilibrium-pose lag using the smallest F_m / K_p ratio in each
        translation/rotation block. This bounds the elastic K_p term; damping and
        feed-forward terms are applied separately. Zero-stiffness axes do not constrain lag."""
        eps = 1e-9
        lin_lim = (self._f_max[:, :3] / self._kp[:, :3].clamp(min=eps)).amin(dim=-1, keepdim=True)
        d = p_des - ee_pos
        n = torch.norm(d, dim=-1, keepdim=True)
        p_out = torch.where(n > lin_lim, ee_pos + d * (lin_lim / n.clamp(min=eps)), p_des)

        rot_lim = (self._f_max[:, 3:] / self._kp[:, 3:].clamp(min=eps)).amin(dim=-1, keepdim=True)
        aa = cm.axis_angle_from_quat(cm.quat_mul(q_des, cm.quat_inv(ee_quat)))
        ang = torch.norm(aa, dim=-1, keepdim=True)
        q_clamped = cm.quat_mul(cm.axis_angle_to_quat(aa * (rot_lim / ang.clamp(min=eps))), ee_quat)
        q_out = torch.where(ang > rot_lim, q_clamped, q_des)
        return p_out, q_out

    def _update_active_axes(self) -> None:
        """Envs with zero rotational stiffness are orientation-free: rotational axes inactive."""
        self._rot_free = (self._kp[:, 3:] <= 0.0).all(dim=-1)
        self._active = torch.ones(self.num_envs, 6, dtype=torch.bool, device=self.device)
        self._active[:, 3:] = ~self._rot_free.unsqueeze(-1)
