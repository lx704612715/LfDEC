# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared tensor math for poses, wrench transforms, interpolation and joint torques.

Functions are batched over environments and independent of Isaac Lab. Quaternions
use wxyz order; pose deltas contain translation followed by axis-angle rotation."""

from __future__ import annotations

import torch

# ----------------------------------------------------------------------------- quaternions


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion (its conjugate)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (N, 3) by unit quaternions ``q`` (N, 4)."""
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_inv(q), v)


def axis_angle_to_quat(aa: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rotation vector (N, 3) -> unit quaternion (N, 4)."""
    angle = torch.norm(aa, dim=-1, keepdim=True)
    scale = torch.where(angle > eps, torch.sin(0.5 * angle) / angle.clamp(min=eps),
                        0.5 - angle.square() / 48.0)
    return torch.cat([torch.cos(0.5 * angle), aa * scale], dim=-1)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit quaternion (N, 4) -> rotation vector (N, 3) with angle in [0, pi]."""
    q = torch.where(q[..., :1] < 0.0, -q, q)
    sin_half = torch.norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half > eps, angle / sin_half.clamp(min=eps), torch.full_like(angle, 2.0))
    return q[..., 1:] * scale


def rotate_wrench(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Rotate both halves of a (N, 6) wrench by ``q``."""
    return torch.cat([quat_apply(q, w[:, :3]), quat_apply(q, w[:, 3:])], dim=-1)


def point_mass_gravity_torques(jacobian: torch.Tensor, gravity: torch.Tensor,
                              mass: float | torch.Tensor, com: torch.Tensor) -> torch.Tensor:
    """Compensate a carried point mass using the TCP Jacobian. Gravity and the TCP-to-COM
    lever are in robot-root coordinates, (N,3); mass is scalar or (N,1). The mass must not
    already be part of the controlled articulation's gravity model."""
    force = -mass * gravity
    wrench = torch.cat([force, torch.cross(com, force, dim=-1)], dim=-1)
    return torch.bmm(jacobian.transpose(1, 2), wrench.unsqueeze(-1)).squeeze(-1)


# ----------------------------------------------------------------------------- filters


def lpf_update(state: torch.Tensor, x: torch.Tensor, dt: float, tau: float) -> torch.Tensor:
    """First-order low-pass, updated in place. ``tau <= 0`` passes ``x`` through."""
    if tau <= 0.0:
        return state.copy_(x)
    return state.lerp_(x, dt / (tau + dt))


# ----------------------------------------------------------------------------- Eq. 1-3


def _blocks(coupling: str) -> tuple[int, int]:
    """Axis grouping for the limits: 6 single axes (paper, box of admissible wrenches) or
    the linear and angular triples (direction-preserving, box-inscribed ellipsoid)."""
    if coupling == "elementwise":
        return 6, 1
    if coupling == "norm":
        return 2, 3
    raise ValueError(f"coupling must be 'norm' or 'elementwise', got {coupling!r}")


def ee_frame_delta(p_des: torch.Tensor, q_des: torch.Tensor, p_goal: torch.Tensor, q_goal: torch.Tensor) -> torch.Tensor:
    """se(3) step from the desired pose to the goal, in the desired-pose frame (N, 6)."""
    dp = quat_apply_inverse(q_des, p_goal - p_des)
    dq = quat_mul(quat_inv(q_des), q_goal)
    w = axis_angle_from_quat(dq)
    theta = torch.norm(w, dim=-1, keepdim=True)
    safe = theta.clamp(min=1e-3)
    A = torch.where(theta > 1e-3, (1.0 - 0.5 * safe / torch.tan(0.5 * safe)) / safe.square(),
                    1.0 / 12.0 + theta.square() / 720.0)
    wxdp = torch.cross(w, dp, dim=-1)
    v = dp - 0.5 * wxdp + A * torch.cross(w, wxdp, dim=-1)
    return torch.cat([v, w], dim=-1)


def clip_twist(dpsi: torch.Tensor, v_step: torch.Tensor, coupling: str = "norm") -> torch.Tensor:
    """Eq. 1: limit the step to ``V_m Δt``. Each block is scaled by one factor, so with
    ``"elementwise"`` this is the paper's per-axis clip and with ``"norm"`` the direction
    of the linear and angular blocks is preserved."""
    b, k = _blocks(coupling)
    x = dpsi.reshape(-1, b, k)
    lim = v_step.expand_as(dpsi).reshape(-1, b, k).clamp(min=1e-12)
    s = (x.abs() / lim).amax(dim=-1, keepdim=True).clamp(min=1.0)
    return (x / s).reshape_as(dpsi)


def wrench_gate(
    dpsi: torch.Tensor, wrench: torch.Tensor, f_max: torch.Tensor, v_step: torch.Tensor, coupling: str = "norm"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 2: ``tanh(|F_m − |F||)`` gate with escape branch, per block.

    Inside the limit the step is scaled by the gate; outside it the step is replaced by an
    escape of ``V_m Δt`` along the force. ``"elementwise"`` gates each axis on its own
    component (paper); ``"norm"`` gates the linear and angular blocks on their norms
    against the box-inscribed ellipsoid, so ``|F|`` never exceeds the limit.

    Returns ``(dpsi_gated, slack)`` with ``slack = F_m − |F|`` broadcast per block.
    """
    b, k = _blocks(coupling)
    x = dpsi.reshape(-1, b, k)
    f = wrench.reshape(-1, b, k)
    fm = f_max.expand_as(wrench).reshape(-1, b, k)
    vs = v_step.expand_as(dpsi).reshape(-1, b, k)
    mag = torch.norm(f, dim=-1, keepdim=True)
    u = torch.where(mag > 1e-9, f / mag.clamp(min=1e-9), torch.full_like(f, k**-0.5))
    limit = ((u / fm) ** 2).sum(dim=-1, keepdim=True).rsqrt()  # ellipsoid radius along u (= F_m for k = 1)
    slack = limit - mag
    g = torch.tanh(slack.abs())
    escape = ((u / vs) ** 2).sum(dim=-1, keepdim=True).rsqrt()
    out = torch.where(slack >= 0.0, g * x, g * escape * u)
    return out.reshape_as(dpsi), slack.expand(-1, b, k).reshape_as(dpsi)


def compose(p_des: torch.Tensor, q_des: torch.Tensor, dpsi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 3: ``T* <- T* exp(Δψ)`` (right-compose in the desired frame)."""
    v, w = dpsi[:, :3], dpsi[:, 3:]
    theta = torch.norm(w, dim=-1, keepdim=True)
    safe = theta.clamp(min=1e-3)
    B = torch.where(theta > 1e-3, (1.0 - torch.cos(safe)) / safe.square(),
                    0.5 - theta.square() / 24.0)
    C = torch.where(theta > 1e-3, (safe - torch.sin(safe)) / safe.pow(3),
                    1.0 / 6.0 - theta.square() / 120.0)
    wxv = torch.cross(w, v, dim=-1)
    dp = v + B * wxv + C * torch.cross(w, wxv, dim=-1)
    p_new = p_des + quat_apply(q_des, dp)
    q_new = quat_mul(q_des, axis_angle_to_quat(dpsi[:, 3:]))
    return p_new, q_new / torch.norm(q_new, dim=-1, keepdim=True)


def compliant_interpolation_step(
    p_des: torch.Tensor,
    q_des: torch.Tensor,
    p_goal: torch.Tensor,
    q_goal: torch.Tensor,
    wrench: torch.Tensor,
    v_max: torch.Tensor,
    f_max: torch.Tensor,
    dt: float,
    gate_on: torch.Tensor,
    coupling: str = "norm",
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One inner-loop tick of Eq. 1-3.

    Args:
        wrench: (N, 6) wrench already expressed in the desired-pose frame.
        gate_on: (N,) bool, False while the F/T bias is being homed.
        active: optional (N, 6) bool; inactive axes (rotational Kp = 0) get no step and no gate.

    Returns:
        ``(p_des, q_des, dpsi, slack)`` after the tick.
    """
    if active is not None:
        # An orientation-free controller must ignore the rotational target before the SE3 log:
        # its inverse Jacobian otherwise changes the translation even after angular masking.
        q_goal = torch.where(active[:, 3:].any(dim=-1, keepdim=True), q_goal, q_des)
    dpsi = ee_frame_delta(p_des, q_des, p_goal, q_goal)
    mask = gate_on.unsqueeze(-1).expand_as(dpsi)
    if active is not None:
        dpsi = dpsi * active
        mask = mask & active
    v_step = v_max * dt
    dpsi = clip_twist(dpsi, v_step, coupling)
    gated, slack = wrench_gate(dpsi, wrench, f_max, v_step, coupling)
    dpsi = torch.where(mask, gated, dpsi)
    p_new, q_new = compose(p_des, q_des, dpsi)
    return p_new, q_new, dpsi, slack


# ----------------------------------------------------------------------------- Eq. 4 extras


def nullspace_torques(
    jacobian: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_pos_ref: torch.Tensor,
    kp: float,
    kd: float,
    mass_matrix: torch.Tensor,
    reg: float = 1e-6,
) -> torch.Tensor:
    """Dynamically consistent null-space posture PD (Khatib) for redundant arms.

    ``τ = Nᵀ M a`` with ``a = kp (q_ref − q) − kd q̇`` and ``Nᵀ = I − Jᵀ Λ J M⁻¹``,
    ``Λ = (J M⁻¹ Jᵀ + reg I)⁻¹``. The task-space acceleration ``J M⁻¹ τ`` is zero, so the
    posture control adds no static task error. Returns zeros for arms with <= 6 joints.
    """
    n = jacobian.shape[-1]
    if n <= 6 or (kp <= 0.0 and kd <= 0.0):
        return torch.zeros_like(joint_pos)
    a = (kp * (joint_pos_ref - joint_pos) - kd * joint_vel).unsqueeze(-1)  # (N, n, 1)
    jt = jacobian.transpose(-1, -2)
    lam_inv = jacobian @ torch.linalg.solve(mass_matrix, jt)  # J M⁻¹ Jᵀ
    lam_inv.diagonal(dim1=-2, dim2=-1).add_(reg)
    return (mass_matrix @ a - jt @ torch.linalg.solve(lam_inv, jacobian @ a)).squeeze(-1)


def joint_damping_torques(joint_vel: torch.Tensor, damping: float) -> torch.Tensor:
    """Viscous joint damping ``τ = −D q̇`` (stands in for the real arm's joint friction)."""
    return -damping * joint_vel if damping > 0.0 else torch.zeros_like(joint_vel)


def pose_error(
    p: torch.Tensor, q: torch.Tensor, p_des: torch.Tensor, q_des: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position and axis-angle orientation error of a pose against its target (both (N, 3)).

    Isaac-free equivalent of ``isaaclab.utils.math.compute_pose_error(..., "axis_angle")``:
    the shortest-path branch lives in :func:`axis_angle_from_quat`.
    """
    return p_des - p, axis_angle_from_quat(quat_mul(q_des, quat_inv(q)))
