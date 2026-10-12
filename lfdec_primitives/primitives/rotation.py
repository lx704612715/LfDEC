# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Revolute and screw primitives with online screw-axis estimation.

Each iteration samples a motion, executes it, observes the displacement and updates
the EKF. Axis and point are refined online; demonstrated screw pitch remains fixed."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import (
    PAPER_BCE_DRAWER,
    PAPER_MCE_T,
    PAPER_MIN_ROT,
    PAPER_MIN_TRAN,
    ECState,
    Demo,
    Primitive,
    RobotLike,
    angle_between,
    clamp_to_cone,
    make_transform,
    matrix_from_rotvec,
    point_line_distance,
    point_twist,
    project_point_to_axis,
    project_to_plane,
    rotation_taking,
    rotvec_from_matrix,
    screw_transform,
    se3_log,
    spherical_from_vector,
    transform_inv,
    unit,
    vector_from_spherical,
)

_AZI, _ELE, _PT, _THETA, _RATE = 0, 1, slice(2, 5), 5, 6


# ----------------------------------------------------------------------------- screw EKF

@dataclass
class ScrewCfg:
    """Noise model of :class:`ScrewEKF`. Every entry is a **variance**, not a standard deviation."""

    pitch: float = 0.0
    """[m/rad] advance along the axis per radian. 0 is a plain revolute joint. Fixed, not
    estimated online: the demonstration compiler learns it from measured screw motion.
    Diagnostics may instead supply the nominal part pitch."""

    init_axis: float = 0.04       # (0.2 rad)^2 around the initial guess
    init_point: float = 0.01      # (0.1 m)^2
    init_theta: float = 2.5e-3    # (0.05 rad)^2
    init_rate: float = 0.25       # (0.5 rad/s)^2

    q_axis: float = 2.0e-4        # process noise per second; the axis is fixed, so this is only
    q_point: float = 2.0e-4       # enough to keep the filter from locking onto a wrong estimate
    q_theta: float = 1.0e-4
    q_rate: float = 1.0

    r_trans: float = 1.0e-5
    """(3.2 mm)^2 on the exponential-coordinate translation. The *position* of a grasped point is
    rigidly tied to the constrained body whatever the grasp does, so this really is sensor noise."""

    r_rot: float = 10.0
    """Rotational observation variance [rad^2]. Increase it for grasps that allow
    hand orientation to slip relative to the constrained object."""

    stall_axis: float = 2.0e-4
    """Variance added to the axis block on an iteration that produced no motion. A stall carries no
    information about the axis but does mean the current estimate is not working, so the filter is
    re-opened rather than left over-confident -- the reference does the same."""

    sample_std: float = 0.01
    """[rad] dither floor on the executed axis. The primitive executes a *sample* of the posterior,
    as the paper's algorithm does; once the covariance has collapsed this keeps a little excitation
    in the command, which is what makes the estimate identifiable at all."""

    reanchor_theta: float = 2.0
    """[rad] re-anchor once the estimated angle passes this. ``se3_log`` is 2*pi-ambiguous and
    ill-conditioned near pi, so the observation must never be allowed to get there."""


class ScrewEKF:
    """EKF over a screw axis in the robot base frame.

    State ``[azi, ele, px, py, pz, theta, rate]`` in an estimation frame rotated so the initial
    axis guess is +x. Observation: the grasped point's displacement and the hand's rotation vector
    since the anchor pose -- see :meth:`_h`.
    """

    def __init__(self, axis: np.ndarray, point: np.ndarray, cfg: ScrewCfg,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.R0 = rotation_taking(unit(np.asarray(axis, dtype=float)), np.array([1.0, 0.0, 0.0]))
        azi, ele = spherical_from_vector(self.R0 @ unit(np.asarray(axis, dtype=float)))
        self.x = np.zeros(7)
        self.x[_AZI], self.x[_ELE] = azi, ele
        self.x[_PT] = self.R0 @ np.asarray(point, dtype=float)
        self.P = np.diag([cfg.init_axis, cfg.init_axis,
                          cfg.init_point, cfg.init_point, cfg.init_point,
                          cfg.init_theta, cfg.init_rate])
        self.R = np.diag([cfg.r_trans] * 3 + [cfg.r_rot] * 3)
        self.T_anchor = np.eye(4)
        self._p_anchor = np.zeros(3)   # EE position at anchor time, estimation frame (gauge)
        self.theta_total = 0.0
        self.n_updates = 0
        self.n_reanchors = 0

    # -- frames ------------------------------------------------------------------
    @property
    def pitch(self) -> float:
        return self.cfg.pitch

    @property
    def axis(self) -> np.ndarray:
        """Estimated axis direction, base frame."""
        return self.R0.T @ vector_from_spherical(self.x[_AZI], self.x[_ELE])

    @property
    def point(self) -> np.ndarray:
        """Estimated point on the axis, base frame."""
        return self.R0.T @ self.x[_PT]

    @property
    def theta(self) -> float:
        """Estimated angle since the current anchor [rad]."""
        return float(self.x[_THETA])

    @property
    def angle(self) -> float:
        """Estimated angle since the *first* anchor [rad], across re-anchorings."""
        return self.theta_total + float(self.x[_THETA])

    def axis_std(self) -> float:
        """[rad] one-sigma spread of the axis direction.

        The two angles are combined as if orthonormal, which holds because the estimation frame
        keeps ``ele`` near pi/2 where ``d(axis)/d(azi)`` has unit length.
        """
        return float(math.sqrt(max(self.P[_AZI, _AZI] + self.P[_ELE, _ELE], 0.0)))

    def point_std(self) -> float:
        """[m] one-sigma spread of the axis position, ignoring the unobservable along-axis mode."""
        u = vector_from_spherical(self.x[_AZI], self.x[_ELE])
        perp = np.eye(3) - np.outer(u, u)
        return float(math.sqrt(max(np.trace(perp @ self.P[_PT, _PT] @ perp), 0.0)))

    # -- model -------------------------------------------------------------------
    def _h(self, x: np.ndarray) -> np.ndarray:
        """Predicted ``[displacement of the grasped point, rotation vector]`` since the anchor.

        Deliberately *not* the exponential coordinates of the relative pose, which is what the
        reference observes. Those compute the translation through the observed rotation (the
        ``V^-1`` factor of :func:`~.base.se3_log`), so a grasp that lets the hand roll corrupts both
        halves and no amount of loosening ``r_rot`` can isolate the clean one. Here the position
        channel is the rigid-body displacement of the grasped point and involves the measured
        orientation nowhere, so the two really are separable -- which is the whole reason
        ``r_rot >> r_trans`` buys anything.
        """
        u = vector_from_spherical(x[_AZI], x[_ELE])
        w = x[_THETA] * u
        dp = (matrix_from_rotvec(w) - np.eye(3)) @ (self._p_anchor - x[_PT]) + self.cfg.pitch * w
        return np.concatenate([dp, w])

    def _H(self, x: np.ndarray) -> np.ndarray:
        """Observation Jacobian by central differences -- see the module docstring."""
        H = np.zeros((6, 7))
        for i in range(7):
            step = 1e-6
            dx = np.zeros(7)
            dx[i] = step
            H[:, i] = (self._h(x + dx) - self._h(x - dx)) / (2.0 * step)
        return H

    # -- cycle -------------------------------------------------------------------
    def anchor(self, T: np.ndarray) -> None:
        """Set the reference pose the observation is measured against; ``theta`` restarts at 0."""
        self.theta_total += float(self.x[_THETA])
        self.T_anchor = np.asarray(T, dtype=float).copy()
        self._p_anchor = self.R0 @ self.T_anchor[:3, 3]
        self.x[_THETA] = 0.0
        self.P[_THETA, :] = 0.0
        self.P[:, _THETA] = 0.0
        self.P[_THETA, _THETA] = self.cfg.init_theta

    def predict(self, dt: float) -> None:
        c = self.cfg
        F = np.eye(7)
        F[_THETA, _RATE] = dt
        self.x = F @ self.x
        Q = np.diag([c.q_axis, c.q_axis, c.q_point, c.q_point, c.q_point, c.q_theta, c.q_rate]) * dt
        self.P = F @ self.P @ F.T + Q

    def update(self, T_now: np.ndarray) -> float:
        """One EKF correction from the observed pose. Returns the innovation norm."""
        T_now = np.asarray(T_now, dtype=float)
        z = np.concatenate([
            self.R0 @ (T_now[:3, 3] - self.T_anchor[:3, 3]),
            self.R0 @ rotvec_from_matrix(T_now[:3, :3] @ self.T_anchor[:3, :3].T),
        ])
        H = self._H(self.x)
        y = z - self._h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        A = np.eye(7) - K @ H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T   # Joseph form: stays symmetric positive
        self._sanitize()
        self.n_updates += 1
        if abs(self.x[_THETA]) >= self.cfg.reanchor_theta:
            self.anchor(T_now)
            self.n_reanchors += 1
        return float(np.linalg.norm(y))

    def stalled(self) -> None:
        """Nothing moved this iteration: re-open the axis instead of staying over-confident."""
        self.P[_AZI, _AZI] += self.cfg.stall_axis
        self.P[_ELE, _ELE] += self.cfg.stall_axis

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw an axis from the posterior (with the dither floor); the point is taken at the mean."""
        cov = self.P[:2, :2] + self.cfg.sample_std**2 * np.eye(2)
        azi, ele = self.rng.multivariate_normal(self.x[:2], 0.5 * (cov + cov.T))
        return self.R0.T @ vector_from_spherical(azi, ele), self.point

    def _sanitize(self) -> None:
        eps = 1e-3
        self.x[_AZI] = (self.x[_AZI] + math.pi) % (2 * math.pi) - math.pi
        self.x[_ELE] = float(np.clip(self.x[_ELE], eps, math.pi - eps))
        # Sliding the stored point along the axis is unobservable, so pin the gauge at the anchor.
        u = vector_from_spherical(self.x[_AZI], self.x[_ELE])
        self.x[_PT] = project_point_to_axis(u, self.x[_PT], self._p_anchor)
        self.P = 0.5 * (self.P + self.P.T)


# ----------------------------------------------------------------------------- primitive

@dataclass
class RotCfg:
    period: float = 0.5           # s per iteration
    stall_period: float = 0.25    # s per iteration once the EE has stopped moving
    delta_theta: float = 0.25
    """[rad] screw advance commanded ahead of the equilibrium pose each iteration.

    The rotational analogue of ``TransCfg.delta_p``, and it has to clear the same bar: the wrench
    gate stalls the equilibrium pose once the pose lag reaches ``F_m / K_p``, so a goal closer than
    that is reached before the constraint is ever loaded and the primitive learns nothing. With the
    drawer-task gains that is ``5 N.m / 30 = 0.167 rad`` in rotation and, at a 0.35 m radius,
    ``0.05 m / 0.35 = 0.143 rad`` in translation."""

    min_rot: float = PAPER_MIN_ROT      # rad of observed rotation that admits an EKF update
    min_tran: float = PAPER_MIN_TRAN    # m of observed translation that admits one as well
    mce_attempts: int = PAPER_MCE_T
    mce_force_attempts: int = 10
    mce_force_frac: float = 0.8
    """No-move attempts at or above this fraction of ``F_m,rot`` -> ``MakingContact``. Checked on
    the *torque* channel: a revolute EC pushed to its end stop resists with a moment about the
    axis, which at the grasp shows up in the wrench's rotational half plus a lever force."""

    bce_disp: float = PAPER_BCE_DRAWER
    """[m] signed successor-maintenance displacement that constitutes BC. A separate grasp-loss
    check accumulates residual against the latched finite screw motion, preserving gradual
    departure without mistaking normal arc curvature for a broken constraint. These detectors
    depend on a trustworthy prior/posterior; they do not certify a physically retained grasp."""

    bce_latch_std_deg: float = 5.0
    bce_latch_updates: int = 10
    anchor_beta: float = 0.5
    """Per-iteration decay of equilibrium-pose error perpendicular to the screw.
    Position is projected against the tangent; orientation against the axis."""

    follow_orientation: bool = True
    """Follow screw rotation with the hand orientation. Disable for a grasp that
    transmits force while allowing orientation to slip, such as a ball knob."""

    max_allowed_deg: float = 30.0       # executed axis kept inside this cone around the initial guess
    terminate_theta: float | None = None   # rad of screw progress -> Done
    terminate_extra: tuple[str, float] | None = None
    min_reach_margin: float = 0.10
    """[rad] stop with ``JointLimit`` once the arm has only this much screw travel left. See
    ``TransCfg.min_reach_margin``; the unit is radians here because the twist handed to
    :func:`~.base.reach_along` is a screw rate, not a translation."""

    jump_factor: float = 3.0
    early_exit_disp: float = 0.005      # m; end the iteration once the EE moved this far
    eqp_dist: float = 0.015
    grip_settle: float = 0.5


class RotationPrimitive(Primitive):
    """Turn a grasped body along a revolute or screw EC whose axis is only approximately known.

    Each iteration commands a small screw increment about a *sample* of the current axis posterior,
    lets the compliant controller resolve the mismatch into bounded contact wrench, and feeds the
    observed pose back into the EKF.

    The off-screw part of the equilibrium-pose gap is decayed every iteration, as in the
    translation primitive -- see ``RotCfg.anchor_beta`` for why that is what keeps the commanded
    motion on the arc.
    """

    def __init__(self, robot: RobotLike, ekf: ScrewEKF, axis_init: np.ndarray, cfg: RotCfg,
                 direction: float = 1.0, axis_true: np.ndarray | None = None,
                 point_true: np.ndarray | None = None, demo: Demo | None = None):
        super().__init__(robot, cfg, demo)
        self.kj_tracker = ekf
        self.u_init = unit(np.asarray(axis_init, dtype=float))
        # The EKF is handed over already seeded with the demonstration's hinge, so its point at
        # construction is the prior -- the half of a revolute EC parameter that is a position.
        self.c_init = np.asarray(ekf.point, dtype=float).copy()
        self.demo_base_EC_parameters = np.r_[self.u_init, self.c_init]
        self.sign = 1.0 if direction >= 0.0 else -1.0
        self.u_true = None if axis_true is None else unit(np.asarray(axis_true, dtype=float))
        self.c_true = None if point_true is None else np.asarray(point_true, dtype=float)

    def _base_parameters(self, inference_ht_demo=None):
        if inference_ht_demo is None and self.inf_base_EC_parameters is not None:
            return self.kj_tracker.axis, self.kj_tracker.point
        G = np.eye(4) if inference_ht_demo is None else inference_ht_demo
        return (G[:3, :3] @ self.demo_base_EC_parameters[:3],
                G[:3, :3] @ self.demo_base_EC_parameters[3:6] + G[:3, 3])

    def get_curt_constraint_direction(self, curt_base_p_ee, inference_ht_demo=None) -> np.ndarray | None:
        """Toward the estimated axis, perpendicular to it -- the reference's
        ``direction_to_rotation_axis``: the one direction a revolute EC blocks at ``p``."""
        base_r_axis, base_r_center = self._base_parameters(inference_ht_demo)
        d = project_to_plane(base_r_center - np.asarray(curt_base_p_ee, dtype=float), base_r_axis)
        return None if float(np.linalg.norm(d)) < 1e-9 else unit(d)

    def ec_state(self) -> ECState:
        """Each axis drawn at the point on *its own* line closest to the hand, so a wrong guess
        shows up as a line in the wrong place as well as pointing the wrong way -- which is the
        half of a revolute estimate a direction-only arrow would hide."""
        _, p = self.robot.ee_pose()
        directions, origins = {}, {}
        for key, axis, point in (("initial", self.u_init, self.c_init),
                                 ("estimate", self.kj_tracker.axis, self.kj_tracker.point),
                                 ("true", self.u_true, self.c_true)):
            if axis is None or point is None:
                continue
            directions[key] = unit(np.asarray(axis, dtype=float))
            origins[key] = project_point_to_axis(axis, point, p)
        return ECState("revolute", directions, origins, math.degrees(self.kj_tracker.axis_std()))

    def get_curt_admissible_motion_direction(self, curt_base_p_ee, inference_ht_demo=None) -> np.ndarray:
        """The tangent of the circle ``p`` rides, in the direction the primitive turns."""
        base_r_axis, base_r_center = self._base_parameters(inference_ht_demo)
        return unit(self.sign * point_twist(base_r_axis, base_r_center, curt_base_p_ee, self.kj_tracker.pitch)[:3])

    def maintain_contact_action(self, curt_base_ht_ee):
        base_maintain_contact_direction = np.zeros(3)
        if self.next_EC_primitive is not None:
            next_admissible_motion_direction = self.next_EC_primitive.get_curt_admissible_motion_direction(
                curt_base_ht_ee[:3, 3], self.inference_ht_demo)
            if next_admissible_motion_direction is not None:
                next_admissible_motion_direction = np.asarray(next_admissible_motion_direction).reshape(3, -1)[:, 0]
                curt_admissible_motion_direction = self.get_curt_admissible_motion_direction(curt_base_ht_ee[:3, 3])
                base_maintain_contact_direction = (next_admissible_motion_direction -
                    np.dot(next_admissible_motion_direction, curt_admissible_motion_direction) * curt_admissible_motion_direction)
        elif self.demo is not None and self.demo.ee_ft_robot is not None:
            demo_ee_force = self.demo.ee_ft_robot[:, :3].mean(axis=0)
            if np.linalg.norm(demo_ee_force) >= 10.0:
                base_maintain_contact_direction = curt_base_ht_ee[:3, :3] @ (-unit(demo_ee_force))
        # The reference has contradictory ±z fallbacks when no force evidence is available.
        # Omit that unsupported terminal preload instead of inventing a direction.
        return make_transform(np.eye(3), 3.0 * self.cfg.eqp_dist *
                              curt_base_ht_ee[:3, :3].T @ base_maintain_contact_direction)

    def seeking_contact_action(self, curt_base_ht_ee):
        u_s, c_s = self.kj_tracker.sample()
        u_s = clamp_to_cone(u_s, self.u_init, math.radians(self.cfg.max_allowed_deg))
        self.sampled_base_r_axis, self.sampled_base_r_center = u_s, c_s
        R_des, p_des = self.robot.desired_pose()
        d_base_ht_ee = screw_transform(u_s, c_s, self.sign * self.cfg.delta_theta, self.kj_tracker.pitch) @ make_transform(R_des, p_des)
        if not self.cfg.follow_orientation:
            d_base_ht_ee[:3, :3] = R_des
        if self.cfg.anchor_beta > 0.0:
            t_hat = unit(point_twist(u_s, c_s, curt_base_ht_ee[:3, 3], self.kj_tracker.pitch)[:3])
            gap_p = p_des - curt_base_ht_ee[:3, 3]
            d_base_ht_ee[:3, 3] -= self.cfg.anchor_beta * (gap_p - np.dot(gap_p, t_hat) * t_hat)
            if self.cfg.follow_orientation:
                gap_r = rotvec_from_matrix(R_des @ curt_base_ht_ee[:3, :3].T)
                d_base_ht_ee[:3, :3] = matrix_from_rotvec(-self.cfg.anchor_beta *
                    (gap_r - np.dot(gap_r, u_s) * u_s)) @ d_base_ht_ee[:3, :3]
        return d_base_ht_ee

    def update_kj_tracker_state(self, obs_base_ht_ee, curt_timestamp_sec,
                                ee_maintain_contact_direction, vis=False, verbose=False):
        delta_ee_SE3_ee = transform_inv(self.last_kj_update_pose) @ obs_base_ht_ee
        delta_ee_se3_ee = se3_log(delta_ee_SE3_ee)
        translation = np.linalg.norm(delta_ee_se3_ee[:3])
        rotation_angle = np.linalg.norm(delta_ee_se3_ee[3:])
        updated = translation > self.cfg.min_tran or rotation_angle > self.cfg.min_rot
        if updated:
            innov = self.kj_tracker.update(obs_base_ht_ee)
            base_maintain_contact_direction = self._maintain_base_ht_ee[:3, :3] @ ee_maintain_contact_direction[:3, 3]
            norm = np.linalg.norm(base_maintain_contact_direction)
            obs_base_disp_ee = obs_base_ht_ee[:3, 3] - self.last_kj_update_pose[:3, 3]
            self.movement_in_maintain_contact_direction.append(
                0.0 if norm < 1e-9 else float(obs_base_disp_ee @ (base_maintain_contact_direction / norm)))
            self.last_kj_update_pose = obs_base_ht_ee.copy()
            self.last_kj_update_timestamp = curt_timestamp_sec
            self.no_movement_count = 0
            self.inf_base_EC_parameters = np.r_[self.kj_tracker.axis, self.kj_tracker.point]
        else:
            innov = float("nan")
            self.kj_tracker.stalled()
            self.no_movement_count += 1
        return updated, innov

    def check_contact_changing_events(self):
        if self.no_movement_count >= self.cfg.mce_attempts:
            self.event = "MakingContact"
        if self.next_EC_primitive is not None and abs(sum(self.movement_in_maintain_contact_direction)) >= self.cfg.bce_disp:
            self.event = "BreakingContact"

    def _err_axis(self, u: np.ndarray) -> float:
        if self.u_true is None:
            return float("nan")
        a = math.degrees(angle_between(u, self.u_true))
        return min(a, 180.0 - a)   # an axis is a line: antiparallel is the same constraint

    def _err_point(self, c: np.ndarray) -> float:
        if self.u_true is None or self.c_true is None:
            return float("nan")
        return point_line_distance(c, self.u_true, self.c_true)

    def exploitation(self, max_iters: int = 200, verbose: bool = True) -> str:
        c, rb, ekf = self.cfg, self.robot, self.kj_tracker
        self.begin_exploitation()
        self.transfer_demonstration()
        self.u_init = self.inference_ht_demo[:3, :3] @ self.demo_base_EC_parameters[:3]
        self.c_init = (self.inference_ht_demo[:3, :3] @ self.demo_base_EC_parameters[3:6] +
                       self.inference_ht_demo[:3, 3])
        ekf.__init__(self.u_init, self.c_init, ekf.cfg, ekf.rng)
        self.inf_base_EC_parameters = np.r_[self.u_init, self.c_init]
        R_prev, p_prev = rb.ee_pose()
        T_prev = make_transform(R_prev, p_prev)
        ekf.anchor(T_prev)
        self.last_kj_update_pose = T_prev.copy()
        self.last_kj_update_timestamp = rb.t
        self.no_movement_count = 0
        self.movement_in_maintain_contact_direction = []
        no_move, travel = 0, 0.0
        angle0 = ekf.angle
        perp_departure = np.zeros(3)
        latch: tuple[np.ndarray, np.ndarray] | None = None
        for k in range(max_iters):
            period = c.stall_period if no_move > 0 else c.period
            last_timestamp_sec = rb.t
            curt_base_ht_ee = make_transform(*rb.ee_pose())
            self._maintain_base_ht_ee = curt_base_ht_ee
            d_seeking_base_ht_ee = self.seeking_contact_action(curt_base_ht_ee)
            d_maintain_ee_ht_ee = self.maintain_contact_action(curt_base_ht_ee)
            d_base_ht_ee = d_seeking_base_ht_ee @ d_maintain_ee_ht_ee
            u_s, c_s = self.sampled_base_r_axis, self.sampled_base_r_center
            rb.set_goal(d_base_ht_ee[:3, 3], d_base_ht_ee[:3, :3])
            rb.run(period, c.early_exit_disp, p_prev)
            ekf.predict(rb.t - last_timestamp_sec)

            R_now, p_now = rb.ee_pose()
            T_now = make_transform(R_now, p_now)
            d6 = se3_log(T_now @ transform_inv(T_prev))
            d_rot = float(np.linalg.norm(d6[3:]))
            d_tran = float(np.linalg.norm(p_now - p_prev))
            if d_tran > c.jump_factor * rb.v_max_lin * period or d_rot > c.jump_factor * rb.v_max_rot * period:
                self.event = "BreakingContact(jump)"  # moved faster than the equilibrium pose can: grasp lost
                break

            # Observe motion in either direction above the update thresholds.
            # The EKF determines signed screw progress; orientation may slip at the grasp.
            updated, innov = self.update_kj_tracker_state(T_now, rb.t, d_maintain_ee_ht_ee)
            no_move = self.no_movement_count
            u_mean, c_mean = ekf.axis, ekf.point
            travel = self.sign * (ekf.angle - angle0)

            if (latch is None and ekf.n_updates >= c.bce_latch_updates
                    and ekf.axis_std() <= math.radians(c.bce_latch_std_deg)):
                latch = (u_mean.copy(), c_mean.copy())
            if latch is None:
                off_arc = 0.0
            else:
                # A finite circular chord is not its starting tangent. Compare with the finite
                # screw step, inferred from position so a slipping/frozen hand rotation is valid.
                radial_prev = project_to_plane(p_prev - latch[1], latch[0])
                radial_now = project_to_plane(p_now - latch[1], latch[0])
                if min(np.linalg.norm(radial_prev), np.linalg.norm(radial_now)) > 1e-6:
                    delta_r_state = math.atan2(float(latch[0] @ np.cross(radial_prev, radial_now)),
                                              float(radial_prev @ radial_now))
                else:
                    delta_r_state = float(rotvec_from_matrix(R_now @ R_prev.T) @ latch[0])
                d_base_ht_ee = screw_transform(latch[0], latch[1], delta_r_state, ekf.pitch)
                predicted_base_p_ee = d_base_ht_ee[:3, :3] @ p_prev + d_base_ht_ee[:3, 3]
                perp_departure += p_now - predicted_base_p_ee
                off_arc = float(np.linalg.norm(perp_departure))

            F = rb.wrench()
            F_lin, T_rot = float(np.linalg.norm(F[:3])), float(np.linalg.norm(F[3:]))
            twist = self.sign * point_twist(u_mean, c_mean, p_now, ekf.pitch)
            if not c.follow_orientation:
                twist[3:] = 0.0   # the arm only has to carry the grasp along the arc
            reach = rb.reach_margin(twist)
            row = dict(
                iter=k, t=rb.t, ee_x=p_now[0], ee_y=p_now[1], ee_z=p_now[2],
                step_rot=d_rot, step_tran=d_tran, updated=int(updated), no_move=no_move,
                travel=travel, travel_deg=math.degrees(travel), period=period,
                u_s_x=u_s[0], u_s_y=u_s[1], u_s_z=u_s[2],
                mu_x=u_mean[0], mu_y=u_mean[1], mu_z=u_mean[2],
                pt_x=c_mean[0], pt_y=c_mean[1], pt_z=c_mean[2],
                axis_std_deg=math.degrees(ekf.axis_std()), point_std=ekf.point_std(),
                err_axis_deg=self._err_axis(u_mean), err_sample_deg=self._err_axis(u_s),
                err_point=self._err_point(c_mean), theta_est=ekf.angle, innov=innov, off_arc=off_arc,
                F_norm=F_lin, F_x=F[0], F_y=F[1], F_z=F[2],
                T_norm=T_rot, T_x=F[3], T_y=F[4], T_z=F[5],
                reach_margin=reach, **rb.extra(),
            )
            self.rows.append(row)
            if verbose and (k % 5 == 0 or not updated):
                print(f"  it {k:3d} t={rb.t:6.2f}s |dw|={math.degrees(d_rot):5.2f}deg |dp|={d_tran*1e3:5.1f}mm "
                      f"upd={int(updated)} nomove={no_move:2d} turn={math.degrees(travel):6.1f}deg "
                      f"err(axis)={row['err_axis_deg']:5.1f}deg err(pt)={100*row['err_point']:5.1f}cm "
                      f"std={row['axis_std_deg']:4.1f}deg |F|={F_lin:5.1f}N |T|={T_rot:4.1f}Nm "
                      f"off={100*off_arc:4.1f}cm reach={math.degrees(reach):5.1f}deg"
                      + "".join(f" {k_}={v:.3g}" for k_, v in rb.extra().items()))

            if (done := self.terminate_extra_reached(row)) is not None:
                self.event = done
                break
            if c.min_reach_margin > 0.0 and reach <= c.min_reach_margin:
                self.event = "JointLimit"
                break
            self.check_contact_changing_events()
            if self.event != "Running":
                break
            if no_move >= c.mce_force_attempts and T_rot >= c.mce_force_frac * rb.f_max_rot:
                self.event = "MakingContact(force-confirmed)"
                break
            if off_arc > c.bce_disp:
                self.event = "BreakingContact"
                break
            if c.terminate_theta is not None and travel >= c.terminate_theta:
                self.event = "Done"
                break
            T_prev, R_prev, p_prev = T_now, R_now, p_now
        else:
            self.event = "MaxIters"
        if self.cg_event in ("MC", "BC") or self.event.startswith("Done"):
            self.check_grasp_state(verbose)
        return self.event

    def summary(self) -> dict:
        rows = self.rows
        last = rows[-1] if rows else {}
        return dict(
            **super().summary(),
            init_err_deg=self._err_axis(self.u_init),
            final_err_deg=last.get("err_axis_deg", float("nan")),
            best_err_deg=min((r["err_axis_deg"] for r in rows), default=float("nan")),
            final_point_err_cm=100.0 * last.get("err_point", float("nan")),
            best_point_err_cm=100.0 * min((r["err_point"] for r in rows), default=float("nan")),
            final_std_deg=last.get("axis_std_deg", float("nan")),
            iters_to_5deg=next((r["iter"] for r in rows if r["err_axis_deg"] < 5.0), None),
            turn_deg=math.degrees(last.get("travel", 0.0)), n_updates=self.kj_tracker.n_updates,
            F_max_N=max((r["F_norm"] for r in rows), default=0.0),
            T_max_Nm=max((r["T_norm"] for r in rows), default=0.0),
        )

    def plot(self, path: Path | str, title: str, f_max_rot: float, progress_key: str | None = None) -> None:
        if not self.rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.rows
        it = np.array([r["iter"] for r in rows])
        col = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
        fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
        ax[0].plot(it, col("err_axis_deg"), "b-", label="angle(EKF axis, true axis)")
        ax[0].plot(it, col("err_sample_deg"), "c.", ms=3, alpha=.6, label="angle(executed sample, true axis)")
        ax[0].fill_between(it, 0, col("axis_std_deg"), color="grey", alpha=.25, label="EKF axis std")
        ax[0].axhline(5, color="k", ls=":"); ax[0].set_ylabel("[deg]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title(f"rotation primitive — event: {self.event}\n{title}", fontsize=8)
        ax[1].plot(it, 100 * col("err_point"), "b-", label="distance(EKF axis point, true axis) [cm]")
        ax[1].plot(it, 100 * col("point_std"), "grey", alpha=.6, label="EKF point std [cm]")
        ax[1].set_ylabel("[cm]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[2].plot(it, col("travel_deg"), "g-", label="turned along the estimated axis [deg]")
        if progress_key is not None and progress_key in rows[0]:
            ax[2].plot(it, np.degrees(col(progress_key)), "g--", label=f"{progress_key} [deg]")
        ax[2].plot(it, np.degrees(col("step_rot")), "k.-", ms=3, alpha=.6, label="|Δω| per iteration [deg]")
        ax[2].plot(it, col("no_move"), "m-", alpha=.7, label="no-move counter")
        ax[2].set_ylabel("[deg] / count"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        ax[3].plot(it, col("T_norm"), "c-", label="|T| (gate input)")
        ax[3].plot(it, col("F_norm"), "r-", alpha=.6, label="|F|")
        ax[3].plot(it, 100 * col("off_arc"), "b-", alpha=.6, label="off-arc deviation (BCE) [cm]")
        ax[3].axhline(f_max_rot, color="k", ls=":", label="T_m")
        ax[3].set_ylabel("[Nm] / [N] / [cm]"); ax[3].set_xlabel("primitive iteration")
        ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
