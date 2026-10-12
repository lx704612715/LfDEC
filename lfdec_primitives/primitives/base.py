# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Shared numpy machinery for the LfDEC motion primitives (paper §3.2).

Rotations, SE(3) exponential coordinates, the screw and plane geometry the primitives are built
on, the :class:`RobotLike` seam every primitive drives, the :class:`Primitive` base they share
(the reference's ``BasePrimitive``), the :class:`Demo` segment they replay, and CSV helpers. Pure numpy: the unit tests import this on machines with no GPU.

**Twist and wrench ordering is ``[linear(3), angular(3)]`` everywhere in this package**, matching
Isaac Lab's geometric Jacobians and the F/T convention used by the controller. SE(3) exponential
coordinates follow the same order, so a twist, a wrench, a Jacobian column and a pose error can be
projected onto one another without a permutation anywhere.
"""
from __future__ import annotations

import csv
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

PAPER_MCE_T = 30          # sampling attempts without an update -> making-contact event
PAPER_BCE_DRAWER = 0.02   # [m] displacement in a constrained direction -> breaking-contact event
PAPER_MIN_TRAN = 0.003    # [m] observed motion needed for a PF update
PAPER_MIN_ROT = 0.004     # [rad] observed rotation needed for an EKF update
PAPER_BCE_PLANE = 0.008   # [m] displacement along the plane normal since the touch -> breaking contact
PAPER_MCE_FORCE_PLANE = 5.0   # [N] tangential force on the plane -> making contact
# check_transition_to_the_next_EC: push this far into the next EC's constraint direction, for at
# most this long; it is there if the hand moved less than PAPER_VERIFY_DISP and felt PAPER_VERIFY_FORCE.
PAPER_VERIFY_PUSH = 0.06
PAPER_VERIFY_TIME = 3.0
PAPER_VERIFY_DISP = 0.02
PAPER_VERIFY_FORCE = 3.0


# ----------------------------------------------------------------------------- rotations (numpy)

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0)))


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def matrix_from_rotvec(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    K = skew(v / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * K @ K


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    q = quat_from_matrix(R)
    if q[0] < 0.0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-9:
        return 2.0 * q[1:]
    return q[1:] * (2.0 * math.atan2(s, q[0]) / s)


def rotation_taking(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation R with R @ unit(a) = unit(b)."""
    a, b = unit(a), unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:  # antiparallel: half turn about any axis orthogonal to a
        axis = unit(np.cross(a, [1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = unit(np.cross(a, [0.0, 1.0, 0.0]))
        return matrix_from_rotvec(math.pi * axis)
    return matrix_from_rotvec(math.acos(c) * unit(np.cross(a, b)))


def clamp_to_cone(u: np.ndarray, axis: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate ``u`` toward ``axis`` until the angle between them is <= ``max_angle``."""
    if angle_between(u, axis) <= max_angle:
        return unit(u)
    k = np.cross(axis, u)
    if np.linalg.norm(k) < 1e-9:
        return unit(axis)
    return matrix_from_rotvec(max_angle * unit(k)) @ unit(axis)


def rx(roll: float) -> np.ndarray:
    c, s = math.cos(roll), math.sin(roll)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(pitch: float) -> np.ndarray:
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def spherical_from_vector(u: np.ndarray) -> tuple[float, float]:
    """Unit vector -> ``(azimuth, elevation)``, the parameterisation the screw EKF estimates in."""
    u = unit(np.asarray(u, dtype=float))
    return float(math.atan2(u[1], u[0])), float(math.acos(float(np.clip(u[2], -1.0, 1.0))))


def vector_from_spherical(azi: float, ele: float) -> np.ndarray:
    return np.array([math.cos(azi) * math.sin(ele), math.sin(azi) * math.sin(ele), math.cos(ele)])


# ----------------------------------------------------------------------------- SE(3)

def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def transform_inv(T: np.ndarray) -> np.ndarray:
    Ri = T[:3, :3].T
    return make_transform(Ri, -Ri @ T[:3, 3])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential coordinates ``[v(3), w(3)]`` -> homogeneous transform."""
    v, w = np.asarray(xi, dtype=float)[:3], np.asarray(xi, dtype=float)[3:]
    th = float(np.linalg.norm(w))
    R = matrix_from_rotvec(w)
    if th < 1e-8:
        return make_transform(R, v)
    K = skew(w)
    V = np.eye(3) + (1.0 - math.cos(th)) / th**2 * K + (th - math.sin(th)) / th**3 * K @ K
    return make_transform(R, V @ v)


def se3_log(T: np.ndarray) -> np.ndarray:
    """Homogeneous transform -> exponential coordinates ``[v(3), w(3)]``.

    The rotation branch is 2*pi-ambiguous and ill-conditioned near pi, which is why the screw EKF
    re-anchors well before its estimated angle gets there.
    """
    w = rotvec_from_matrix(np.asarray(T, dtype=float)[:3, :3])
    t = np.asarray(T, dtype=float)[:3, 3]
    th = float(np.linalg.norm(w))
    if th < 1e-8:
        return np.concatenate([t, w])
    K = skew(w)
    c = 1.0 / th**2 - (1.0 + math.cos(th)) / (2.0 * th * math.sin(th))
    V_inv = np.eye(3) - 0.5 * K + c * K @ K
    return np.concatenate([V_inv @ t, w])


def adjoint(T: np.ndarray) -> np.ndarray:
    """(6, 6) adjoint of ``T_ab`` for ``[v, w]`` twists: ``xi_a = adjoint(T_ab) @ xi_b``.

    Re-expresses the body twist of one rigid body from frame ``b`` to frame ``a`` on the same
    body -- how a camera twist becomes the twist of the hand that carries it.
    """
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(t) @ R
    A[3:, 3:] = R
    return A


# ----------------------------------------------------------------------------- screw geometry

def screw_twist(axis: np.ndarray, point: np.ndarray, pitch: float = 0.0) -> np.ndarray:
    """Unit-rate spatial twist ``[v(3), w(3)]`` of the screw (``axis`` through ``point``).

    ``pitch`` is metres of advance along the axis per radian of rotation (``lead / 2*pi``), so 0 is
    a plain revolute joint and a positive value is a right-handed screw. Turning the screw by
    ``theta`` displaces the body by ``se3_exp(theta * twist)``.
    """
    u = unit(np.asarray(axis, dtype=float))
    return np.concatenate([np.cross(np.asarray(point, dtype=float), u) + pitch * u, u])


def screw_transform(axis: np.ndarray, point: np.ndarray, theta: float, pitch: float = 0.0) -> np.ndarray:
    """Displacement of a body carried ``theta`` radians along the screw, in the same frame."""
    return se3_exp(theta * screw_twist(axis, point, pitch))


def point_twist(axis: np.ndarray, point: np.ndarray, p_ee: np.ndarray, pitch: float = 0.0) -> np.ndarray:
    """Unit-rate twist ``[v(3), w(3)]`` *of the control point* ``p_ee`` on the screw.

    Same rotation as :func:`screw_twist`, but the linear part is the velocity of ``p_ee`` rather
    than of the frame origin -- which is what a geometric Jacobian of that point returns, so this
    is the vector to hand to :func:`reach_along`.
    """
    u = unit(np.asarray(axis, dtype=float))
    r = np.asarray(p_ee, dtype=float) - np.asarray(point, dtype=float)
    return np.concatenate([np.cross(u, r) + pitch * u, u])


def axis_coordinates(axis: np.ndarray, point: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """``(radius, height)`` of ``p`` in the cylindrical frame of the screw axis.

    Both are invariant while a rigid body rides a revolute joint, so a jump in either is the body
    leaving the constraint -- the breaking-contact test of :class:`~.rotation.RotationPrimitive`.
    """
    u = unit(np.asarray(axis, dtype=float))
    r = np.asarray(p, dtype=float) - np.asarray(point, dtype=float)
    h = float(np.dot(r, u))
    return float(np.linalg.norm(r - h * u)), h


def project_point_to_axis(axis: np.ndarray, point: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The point of the screw axis closest to ``p``.

    Sliding the stored point along the axis changes no observable: ``cross(point, u)`` in
    :func:`screw_twist` annihilates the parallel component. The EKF therefore cannot estimate it
    and re-projects after every correction so that the unobservable direction does not wander.
    """
    u = unit(np.asarray(axis, dtype=float))
    q = np.asarray(point, dtype=float)
    return q + float(np.dot(np.asarray(p, dtype=float) - q, u)) * u


def point_line_distance(p: np.ndarray, axis: np.ndarray, point: np.ndarray) -> float:
    """Perpendicular distance from ``p`` to the line ``(axis, point)`` [m].

    The positional error metric for an estimated revolute axis, and the companion to the angular
    one: two lines can be exactly parallel and still be half a door apart.
    """
    u = unit(np.asarray(axis, dtype=float))
    d = np.asarray(p, dtype=float) - np.asarray(point, dtype=float)
    return float(np.linalg.norm(d - float(np.dot(d, u)) * u))


# ----------------------------------------------------------------------------- plane geometry

def project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Component of ``v`` tangent to the plane with the given normal."""
    n = unit(np.asarray(normal, dtype=float))
    v = np.asarray(v, dtype=float)
    return v - float(np.dot(v, n)) * n


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes ``(e1, e2)`` with ``e1 x e2 = normal``.

    The choice of ``e1`` is arbitrary but has to be *continuous* in ``normal``, because the
    spiral of :class:`~.plane.PlanePrimitive` is regenerated from the live estimate every
    iteration: a basis that flipped when the estimate moved would send the search back over
    ground it had already covered. Seeding from whichever world axis is least parallel to the
    normal gives that.
    """
    n = unit(np.asarray(normal, dtype=float))
    seed = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = unit(np.cross(seed, n))
    return e1, np.cross(n, e1)


# ----------------------------------------------------------------------------- arm reach

def reach_along(jacobian: np.ndarray, q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                twist: np.ndarray) -> float:
    """Units of motion along ``twist`` before some joint reaches its limit.

    ``dq/ds = J^+ twist`` is the joint velocity for a unit rate along the commanded twist, so
    ``headroom_j / |dq_j/ds|`` is how far joint ``j`` allows and the smallest of those is the
    reach. The unit follows the twist: metres for a pure translation (``[u, 0]``), radians for a
    screw (:func:`point_twist`). First-order only -- the Jacobian turns along the path -- but it is
    re-evaluated every iteration and tightens as a limit is approached, which is the direction a
    stopping test wants to err in.

    Args:
        jacobian: (6, n) geometric Jacobian of the control point, ``[linear; angular]`` rows, in
            the same frame as ``twist``.
        q / q_lo / q_hi: (n,) joint positions and their limits.
        twist: (6,) unit-rate ``[v, w]`` of the control point. Not normalised: its magnitude sets
            the unit of the answer.
    """
    dq = np.linalg.pinv(np.asarray(jacobian, dtype=float)) @ np.asarray(twist, dtype=float)
    headroom = np.where(dq > 0.0, np.asarray(q_hi) - np.asarray(q), np.asarray(q) - np.asarray(q_lo))
    return float(np.min(headroom / np.maximum(np.abs(dq), 1e-9)))


def fit_line_direction(points: np.ndarray) -> np.ndarray:
    """Principal axis of a position trajectory, signed from first to last point."""
    P = np.asarray(points, dtype=float)
    _, _, vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
    u = vt[0]
    return unit(-u if np.dot(u, P[-1] - P[0]) < 0 else u)


def fit_plane(points: np.ndarray, init_cond=None, fit_samples: int = 10,
              fit_with_best_n: int = 10, iterations: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Robust plane fitting with the reference's sampling and optional force-filtering inputs.

    Fit unit normals by SVD rather than minimizing an unconstrained plane scale (whose trivial
    zero solution is invalid). Rank sampled models by median geometric residual, then refit
    their consensus. The deterministic sampler makes compiled task parameters reproducible.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 3 or not np.isfinite(P).all():
        raise ValueError("plane fitting needs at least three finite 3D points")
    if init_cond is not None:
        force_direction = unit(np.asarray(init_cond)[:3])
        delta_points = np.diff(P, axis=0)
        norms = np.linalg.norm(delta_points, axis=1)
        keep = np.r_[True, (norms < 1e-10) | (np.abs(delta_points @ force_direction) <=
                                            norms * math.cos(math.radians(65)))]
        if keep.sum() >= min(fit_with_best_n, len(P)):
            P = P[keep]
    def model(X):
        center = X.mean(axis=0)
        _, singular, vt = np.linalg.svd(X - center, full_matrices=False)
        if singular[1] < 1e-9:
            return None
        return vt[-1], center
    best = model(P)
    if best is None:
        raise ValueError("plane fitting needs two independent position directions")
    best_score = float(np.median(np.abs((P - best[1]) @ best[0])))
    rng = np.random.default_rng(0)
    for _ in range(iterations if len(P) > fit_samples else 0):
        params = model(P[rng.choice(len(P), min(fit_samples, len(P)), replace=False)])
        if params is not None:
            score = float(np.median(np.abs((P - params[1]) @ params[0])))
            if score < best_score:
                best, best_score = params, score
    errors = np.abs((P - best[1]) @ best[0])
    robust_X = P[errors <= max(1e-6, 3.0 * 1.4826 * best_score)]
    if len(robust_X) < 3:
        robust_X = P[np.argsort(errors)[:min(fit_with_best_n, len(P))]]
    params = model(robust_X)
    return best if params is None else params


def fit_revolute_axis(poses: np.ndarray, axis_source: str = "position") -> tuple[np.ndarray, np.ndarray]:
    """``(axis, point)`` of the hinge a pose trajectory turned about -- the demonstration's
    revolute EC parameters.

    The reference's default is a position circle within a fitted plane; hand rotation is not
    required. ``axis_source='orientation'`` is an explicit alternative for rigid grasps with
    noisy short position arcs. Both paths sign the axis from the ordered position motion.
    """
    T = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    P = T[:, :3, 3]
    centered = P - P.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 3 or singular[1] < max(1e-6, 1e-3 * singular[0]):
        raise ValueError("revolute learning needs an identifiable position arc to determine its turn")
    if axis_source == "position":
        axis, plane_center = fit_plane(P)
    elif axis_source == "orientation":
        rotvec = rotvec_from_matrix(T[-1, :3, :3] @ T[0, :3, :3].T)
        if np.linalg.norm(rotvec) < math.radians(1.0):
            raise ValueError("orientation-based revolute learning needs measurable hand rotation")
        axis = unit(rotvec)
    else:
        raise ValueError("axis_source must be 'position' or 'orientation'")
    e1, e2 = tangent_basis(axis)
    c0 = plane_center if axis_source == "position" else P.mean(axis=0)
    Q = np.stack([(P - c0) @ e1, (P - c0) @ e2], axis=1)
    # |q - c|^2 = r^2  <=>  2 c . q + (r^2 - |c|^2) = |q|^2, linear in (c, r^2 - |c|^2)
    A = np.concatenate([2.0 * Q, np.ones((Q.shape[0], 1))], axis=1)
    if np.linalg.norm(P - c0) < 1e-5 or np.linalg.matrix_rank(A) < 3:
        raise ValueError("revolute learning needs an identifiable position arc")
    sol = np.linalg.lstsq(A, np.sum(Q * Q, axis=1), rcond=None)[0]
    revolute_parameter = np.r_[sol[:2], np.linalg.norm(Q - sol[:2], axis=1).mean()]
    # Refine the algebraic initialization with the reference's geometric circle residual.
    for _ in range(30 if axis_source == "position" else 0):
        delta = revolute_parameter[:2] - Q
        distances = np.linalg.norm(delta, axis=1).clip(min=1e-12)
        residuals = distances - revolute_parameter[2]
        J = np.column_stack([delta / distances[:, None], -np.ones(len(Q))])
        correction = np.linalg.lstsq(J, -residuals, rcond=None)[0]
        revolute_parameter += correction
        if np.linalg.norm(correction) < 1e-10:
            break
    point = c0 + revolute_parameter[0] * e1 + revolute_parameter[1] * e2
    radial = P - point
    turn = np.arctan2(np.cross(radial[:-1], radial[1:]) @ axis,
                      np.sum(radial[:-1] * radial[1:], axis=1))
    cumulative = np.r_[0., np.cumsum(turn)]
    if np.ptp(cumulative) >= np.pi - 1e-6:
        raise ValueError("revolute learning requires a turn below pi; split longer demonstrations")
    if abs(cumulative[-1]) < math.radians(1.0):
        raise ValueError("revolute learning needs a signed position turn of at least one degree")
    return (-axis if cumulative[-1] < 0 else axis), point


# ----------------------------------------------------------------------------- trajectories

def line_trajectory(T_start: np.ndarray, T_goal: np.ndarray, max_step: float = 0.02,
                    max_rot: float = 0.15) -> list[np.ndarray]:
    """Straight line in position and constant-axis rotation from ``T_start`` (excluded) to
    ``T_goal`` (included), sampled to the given step limits.

    The trajectory a scripted task needs, and the stand-in for a demonstration. Sampling is
    uniform in the interpolation parameter, with the number of waypoints set by whichever of the
    two limits binds; both are *spacings*, not speeds -- how fast the hand travels is the
    controller's ``V_m``. A :class:`Demo` built from it must put ``T_start`` in front, because a
    segment's first pose is the one the transfer is anchored on.
    """
    T_start, T_goal = np.asarray(T_start, dtype=float), np.asarray(T_goal, dtype=float)
    dp = T_goal[:3, 3] - T_start[:3, 3]
    dw = rotvec_from_matrix(T_goal[:3, :3] @ T_start[:3, :3].T)
    n = max(1, math.ceil(max(float(np.linalg.norm(dp)) / max_step,
                                 float(np.linalg.norm(dw)) / max_rot)))
    return [make_transform(matrix_from_rotvec((i / n) * dw) @ T_start[:3, :3],
                           T_start[:3, 3] + (i / n) * dp) for i in range(1, n + 1)]


def project_trajectory(waypoints, T_start: np.ndarray) -> list[np.ndarray]:
    """Re-anchor a recorded trajectory on the pose the hand is in now.

    ``T_i -> T_start (T_0^-1 T_i)``: the reference's ``inference_ht_demo @ demo_base_ht_ee``, which
    keeps the demonstrated *shape* and moves its start onto the current pose. Left-composition, so
    the result is in the base frame directly; the first waypoint comes back as ``T_start``.
    """
    W = [np.asarray(T, dtype=float) for T in waypoints]
    if not W:
        return []
    rel = transform_inv(W[0])
    return [np.asarray(T_start, dtype=float) @ rel @ T for T in W]


# ----------------------------------------------------------------------------- the robot seam

class RobotLike(Protocol):
    """What a primitive needs from a robot. Poses are in the robot base frame.

    IsaacECRobot implements this for Isaac Lab on CPU or GPU. Plant-free test robots implement
    the same contract to check primitive behavior independently of the simulator.
    """

    t: float                 # simulated time [s]
    v_max_lin: float         # translational V_m [m/s]
    v_max_rot: float         # rotational V_m [rad/s]
    f_max_lin: float         # translational F_m [N]
    f_max_rot: float         # rotational F_m [N.m]

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]: ...        # (R, p) of the control point
    def desired_pose(self) -> tuple[np.ndarray, np.ndarray]: ...   # (R, p) of the equilibrium pose T*
    def set_goal(self, p: np.ndarray, R: np.ndarray | None = None) -> None: ...
    """Latch an absolute goal pose. ``R = None`` holds the current desired orientation."""
    def run(self, seconds: float, stop_disp: float, p_ref: np.ndarray) -> None: ...
    """Step the plant; may return early once ``|p - p_ref| >= stop_disp``."""
    def wrench(self) -> np.ndarray: ...                            # (6,) gate input [F, T], EE frame
    def reach_margin(self, twist: np.ndarray) -> float: ...        # see reach_along
    def extra(self) -> dict: ...                                   # diagnostics appended to each row
    def set_gripper(self, closed: bool) -> None: ...               # command the fingers
    def gripper_gap(self) -> float: ...                            # [m] finger opening, per finger


# ----------------------------------------------------------------------------- demonstrations

@dataclass
class View:
    """One RGB-D frame with its intrinsics: what the wrist camera saw at a demonstration's start.

    Stored raw rather than as features, as the reference stores ``color_img``/``depth_img``, so
    the features can be re-extracted with whatever extractor the servo is using.
    """

    rgb: np.ndarray     # (H, W, 3) uint8
    depth: np.ndarray   # (H, W) float32, metres along the optical axis; 0 or non-finite = no reading
    K: np.ndarray       # (3, 3) pinhole intrinsics

    def __post_init__(self):
        self.rgb, self.depth, self.K = map(np.asarray, (self.rgb, self.depth, self.K))
        if (self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8
                or min(self.rgb.shape[:2]) == 0):
            raise ValueError("rgb must be a nonempty (H, W, 3) uint8 image")
        if self.depth.shape != self.rgb.shape[:2] or not np.issubdtype(self.depth.dtype, np.number):
            raise ValueError("depth must be a numeric (H, W) image matching rgb")
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])):
            raise ValueError("K must contain finite pinhole intrinsics with positive focal lengths")


@dataclass
class Demo:
    """One demonstrated segment: the reference's ``base_ht_ee`` / ``gripper_state`` arrays and the
    visual-servoing reference it saves alongside them, in one type.

    ``poses[0]`` is the segment start. At execution the primitive computes
    ``T_inf = T_now @ inv(poses[0])`` and carries the whole segment over with it -- the paper's
    ``inference_ht_demo`` -- which is what makes a demonstration reusable at a new object pose once
    the hand has been servoed to the same *view* of the object.
    """

    poses: np.ndarray               # (N, 4, 4) T_base_ee along the segment
    grip_closed: np.ndarray         # (N,) bool, the gripper state along the segment
    view: View | None = None        # what the camera saw at poses[0]; None = nothing to servo to
    ee_ft_robot: np.ndarray | None = None  # (N, 6) environment wrench, in the demonstrated EE frame
    expl_direction: np.ndarray | None = None  # (3,) demonstrated contact-seeking direction, base frame

    def __post_init__(self):
        self.poses = np.asarray(self.poses, dtype=float)
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4) or len(self.poses) == 0:
            raise ValueError("poses must be a nonempty (N, 4, 4) trajectory")
        if not np.isfinite(self.poses).all():
            raise ValueError("poses must be finite")
        R = self.poses[:, :3, :3]
        if (not np.allclose(self.poses[:, 3, :], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(R), 1, atol=1e-5)):
            raise ValueError("poses must be rigid transforms in SE(3)")
        grip = np.asarray(self.grip_closed)
        if grip.shape != (len(self.poses),):
            raise ValueError("grip_closed must have one entry per pose")
        if (grip.dtype.kind not in "biuf" or not np.isfinite(grip).all()
                or not np.isin(grip, [0, 1]).all()):
            raise ValueError("grip_closed entries must be Boolean commands (0 or 1)")
        self.grip_closed = grip.astype(bool)
        if self.ee_ft_robot is not None:
            self.ee_ft_robot = np.asarray(self.ee_ft_robot, dtype=float)
            if self.ee_ft_robot.shape != (len(self.poses), 6) or not np.isfinite(self.ee_ft_robot).all():
                raise ValueError("ee_ft_robot must be a finite (N, 6) wrench history")
        if self.expl_direction is not None:
            self.expl_direction = np.asarray(self.expl_direction, dtype=float)
            if self.expl_direction.shape != (3,) or not np.isfinite(self.expl_direction).all():
                raise ValueError("expl_direction must be a finite base-frame three-vector")

    @property
    def T_start(self) -> np.ndarray:
        return self.poses[0]

    @classmethod
    def from_poses(cls, poses, grip_closed: bool = False, close_at_end: bool | None = None,
                   view: View | None = None) -> Demo:
        """A scripted segment: the gripper holds ``grip_closed`` throughout, or -- with
        ``close_at_end`` given -- switches to that state on the last pose."""
        poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
        grip = np.full(poses.shape[0], grip_closed, dtype=bool)
        if close_at_end is not None:
            grip[-1] = close_at_end
        return cls(poses, grip, view)

    def save(self, path: Path | str) -> None:
        extra = {} if self.view is None else dict(view_rgb=self.view.rgb, view_depth=self.view.depth,
                                                    view_K=self.view.K)
        if self.ee_ft_robot is not None:
            extra["ee_ft_robot"] = self.ee_ft_robot
        if self.expl_direction is not None:
            extra["expl_direction"] = self.expl_direction
        np.savez_compressed(str(path), poses=self.poses, grip_closed=self.grip_closed, **extra)

    @classmethod
    def load(cls, path: Path | str) -> Demo:
        with np.load(str(path), allow_pickle=False) as z:
            required = {"poses", "grip_closed"}
            if not required.issubset(z.files):
                raise ValueError(f"{path}: missing demonstration arrays {sorted(required - set(z.files))}")
            view_keys = {"view_rgb", "view_depth", "view_K"}
            if view_keys.intersection(z.files) and not view_keys.issubset(z.files):
                raise ValueError(f"{path}: a reference view requires RGB, depth and intrinsics")
            view = View(z["view_rgb"], z["view_depth"], z["view_K"]) if "view_rgb" in z else None
            return cls(z["poses"], z["grip_closed"], view,
                       z["ee_ft_robot"] if "ee_ft_robot" in z else None,
                       z["expl_direction"] if "expl_direction" in z else None)


# ----------------------------------------------------------------------------- the primitive seam

@dataclass(frozen=True)
class ECState:
    """What a primitive believes about the constraint it is exploiting, in one shape.

    The EC parameters are the point of the method, so they are worth watching while a run
    happens -- as arrows in the simulator and as a curve in the live window. Both want the same
    three things from every EC type, so they get one type instead of a case per primitive:

    ============  ==============================================================================
    key           meaning
    ============  ==============================================================================
    ``initial``   the prior, measured from the demonstration and fixed for the run
    ``estimate``  what the estimator believes now; the thing that moves
    ``true``      the scene's truth, present only when the run was given it (never in a run
                  driven by a task file, which knows nothing the demonstration did not say)
    ============  ==============================================================================

    Args:
        kind: the EC type, for the labels -- ``prismatic``, ``revolute``, ``planar``.
        directions: the three keys above to unit directions in the robot base frame.
        origins: a point each of those lines passes through, for an EC whose parameter has a
            position as well as a direction (a hinge). Empty for one that does not, and then
            the drawing anchors every arrow at the control point.
        std_deg: the estimator's angular spread, its own measure of how sure it is. ``nan``
            when nothing is being estimated -- a plane normal is transferred, not learnt.
    """

    kind: str
    directions: dict[str, np.ndarray]
    origins: dict[str, np.ndarray] = field(default_factory=dict)
    std_deg: float = float("nan")


def contact_event(event: str) -> str:
    """Fold a primitive's detailed event onto the paper's ``ContactChangingEvent``.

    ``MakingContact``, ``MakingContact(force-confirmed)``, ``MakingContact(early)`` -> ``"MC"``;
    ``BreakingContact``, ``BreakingContact(jump)`` -> ``"BC"``; everything else -- ``Done``,
    ``JointLimit``, ``NoContact``, ``MaxIters``, a failed servo -- is not a change of contact state
    and folds to ``"NonC"``. This label describes contact, not execution success. The planner
    checks the detailed result separately so a failed primitive cannot take a ``NonC`` edge.
    """
    if event.startswith("MakingContact"):
        return "MC"
    if event.startswith("BreakingContact"):
        return "BC"
    return "NonC"


class Primitive(ABC):
    """What every EC primitive is to the planner: the reference's ``BasePrimitive``.

    A primitive exploits one environmental constraint until a contact-changing event, and knows
    two things about the constraint it exploits -- the direction it blocks and a direction it
    admits -- so that the primitive *before* it can seek it and verify the handover. ``next`` is
    set by the planner (the reference's ``set_next_EC_primitive``) and is the demonstration's
    successor.
    """

    servo = None
    """The visual alignment this primitive starts with, if it has one -- set by the primitives
    that servo. Declared here because the live window asks every node for it."""

    def __init__(self, robot: RobotLike, cfg, demo: Demo | None = None):
        self.robot, self.cfg = robot, cfg
        self.demo = demo
        self.demo_base_ht_ee = None if demo is None else demo.poses
        self.demo_base_trajectory_ee = {} if demo is None else {
            "base_ht_ee": demo.poses, "gripper_state": demo.grip_closed,
            "ee_ft_robot": demo.ee_ft_robot,
        }
        self.demo_base_expl_direction = None if demo is None else demo.expl_direction
        self.init_demo_ht_ee = None if demo is None else demo.T_start
        self.init_base_ht_ee = np.eye(4)
        self.inference_ht_demo = np.eye(4)
        self.inf_base_EC_parameters = None
        self.d_cg_event = "NonC"
        self.transition_successors: dict[str, Primitive] = {}
        self._gripper_state: bool | None = None
        self.rows: list[dict] = []
        self.event = "Running"
        self.next_EC_primitive: Primitive | None = None
        self.verification: dict | None = None
        self._extra_start: float | None = None   # see terminate_extra_reached

    def begin_exploitation(self) -> None:
        """A graph may revisit a node; observations and termination baselines belong to one run."""
        self.event = "Running"
        self.rows = []
        self.verification = None
        self._extra_start = None
        spec = getattr(self.cfg, "terminate_extra", None)
        if spec is not None:
            key, _ = spec
            values = self.robot.extra()
            if key not in values:
                raise KeyError(f"terminate_extra watches {key!r}, which the robot's extra() does not report")
            self._extra_start = float(values[key])
        self.inf_base_EC_parameters = None
        self._gripper_state = None

    def transfer_demonstration(self) -> None:
        """Anchor the demonstration in the execution situation, using the reference names."""
        self.init_base_ht_ee = make_transform(*self.robot.ee_pose())
        self.inference_ht_demo = (np.eye(4) if self.init_demo_ht_ee is None else
                                  self.init_base_ht_ee @ transform_inv(self.init_demo_ht_ee))

    def check_grasp_state(self, verbose: bool = True) -> bool:
        """Apply the demonstrated boundary state, including the successor's first samples.

        The reference inspects the current tail plus the successor's first three samples.
        Select the last command in that window explicitly: toggling the current state on any
        mismatch can close twice or reverse a valid command when the window contains both states.
        """
        if self.demo is None:
            return False
        gripper_states = self.demo.grip_closed
        successor = self.next_EC_primitive
        if self.transition_successors:
            if self.event == "Running":
                if len(self.transition_successors) > 1:
                    successor = None  # the observed event has not selected a branch yet
            else:
                successor = self.transition_successors.get(self.cg_event)
        next_demo = getattr(successor, "demo", None)
        if next_demo is not None:
            gripper_states = np.concatenate([gripper_states, next_demo.grip_closed[:3]])
        d_gripper_state = bool(gripper_states[-1])
        curt_gripper_state = (bool(self.demo.grip_closed[0]) if self._gripper_state is None else
                              self._gripper_state)
        if curt_gripper_state == d_gripper_state:
            return False
        self.robot.set_gripper(d_gripper_state)
        self._gripper_state = d_gripper_state
        self.robot.run(getattr(self.cfg, "grip_settle", 0.5), 0.0, self.robot.ee_pose()[1])
        if verbose:
            print(f"  [grip] demonstrated boundary: {'closed' if d_gripper_state else 'opened'}")
        return True

    @abstractmethod
    def exploitation(self, max_iters: int = 200, verbose: bool = True) -> str:
        """Exploit the constraint until a contact-changing event. Returns the detailed event
        string; :attr:`cg_event` folds it onto the reference's three."""

    @property
    def cg_event(self) -> str:
        """The observed contact-changing event: ``MC``, ``BC`` or ``NonC``. The reference keeps
        this as a ``ContactChangingEvent`` enum member on the primitive; here the detailed event
        string is the primitive's own record and this is the planner's view of it."""
        return contact_event(self.event)

    def contact_event(self) -> str:
        """:attr:`cg_event` as a call, for callers that had it that way."""
        return self.cg_event

    def set_next_EC_primitive(self, next_EC_primitive: Primitive | None) -> None:  # noqa: N802
        """Link the successor, as the reference's ``funnel_sequence_ec_primitives`` does, so this
        primitive can ask what comes after it which direction to seek."""
        self.next_EC_primitive = next_EC_primitive

    def get_next_constraint_direction(self, curt_base_p_ee: np.ndarray) -> np.ndarray | None:
        """The direction the *next* EC blocks at ``curt_base_p_ee`` -- what this primitive presses
        along to hand over. ``None`` when there is no successor or it blocks nothing, which is the
        reference's backstep case."""
        nxt = self.next_EC_primitive
        if nxt is None:
            return None
        d = nxt.get_curt_constraint_direction(curt_base_p_ee, self.inference_ht_demo)
        return None if d is None or float(np.linalg.norm(d)) < 1e-9 else unit(np.asarray(d, dtype=float))

    def get_curt_constraint_direction(self, curt_base_p_ee: np.ndarray,
                                     inference_ht_demo: np.ndarray | None = None) -> np.ndarray | None:
        """Unit direction this EC blocks at ``p`` -- the direction to *move in* to make contact
        with it, base frame. ``None`` when the EC does not block one (free space) or the
        primitive does not know it."""
        return None

    def get_curt_admissible_motion_direction(self, curt_base_p_ee: np.ndarray,
                                           inference_ht_demo: np.ndarray | None = None) -> np.ndarray | None:
        """A unit direction this EC admits at ``p``, base frame; ``None`` for free space."""
        return None

    def ec_state(self) -> ECState | None:
        """The constraint's parameters as they stand, for the visualisation; ``None`` when there
        is nothing to show -- free space has no constraint to draw."""
        return None

    def terminate_extra_reached(self, row: dict) -> str | None:
        """``Done(key->threshold)`` once the diagnostic named by ``cfg.terminate_extra`` has
        reached its threshold from the side it started on; ``None`` otherwise, and always
        ``None`` for a config without the entry.

        The side is read off the first row, which is what lets a task file give the threshold as
        the joint's own value: a drawer joint opening toward 0.2 m is done at >= 0.2, a door joint
        opening toward -0.5 rad at <= -0.5, and neither the file nor the adapter carries a sign.
        A key the robot does not report is a configuration error, not a run that never ends.
        """
        spec = getattr(self.cfg, "terminate_extra", None)
        if spec is None:
            return None
        key, thresh = spec
        if key not in row:
            raise KeyError(f"terminate_extra watches {key!r}, which the robot's extra() does not report")
        value = float(row[key])
        if self._extra_start is None:
            self._extra_start = value
        toward = thresh - self._extra_start
        if toward == 0.0 or (value - thresh) * toward >= 0.0:
            return f"Done({key}->{thresh:g})"
        return None

    def check_transition_to_the_next_EC(self, push: float = PAPER_VERIFY_PUSH, seconds: float = PAPER_VERIFY_TIME,
                          max_disp: float = PAPER_VERIFY_DISP, force: float = PAPER_VERIFY_FORCE,
                          period: float = 0.1, verbose: bool = True) -> bool:
        """The reference's ``check_transition_to_the_next_EC``: push into the next EC's constraint
        direction and see whether something is there.

        The goal is set ``push`` beyond the equilibrium pose along the direction; the handover is
        confirmed when the hand has moved less than ``max_disp`` that way while feeling at least
        ``force`` against it. Moving the full way, or the time running out, means the constraint
        the demonstration promised is not where the transfer put it. The equilibrium pose is
        restored afterwards. ``True`` without a test when there is no successor or it blocks no
        direction. The measurements go in :attr:`verification`.
        """
        rb = self.robot
        _, p0 = rb.ee_pose()
        d = self.get_next_constraint_direction(p0)
        if d is None:
            self.verification = dict(tested=False, ok=True)
            return True
        R_des, p_des = rb.desired_pose()
        rb.set_goal(p_des + push * d)
        t0, ok, disp, f = rb.t, False, 0.0, 0.0
        while rb.t - t0 < seconds:
            rb.run(period, 0.0, p0)
            R_now, p_now = rb.ee_pose()
            disp = float(np.dot(p_now - p0, d))
            f = -float(np.dot(R_now @ rb.wrench()[:3], d))   # resistance against the push
            if f >= force and disp <= max_disp:
                ok = True
                break
            if disp > max_disp:
                break
        rb.set_goal(p_des, R_des)
        t1 = rb.t
        while rb.t - t1 < seconds:
            rb.run(period, 0.0, p0)
            if float(np.linalg.norm(rb.ee_pose()[1] - p0)) < 0.005:
                break
        self.verification = dict(tested=True, ok=ok, disp=disp, force=f, seconds=rb.t - t0)
        if verbose:
            print(f"  [verify] pushed along {d.round(3)}: moved {1e3 * disp:.1f} mm, felt {f:.1f} N -> "
                  f"{'confirmed' if ok else 'nothing there'}")
        return ok

    def summary(self) -> dict:
        return dict(event=self.event, contact_event=self.contact_event(), iters=len(self.rows))

    def write_csv(self, path: Path | str) -> None:
        write_rows(path, self.rows)


# ----------------------------------------------------------------------------- output

def write_rows(path: Path | str, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
