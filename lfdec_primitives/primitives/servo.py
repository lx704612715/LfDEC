# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Visual alignment to a demonstration reference using matched RGB-D features.

The servo estimates pose error and a correction twist, rejects outliers, and terminates
on convergence, lost features, timeout or unexpected contact."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .base import (
    RobotLike,
    View,
    adjoint,
    make_transform,
    rotvec_from_matrix,
    se3_exp,
    skew,
)

__all__ = [
    "CameraLike",
    "FeatureMatcher",
    "Features",
    "ServoCfg",
    "VisualServo",
    "backproject",
    "draw_matches",
    "feature_pattern",
    "kabsch",
    "m_estimate",
    "pbvs_twist",
    "ibvs_twist",
]


class CameraLike(Protocol):
    """An eye-in-hand RGB-D camera. ``T_ee_cam`` is its pose in the control-point frame."""

    T_ee_cam: np.ndarray

    def capture(self) -> View: ...


@dataclass
class Features:
    """Keypoints of one frame: pixel positions, camera-frame 3D positions and descriptors."""

    uv: np.ndarray       # (M, 2)
    xyz: np.ndarray      # (M, 3)
    desc: np.ndarray     # (M, D) float32


def backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray,
                z_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Lift pixel positions to camera-frame points with the depth at the nearest pixel.

    Returns ``(xyz, keep)``: the points, and the mask of the input rows that had a usable depth
    (finite and inside ``z_range``). The reference's
    ``transform_keypoints_to_image_plane_coordinate``.
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    h, w = depth.shape
    c = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    r = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    z = np.asarray(depth, dtype=float)[r, c]
    keep = np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=1)[keep], keep


@dataclass
class ServoCfg:
    """Configuration of :class:`VisualServo`."""

    gain: float = 1.0
    servo_mode: str = "pbvs"  # free-space PBVS; the plane requests IBVS
    """[1/s] the point error is asked to decay at this rate."""
    period: float = 0.1
    """[s] of plant per update; also the integration step for the goal pose."""
    max_step: float = 0.01
    max_rot_step: float = 0.05
    """[m] / [rad] cap on one update's goal advance -- the reference's ``max_displacement`` /
    ``max_rotation``. A wrong match can ask for anything; this is what bounds the damage."""
    tol: float = 0.005
    rot_tol: float = 0.015
    """Translation [m] and rotation [rad] error thresholds for visual convergence."""
    error_window: int = 10
    """Observations the pose error is averaged over before the tolerance test, the reference's
    ``history_error`` deque. One good frame is not an alignment, and the test is on the average
    so a single bad match cannot end the servo either. The servo returns ``Converged`` on the
    first update where the average is inside tolerance; ``max_time`` is a fallback, not the
    normal way out."""
    min_points: int = 5
    """Fewer matched points than this is not an observation. The reference's ``min_num_feature``."""
    lost_after: int = 10
    """Consecutive updates without an observation -> ``Lost``."""
    max_time: float = 20.0
    """[s] -> ``Timeout``."""
    collision_force: float = 8.0
    """[N] -> ``MakingContact(early)``; servoing happens in free space and a contact is news."""
    ratio: float = 0.75
    """Lowe ratio for the descriptor match."""
    max_features: int = 1000
    z_range: tuple[float, float] = (0.15, 1.0)
    """Valid feature depth interval [m]. Neighborhood depth checks exclude gripper
    features and background points from the alignment observations."""
    tukey_c: float | None = 4.6851
    """Tukey biweight constant of the M-estimator, 4.685 for 95 % efficiency on Gaussian residuals
    (the reference's ``r_estimation_efficiency: 0.95``). ``None`` is plain least squares."""


# ----------------------------------------------------------------------------- synthetic texture

def feature_pattern(n: int = 256, seed: int = 7, shapes: int = 120) -> np.ndarray:
    """Generate a deterministic uint8 RGB texture of irregular polygons and ellipses.
    Random sizes, colors and overlap provide distinct keypoint neighborhoods."""
    import cv2

    rng = np.random.default_rng(seed)
    tex = np.full((n, n, 3), 128, dtype=np.uint8)
    for _ in range(shapes):
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        size = float(rng.uniform(0.06, 0.30) * n)
        cx, cy = rng.uniform(0, n, 2)
        if rng.random() < 0.5:
            k = int(rng.integers(3, 7))
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            rad = size * rng.uniform(0.4, 1.0, k)
            pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1).astype(np.int32)
            cv2.fillPoly(tex, [pts], colour)
        else:
            axes = (max(1, int(size)), max(1, int(size * rng.uniform(0.3, 1.0))))
            cv2.ellipse(tex, (int(cx), int(cy)), axes, float(rng.uniform(0, 180)), 0, 360, colour, -1)
    return tex


def draw_matches(ref_rgb: np.ndarray, uv_ref: np.ndarray, obs_rgb: np.ndarray, uv_obs: np.ndarray,
                 inliers: np.ndarray | None = None, radius: int = 4, thickness: int = 2,
                 alpha: float = 0.6) -> np.ndarray:
    """Reference and current frame side by side, each match drawn as a line between them.

    The reference's ``vision_utils.drawMatches``, with its two output images folded into one:
    ``inliers`` is the M-estimator's verdict, and the matches it kept are drawn green while the
    ones it down-weighted are red. That is the picture that tells you whether an alignment that
    will not converge is looking at the object or at the gripper.

    Returns a BGR image, ready for ``cv2.imshow`` or ``cv2.imwrite``.
    """
    import cv2

    a, b = np.ascontiguousarray(ref_rgb[..., :3]), np.ascontiguousarray(obs_rgb[..., :3])
    h = max(a.shape[0], b.shape[0])
    out = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    out[:a.shape[0], :a.shape[1]] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    out[:b.shape[0], a.shape[1]:] = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
    plain = out.copy()
    keep = np.ones(len(uv_ref), bool) if inliers is None else np.asarray(inliers, bool)
    for (x1, y1), (x2, y2), ok in zip(uv_ref, uv_obs, keep, strict=False):
        colour = (0, 255, 0) if ok else (0, 0, 255)
        p1, p2 = (int(x1), int(y1)), (int(x2) + a.shape[1], int(y2))
        cv2.circle(out, p1, radius, colour, thickness)
        cv2.circle(out, p2, radius, colour, thickness)
        cv2.line(out, p1, p2, colour, thickness)
    return cv2.addWeighted(out, alpha, plain, 1.0 - alpha, 0.0)


# ----------------------------------------------------------------------------- the servo law

def pbvs_twist(p_obs: np.ndarray, p_ref: np.ndarray, gain: float,
               tukey_c: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Camera twist ``[v, w]`` that drives matched 3D points onto their reference positions.

    A point fixed in the world and seen from a camera moving with body twist ``(v, w)`` moves in
    the camera frame as ``dp/dt = -v - w x p = [-I, [p]x] [v; w]``. Asking every matched point's
    error ``e = p_obs - p_ref`` to decay as ``de/dt = -gain e`` and solving the stacked system in
    the least-squares sense gives the twist; the reference's ``compute_pbvs_goal`` with
    ``d = ref - obs``. Returns the twist and the M-estimator's per-point weights (1 = inlier).
    """
    p_obs = np.asarray(p_obs, dtype=float).reshape(-1, 3)
    p_ref = np.asarray(p_ref, dtype=float).reshape(-1, 3)
    n = p_obs.shape[0]
    L = np.zeros((3 * n, 6))
    for i in range(n):
        L[3 * i:3 * i + 3, :3] = -np.eye(3)
        L[3 * i:3 * i + 3, 3:] = skew(p_obs[i])
    d = -gain * (p_obs - p_ref).reshape(-1)
    xi, w = m_estimate(L, d, tukey_c)
    return xi, w.reshape(n, 3).min(axis=1)


def ibvs_twist(p_obs: np.ndarray, p_ref: np.ndarray, gain: float,
               tukey_c: float | None) -> tuple[np.ndarray, np.ndarray]:
    """The reference's compute_ibvs_goal: normalized image interaction matrix and M-estimate.

    Depth scales translation in the interaction matrix; the error is measured in normalized
    image coordinates, not pixels or 3D positions. Twists use the same [v,w] convention as PBVS.
    """
    p_obs, p_ref = np.asarray(p_obs), np.asarray(p_ref)
    if (p_obs.shape != p_ref.shape or p_obs.ndim != 2 or p_obs.shape[1] != 3
            or not np.isfinite(p_obs).all() or not np.isfinite(p_ref).all()
            or np.any(p_obs[:, 2] <= 0) or np.any(p_ref[:, 2] <= 0)):
        raise ValueError("IBVS needs paired finite 3D points with positive depth")
    n = len(p_obs)
    positions_2d_img_matched = p_obs[:, :2] / p_obs[:, 2:3]
    reference_2d = p_ref[:, :2] / p_ref[:, 2:3]
    L = np.zeros((2 * n, 6))
    for i, ((x, y), z) in enumerate(zip(positions_2d_img_matched, p_obs[:, 2])):
        L[2*i:2*i+2] = [[-1/z, 0, x/z, x*y, -(1+x*x), y],
                        [0, -1/z, y/z, 1+y*y, -x*y, -x]]
    errors_2d = -gain * (positions_2d_img_matched - reference_2d).reshape(-1)
    xi, weights = m_estimate(L, errors_2d, tukey_c)
    return xi, weights.reshape(n, 2).min(axis=1)


def kabsch(p_from: np.ndarray, p_to: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform ``(R, t)`` with ``p_to ~= R p_from + t`` in the least-squares sense.

    The pose error between the current view and the reference, from the matched points: what the
    convergence test is on. Arun/Kabsch via SVD, with the reflection guard.
    """
    A = np.asarray(p_from, dtype=float).reshape(-1, 3)
    B = np.asarray(p_to, dtype=float).reshape(-1, 3)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def m_estimate(L: np.ndarray, d: np.ndarray, tukey_c: float | None,
               iters: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted least squares with Tukey's biweight -- the reference's ``m_estimate``.

    Residuals are scaled by their median absolute deviation, so the outlier threshold adapts to
    how good the matches are on this frame. Returns ``(solution, weights)``; weights are per row.
    """
    m = np.linalg.lstsq(L, d, rcond=None)[0]
    w = np.ones(d.shape[0])
    if tukey_c is None:
        return m, w
    for _ in range(iters):
        r = np.abs(L @ m - d)
        mad = float(np.median(r))
        eps = tukey_c * mad / 0.6745
        if eps <= 1e-12:
            break
        w = np.where(r <= eps, (1.0 - (r / eps) ** 2) ** 2, 0.0)
        LtW = L.T * w
        A = LtW @ L
        if np.linalg.matrix_rank(A) < L.shape[1]:
            continue
        m = np.linalg.solve(A, LtW @ d)
    return m, w


# ----------------------------------------------------------------------------- features

class FeatureMatcher:
    """SIFT keypoints lifted to 3D by depth, matched by descriptor with Lowe's ratio test.

    The reference's ``ImageExtractionAndMatch`` with OpenCV in place of LightGlue. ``cv2`` is
    imported here and nowhere else in the package, so everything that does not servo stays
    numpy-only.
    """

    def __init__(self, cfg: ServoCfg):
        import cv2

        self.cfg = cfg
        self._cv2 = cv2
        self._sift = cv2.SIFT_create(nfeatures=cfg.max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, view: View) -> Features:
        cv2 = self._cv2
        rgb = np.ascontiguousarray(view.rgb[..., :3])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if not kps:
            return Features(np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 128), np.float32))
        uv = np.array([k.pt for k in kps], dtype=float)
        xyz, keep = backproject(uv, view.depth, view.K, self.cfg.z_range)
        # A silhouette feature can sit just outside the gripper: its centre pixel sees the
        # surface behind it, while its image neighbourhood follows the hand. Require the
        # keypoint's neighbourhood to pass the depth filter too, in both views. Otherwise
        # changing surface depth gives a fixed gripper edge a fictitious 3D motion.
        depth = np.asarray(view.depth)
        valid = np.isfinite(depth) & (depth > self.cfg.z_range[0]) & (depth < self.cfg.z_range[1])
        integral = cv2.integral(valid.astype(np.uint8))
        radius = np.ceil(0.5 * np.array([k.size for k in kps])).astype(int)[:, None]
        centre = np.rint(uv).astype(int)
        limit = np.array([depth.shape[1], depth.shape[0]])
        lo, hi = np.maximum(centre - radius, 0), np.minimum(centre + radius + 1, limit)
        x0, y0 = lo.T
        x1, y1 = hi.T
        count = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        supported = (count == np.prod(hi - lo, axis=1))[keep]
        return Features(uv[keep][supported], xyz[supported], np.asarray(desc, dtype=np.float32)[keep][supported])

    def match(self, obs: Features, ref: Features) -> tuple[np.ndarray, np.ndarray]:
        """``(idx_obs, idx_ref)`` of the matches that pass the ratio test."""
        if obs.desc.shape[0] < 2 or ref.desc.shape[0] < 2:
            return np.zeros(0, int), np.zeros(0, int)
        pairs = self._bf.knnMatch(obs.desc, ref.desc, k=2)
        good = [m for m, n_ in (p for p in pairs if len(p) == 2) if m.distance < self.cfg.ratio * n_.distance]
        if not good:
            return np.zeros(0, int), np.zeros(0, int)
        return (np.array([m.queryIdx for m in good], int), np.array([m.trainIdx for m in good], int))


# ----------------------------------------------------------------------------- the servo

class VisualServo:
    """Drive the hand until the camera sees the reference view.

    Args:
        camera: the eye-in-hand camera.
        reference: the view to reach -- ``Demo.view``.
        cfg: see :class:`ServoCfg`.
    """

    def __init__(self, camera: CameraLike, reference: View, cfg: ServoCfg | None = None,
                 matcher: FeatureMatcher | None = None):
        self.cam = camera
        self.cfg = cfg if cfg is not None else ServoCfg()
        if self.cfg.servo_mode not in ("ibvs", "pbvs"):
            raise ValueError("servo_mode must be 'ibvs' or 'pbvs'")
        self.matcher = matcher if matcher is not None else FeatureMatcher(self.cfg)
        self.ref_view = reference
        self.ref = self.matcher.extract(reference)
        self.last_view: View | None = None   # the most recent frame, for looking at what it saw
        self.last_match: dict | None = None  # its matched keypoints and the M-estimator's verdict
        self.rows: list[dict] = []
        self.event = "Ready"

    def twist(self) -> tuple[np.ndarray | None, dict]:
        """One observation: the control-point twist ``[v, w]`` (its own frame) that reduces the
        view error, or ``None`` when too few features matched; plus diagnostics."""
        c = self.cfg
        self.last_view = self.cam.capture()
        obs = self.matcher.extract(self.last_view)
        io, ir = self.matcher.match(obs, self.ref)
        info = dict(n_features=obs.xyz.shape[0], n_matched=int(io.shape[0]), n_inliers=0,
                    err=float("nan"), err_pos=float("nan"), err_rot=float("nan"), depth_med=float("nan"))
        if io.shape[0] < c.min_points:
            self.last_match = None
            return None, info
        p_obs, p_ref = obs.xyz[io], self.ref.xyz[ir]
        compute_goal = ibvs_twist if c.servo_mode == "ibvs" else pbvs_twist
        xi_cam, w = compute_goal(p_obs, p_ref, c.gain, c.tukey_c)
        inl = w > 0.0
        self.last_match = dict(uv_ref=self.ref.uv[ir], uv_obs=obs.uv[io], inliers=inl)
        if int(inl.sum()) < c.min_points:
            return None, info
        R, t = kabsch(p_obs[inl], p_ref[inl])   # the reference camera's pose in the current one
        info["n_inliers"] = int(inl.sum())
        info["err"] = float(np.mean(np.linalg.norm((p_obs - p_ref)[inl], axis=1)))
        info["err_pos"] = float(np.linalg.norm(t))
        info["err_rot"] = float(np.linalg.norm(rotvec_from_matrix(R)))
        # Where the inliers were. The servo is only as good as what it matched, and things
        # fixed to the hand match perfectly and mean nothing; their depth gives them away.
        info["depth_med"] = float(np.median(p_obs[inl, 2]))
        return adjoint(self.cam.T_ee_cam) @ xi_cam, info

    def goal(self, R_ee: np.ndarray, p_ee: np.ndarray, xi_ee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert a correction twist into a goal composed on the equilibrium pose."""
        c = self.cfg
        v, w = xi_ee[:3] * c.period, xi_ee[3:] * c.period
        nv, nw = float(np.linalg.norm(v)), float(np.linalg.norm(w))
        if nv > c.max_step:
            v = v * (c.max_step / nv)
        if nw > c.max_rot_step:
            w = w * (c.max_rot_step / nw)
        T = make_transform(R_ee, p_ee) @ se3_exp(np.concatenate([v, w]))
        return T[:3, :3], T[:3, 3]

    def match_image(self) -> np.ndarray | None:
        """The reference and the current frame side by side with the matches drawn, or ``None``
        before anything has matched. BGR, for ``cv2.imwrite`` or
        :class:`~lfdec_primitives.vis.monitor.ECMonitor`'s alignment panel, which is what shows it live."""
        m = self.last_match
        if m is None or self.last_view is None:
            return None
        return draw_matches(self.ref_view.rgb, m["uv_ref"], self.last_view.rgb, m["uv_obs"], m["inliers"])

    def run(self, robot: RobotLike, verbose: bool = True) -> str:
        """Servo until the averaged view error is inside tolerance (``Converged``), or until
        ``Lost``, ``MakingContact(early)`` or -- as a fallback only -- ``Timeout``."""
        c = self.cfg
        t0, lost = robot.t, 0
        history: deque[tuple[float, float]] = deque(maxlen=c.error_window)
        while robot.t - t0 < c.max_time:
            xi, info = self.twist()
            self.event = "Holding" if xi is None else "Running"
            self.record_observation(robot, info, held=len(history))
            if xi is None:
                lost += 1
                history.clear()
                robot.run(c.period, 0.0, robot.ee_pose()[1])
                if lost >= c.lost_after:
                    self.event = "Lost"
                    return self.event
                continue
            lost = 0
            # The step marches from the equilibrium pose the controller is already tracking.
            R_des, p_des = robot.desired_pose()
            Rg, pg = self.goal(R_des, p_des, xi)
            robot.set_goal(pg, Rg)
            robot.run(c.period, 0.0, robot.ee_pose()[1])
            if float(np.linalg.norm(robot.wrench()[:3])) >= c.collision_force:
                self.event = "MakingContact(early)"
                return self.event
            history.append((info["err_pos"], info["err_rot"]))
            avg_pos, avg_rot = (float(np.mean([h[0] for h in history])),
                                float(np.mean([h[1] for h in history])))
            arrived = len(history) == c.error_window and avg_pos <= c.tol and avg_rot <= c.rot_tol
            if verbose and (len(self.rows) % 10 == 1 or arrived):
                print(f"  [servo] t={robot.t:6.2f}s matched={info['n_matched']:3d} inliers={info['n_inliers']:3d} "
                      f"pose err={1e3 * info['err_pos']:5.1f}mm/{np.degrees(info['err_rot']):4.1f}deg "
                      f"avg={1e3 * avg_pos:5.1f}mm/{np.degrees(avg_rot):4.1f}deg "
                      f"|v|={1e3 * float(np.linalg.norm(xi[:3])):5.1f}mm/s "
                      f"|w|={np.degrees(float(np.linalg.norm(xi[3:]))):4.1f}deg/s")
            if arrived:
                self.event = "Converged"
                if verbose:
                    print(f"  [servo] inside tolerance ({1e3 * c.tol:.0f} mm / {np.degrees(c.rot_tol):.1f} deg) "
                          f"after {robot.t - t0:.1f}s -- handing over")
                return self.event
        self.event = "Timeout"
        return self.event

    def record_observation(self, robot: RobotLike, info: dict, held: int = 0) -> None:
        """Publish a twist observation before moving, including when a primitive owns the loop.

        The monitor runs inside ``robot.run``: logging after that call would pair the new match
        image with the previous observation's errors. Timestamp and pose belong to capture time,
        not the end of the commanded motion. ``held`` is the free-space convergence-window size;
        the plane's bounded correction does not use that convergence test.
        """
        _, p = robot.ee_pose()
        F = robot.wrench()
        self.rows.append(dict(iter=len(self.rows), t=robot.t, ee_x=p[0], ee_y=p[1], ee_z=p[2],
                              held=held, F_norm=float(np.linalg.norm(F[:3])), **info))

    def summary(self) -> dict:
        rows = self.rows
        last = rows[-1] if rows else {}
        return dict(servo_event=self.event, servo_iters=len(rows),
                    servo_err_mm=1e3 * last.get("err_pos", float("nan")),
                    servo_err_deg=float(np.degrees(last.get("err_rot", float("nan")))),
                    servo_inliers=last.get("n_inliers", 0), servo_depth_med=last.get("depth_med", float("nan")),
                    servo_matched=last.get("n_matched", 0), servo_seconds=(rows[-1]["t"] - rows[0]["t"]) if rows else 0.0)
