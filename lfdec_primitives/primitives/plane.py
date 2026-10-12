# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Plane primitive: transfer the demonstrated normal and trajectory to the current EE frame,
replay the projected trajectory under normal preload, then visually align on the plane.
Surface servoing freezes orientation. Spiral search follows bounded alignment when
contact remains; breaking contact hands execution to the successor."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import (
    PAPER_BCE_PLANE,
    PAPER_MCE_FORCE_PLANE,
    PAPER_MCE_T,
    PAPER_MIN_TRAN,
    Demo,
    ECState,
    Primitive,
    RobotLike,
    angle_between,
    make_transform,
    project_to_plane,
    project_trajectory,
    rotvec_from_matrix,
    tangent_basis,
    unit,
)
from .servo import VisualServo

__all__ = ["PlaneCfg", "PlanePrimitive", "spiral_offsets"]


# ----------------------------------------------------------------------------- search pattern

def spiral_offsets(radius: float, pitch: float, step: float) -> np.ndarray:
    """``(N, 2)`` in-plane offsets of an Archimedean spiral out to ``radius``: successive turns
    ``pitch`` apart radially, successive waypoints ``step`` apart along the arc, the last one on
    the rim.

    The reference's ``spiral_3D_path_generator`` spaces waypoints by *angle*, which puts the
    first turn's waypoints a fraction of a millimetre apart. Indexed by the clock that is
    harmless; issued on progress (departure 2) it is not, since a part that stiction holds
    still is then asked for the same unreachable fraction of a millimetre thirty times over
    -- see ``PlaneCfg.spiral_step``.
    """
    if min(radius, pitch, step) <= 0.0:
        raise ValueError("radius, pitch and step must be positive")
    theta_max = 2.0 * math.pi * radius / pitch
    theta = np.linspace(0.0, theta_max, max(2, int(64 * theta_max)) + 1)
    r = pitch * theta / (2.0 * math.pi)
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    s_way = np.arange(step, s[-1], step)
    if s_way.size == 0 or s[-1] - s_way[-1] > 0.5 * step:
        s_way = np.append(s_way, s[-1])
    else:
        s_way[-1] = s[-1]
    theta_way = np.interp(s_way, s, theta)
    r_way = pitch * theta_way / (2.0 * math.pi)
    return np.stack([r_way * np.cos(theta_way), r_way * np.sin(theta_way)], axis=1)


# ----------------------------------------------------------------------------- primitive

@dataclass
class PlaneCfg:
    """Configuration of :class:`PlanePrimitive`."""

    period: float = 0.5           # s per iteration
    stall_period: float = 0.25    # s per iteration once the control point has stopped moving

    press_depth: float = 0.01
    """Normal equilibrium-pose offset into the plane [m] to maintain contact.
    The compliant controller limits the resulting force."""

    settle: float = 0.15
    """[s] the plant is stepped this long with the goal unchanged before the wrench is read.

    The observation has to be taken at rest. An iteration ends as soon as the control point has
    moved ``early_exit_disp``, i.e. while it is still moving, and a joint-reaction wrench sensor
    reports the distal segment's inertia along with the contact -- several newtons at the
    accelerations a 0.1 m/s command produces, the size of the tangential force this primitive's
    making-contact test keys on."""

    contact_force: float = 2.0
    """[N] along the normal that counts as being on the surface. Checked once, at the start: the
    reference's plane segment begins in contact, delivered by the free-space primitive before it,
    and a plane primitive started in the air has nothing to exploit (``NoContact``)."""

    waypoint_tol: float = 0.008
    waypoint_rot_tol: float = 0.10
    """[m] how close the part must get to a waypoint before the next one is issued.

    Size it at or above the stiction dead band ``mu * press_depth``, where the part stops with
    the spring balanced against friction -- 1 cm at ``mu = 1``. Too small and the search waits
    at a waypoint it cannot reach until the no-move counter calls it a making-contact event."""
    waypoint_attempts: int = 12
    """Iterations spent on one waypoint before moving to the next regardless."""

    servo_time: float = 3.0
    servo_max_disp: float = 0.03
    """Maximum surface-servo duration [s] and displacement [m]. Exhausting either
    budget switches to spiral search when contact remains."""
    servo_z_range: tuple[float, float] = (0.08, 1.0)
    """[m] depth range of the surface servo's features, in place of ``ServoCfg.z_range``. The
    free-space servo's 15 cm floor keeps the gripper out of the match; on a surface the camera
    is closer than that to what it servoes on (13 cm on the peg task: the peg's length plus the
    camera's offset behind the control point), so the floor has to sit between the fingertips
    and the surface."""

    spiral_radius: float = 0.05
    """[m] how far out the search goes. Sized against the *landing* error -- where the
    demonstration's end pose lands after the transfer, plus the distance the demonstrator
    stopped short of the feature -- not against the nominal distance to the feature; and against
    the normal error too: the goal leaves the true surface by ``r sin(theta)`` at the rim, which
    ``bce_depth`` bounds."""
    spiral_pitch: float = 0.0075
    """[m] of radial growth per turn: the search passes within half of this of every point
    inside the radius, so size it against the feature's capture width (an opening's clearance
    around the part)."""
    spiral_step: float = 0.005
    """[m] of arc between waypoints. A waypoint is issued when the previous one is reached *or
    given up on*, so a part that stiction holds still is asked for ``spiral_step`` more every
    iteration and the spring crosses the dead band ``mu * press_depth`` after
    ``mu * press_depth / spiral_step`` of them. Keep that count well under ``mce_attempts``, or
    the no-move counter reports stiction as ``MakingContact``: measured on the peg (mu 0.6,
    15 mm press, 9 mm dead band) with the reference's angular spacing, whose first thirty
    waypoints added up to 7.5 mm."""

    min_tran: float = PAPER_MIN_TRAN
    mce_attempts: int = PAPER_MCE_T
    """No-move iterations -> ``MakingContact``: the paper's general making-contact definition."""
    mce_force: float = PAPER_MCE_FORCE_PLANE
    """[N] tangential force, over what the part felt at the start of the segment, ->
    ``MakingContact``: the part has run into something on the surface. The reference's
    ``min_force_tangent_direction``."""
    bce_depth: float = PAPER_BCE_PLANE
    """Minimum displacement through the plane [m] for BreakingContact,
    combined with the contact-force check."""

    terminate_extra: tuple[str, float] | None = None
    min_reach_margin: float = 0.03
    """[m] stop with ``JointLimit`` once the arm has only this much travel left along the current
    tangential direction. See ``TransCfg.min_reach_margin``."""
    jump_factor: float = 3.0
    early_exit_disp: float = 0.005  # m; end the iteration once the control point moved this far


class PlanePrimitive(Primitive):
    """Exploit a planar EC carried over from a demonstration until the contact changes.

    ======  ==========================================================================
    phase   what it does
    ======  ==========================================================================
    replay  the demonstrated segment, transferred by ``T_inf`` and pressed onto the plane
    servo   the visual correction toward the demonstration's end view, in-plane, bounded
    spiral  an Archimedean spiral about the pose the servo ended at, until the part sinks
            (``BreakingContact`` -- the opening) or jams (``MakingContact`` -- a wall)
    ======  ==========================================================================

    Args:
        robot: the :class:`~.base.RobotLike` adapter, already in contact with the surface.
        normal_demo: outward normal of the plane *in the demonstration's base frame* -- the
            segment's EC parameter (:func:`~.base.fit_plane` on a recorded slide, or the scene's
            truth for a scripted task).
        demo: the segment to replay. ``None`` skips the replay: the plane is then taken to be
            where the normal says, through the current position, and the search starts at once.
        cfg: see :class:`PlaneCfg`.
        servo: a :class:`~.servo.VisualServo` on the view the *next* segment starts from -- the
            reference's ``D1S{k+1}_start``, the surface as seen where the demonstrated slide
            ended -- or ``None`` to go from the replay straight to the spiral.
        normal_true / point_true: the true surface, for the diagnostics only.
    """

    def __init__(self, robot: RobotLike, normal_demo: np.ndarray, demo: Demo | None = None,
                 cfg: PlaneCfg | None = None, servo: VisualServo | None = None,
                 normal_true: np.ndarray | None = None, point_true: np.ndarray | None = None):
        super().__init__(robot, cfg if cfg is not None else PlaneCfg(), demo)
        self.n_demo = unit(np.asarray(normal_demo, dtype=float))
        self.demo_base_EC_parameters = self.n_demo.copy()
        self.demo, self.servo = demo, servo
        self.n_true = None if normal_true is None else unit(np.asarray(normal_true, dtype=float))
        self.p_true = None if point_true is None else np.asarray(point_true, dtype=float)
        self.normal = self.n_demo.copy()        # transferred in run()
        self.p_touch: np.ndarray | None = None  # where the segment started, on the plane
        self.p_centre: np.ndarray | None = None  # where the spiral is centred
        self.phase = "replay"
        self._F0 = np.zeros(3)
        self._p_prev: np.ndarray | None = None
        self._no_move, self._path, self._max_iters = 0, 0.0, 0
        self._target = np.zeros(3)

    # ------------------------------------------------------------------ Primitive hooks

    def get_curt_constraint_direction(self, curt_base_p_ee=None, inference_ht_demo=None) -> np.ndarray:
        """Into the surface: the direction the plane blocks, and the one to seek it along."""
        if inference_ht_demo is not None or self.inf_base_EC_parameters is None:
            R = np.eye(3) if inference_ht_demo is None else inference_ht_demo[:3, :3]
            return -R @ self.demo_base_EC_parameters
        return -self.inf_base_EC_parameters[:3]

    def get_curt_admissible_motion_direction(self, curt_base_p_ee=None, inference_ht_demo=None) -> np.ndarray:
        normal = -self.get_curt_constraint_direction(curt_base_p_ee, inference_ht_demo)
        return np.stack(tangent_basis(normal), axis=1)

    # ------------------------------------------------------------------ diagnostics

    def ec_state(self) -> ECState:
        """``initial`` is the demonstration's normal, ``estimate`` the same normal transferred
        onto where the surface is now (they coincide until :meth:`exploitation` transfers it).
        Nothing is estimated on a plane, so there is no spread to report."""
        d = {"initial": self.n_demo, "estimate": self.normal}
        if self.n_true is not None:
            d["true"] = self.n_true
        return ECState("planar", d)

    def _err_normal(self) -> float:
        return math.degrees(angle_between(self.normal, self.n_true)) if self.n_true is not None else float("nan")

    def _height_true(self, p: np.ndarray) -> float:
        """[m] of the control point above the true surface (NaN without a truth)."""
        if self.n_true is None or self.p_true is None:
            return float("nan")
        return float(np.dot(self.n_true, p - self.p_true))

    # ------------------------------------------------------------------ run

    def exploitation(self, max_iters: int = 300, verbose: bool = True) -> str:
        c, rb = self.cfg, self.robot
        self.begin_exploitation()
        self.transfer_demonstration()
        self._no_move, self._path = 0, 0.0
        self._max_iters = max_iters
        R0, p0 = rb.ee_pose()
        self.normal = unit(self.inference_ht_demo[:3, :3] @ self.demo_base_EC_parameters)
        self.inf_base_EC_parameters = np.r_[self.normal, -np.dot(self.normal, p0)]
        self.proj_base_ht_ee = ([] if self.demo is None else
                                project_trajectory(self.demo_base_ht_ee, self.init_base_ht_ee))
        self.p_touch, self._p_prev = p0.copy(), p0.copy()
        f0 = R0 @ rb.wrench()[:3]
        if float(np.dot(f0, self.normal)) < c.contact_force:
            self.event = "NoContact"
            return self.event
        # The making-contact test is on the *change* of tangential force from here, as in the
        # reference (init_base_ft_robot): whatever the approach left in the wrench is the baseline.
        self._F0 = f0.copy()
        if verbose:
            print(f"  [plane] transferred normal {self.normal.round(4)}"
                  + (f", {self._err_normal():.1f} deg off the truth" if self.n_true is not None else "")
                  + f"; F.n = {float(np.dot(f0, self.normal)):.1f} N at the start")

        for action in (self.replay_demonstration_trajectory, self.visual_servoing_on_the_plane,
                       self.spiral_exploration_for_seeking_contact_changing_event):
            if not action(verbose=verbose):
                break
        if self.cg_event in ("MC", "BC") or self.event.startswith("Done"):
            self.check_grasp_state(verbose)
        return self.event

    def replay_demonstration_trajectory(self, update_plane_estimator=False, verbose=True):
        self.phase = "replay"
        for d_base_ht_ee in self.proj_base_ht_ee[1:]:
            if not self._track(d_base_ht_ee[:3, 3], verbose, R=d_base_ht_ee[:3, :3]):
                return False
        return True

    def spiral_exploration_for_seeking_contact_changing_event(self, update_plane_estimator=False, verbose=True):
        c, rb = self.cfg, self.robot
        self.phase = "spiral"
        self.p_centre = rb.ee_pose()[1].copy()   # the reference spirals about where it stands
        for uv in spiral_offsets(c.spiral_radius, c.spiral_pitch, c.spiral_step):
            e1, e2 = tangent_basis(self.normal)
            if not self._track(self.p_centre + uv[0] * e1 + uv[1] * e2, verbose):
                return False
        self.event = "SpiralExhausted"
        return False

    def _track(self, target: np.ndarray, verbose: bool, R=None) -> bool:
        """Issue one waypoint until reached or given up on. False once an event has ended the run."""
        for _ in range(self.cfg.waypoint_attempts):
            reached = self._step(target, verbose, R=R)
            if self.event != "Running":
                return False
            if reached:
                break
        return True

    def visual_servoing_on_the_plane(self, update_plane_estimator=False, verbose=True) -> bool:
        """The reference's ``visual_servoing_on_the_plane``: the servo's translation, projected
        onto the plane, until the part has moved ``servo_max_disp`` or ``servo_time`` is up.

        The command accumulates the servo's steps from where the part started rather than being
        re-anchored on the measured pose each iteration, for the reason the free-space servo
        marches from the equilibrium pose: under stiction the part does not move until the
        command leads it by the dead band, and a command re-anchored on a part that has not
        moved never gets there."""
        c, rb, sv, n = self.cfg, self.robot, self.servo, self.normal
        if sv is None:
            return True
        self.phase = "servo"
        init_base_ht_ee = make_transform(*rb.ee_pose())
        _, p_s0 = rb.ee_pose()
        target = p_s0.copy()
        t0 = rb.t
        while rb.t - t0 < c.servo_time:
            xi, info = sv.twist()
            sv.event = "Holding" if xi is None else "Running"
            sv.record_observation(rb, info)
            if xi is not None:                   # target lost: hold the command, as the reference does
                R_now, _ = rb.ee_pose()
                dp = project_to_plane(R_now @ xi[:3], n) * c.period
                dn = float(np.linalg.norm(dp))
                if dn > sv.cfg.max_step:
                    dp = dp * (sv.cfg.max_step / dn)
                target = target + dp
            self._step(target, verbose, servo_err=info.get("err", float("nan")), R=init_base_ht_ee[:3, :3])
            if self.event != "Running":
                sv.event = self.event
                return False
            moved = float(np.linalg.norm(project_to_plane(rb.ee_pose()[1] - p_s0, n)))
            if moved >= c.servo_max_disp:
                sv.event = "DisplacementLimit"
                if verbose:
                    print(f"  [servo] moved {1e3 * moved:.1f} mm, the {1e3 * c.servo_max_disp:.0f} mm bound")
                return True
        sv.event = "Timeout"
        return True

    def _step(self, target: np.ndarray, verbose: bool, servo_err: float = float("nan"), R=None) -> bool:
        """One iteration toward ``target`` on the plane: press, run, observe at rest, test events.
        Returns whether the waypoint counts as reached."""
        c, rb, n = self.cfg, self.robot, self.normal
        p_prev = self._p_prev
        # Onto the transferred plane through the touch point, then in. The surface decides the
        # third coordinate; commanding it from the plane keeps the whole normal error out of the
        # contact except for what press_depth puts there.
        target = target - float(np.dot(n, target - self.p_touch)) * n
        self._target = target
        period = c.stall_period if self._no_move > 0 else c.period
        rb.set_goal(target - c.press_depth * n, R)
        rb.run(period, c.early_exit_disp, p_prev)
        rb.run(c.settle, 0.0, p_prev)   # observe at rest -- see PlaneCfg.settle

        R_now, p_now = rb.ee_pose()
        d_p = p_now - p_prev
        dn = float(np.linalg.norm(d_p))
        F = rb.wrench()
        f_base = R_now @ F[:3]
        f_n = float(np.dot(f_base, n))
        f_t = float(np.linalg.norm(project_to_plane(f_base - self._F0, n)))
        in_contact = f_n >= c.contact_force
        # A stall is a commanded move that did not happen, not simply a small step: a spiral's
        # first turns legitimately ask for sub-millimetre steps.
        asked = float(np.linalg.norm(project_to_plane(target - p_prev, n)))
        blocked = asked > c.min_tran and dn <= c.min_tran
        self._no_move = self._no_move + 1 if blocked else 0
        self._path += float(np.linalg.norm(project_to_plane(d_p, n)))
        sink = -float(np.dot(n, p_now - self.p_touch))
        track_err = float(np.linalg.norm(project_to_plane(p_now - target, n)))
        centre = self.p_centre if self.p_centre is not None else self.p_touch
        t_hat = unit(project_to_plane(d_p, n)) if dn > 1e-9 else tangent_basis(n)[0]
        reach = rb.reach_margin(np.concatenate([t_hat, np.zeros(3)]))
        k = len(self.rows)
        row = dict(
            iter=k, t=rb.t, phase=self.phase, ee_x=p_now[0], ee_y=p_now[1], ee_z=p_now[2],
            step_norm=dn, in_contact=int(in_contact), no_move=self._no_move, path=self._path, period=period,
            # What was asked against what the surface gave: ``spiral_r`` is the commanded radius
            # about the search centre, ``swept_r`` the radius actually reached, ``track_err`` the
            # in-plane gap between the two. A search is only as good as ``swept_r``.
            spiral_r=float(np.linalg.norm(project_to_plane(target - centre, n))),
            swept_r=float(np.linalg.norm(project_to_plane(p_now - centre, n))),
            asked=asked, track_err=track_err,
            n_x=n[0], n_y=n[1], n_z=n[2], sink=sink, height_true=self._height_true(p_now),
            err_normal_deg=self._err_normal(), servo_err=servo_err,
            F_norm=float(np.linalg.norm(F[:3])), F_n=f_n, F_t=f_t,
            T_norm=float(np.linalg.norm(F[3:])), reach_margin=reach, **rb.extra(),
        )
        self.rows.append(row)
        if verbose and (k % 5 == 0 or blocked):
            print(f"  it {k:3d} t={rb.t:6.2f}s [{self.phase:6s}] |dp|={dn * 1e3:5.1f}mm "
                  f"r={100 * row['spiral_r']:4.1f}cm nomove={self._no_move:2d} "
                  f"F_n={f_n:5.1f}N F_t={f_t:5.1f}N sink={1e3 * sink:5.1f}mm "
                  f"track={1e3 * track_err:4.1f}mm reach={100 * reach:4.1f}cm"
                  + "".join(f" {k_}={v:.3g}" for k_, v in rb.extra().items()))

        if dn > c.jump_factor * rb.v_max_lin * period:
            self.event = "BreakingContact(jump)"   # moved faster than the equilibrium pose can
        elif (done := self.terminate_extra_reached(row)) is not None:
            self.event = done
        elif sink >= c.bce_depth and f_n < c.contact_force:
            self.event = "BreakingContact"
        elif c.min_reach_margin > 0.0 and reach <= c.min_reach_margin:
            self.event = "JointLimit"
        elif self._no_move >= c.mce_attempts:
            self.event = "MakingContact"
        elif f_t >= c.mce_force:
            self.event = "MakingContact(force-confirmed)"
        elif len(self.rows) >= self._max_iters:
            self.event = "MaxIters"
        self._p_prev = p_now
        rotation_error = 0.0 if R is None else np.linalg.norm(rotvec_from_matrix(R @ R_now.T))
        return track_err <= c.waypoint_tol and rotation_error <= c.waypoint_rot_tol

    # ------------------------------------------------------------------ output

    def summary(self) -> dict:
        rows = self.rows
        last = rows[-1] if rows else {}
        out = dict(
            **super().summary(), phase=self.phase,
            normal_err_deg=self._err_normal(),
            replay_waypoints=0 if self.demo is None else self.demo.poses.shape[0] - 1,
            path_cm=100.0 * last.get("path", 0.0),
            search_r_cm=100.0 * max((r["spiral_r"] for r in rows if r["phase"] == "spiral"), default=0.0),
            swept_r_cm=100.0 * max((r["swept_r"] for r in rows if r["phase"] == "spiral"), default=0.0),
            track_err_cm=100.0 * max((r["track_err"] for r in rows), default=0.0),
            sink_max_mm=1e3 * max((r["sink"] for r in rows), default=0.0),
            F_max_N=max((r["F_norm"] for r in rows), default=0.0),
            F_t_max_N=max((r["F_t"] for r in rows), default=0.0),
        )
        if self.servo is not None:
            out.update(self.servo.summary())
        return out

    def plot(self, path: Path | str, title: str, f_max: float, progress_key: str | None = None) -> None:
        if not self.rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.rows
        it = np.array([r["iter"] for r in rows])
        col = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        ax[0].plot(it, 1e3 * col("sink"), "r-", label="sink through the transferred plane [mm]")
        if np.isfinite(col("height_true")).any():
            ax[0].plot(it, 1e3 * col("height_true"), "g--", alpha=.7, label="control point above the true surface [mm]")
        ax[0].axhline(1e3 * self.cfg.bce_depth, color="k", ls=":", label="breaking-contact depth")
        ax[0].set_ylabel("[mm]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title(f"plane primitive — event: {self.event}\n{title}", fontsize=8)
        ax[1].plot(it, 100 * col("spiral_r"), "g-", label="commanded radius [cm]")
        ax[1].plot(it, 100 * col("swept_r"), "g--", alpha=.7, label="swept radius [cm]")
        ax[1].plot(it, 100 * col("path"), "b--", alpha=.5, label="tangential path [cm]")
        ax[1].plot(it, 1e3 * col("step_norm"), "k.-", ms=3, alpha=.6, label="|Δp| per iteration [mm]")
        ax[1].plot(it, col("no_move"), "m-", alpha=.7, label="no-move counter")
        ax[1].set_ylabel("[cm] / [mm] / count"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[2].plot(it, col("F_n"), "r-", label="F along the normal (gate input)")
        ax[2].plot(it, col("F_t"), "b-", alpha=.7, label="F tangential, over the start reading")
        ax[2].plot(it, col("T_norm"), "c-", alpha=.6, label="|T|")
        ax[2].axhline(self.cfg.mce_force, color="b", ls=":", label="making-contact force")
        ax[2].axhline(f_max, color="k", ls=":", label="F_m")
        if progress_key is not None and progress_key in rows[0]:
            ax[2].plot(it, 1e3 * col(progress_key), "g-", alpha=.7, label=f"{progress_key} [mm]")
        ax[2].set_ylabel("[N] / [Nm]"); ax[2].set_xlabel("primitive iteration")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        phases = [r["phase"] for r in rows]
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                for a in ax:
                    a.axvline(i - 0.5, color="k", lw=.8, alpha=.5)
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
