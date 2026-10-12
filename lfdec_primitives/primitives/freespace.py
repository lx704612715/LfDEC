# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Free-space primitive: visually align, transfer and replay the demonstration, then apply
the demonstrated gripper action. A successor-directed approach detects contact
when the next EC requires it. Waypoints advance when reached."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import (
    Demo,
    Primitive,
    RobotLike,
    project_trajectory,
    rotvec_from_matrix,
    unit,
)
from .servo import VisualServo

__all__ = ["FreeCfg", "FreeSpacePrimitive"]


@dataclass
class FreeCfg:
    """Configuration of :class:`FreeSpacePrimitive`."""

    period: float = 0.5           # s per iteration
    waypoint_tol: float = 0.01    # m within which a transit waypoint counts as reached
    waypoint_rot_tol: float = 0.10  # rad, likewise for its orientation
    waypoint_attempts: int = 12
    """Retries per waypoint. Intermediate waypoints may be skipped, but the final pose must
    actually satisfy both tolerances unless expected contact completes the replay."""

    collision_force: float = 8.0
    """[N] contact during transit that is not supposed to be there -> ``MakingContact(early)``.

    Must sit below the controller's ``F_m``. The wrench gate gives way at ``F_m``, so a threshold
    at or above it is unreachable by construction: the equilibrium pose retreats along the contact
    force and the measured wrench plateaus just under the limit, and the primitive would sit
    there pressing on the obstacle and calling it free space."""

    grip_settle: float = 0.5
    """[s] the plant runs after a gripper command before the finger gap is read."""

    approach_dir: np.ndarray | None = None
    """Direction to *move* in once the trajectory is finished, base frame -- into the next EC,
    along the one direction that EC does not admit. ``None`` asks the next primitive
    (``next.get_curt_constraint_direction``) when the planner has linked one, and skips the phase
    otherwise: the primitive is then the reference's backstep."""

    approach_step: float = 0.01   # m of goal advance per iteration while seeking
    touch_force: float = 3.0
    """[N] of resistance *against* the approach direction that confirms the next constraint ->
    ``MakingContact``. The wrench is the F/T-sensor convention -- what the environment does to the
    end-effector -- so pressing into a surface reads as a force along ``-approach_dir``."""
    max_approach: float = 0.06
    """[m] give up with ``NoContact`` once the hand has moved this far along the direction without
    meeting anything. The reference's 6 cm. ``0`` turns the seek off: the segment ends where the
    demonstration ended and the primitive reports ``Done`` -- a grasp is such a segment, and
    seeking after it would read the grip's own preload as a contact with the next EC."""

    min_reach_margin: float = 0.03
    """[m] stop with ``JointLimit`` once the arm has this much travel left along the direction it
    is currently moving in. See ``TransCfg.min_reach_margin``."""

    early_exit_disp: float = 0.005  # m; end the iteration once the hand moved this far


class FreeSpacePrimitive(Primitive):
    """Carry the hand to the next constraint, as the demonstration did, and confirm the handover.

    Args:
        robot: the :class:`~.base.RobotLike` adapter.
        demo: the segment to reproduce. ``demo.poses[0]`` is the pose the transfer is anchored
            on, so a scripted segment has to start with the pose the hand is in.
        cfg: see :class:`FreeCfg`.
        servo: a :class:`~.servo.VisualServo` on ``demo.view``, or ``None`` to start the transfer
            from wherever the hand is.
    """

    def __init__(self, robot: RobotLike, demo: Demo, cfg: FreeCfg | None = None,
                 servo: VisualServo | None = None):
        super().__init__(robot, cfg if cfg is not None else FreeCfg(), demo)
        self.demo, self.servo = demo, servo
        self.dir = None if self.cfg.approach_dir is None else unit(np.asarray(self.cfg.approach_dir, dtype=float))
        self.proj_base_ht_ee: list[np.ndarray] = []
        self.phase = "servo" if servo is not None else "transit"
        self.grip_gap = float("nan")
        self.init_base_ht_ee = np.eye(4)     # where the replay starts, after the alignment
        self.inference_ht_demo = np.eye(4)   # T_inf: the demonstration's frame moved onto it

    def exploitation(self, max_iters: int = 200, verbose: bool = True) -> str:
        """The reference's ``ApproachPrimitive.exploitation``: servo to the pose the
        demonstration started from, carry the segment over, work the gripper, then seek the
        next constraint."""
        rb = self.robot
        self.begin_exploitation()
        self.dir = None if self.cfg.approach_dir is None else unit(np.asarray(self.cfg.approach_dir, dtype=float))
        if not self.visual_servoing_to_starting_pose(verbose):
            return self.event

        # T_inf: where the demonstration's frame has moved to, the reference's inference_ht_demo
        self.transfer_demonstration()
        self.proj_base_ht_ee = project_trajectory(self.demo.poses, self.init_base_ht_ee)

        self.replay_cartesian_demonstrated_trajectory(max_iters, verbose)
        if self.event not in ("Running", "MakingContact(early)"):
            return self.event
        if self.event == "MakingContact(early)":
            self.check_grasp_state(verbose)
            return self.event

        if self.cfg.max_approach <= 0.0:
            self.event = "Done"
            self.check_grasp_state(verbose)
            return self.event
        d_constraint_direction = self.get_next_constraint_direction(rb.ee_pose()[1]) if self.dir is None else self.dir
        if d_constraint_direction is None or float(np.linalg.norm(d_constraint_direction)) < 1e-9:
            self.event = "Done"
            self.check_grasp_state(verbose)
            return self.event
        self.check_grasp_state(verbose)
        self.dir = unit(np.asarray(d_constraint_direction, dtype=float))
        self.moving_towards_the_next_constraint_direction(max_iters, verbose)
        if self.cg_event in ("MC", "BC"):
            self.check_grasp_state(verbose)
        return self.event

    # ------------------------------------------------------------------ phases

    def start_visual_servoing_mode(self) -> bool:
        """Is there an alignment to do? The reference publishes the reference-image name to its
        servo node here; the servo object is handed in directly, so this only says whether both
        it and a demonstrated view are present."""
        return self.servo is not None and self.demo.view is not None

    def visual_servoing_to_starting_pose(self, verbose: bool = True) -> bool:
        """Drive the hand until the camera sees ``demo.view``, the pose the demonstration started
        from. ``True`` when the alignment is good enough to carry the segment over.

        The servo returns as soon as the averaged view error is inside
        ``ServoCfg.tol`` / ``rot_tol``, which is what switches the primitive to the replay."""
        if not self.start_visual_servoing_mode():
            return True
        self.phase = "servo"
        ev = self.servo.run(self.robot, verbose)
        if ev == "Converged":
            return True
        # A contact met while servoing is an event in its own right; a lost or slow servo is not,
        # and folds to NonC for the planner.
        self.event = ev if ev.startswith("MakingContact") else f"Servo{ev}"
        return False

    def check_grasp_state(self, verbose: bool = True) -> None:
        """Set the gripper to the demonstration's final state if it changed over the segment --
        the reference's ``check_grasp_state``. This is where a grasp happens."""
        c, rb = self.cfg, self.robot
        previous_phase, self.phase = self.phase, "grip"
        if not super().check_grasp_state(verbose):
            self.phase = previous_phase
            return
        self.grip_gap = rb.gripper_gap()
        F = rb.wrench()
        self._log(len(self.rows), rb, rb.ee_pose()[1], "grip", waypoint=len(self.proj_base_ht_ee) - 1,
                  gap=0.0, gap_rot=0.0, path=self.rows[-1]["path"] if self.rows else 0.0,
                  seek=0.0, F=F, reach=rb.reach_margin(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])))

    def replay_cartesian_demonstrated_trajectory(self, max_iters: int, verbose: bool) -> str:
        """Track the demonstrated segment re-anchored on the aligned pose, watching for a contact
        that should not be there. The reference's ``replay_cartesian_demonstrated_trajectory``."""
        self.phase = "transit"
        c, rb = self.cfg, self.robot
        # Start replay with the demonstrated gripper state; subsequent waypoints
        # apply any recorded changes, and check_grasp_state applies the terminal state.
        rb.set_gripper(bool(self.demo.grip_closed[0]))
        self._gripper_state = bool(self.demo.grip_closed[0])
        idx, tries, path = 1, 0, 0.0       # waypoints[0] is the pose the hand is already in
        _, p_prev = rb.ee_pose()
        n_wp = len(self.proj_base_ht_ee)
        for k in range(len(self.rows), len(self.rows) + max_iters):
            if idx >= n_wp:
                break
            T_w = self.proj_base_ht_ee[idx]
            rb.set_goal(T_w[:3, 3], T_w[:3, :3])
            rb.run(c.period, c.early_exit_disp, p_prev)
            R_now, p_now = rb.ee_pose()
            gap = float(np.linalg.norm(T_w[:3, 3] - p_now))
            gap_rot = float(np.linalg.norm(rotvec_from_matrix(T_w[:3, :3] @ R_now.T)))
            path += float(np.linalg.norm(p_now - p_prev))
            tries += 1
            reached = gap <= c.waypoint_tol and gap_rot <= c.waypoint_rot_tol
            exhausted = tries >= c.waypoint_attempts
            was_final = idx == n_wp - 1
            if reached or (exhausted and idx < n_wp - 1):
                idx, tries = idx + 1, 0
            F = rb.wrench()
            F_lin = float(np.linalg.norm(F[:3]))
            step = p_now - p_prev
            heading = unit(step) if float(np.linalg.norm(step)) > 1e-9 else unit(T_w[:3, 3] - p_now)
            reach = rb.reach_margin(np.concatenate([heading, np.zeros(3)]))
            self._log(k, rb, p_now, "transit", waypoint=idx - 1, gap=gap, gap_rot=gap_rot, path=path,
                      seek=0.0, F=F, reach=reach)
            if verbose and (k % 5 == 0 or idx >= n_wp):
                print(f"  it {k:3d} t={rb.t:6.2f}s [transit] wp {idx - 1:3d}/{n_wp - 1} "
                      f"gap={100 * gap:5.2f}cm/{math.degrees(gap_rot):5.1f}deg path={100 * path:5.1f}cm "
                      f"|F|={F_lin:5.1f}N reach={100 * reach:5.1f}cm")
            self.check_contact_changing_events(R_now, p_now, F)
            if self.event != "Running":
                return self.event
            if exhausted and not reached and was_final:
                self.event = "ReplayUnreached"
                return self.event
            if c.min_reach_margin > 0.0 and reach <= c.min_reach_margin:
                self.event = "JointLimit"
                return self.event
            p_prev = p_now
            if idx >= n_wp:
                return "Running"
        else:
            self.event = "MaxIters"
            return self.event
        return "Running"

    def check_contact_changing_events(self, R_now, curt_base_p_ee, obs_ee_ft_robot) -> None:
        """Expected next-EC contact completes replay; other collisions fail before gripping."""
        d_constraint_direction = (self.dir if self.dir is not None else
                                  self.get_next_constraint_direction(curt_base_p_ee))
        if d_constraint_direction is not None and (self.d_cg_event == "MC" or self.dir is not None):
            force_in_constraint_direction = -float(np.dot(R_now @ obs_ee_ft_robot[:3],
                                                          d_constraint_direction))
            if force_in_constraint_direction >= self.cfg.touch_force:
                self.event = "MakingContact(early)"
                return
        if np.linalg.norm(obs_ee_ft_robot[:3]) >= self.cfg.collision_force:
            self.event = "UnexpectedContact"

    def moving_towards_the_next_constraint_direction(self, max_iters: int, verbose: bool) -> None:
        """Press along the next constraint's direction until the force confirms it is there --
        the reference's method of the same name."""
        self.phase = "approach"
        c, rb = self.cfg, self.robot
        _, p0 = rb.ee_pose()
        for k in range(len(self.rows), len(self.rows) + max_iters):
            _, p_prev = rb.ee_pose()
            _, p_des = rb.desired_pose()
            rb.set_goal(p_des + c.approach_step * self.dir)
            rb.run(c.period, c.early_exit_disp, p_prev)
            R_now, p_now = rb.ee_pose()
            # Count measured travel because motion may stop before the commanded period.
            sought = float(np.dot(p_now - p0, self.dir))
            F = rb.wrench()
            f_resist = -float(np.dot(R_now @ F[:3], self.dir))
            reach = rb.reach_margin(np.concatenate([self.dir, np.zeros(3)]))
            self._log(k, rb, p_now, "approach", waypoint=len(self.proj_base_ht_ee) - 1,
                      gap=float(np.linalg.norm(p_now - p0)), gap_rot=0.0,
                      path=self.rows[-1]["path"] if self.rows else 0.0,
                      seek=sought, F=F, reach=reach)
            if verbose:
                print(f"  it {k:3d} t={rb.t:6.2f}s [approach] sought={100 * sought:4.1f}cm "
                      f"resist={f_resist:5.1f}N reach={100 * reach:5.1f}cm")
            if f_resist >= c.touch_force:
                self.event = "MakingContact"
                return
            if c.min_reach_margin > 0.0 and reach <= c.min_reach_margin:
                self.event = "JointLimit"
                return
            if sought >= c.max_approach:
                self.event = "NoContact"
                return
        self.event = "MaxIters"

    # ------------------------------------------------------------------ output

    def _log(self, k: int, rb: RobotLike, p: np.ndarray, phase: str, *, waypoint: int, gap: float,
             gap_rot: float, path: float, seek: float, F: np.ndarray, reach: float) -> None:
        self.rows.append(dict(
            iter=k, t=rb.t, phase=phase, ee_x=p[0], ee_y=p[1], ee_z=p[2],
            waypoint=waypoint, gap=gap, gap_rot_deg=math.degrees(gap_rot), path=path, seek=seek,
            F_norm=float(np.linalg.norm(F[:3])), F_x=F[0], F_y=F[1], F_z=F[2],
            T_norm=float(np.linalg.norm(F[3:])), reach_margin=reach, **rb.extra(),
        ))

    def summary(self) -> dict:
        rows = self.rows
        last = rows[-1] if rows else {}
        out = dict(
            **super().summary(), phase=self.phase,
            waypoints=max(len(self.proj_base_ht_ee) - 1, 0), reached=last.get("waypoint", 0),
            final_gap_cm=100.0 * last.get("gap", float("nan")),
            final_gap_deg=last.get("gap_rot_deg", float("nan")),
            path_cm=100.0 * last.get("path", 0.0), seek_cm=100.0 * last.get("seek", 0.0),
            grip_gap_mm=1e3 * self.grip_gap,
            F_max_N=max((r["F_norm"] for r in rows), default=0.0),
            F_transit_max_N=max((r["F_norm"] for r in rows if r["phase"] == "transit"), default=0.0),
        )
        if self.servo is not None:
            out.update(self.servo.summary())
        return out

    def plot(self, path: Path | str, title: str, f_max: float) -> None:
        if not self.rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.rows
        it = np.array([r["iter"] for r in rows])
        col = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        ax[0].plot(it, 100 * col("gap"), "b-", label="distance to the waypoint [cm]")
        ax[0].plot(it, col("gap_rot_deg"), "c--", alpha=.7, label="orientation gap [deg]")
        ax[0].axhline(100 * self.cfg.waypoint_tol, color="k", ls=":", label="waypoint tolerance")
        ax[0].set_ylabel("[cm] / [deg]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title(f"free-space primitive — event: {self.event}\n{title}", fontsize=8)
        ax[1].plot(it, col("waypoint"), "g-", label="waypoint reached")
        ax[1].plot(it, 100 * col("path"), "g--", alpha=.6, label="path travelled [cm]")
        ax[1].plot(it, 100 * col("seek"), "m-", alpha=.7, label="pressed along the next EC [cm]")
        ax[1].set_ylabel("count / [cm]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[2].plot(it, col("F_norm"), "r-", label="|F|")
        ax[2].plot(it, col("T_norm"), "c-", alpha=.7, label="|T|")
        ax[2].axhline(self.cfg.collision_force, color="tab:orange", ls="--", label="unexpected-contact threshold")
        ax[2].axhline(f_max, color="k", ls=":", label="F_m")
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
