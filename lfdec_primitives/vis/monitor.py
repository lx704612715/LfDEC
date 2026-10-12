# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""One live window over a whole sequence of EC primitives.

Three things decide whether a run worked, and all three are invisible in a viewer that only
shows the arm moving:

==========  ====================================================================================
panel       what it answers
==========  ====================================================================================
``align``   *Is the hand looking at the object?* The demonstrated view beside what the wrist
            camera sees now, the matched keypoints joined, green where the M-estimator kept the
            match and red where it threw it out. A servo that will not converge is almost always
            matching the gripper or the background, and that is visible here and nowhere else.
``force``   *What is the contact doing?* The gate-input wrench at the control rate against
            ``F_m``. A primitive stalls when the gate closes, so the force curve is the
            explanation for most of what the arm does or fails to do.
``ec``      *Is the constraint being learnt?* The angle between the live estimate and the
            demonstration's prior, with the estimator's own spread as a band around it -- the EC
            parameters arriving. Truth is drawn too when the run was given it; a run driven by a
            task file never is.
==========  ====================================================================================

The window spans the whole sequence rather than one primitive: :meth:`ECMonitor.begin` marks
each node on the time axes, so the handover from a grasp to a pull is one picture. Hand it to
:class:`~.planner.ECPlanner` as its ``on_node`` hook (or, in Isaac, through the robot adapter,
which already refreshes the simulator's arrows at the same rate) and it needs nothing else.

This is matplotlib, which puts it in the same process as the plant and the simulator's own
viewer; :meth:`ECMonitor.update` redraws at most ``hz`` times a second, so calling it every
control tick costs nothing measurable. Without a display matplotlib falls back to a
non-interactive backend, nothing appears, and :meth:`ECMonitor.close` still writes the final
frame -- which is the picture worth keeping anyway.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..primitives.base import Primitive, angle_between, unit

__all__ = ["ECMonitor"]

PANELS = ("align", "force", "ec")


class ECMonitor:
    """The live window, and the planner's ``on_node`` observer.

    Args:
        f_max_lin / f_max_rot: the wrench gate's limits [N] and [N.m], drawn as the saturation
            lines the force panel is read against.
        panels: which of ``align``, ``force``, ``ec`` to build, in that order. A run with no
            camera has nothing to put in ``align`` and an empty axis is a distraction.
        window: [s] of history on the time axes; ``0`` keeps the whole run.
        hz: redraw rate.
        title: window and figure title.
    """

    def __init__(self, f_max_lin: float, f_max_rot: float, panels: tuple[str, ...] = PANELS,
                 window: float = 0.0, hz: float = 5.0, title: str = "EC execution"):
        import matplotlib.pyplot as plt

        unknown = [p for p in panels if p not in PANELS]
        if unknown or not panels:
            raise ValueError(f"panels must be a non-empty selection from {PANELS}, got {panels}")

        self._plt = plt
        self.panels = tuple(panels)
        self.window, self.f_max_lin, self.f_max_rot = window, f_max_lin, f_max_rot
        self._min_dt, self._last = 1.0 / hz, -1e9
        self.node, self.primitive = "", None
        self.ec_rows: list[dict] = []
        """One row per redraw while a node with an :class:`~.base.ECState` is running: the
        estimate's angle from the prior, its spread, and its error against the truth."""
        self._pending_marks: list[str] = []

        plt.ion()
        heights = {"align": 4.0, "force": 2.6, "ec": 2.2}   # inches, and the row ratios
        rows = [heights[p] for p in self.panels]
        self.fig, axes = plt.subplots(len(self.panels), 1, num=title, figsize=(10.5, sum(rows)),
                                      gridspec_kw=dict(height_ratios=rows))
        self.ax = dict(zip(self.panels, np.atleast_1d(axes), strict=True))
        self._time_axes = [self.ax[p] for p in ("force", "ec") if p in self.ax]
        for ax in self._time_axes[1:]:
            # One time axis for the whole sequence: the point of the window is that the force at
            # the handover and the estimate after it are read off the same instant.
            ax.sharex(self._time_axes[0])
        if "align" in self.ax:
            self._build_align()
        if "force" in self.ax:
            self._build_force()
        if "ec" in self.ax:
            self._build_ec()
        if self._time_axes:
            self._time_axes[-1].set_xlabel("t [s]")
        self.fig.suptitle(title, fontsize=10)
        self.fig.tight_layout()

    # ------------------------------------------------------------------ construction

    def _build_align(self) -> None:
        ax = self.ax["align"]
        ax.set_axis_off()
        self._align_im = None
        self._align_title = ax.set_title("visual alignment: demonstrated view | now -- waiting",
                                         fontsize=8)

    def _build_force(self) -> None:
        ax = self.ax["force"]
        self._force_lines = {}
        for lab, c in zip("xyz", ("tab:red", "tab:green", "tab:blue"), strict=True):
            self._force_lines[f"F_{lab}"] = ax.plot([], [], color=c, lw=.8, label=f"F_{lab}")[0]
        self._force_lines["F"] = ax.plot([], [], "k-", lw=1.5, label="|F| (gate input)")[0]
        ax.axhline(self.f_max_lin, color="k", ls=":", lw=1.1, label=f"F_m = {self.f_max_lin:g} N")
        ax.axhline(-self.f_max_lin, color="k", ls=":", lw=1.1)
        ax.set_ylabel("force [N]")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=5, loc="upper left")

    def _build_ec(self) -> None:
        ax = self.ax["ec"]
        self._ec_lines = {
            "moved": ax.plot([], [], color="tab:blue", lw=1.4,
                             label="estimate vs the demonstration's prior (band: estimator spread)")[0],
            "err": ax.plot([], [], color="tab:green", lw=1.2, label="estimate vs the truth")[0],
        }
        self._ec_band = None
        ax.set_ylabel("EC parameter [deg]")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3, loc="upper left")

    # ------------------------------------------------------------------ driving it

    def begin(self, node: str, primitive: Primitive) -> None:
        """The node about to run. Marks the handover on the time axes.

        This is :class:`~.planner.ECPlanner`'s ``on_node`` signature, so the planner can be
        handed the monitor directly.
        """
        self.node, self.primitive = node, primitive
        self._pending_marks.append(node)

    def update(self, ticks: list[dict], force: bool = False) -> None:
        """Redraw from the robot's control-rate tick list, at most ``hz`` times a second.

        The EC panel samples the active primitive here rather than reading its per-iteration
        rows: an iteration is half a second and every EC type logs different columns, while
        :meth:`~.base.Primitive.ec_state` is the same for all of them.
        """
        now = time.monotonic()
        if not force and now - self._last < self._min_dt:
            return
        self._last = now
        t_now = float(ticks[-1]["t"]) if ticks else 0.0
        state = None if self.primitive is None else self.primitive.ec_state()
        self._sample_ec(t_now, state)
        if "align" in self.ax:
            self._draw_align()
        if "force" in self.ax and ticks:
            self._draw_force(ticks)
        if "ec" in self.ax:
            self._draw_ec()
        self._draw_marks(t_now)
        self.fig.suptitle(self._headline(t_now, state), fontsize=10)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

    def frame(self) -> np.ndarray:
        """The figure as it stands, ``(H, W, 3)`` uint8 RGB -- what :class:`~.video.RunVideo`
        puts beside the simulator's view. Works on any backend, window or not."""
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self, path: Path | str | None = None) -> None:
        """Save the final frame if asked, then close the window."""
        if path is not None:
            self.fig.savefig(str(path), dpi=130)
        self._plt.close(self.fig)

    # ------------------------------------------------------------------ panels

    def _headline(self, t: float, state) -> str:
        prim = self.primitive
        head = f"t = {t:5.1f} s"
        if prim is not None:
            head += f"   [{self.node}] {type(prim).__name__}  {prim.event}"
            if (phase := getattr(prim, "phase", None)) is not None:
                head += f"  phase={phase}"
        if state is not None:
            head += (f"   {state.kind} EC: " + " ".join(f"{v:+.3f}" for v in state.directions["estimate"]))
            if math.isfinite(state.std_deg):
                head += f" +- {state.std_deg:.1f} deg"
        return head

    def _draw_align(self) -> None:
        servo = None if self.primitive is None else self.primitive.servo
        img = None if servo is None else servo.match_image()
        if img is None:
            if self._align_im is not None:
                self._align_im.set_visible(False)
            message = "no servo for this node" if servo is None else f"no current matches [{servo.event}]"
            self._align_title.set_text(f"visual alignment: {message}")
            return
        rgb = img[..., ::-1]   # match_image is BGR, for cv2
        if self._align_im is None:
            self._align_im = self.ax["align"].imshow(rgb)
        else:
            self._align_im.set_data(rgb)
            self._align_im.set_visible(True)
        last = servo.rows[-1] if servo.rows else {}
        # Outside the servo phase this is a retained observation, not a live camera feed.
        mode = "last observation" if getattr(self.primitive, "phase", None) != "servo" else "now"
        self._align_title.set_text(
            f"visual alignment: demonstrated view | {mode} -- {last.get('n_inliers', 0)} inliers of "
            f"{last.get('n_matched', 0)} matched, {1e3 * last.get('err_pos', float('nan')):.1f} mm / "
            f"{math.degrees(last.get('err_rot', float('nan'))):.1f} deg to go  [{servo.event}]")

    def _draw_force(self, ticks: list[dict]) -> None:
        ax = self.ax["force"]
        t = np.array([r["t"] for r in ticks], dtype=float)
        keep = t >= (t[-1] - self.window) if self.window > 0 else np.ones_like(t, bool)
        t = t[keep]
        F = np.stack([[r[f"F_{k}"] for r in ticks] for k in "xyz"], axis=1)[keep]
        for i, lab in enumerate("xyz"):
            self._force_lines[f"F_{lab}"].set_data(t, F[:, i])
        self._force_lines["F"].set_data(t, np.linalg.norm(F, axis=1))
        hi = max(float(np.abs(F).max()) * 1.15, self.f_max_lin * 1.3)
        ax.set_ylim(-hi, hi)
        ax.set_xlim(t[0], max(t[-1], t[0] + 1e-3))

    def _sample_ec(self, t: float, state) -> None:
        if state is None:
            return
        est = state.directions["estimate"]
        row = dict(t=t, node=self.node, kind=state.kind, std_deg=state.std_deg,
                   moved_deg=math.degrees(angle_between(est, state.directions["initial"])),
                   err_deg=(math.degrees(angle_between(est, state.directions["true"]))
                            if "true" in state.directions else float("nan")),
                   **{f"u_{k}": float(v) for k, v in zip("xyz", unit(est), strict=True)})
        self.ec_rows.append(row)

    def _draw_ec(self) -> None:
        if not self.ec_rows:
            return
        ax = self.ax["ec"]
        t = np.array([r["t"] for r in self.ec_rows])
        moved = np.array([r["moved_deg"] for r in self.ec_rows])
        std = np.array([r["std_deg"] for r in self.ec_rows])
        err = np.array([r["err_deg"] for r in self.ec_rows])
        self._ec_lines["moved"].set_data(t, moved)
        self._ec_lines["err"].set_data(t, err)
        if self._ec_band is not None:
            self._ec_band.remove()
        self._ec_band = None
        if np.isfinite(std).any():
            self._ec_band = ax.fill_between(t, moved - std, moved + std, color="tab:blue", alpha=.2)
        hi = float(np.nanmax(np.concatenate([moved + np.nan_to_num(std), err, [1.0]])))
        ax.set_ylim(min(0.0, float(np.nanmin(moved - np.nan_to_num(std)))), 1.2 * hi)
        if "force" not in self.ax:   # otherwise the force panel owns the shared time axis
            ax.set_xlim(t[0], max(t[-1], t[0] + 1e-3))

    def _draw_marks(self, t: float) -> None:
        """A dotted line and a label at each handover, on every time axis."""
        while self._pending_marks:
            node = self._pending_marks.pop(0)
            for ax in self._time_axes:
                ax.axvline(t, color="k", ls="--", lw=.8, alpha=.5)
            if self._time_axes:
                self._time_axes[0].text(t, 0.02, f" {node}", transform=self._time_axes[0].get_xaxis_transform(),
                                        va="bottom", ha="left", fontsize=7, alpha=.7)
