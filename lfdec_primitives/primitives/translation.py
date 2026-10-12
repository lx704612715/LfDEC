# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Translation primitive of the LfDEC paper (§3.2): prismatic-joint particle filter and the
sample -> execute -> observe -> update loop. Pure numpy, driven by the Isaac robot adapter
or a plant-free test robot (see :class:`~.base.RobotLike`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import (
    PAPER_BCE_DRAWER,
    PAPER_MCE_T,
    PAPER_MIN_TRAN,
    ECState,
    Demo,
    Primitive,
    RobotLike,
    angle_between,
    clamp_to_cone,
    rotation_taking,
    make_transform,
    unit,
)

# ----------------------------------------------------------------------------- particle filter

@dataclass
class PFCfg:
    n_particles: int = 2000
    init_std: float = 0.10        # rad, around the demo axis
    pred_std: float = 0.02        # rad, per step
    meas_std: float = 0.05        # rad, floor of the measurement noise
    pos_noise: float = 0.0005     # m; a displacement's direction noise is ~ pos_noise / |Δp|


class PrismaticPF:
    """Particles are unit directions in spherical coordinates (θ, φ) of a frame rotated so the
    initial axis is +x (θ = π/2, φ = 0), far from the poles."""

    def __init__(self, u_init: np.ndarray, cfg: PFCfg, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.R0 = rotation_taking(unit(u_init), np.array([1.0, 0.0, 0.0]))  # base -> PF frame
        n = cfg.n_particles
        self.th = math.pi / 2 + rng.normal(0.0, cfg.init_std, n)
        self.ph = rng.normal(0.0, cfg.init_std, n)
        self.w = np.full(n, 1.0 / n)
        self.n_updates = 0
        self._wrap()

    def _wrap(self) -> None:
        th = np.mod(self.th, 2 * math.pi)
        over = th > math.pi
        th[over] = 2 * math.pi - th[over]
        self.ph[over] += math.pi
        self.th = th
        self.ph = (self.ph + math.pi) % (2 * math.pi) - math.pi

    def _u_pf(self) -> np.ndarray:
        st = np.sin(self.th)
        return np.stack([st * np.cos(self.ph), st * np.sin(self.ph), np.cos(self.th)], axis=1)

    def to_base(self, u_pf: np.ndarray) -> np.ndarray:
        return u_pf @ self.R0

    def predict(self) -> None:
        self.th += self.rng.normal(0.0, self.cfg.pred_std, self.th.shape)
        self.ph += self.rng.normal(0.0, self.cfg.pred_std, self.ph.shape)
        self._wrap()

    def mean_direction(self) -> np.ndarray:
        return unit(self.to_base((self.w[:, None] * self._u_pf()).sum(axis=0)))

    def angular_std(self) -> float:
        u = self._u_pf()
        mu = unit((self.w[:, None] * u).sum(axis=0))
        ang = np.arccos(np.clip(u @ mu, -1.0, 1.0))
        return float(math.sqrt((self.w * ang ** 2).sum()))

    def sample(self) -> np.ndarray:
        i = self.rng.choice(self.th.size, p=self.w)
        return self.to_base(self._u_pf()[i])

    def _likelihood(self, u_obs: np.ndarray, sigma: float) -> np.ndarray:
        ang = np.arccos(np.clip(self._u_pf() @ u_obs, -1.0, 1.0))
        return np.exp(-0.5 * (ang / sigma) ** 2)

    def update(self, u_obs_base: np.ndarray, disp_norm: float | None = None) -> None:
        u_obs = self.R0 @ unit(u_obs_base)
        sigma = self.cfg.meas_std
        if disp_norm:
            sigma = math.hypot(sigma, self.cfg.pos_noise / disp_norm)  # short steps carry a noisier direction
        # Preserve the prior particles for unlikely observations; grasp loss is detected
        # against a latched constraint axis by the primitive.
        lik = self._likelihood(u_obs, sigma)
        w = self.w * lik
        s = w.sum()
        self.w = w / s if (s > 0.0 and np.isfinite(s)) else np.full_like(w, 1.0 / w.size)
        self.n_updates += 1
        if 1.0 / float((self.w ** 2).sum()) < 0.5 * self.w.size:
            self._resample()

    def _resample(self) -> None:
        n = self.w.size
        idx = np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(n)) / n)
        idx = np.clip(idx, 0, n - 1)
        self.th, self.ph = self.th[idx].copy(), self.ph[idx].copy()
        self.w = np.full(n, 1.0 / n)


# ----------------------------------------------------------------------------- primitive

@dataclass
class TransCfg:
    period: float = 0.5           # s per iteration
    stall_period: float = 0.25    # s per iteration once the EE has stopped moving
    delta_p: float = 0.10         # m goal distance ahead of the equilibrium point; must exceed F_m/Kp so the gate sets the stall force
    anchor_beta: float = 0.5      # decay per iteration of the side gap (p_des − p_ee) perpendicular to the sampled axis
    min_tran: float = PAPER_MIN_TRAN
    mce_attempts: int = PAPER_MCE_T
    mce_force_attempts: int = 10  # no-move attempts at >= mce_force_frac * F_m -> MakingContact (end stop)
    mce_force_frac: float = 0.8
    bce_disp: float = PAPER_BCE_DRAWER
    bce_latch_std_deg: float = 5.0  # PF angular std at or below which the BCE reference axis is latched
    bce_latch_updates: int = 10     # and this many updates must have gone in first
    max_allowed_deg: float = 30.0 # executed direction is kept inside this cone around the demo axis
    terminate_p: float | None = None
    terminate_extra: tuple[str, float] | None = None
    """``(key, threshold)`` checked against :meth:`RobotLike.extra`: the primitive stops with
    ``Done(key->threshold)`` once the diagnostic has reached the threshold from the side it
    started on (:meth:`~.base.Primitive.terminate_extra_reached`). This is the *task's* success
    criterion, not one of the paper's EC events -- for the drawer, its joint against the stock
    task's own success threshold. Kept separate from ``terminate_p``, which watches EE travel and
    so counts a slipping grasp as progress."""
    min_reach_margin: float = 0.03
    """Stop with JointLimit when remaining arm travel along the estimated axis
    falls below this distance [m]. Zero disables the check."""
    jump_factor: float = 3.0      # |Δp| > jump_factor * V_m * period -> the handle was lost
    early_exit_disp: float = 0.005  # m; end the iteration once the EE moved this far
    eqp_dist: float = 0.015       # reference's successor-directed equilibrium preload [m]
    grip_settle: float = 0.5


class TranslationPrimitive(Primitive):
    """Pull along a prismatic EC whose axis is only approximately known, learning it as it goes."""

    def __init__(self, robot: RobotLike, pf: PrismaticPF, u_init: np.ndarray, cfg: TransCfg,
                 u_true: np.ndarray | None = None, demo: Demo | None = None):
        super().__init__(robot, cfg, demo)
        self.pfo_3d = pf
        self.u_init = unit(u_init)
        self.demo_base_EC_parameters = self.u_init.copy()
        self.u_true = None if u_true is None else unit(u_true)

    def get_curt_admissible_motion_direction(self, curt_base_p_ee=None, inference_ht_demo=None) -> np.ndarray:
        """The estimated axis. Which way along it is the demonstration's business."""
        if inference_ht_demo is not None or self.inf_base_EC_parameters is None:
            R = np.eye(3) if inference_ht_demo is None else inference_ht_demo[:3, :3]
            return R @ self.demo_base_EC_parameters
        return self.pfo_3d.mean_direction()

    def get_curt_constraint_direction(self, curt_base_p_ee=None, inference_ht_demo=None):
        if self.demo_base_expl_direction is None:
            return None
        G = inference_ht_demo if inference_ht_demo is not None else self.inference_ht_demo
        return self.demo_base_expl_direction.copy() if G is None else G[:3, :3] @ self.demo_base_expl_direction

    def maintain_contact_action(self, curt_base_ht_ee):
        d_ee_ht_ee = np.eye(4)
        if self.next_EC_primitive is None or self.d_cg_event == "MC":
            return d_ee_ht_ee
        next_admissible_motion_direction = self.next_EC_primitive.get_curt_admissible_motion_direction(
            curt_base_ht_ee[:3, 3], self.inference_ht_demo)
        if next_admissible_motion_direction is None:
            return d_ee_ht_ee
        next_admissible_motion_direction = np.asarray(next_admissible_motion_direction).reshape(3, -1)[:, 0]
        curt_admissible_motion_direction = self.get_curt_admissible_motion_direction()
        base_maintain_contact_direction = (next_admissible_motion_direction -
            np.dot(next_admissible_motion_direction, curt_admissible_motion_direction) * curt_admissible_motion_direction)
        d_ee_ht_ee[:3, 3] = self.cfg.eqp_dist * curt_base_ht_ee[:3, :3].T @ base_maintain_contact_direction
        return d_ee_ht_ee

    def seeking_contact_action(self, curt_base_ht_ee):
        self.pfo_3d.predict()
        self.d_base_direction = clamp_to_cone(self.pfo_3d.sample(), self.u_init,
                                               math.radians(self.cfg.max_allowed_deg))
        R_des, p_des = self.robot.desired_pose()
        gap = p_des - curt_base_ht_ee[:3, 3]
        d_base_ht_ee = make_transform(R_des, p_des + self.cfg.delta_p * self.d_base_direction -
            self.cfg.anchor_beta * (gap - np.dot(gap, self.d_base_direction) * self.d_base_direction))
        return d_base_ht_ee

    def update_prismatic_tracker(self, obs_base_ht_ee, ee_maintain_contact_direction, verbose=False):
        obs_base_disp_ee = obs_base_ht_ee[:3, 3] - self.last_update_ht_ee[:3, 3]
        translation = float(np.linalg.norm(obs_base_disp_ee))
        updated = translation > self.cfg.min_tran
        if updated:
            self.pfo_3d.update(obs_base_disp_ee / translation, disp_norm=translation)
            self.no_movement_count = 0
            base_maintain_contact_direction = self._maintain_base_ht_ee[:3, :3] @ ee_maintain_contact_direction[:3, 3]
            norm = np.linalg.norm(base_maintain_contact_direction)
            self.movement_in_maintain_contact_direction.append(
                0.0 if norm < 1e-9 else float(obs_base_disp_ee @ (base_maintain_contact_direction / norm)))
            self.total_movement.append(translation)
            self.last_update_ht_ee = obs_base_ht_ee.copy()
            self.inf_base_EC_parameters = self.pfo_3d.mean_direction()
        else:
            self.no_movement_count += 1
        return updated

    def check_contact_changing_events(self):
        if self.no_movement_count >= self.cfg.mce_attempts:
            self.event = "MakingContact"
        if self.next_EC_primitive is not None and abs(sum(self.movement_in_maintain_contact_direction)) >= self.cfg.bce_disp:
            self.event = "BreakingContact"

    def ec_state(self) -> ECState:
        d = {"initial": self.u_init, "estimate": self.pfo_3d.mean_direction()}
        if self.u_true is not None:
            d["true"] = self.u_true
        return ECState("prismatic", d, std_deg=math.degrees(self.pfo_3d.angular_std()))

    def _err(self, u: np.ndarray) -> float:
        return math.degrees(angle_between(u, self.u_true)) if self.u_true is not None else float("nan")

    def exploitation(self, max_iters: int = 200, verbose: bool = True) -> str:
        c, rb, pf = self.cfg, self.robot, self.pfo_3d
        self.begin_exploitation()
        self.transfer_demonstration()
        self.inf_base_EC_parameters = self.inference_ht_demo[:3, :3] @ self.demo_base_EC_parameters
        self.u_init = self.inf_base_EC_parameters.copy()
        pf.__init__(self.u_init, pf.cfg, pf.rng)
        _, p_prev = rb.ee_pose()
        self.last_update_ht_ee = self.init_base_ht_ee.copy()
        self.no_movement_count = 0
        self.total_movement = []
        self.movement_in_maintain_contact_direction = []
        no_move, travel = 0, 0.0
        perp_departure = np.zeros(3)
        u_ref: np.ndarray | None = None  # latched BCE reference axis
        for k in range(max_iters):
            curt_base_ht_ee = make_transform(*rb.ee_pose())
            self._maintain_base_ht_ee = curt_base_ht_ee
            d_maintain_ee_ht_ee = self.maintain_contact_action(curt_base_ht_ee)
            d_seeking_base_ht_ee = self.seeking_contact_action(curt_base_ht_ee)
            d_base_ht_ee = d_seeking_base_ht_ee @ d_maintain_ee_ht_ee
            u_s = self.d_base_direction
            rb.set_goal(d_base_ht_ee[:3, 3])
            period = c.stall_period if no_move > 0 else c.period
            rb.run(period, c.early_exit_disp, p_prev)
            R_now, p_now = rb.ee_pose()
            d = p_now - p_prev
            dn = float(np.linalg.norm(d))
            if dn > c.jump_factor * rb.v_max_lin * period:
                self.event = "BreakingContact(jump)"  # the EE moved faster than the equilibrium point can: handle lost
                break
            backward = bool(np.dot(d, u_s) <= 0.0)
            updated = self.update_prismatic_tracker(make_transform(R_now, p_now), d_maintain_ee_ht_ee)
            no_move = self.no_movement_count
            u_mean = pf.mean_direction()
            travel += float(np.dot(d, u_mean))
            # Latch a converged axis so constraint departure is measured independently
            # of the filter updates that follow the current displacement.
            if (
                u_ref is None
                and pf.n_updates >= c.bce_latch_updates
                and pf.angular_std() <= math.radians(c.bce_latch_std_deg)
            ):
                u_ref = u_mean
            if u_ref is None:
                perp_disp = 0.0
            else:
                perp_departure += d - np.dot(d, u_ref) * u_ref
                perp_disp = float(np.linalg.norm(perp_departure))
            F = rb.wrench()
            F_lin = float(np.linalg.norm(F[:3]))
            # Asked along the PF mean, not the sampled direction: the constraint decides where the
            # EE actually goes, so "how much further can the arm follow this constraint" is the
            # question, not "how far could it go if it were free".
            reach = rb.reach_margin(np.concatenate([u_mean, np.zeros(3)]))
            row = dict(
                iter=k, t=rb.t, ee_x=p_now[0], ee_y=p_now[1], ee_z=p_now[2],
                step_norm=dn, updated=int(updated), no_move=no_move, travel=travel, period=period,
                u_s_x=u_s[0], u_s_y=u_s[1], u_s_z=u_s[2], mu_x=u_mean[0], mu_y=u_mean[1], mu_z=u_mean[2],
                ang_std_deg=math.degrees(pf.angular_std()), err_mean_deg=self._err(u_mean), err_sample_deg=self._err(u_s),
                err_init_deg=math.degrees(angle_between(u_mean, self.u_init)), perp_disp=perp_disp,
                F_norm=F_lin, F_x=F[0], F_y=F[1], F_z=F[2], T_norm=float(np.linalg.norm(F[3:])),
                F_along=float(np.dot(R_now @ F[:3], u_mean)),
                gap_cm=100.0 * float(np.linalg.norm(rb.desired_pose()[1] - p_now)),
                backward=int(backward and dn > c.min_tran), reach_margin=reach, **rb.extra(),
            )
            self.rows.append(row)
            if verbose and (k % 5 == 0 or not updated):
                print(f"  it {k:3d} t={rb.t:6.2f}s |dp|={dn*1e3:5.1f}mm upd={int(updated)} nomove={no_move:2d} "
                      f"travel={travel*100:5.1f}cm err(mean)={row['err_mean_deg']:5.1f}deg std={row['ang_std_deg']:4.1f}deg "
                      f"|F|={F_lin:5.1f}N F∥={row['F_along']:+6.1f}N |T|={row['T_norm']:4.1f}Nm gap={row['gap_cm']:4.1f}cm "
                      f"reach={100 * reach:4.1f}cm"
                      + "".join(f" {k_}={v:.3g}" for k_, v in rb.extra().items()) + ("  <-backward" if row["backward"] else ""))
            if (done := self.terminate_extra_reached(row)) is not None:
                self.event = done
                break
            if c.min_reach_margin > 0.0 and reach <= c.min_reach_margin:
                self.event = "JointLimit"
                break
            self.check_contact_changing_events()
            if self.event != "Running":
                break
            if no_move >= c.mce_force_attempts and F_lin >= c.mce_force_frac * rb.f_max_lin:
                self.event = "MakingContact(force-confirmed)"
                break
            if perp_disp > c.bce_disp:
                self.event = "BreakingContact"
                break
            if c.terminate_p is not None and travel >= c.terminate_p:
                self.event = "Done"
                break
            p_prev = p_now
        else:
            self.event = "MaxIters"
        if self.cg_event in ("MC", "BC") or self.event.startswith("Done"):
            self.check_grasp_state(verbose)
        return self.event

    def summary(self) -> dict:
        rows = self.rows
        last = rows[-1] if rows else {}
        return dict(
            **super().summary(), init_err_deg=self._err(self.u_init),
            final_err_deg=last.get("err_mean_deg", float("nan")), final_std_deg=last.get("ang_std_deg", float("nan")),
            # A run that ends in BreakingContact keeps updating from post-break displacements, so
            # its last row is not its estimate of the constraint. Report the best as well.
            best_err_deg=min((r["err_mean_deg"] for r in rows), default=float("nan")),
            iters_to_5deg=next((r["iter"] for r in rows if r["err_mean_deg"] < 5.0), None),
            travel_cm=100.0 * last.get("travel", 0.0), n_updates=self.pfo_3d.n_updates,
            F_max_N=max((r["F_norm"] for r in rows), default=0.0),
        )

    def plot(self, path: Path | str, title: str, f_max: float, min_tran: float | None = None) -> None:
        if not self.rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.rows
        it = np.array([r["iter"] for r in rows])
        col = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
        fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
        ax[0].plot(it, col("err_mean_deg"), "b-", label="angle(PF mean, true axis)")
        ax[0].plot(it, col("err_sample_deg"), "c.", ms=3, alpha=.6, label="angle(executed sample, true axis)")
        ax[0].fill_between(it, 0, col("ang_std_deg"), color="grey", alpha=.25, label="PF angular std")
        ax[0].axhline(5, color="k", ls=":"); ax[0].set_ylabel("[deg]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title(f"translation primitive — event: {self.event}\n{title}", fontsize=8)
        if "drawer_q" in rows[0]:
            ax[1].plot(it, 100 * col("drawer_q"), "g-", label="drawer joint [cm]")
        ax[1].plot(it, 100 * col("travel"), "g--", label="EE travel along PF axis [cm]")
        ax[1].set_ylabel("[cm]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[2].plot(it, 1e3 * col("step_norm"), "k.-", ms=3, label="|Δp| per iteration [mm]")
        ax[2].axhline(1e3 * (min_tran or PAPER_MIN_TRAN), color="r", ls=":", label="update threshold")
        ax[2].plot(it, col("no_move"), "m-", alpha=.7, label="no-move counter")
        ax[2].set_ylabel("[mm] / count"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        ax[3].plot(it, col("F_norm"), "r-", label="|F| (gate input)")
        ax[3].plot(it, col("F_along"), "m--", label="F along PF axis")
        ax[3].plot(it, col("T_norm"), "c-", label="|T|")
        ax[3].plot(it, 100 * col("perp_disp"), "b-", alpha=.6, label="perp. disp. (BCE) [cm]")
        ax[3].axhline(f_max, color="k", ls=":", label="F_m")
        ax[3].set_ylabel("[N] / [Nm] / [cm]"); ax[3].set_xlabel("primitive iteration"); ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
