# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Force, torque and task-progress plots from control-rate logs."""
from pathlib import Path

import numpy as np


def plot_forces(
    ticks: list[dict],
    path: Path | str,
    title: str,
    f_max_lin: float,
    f_max_rot: float,
    progress_key: str | None = None,
    progress_label: str = "",
    progress_scale: float = 1.0,
) -> None:
    """Per-tick force/torque history of a manipulation, with the gate's headroom.

    Takes the control-rate log rather than the per-iteration rows: an iteration ends as soon as
    the EE has moved ``early_exit_disp``, so the per-iteration force is one sample every few
    hundredths of a second and says nothing about what the contact did in between.

    Args:
        ticks: rows with ``t`` and the gate-input wrench ``F_x/F_y/F_z`` and ``T_x/T_y/T_z``
            (bias-corrected and filtered -- the signal the gate actually sees). ``F_raw_*`` is
            drawn if present.
        path: output image path.
        title: figure subtitle.
        f_max_lin / f_max_rot: the gate's limits [N] and [N.m], drawn as the saturation lines.
        progress_key: optional tick column drawn on a twin axis of the headroom panel -- the task's
            own progress variable (drawer opening, door angle).
        progress_label / progress_scale: axis label and multiplier for that column.
    """
    if not ticks:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = lambda k: np.array([r[k] for r in ticks], dtype=float)  # noqa: E731
    has = lambda k: k in ticks[0]  # noqa: E731
    t = col("t")
    F = np.stack([col("F_x"), col("F_y"), col("F_z")], axis=1)
    T = np.stack([col("T_x"), col("T_y"), col("T_z")], axis=1)
    Fn, Tn = np.linalg.norm(F, axis=1), np.linalg.norm(T, axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].set_title(f"contact wrench at the TCP\n{title}", fontsize=9)
    for i, (lab, c) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
        ax[0].plot(t, F[:, i], color=c, lw=.9, label=f"F_{lab} (EE frame)")
    if has("F_raw_x"):
        ax[0].plot(t, np.linalg.norm(np.stack([col("F_raw_x"), col("F_raw_y"), col("F_raw_z")], axis=1), axis=1),
                   color="grey", lw=.7, alpha=.6, label="|F| raw (before bias)")
    ax[0].plot(t, Fn, "k-", lw=1.4, label="|F| (gate input)")
    ax[0].axhline(f_max_lin, color="k", ls=":", lw=1.2, label=f"F_m = {f_max_lin:g} N")
    ax[0].axhline(-f_max_lin, color="k", ls=":", lw=1.2)
    ax[0].set_ylabel("force [N]"); ax[0].legend(fontsize=7, ncol=3); ax[0].grid(alpha=.3)

    for i, (lab, c) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
        ax[1].plot(t, T[:, i], color=c, lw=.9, label=f"T_{lab}")
    ax[1].plot(t, Tn, "k-", lw=1.4, label="|T|")
    ax[1].axhline(f_max_rot, color="k", ls=":", lw=1.2, label=f"T_m = {f_max_rot:g} N.m")
    ax[1].axhline(-f_max_rot, color="k", ls=":", lw=1.2)
    ax[1].set_ylabel("torque [N.m]"); ax[1].legend(fontsize=7, ncol=3); ax[1].grid(alpha=.3)

    # Negative gate headroom activates the escape branch, which retreats along
    # the contact force to reduce the wrench.
    slack_lin, slack_rot = f_max_lin - Fn, f_max_rot - Tn
    ax[2].plot(t, slack_lin, color="tab:purple", lw=1.1, label="F_m - |F| [N]")
    ax[2].plot(t, slack_rot, color="tab:orange", lw=1.1, label="T_m - |T| [N.m]")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].fill_between(t, slack_lin, 0, where=slack_lin < 0, color="tab:purple", alpha=.25,
                       label="linear escape branch active")
    ax[2].fill_between(t, slack_rot, 0, where=slack_rot < 0, color="tab:orange", alpha=.25,
                       label="angular escape branch active")
    ax[2].set_ylabel("gate headroom"); ax[2].grid(alpha=.3)
    if progress_key is not None and has(progress_key):
        rhs = ax[2].twinx()
        rhs.plot(t, progress_scale * col(progress_key), "g-", lw=1.3, label=progress_label)
        rhs.set_ylabel(progress_label, color="g")
        rhs.tick_params(axis="y", colors="g")
        rhs.legend(fontsize=7, loc="upper right")
    ax[2].legend(fontsize=7, loc="lower left"); ax[2].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
