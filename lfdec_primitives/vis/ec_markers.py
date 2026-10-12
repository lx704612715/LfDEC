# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Arrow markers showing what a primitive believes about the constraint it is exploiting.

Three arrows from the grasp point -- the keys of an :class:`~..primitives.base.ECState` -- so the
estimate can be watched converging while the drawer opens:

======  ==============================================================================
colour  meaning
======  ==============================================================================
red     the initial guess handed to the primitive -- deliberately wrong, and fixed
blue    the particle filter's current mean direction, which is what the arm pulls along
green   the true axis, for reference (hidden unless the run was given it)
======  ==============================================================================

The blue arrow is the interesting one: it starts on top of the red one and swings onto the green
one over the first few iterations as the filter observes which way the drawer actually moves.

Markers are USD-prim based and are drawn by whichever visualizer backend is active, so this works
under Kit and under the Newton viewer. With no visualizer running the calls are cheap no-ops.
"""

from __future__ import annotations

import torch

__all__ = ["ECAxisMarkers"]


def _quat_aiming_x_along(d: torch.Tensor) -> torch.Tensor:
    """Quaternion (w-first) rotating the marker's +x axis onto each unit direction in ``d``.

    Builds a full orthonormal frame rather than the shortest-arc formula so the result is stable
    when ``d`` is near -x, where the shortest arc is degenerate.
    """
    import isaaclab.utils.math as math_utils

    x = torch.nn.functional.normalize(d, dim=-1)
    # Any reference not parallel to x; +z unless x is nearly vertical, then +y.
    ref = torch.where(
        x[:, 2:3].abs() > 0.9,
        torch.tensor([[0.0, 1.0, 0.0]], device=d.device).expand_as(x),
        torch.tensor([[0.0, 0.0, 1.0]], device=d.device).expand_as(x),
    )
    y = torch.nn.functional.normalize(torch.cross(ref, x, dim=-1), dim=-1)
    z = torch.cross(x, y, dim=-1)
    return math_utils.quat_from_matrix(torch.stack([x, y, z], dim=-1))


class ECAxisMarkers:
    """Draws the EC direction estimate as arrows anchored at the grasp point.

    The arrows are deliberately large -- 45 cm long and 3 cm thick by default. They are anchored
    at the grasp point, which is inside the gripper and up against the cabinet, so a thin arrow
    is mostly hidden by the hand and the drawer front.

    Args:
        prim_path: USD scope for the markers, e.g. ``"/Visuals/ECAxis"``.
        length: arrow length [m] at unit scale.
        shaft: arrow thickness [m]; defaults to 7% of ``length`` so one knob scales the whole
            arrow and a longer arrow does not turn into a needle.
        keys: which arrows this instance will ever draw. Dropping ``"true"`` is how a run says
            it does not want the answer on screen: the primitive still gets the truth for its
            diagnostics, it just is not drawn next to the estimate it is meant to judge.
    """

    def __init__(
        self, prim_path: str = "/Visuals/ECAxis", length: float = 0.45, shaft: float | None = None,
        keys: tuple[str, ...] = ("initial", "estimate", "true"),
    ):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        if shaft is None:
            shaft = 0.07 * length

        def arrow(rgb):
            return sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(length, shaft, shaft),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            )

        colours = {"initial": (0.9, 0.1, 0.1), "estimate": (0.1, 0.35, 0.95), "true": (0.1, 0.8, 0.2)}
        unknown = set(keys) - set(colours)
        if unknown:
            raise KeyError(f"no arrow for {sorted(unknown)}; the keys are {sorted(colours)}")
        self._order = list(keys)
        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path=prim_path,
                                    markers={k: arrow(colours[k]) for k in self._order})
        )

    def update(
        self,
        origin_w: torch.Tensor,
        directions_w: dict[str, torch.Tensor],
        origins_w: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Redraw the arrows.

        Args:
            origin_w: (3,) or (1, 3) arrow tail in **world** coordinates -- the grasp point for a
                translation EC, and the default for any key ``origins_w`` leaves out.
            directions_w: maps ``"initial"`` / ``"estimate"`` / ``"true"`` to a (3,) world-frame
                direction. Keys left out, and keys this instance was not built with, are not
                drawn.
            origins_w: per-arrow tails, for an EC whose estimate has a *position* as well as a
                direction. A revolute axis is a line, so its estimate and the truth are drawn at
                the two different places they pass through, and the arrows converging onto each
                other is the whole point.
        """
        keys = [k for k in self._order if k in directions_w]
        if not keys:
            return
        origin = origin_w.reshape(1, 3)
        def pick(v):
            return torch.as_tensor(v, device=origin.device, dtype=origin.dtype).reshape(3)

        d = torch.stack([pick(directions_w[k]) for k in keys])
        o = torch.stack([pick((origins_w or {}).get(k, origin[0])) for k in keys])
        self._markers.visualize(
            translations=o,
            orientations=_quat_aiming_x_along(d),
            marker_indices=[self._order.index(k) for k in keys],
        )
