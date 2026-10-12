# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Simulation views used while recording a manipulation."""
from ..primitives import PlanePrimitive


def viewer_frames(env):
    """The Newton GL viewer's framebuffer, ``(H, W, 3)`` uint8, for :class:`RunVideo`. Headless
    (``DISPLAY`` unset) the viewer renders the frame on demand and draws no side panel."""
    viewers = [v for v in env.sim.visualizers if v.cfg.visualizer_type == "newton_gl"]
    if not viewers:
        raise RuntimeError("--record needs the Newton GL viewer (the default --viz newton_gl)")
    return viewers[0].render_rgb_array


def activate_primitive(robot, env, name, primitive):
    """Select the active primitive and show the Factory peg's contact stages in close-up."""
    from ..tasks.envs.factory_peg_ec_cfg import CONTACT_VIEW_EYE, CONTACT_VIEW_LOOKAT

    robot.set_active_primitive(name, primitive)
    if isinstance(primitive, PlanePrimitive):
        env.sim.set_camera_view(CONTACT_VIEW_EYE, CONTACT_VIEW_LOOKAT)
