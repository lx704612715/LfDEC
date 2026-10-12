# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Panda TCP geometry and finger-command constants."""

PANDA_TCP_OFFSET = [0.0, 0.0, 0.1034]
"""Fingertip pad centre in the ``panda_hand`` frame -- the grasp point, and the control and
wrench reference point. Matches the stock ``ee_frame`` ``ee_tcp`` target."""

PANDA_FINGER_OPEN = 0.04
"""Open finger joint position [m]."""
