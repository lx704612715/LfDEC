# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Geometric checks for rigid bodies passing through an aperture."""
from itertools import product

import numpy as np


def box_aperture_clearance(center, rotation, size, hole_side, slab_thickness):
    """Minimum square-aperture clearance of a box inside the slab, in metres.

    The aperture is centred at the origin; the slab spans z in [-thickness, 0].
    Clip the box to that interval before checking its x/y extent. Parts below
    the slab need not remain inside the aperture. Negative clearance means collision;
    NaN means an invalid pose or no box section inside the slab.
    """
    corners = np.asarray(list(product((-1., 1.), repeat=3))) * (np.asarray(size) / 2)
    vertices = np.asarray(center) + corners @ np.asarray(rotation).T
    points = list(vertices[(vertices[:, 2] >= -slab_thickness) & (vertices[:, 2] <= 0)])
    for first in range(8):
        for bit in (1, 2, 4):
            second = first ^ bit
            if first >= second:
                continue
            a, b = vertices[first], vertices[second]
            dz = b[2] - a[2]
            if abs(dz) < 1e-12:
                continue
            for height in (-slab_thickness, 0.):
                fraction = (height - a[2]) / dz
                if 0 <= fraction <= 1:
                    points.append(a + fraction * (b - a))
    if not np.isfinite(vertices).all() or not points:
        return float("nan")
    return float(hole_side / 2 - np.max(np.abs(np.asarray(points)[:, :2])))
