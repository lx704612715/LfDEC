# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""A printed label for a simulated object, as a USD the scene configs spawn under a link.

The stock Isaac assets are flat grey, and a keypoint detector finds a dozen features on a whole
cabinet front -- nowhere near what the visual servo needs. A label is what a real object would
carry, and it is also what a detector wants: sharp edges, high contrast, no two neighbourhoods
alike. This module turns an image into a thin, visual-only slab wearing it; a scene puts the
slab under the link it belongs to (``{ENV_REGEX_NS}/Cabinet/drawer_top/Sticker``), where Newton
imports it as a visual shape *of that link*, so it rides with the part, blocks nothing and cannot
be touched. A separate body would have to be a collider to render at all under Newton, and a
collider across a drawer front stops the drawer.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["STICKER_IMAGE", "STICKER_THICK", "sticker_texture", "sticker_usd"]

STICKER_IMAGE = Path(__file__).parent / "assets" / "rbo_logo.png"
"""The default label. Any image works; this is the lab's."""

STICKER_THICK = 0.002
STICKER_MARGIN = 0.05
STICKER_RES = 1024
"""Slab thickness [m], the white border kept around the image as a fraction of the slab's height,
and the texels across it. Finer texels do not add features: the shapes then land near the pixel
scale and the renderer's filtering averages them away."""


def sticker_texture(image: Path, aspect: float, res: int = STICKER_RES, margin: float = STICKER_MARGIN):
    """``image`` centred on a white panel of the given width/height ratio, as a PIL image. White
    rather than cropped to the artwork because that is what a sticker looks like."""
    from PIL import Image

    w = int(res)
    h = max(1, round(res / aspect))
    pad = margin * h
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    art = Image.open(image).convert("RGBA")
    s = min((w - 2 * pad) / art.width, (h - 2 * pad) / art.height)
    art = art.resize((max(1, round(s * art.width)), max(1, round(s * art.height))), Image.LANCZOS)
    canvas.paste(art, ((w - art.width) // 2, (h - art.height) // 2), art)
    return canvas


def sticker_usd(size: tuple[float, float], image: Path = STICKER_IMAGE, thick: float = STICKER_THICK,
                res: int = STICKER_RES, margin: float = STICKER_MARGIN) -> Path:
    """A ``size`` (width, height) [m] slab wearing :func:`sticker_texture`, as a USD in the cache.

    Visual only -- no rigid body, no collider -- so it can be spawned under a link and become
    part of it. Textured the way the block's cube is (:func:`..grasp_free_cfg.textured_cube_usd`),
    the one path proven to render on kit-less Newton: a *mesh* with per-face ``st`` coordinates
    and a ``UsdPreviewSurface`` fed by a ``UsdUVTexture``. Every face is authored
    counter-clockwise from outside with the same ``st``, so the label reads correctly from
    whichever side is the front.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    cache = Path(os.environ.get("LFDEC_CACHE", Path.home() / ".cache" / "lfdec")) / "assets"
    cache.mkdir(parents=True, exist_ok=True)
    w, h, t = float(size[0]), float(size[1]), float(thick)
    tag = f"{image.stem}_{1e3 * w:.0f}x{1e3 * h:.0f}x{1e3 * t:.0f}mm_r{res}"
    usd_path = cache / f"sticker_{tag}.usd"
    png_path = cache / f"sticker_{tag}.png"
    if usd_path.exists():
        return usd_path
    sticker_texture(image, w / h, res, margin).save(png_path)

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Sticker")
    stage.SetDefaultPrim(root.GetPrim())
    hx, hy, hz = 0.5 * t, 0.5 * w, 0.5 * h
    quads = [  # four corners of each face, counter-clockwise seen from outside
        [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],
        [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)],
        [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)],
        [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],
        [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
        [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)],
    ]
    mesh = UsdGeom.Mesh.Define(stage, "/Sticker/geometry")
    mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for quad in quads for v in quad])
    mesh.GetFaceVertexCountsAttr().Set([4] * 6)
    mesh.GetFaceVertexIndicesAttr().Set([4 * i + k for i in range(6) for k in range(4)])
    mesh.GetSubdivisionSchemeAttr().Set("none")
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)] * 6)

    material = UsdShade.Material.Define(stage, "/Sticker/Looks/label")
    surface = UsdShade.Shader.Define(stage, "/Sticker/Looks/label/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    tex = UsdShade.Shader.Define(stage, "/Sticker/Looks/label/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{png_path.name}")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.9, 0.9)])
    stage.GetRootLayer().Save()
    return usd_path
