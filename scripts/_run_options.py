# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Display bootstrap and output paths, independent of the simulator."""
from __future__ import annotations

import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

EXAMPLES = ("block_grasp_lift", "drawer_grasp_pull", "door_grasp_open", "peg_insert",
            "factory_peg_insert", "nut_turn")


def configure_display(argv: list[str]) -> None:
    """Run before Isaac/pyglet imports: EGL and Agg unless explicitly watching."""
    if "--show" not in argv:
        os.environ.pop("DISPLAY", None)
        os.environ["MPLBACKEND"] = "Agg"


def configure_outputs(args, repo: Path) -> None:
    """Allocate a distinct run directory; validate recording before loading a scene."""
    if not math.isfinite(args.record_speed) or args.record_speed <= 0:
        raise ValueError("--record-speed must be a finite positive number")
    if args.show and (args.no_monitor or "none" in (args.visualizer or [])):
        raise ValueError("--show requires both the simulation viewer and monitor")
    if args.record and (args.no_record or "newton_gl" not in (args.visualizer or [])):
        raise ValueError("--record requires --viz newton_gl and cannot accompany --no-record")
    name = args.example or Path(args.task_file or "drawer_grasp_pull.yaml").stem
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.out_dir = str(repo / "outputs" / name / f"{stamp}_{uuid4().hex[:8]}")
    out = Path(args.out_dir).resolve()
    recording = not args.no_record and "newton_gl" in (args.visualizer or [])
    if recording:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Video recording requires ffmpeg. Install it (Ubuntu: sudo apt install ffmpeg), "
                               "or use --no-record for a run without video.")
        args.record = str(Path(args.record).resolve()) if args.record else str(out / "run.mp4")
        if Path(args.record).suffix.lower() != ".mp4":
            raise ValueError("--record must name an .mp4 file")
    else:
        args.record = None
    if args.show:
        if os.name == "posix" and not os.environ.get("DISPLAY"):
            raise RuntimeError("--show requires a desktop display; run without --show to record offline")
        # Tk is included with uv's managed Python. Check it before building the simulation.
        import matplotlib
        matplotlib.use("TkAgg", force=True)
        import tkinter
        window = tkinter.Tk()
        window.withdraw()
        window.destroy()
    out.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out)
