# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Write a run to an mp4 as it happens: the simulator's view beside the EC monitor, at ``speed``
times sim time, without a window.

Frames are taken on *sim* time, one per ``speed / fps`` seconds of it, so the video is the same
whatever the machine's frame rate and a run that simulates slower than real time still plays
back at ``speed`` times its own clock. Each frame goes straight to ``ffmpeg`` over a pipe as raw
RGB; nothing is buffered and nothing is shown. The Isaac scripts feed it the Newton viewer's
framebuffer (``render_rgb_array`` of a viewer running headless); any
other source is a callable returning an ``(H, W, 3)`` uint8 array of constant size.
"""
from __future__ import annotations

import subprocess
import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

__all__ = ["RunVideo"]


class RunVideo:
    """Args:
        path: the mp4 to write.
        viewer_frame: returns the simulator's current view, ``(H, W, 3)`` uint8 RGB.
        monitor: an :class:`~lfdec_primitives.vis.monitor.ECMonitor`, drawn beside the view (scaled to its height),
            or ``None`` for the view alone.
        speed: playback speed relative to sim time.
        fps: frames per second of the file.
    """

    def __init__(self, path: Path | str, viewer_frame: Callable[[], np.ndarray], monitor=None,
                 speed: float = 5.0, fps: int = 30):
        if not math.isfinite(speed) or speed <= 0.0 or fps <= 0:
            raise ValueError("speed and fps must be positive")
        self.path = Path(path)
        self.viewer_frame, self.monitor = viewer_frame, monitor
        self.period, self.fps = speed / fps, fps   # sim seconds per video frame
        self.frames = 0
        self._origin: float | None = None
        self._closed = False
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None

    def tick(self, t: float, ticks: list[dict]) -> None:
        """Called every control step with the sim clock; writes a frame when one is due."""
        if self._closed:
            raise RuntimeError("cannot record after closing the video")
        if not math.isfinite(t):
            raise ValueError("simulation time must be finite")
        if self._origin is None:
            self._origin = t
        due = math.floor((t - self._origin) / self.period + 1e-9) + 1
        if due <= self.frames:
            return
        view = np.ascontiguousarray(self.viewer_frame(), dtype=np.uint8)
        if self.monitor is not None:
            self.monitor.update(ticks, force=True)
            panel = self.monitor.frame()
            h = view.shape[0]
            w = round(panel.shape[1] * h / panel.shape[0]) // 2 * 2   # even, for yuv420p
            view = np.concatenate([view, cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)], axis=1)
        # An integer frame clock avoids drift when control dt is not a divisor of 1/fps.
        # If stepping was slower than recording, repeat the latest available observation.
        while self.frames < due:
            self._write(view)

    def close(self) -> None:
        """Finish the file. Raises if ffmpeg failed or nothing was written."""
        if self._closed:
            return
        if self._proc is None:
            raise RuntimeError(f"no frame was written to {self.path}")
        proc, self._proc = self._proc, None
        try:
            proc.stdin.close()
        finally:
            code = proc.wait()
            self._closed = True
        if code != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self.close()

    def _write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._proc is None:
            self._size = (w, h)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE)
        elif (w, h) != self._size:
            raise RuntimeError(f"frame size changed from {self._size} to {(w, h)}")
        self._proc.stdin.write(frame.tobytes())
        self.frames += 1
