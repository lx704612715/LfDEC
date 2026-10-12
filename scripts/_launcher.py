"""Isaac Lab launcher arguments with the viewer on by default.

Isaac Lab's ``--viz`` defaults to headless. Every script here is a run one watches, so the
kit-less Newton GL viewer is on unless told otherwise: ``--viz none`` runs headless, ``--viz kit``
selects the Isaac Sim viewport where Kit is installed. Import this in place of
``isaaclab.app.add_launcher_args``.
"""

import argparse

from isaaclab.app import add_launcher_args as _isaaclab_launcher_args

DEFAULT_VISUALIZER = ["newton_gl"]


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Isaac Lab's launcher flags, with ``--viz`` defaulting to the Newton GL viewer."""
    _isaaclab_launcher_args(parser)
    parser.set_defaults(visualizer=DEFAULT_VISUALIZER)
