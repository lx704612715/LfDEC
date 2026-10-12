"""Put the repo root on ``sys.path`` so ``import lfdec_primitives`` works from a
fresh clone, without needing ``pip install -e .`` into the Isaac Sim interpreter.

Scripts import this for its side effect, before importing ``lfdec_primitives``.
Python already puts a script's own directory on ``sys.path``, which is how this
module itself is found.
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REPO_ROOT = _ROOT
"""Absolute path of the repo root, for scripts that write outputs next to the source."""
