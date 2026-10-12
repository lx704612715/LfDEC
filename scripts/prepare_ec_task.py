# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Compile annotated demonstration segments into EC parameters and runnable planner YAML.

Runs without Isaac Lab or a GPU. Example:
  python scripts/prepare_ec_task.py tasks/drawer_grasp_pull.demo.yaml --output /tmp/drawer/task.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _repo_path  # noqa: F401
from lfdec_primitives import compile_demonstration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demonstration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = compile_demonstration(args.demonstration, args.output)
    print(f"Prepared {output}; learned parameters and source hashes: {output.with_suffix('.learning.json')}")


if __name__ == "__main__":
    main()
