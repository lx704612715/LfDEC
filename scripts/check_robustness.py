# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Run a bounded Isaac/Newton regression matrix and retain every case's evidence.

This samples three seeds and two perturbations per task; it is not an estimate of general
success probability. Run from the project's Python environment. Each case starts a fresh
simulator and uses the shipped demonstrations unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "drawer": ("drawer_grasp_pull", "pull", "cabinet/drawer_top_joint", 0.2),
    "door": ("door_grasp_open", "open", "cabinet/door_left_joint", -0.436),
}
PERTURBATIONS = {
    "drawer": {
        "default": {},
        "moved_fixture": {"cabinet-dx": 0.02, "cabinet-dy": 0.03},
    },
    "door": {
        "default": {},
        "opposite_start": {"hand-dy": -0.02, "hand-dz": 0.02, "hand-dyaw-deg": -4},
    },
    "factory": {
        "positive_start": {"hand-dy": 0.006, "hand-dz": 0.006, "hand-dyaw-deg": -3},
        "negative_start": {"hand-dy": -0.006, "hand-dz": 0.006, "hand-dyaw-deg": 3},
    },
}


def json_safe(value):
    """Unobserved diagnostic errors are NaN in some task summaries; export them as null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def failure_reason(task: str, summary: dict, returncode: int, device: str) -> str | None:
    """Check physical outcomes as well as the process and planner status."""
    if returncode != 0:
        return f"process exited {returncode}: {summary.get('status', 'no status')}"
    if summary.get("device") != device:
        return f"simulation device {summary.get('device')!r}, expected {device!r}"
    if summary.get("status") != "Finished":
        return f"planner: {summary.get('status')}"
    trace = [(s["node"], s["event"]) for s in summary["trace"]]
    if task == "factory":
        if trace != [("approach", "MakingContact"), ("search", "BreakingContact"),
                     ("insert", "Done(insertion_depth->0.024)")]:
            return f"unexpected handovers: {trace}"
        if not summary["servo_events"] or summary["servo_events"][0] != "Converged":
            return "visual alignment did not converge"
        m = summary["metrics"]
        clearance = m["radial_clearance"]
        if not (0.024 <= m["insertion_depth"] <= 0.026
                and 0 <= m["hole_error"] <= clearance
                and 0 <= m["mouth_error"] <= clearance
                and math.isfinite(clearance) and 0 < clearance <= 0.00051):
            return "peg did not remain seated inside the original socket clearance"
        if summary.get("success") is not True:
            return "task runner rejected seating"
    else:
        _, node, joint, target = TASKS[task]
        if trace != [("grasp", "Done"), (node, f"Done({joint}->{target:g})")]:
            return f"unexpected handovers: {trace}"
        if summary["servo_events"] != ["Converged"]:
            return "visual alignment did not converge"
        before, after = summary["progress"][joint]
        direction = 1 if target > 0 else -1
        if not (abs(before) < 0.01 and math.isfinite(after)
                and direction * after >= direction * target):
            return "cabinet did not move from closed to its joint target"
        if not 0.002 < summary["gripper_gap"] < 0.03:
            return "gripper did not retain a physical grasp"
    return None


def run_case(task, seed, name, offsets, directory, device, timeout):
    directory.mkdir()
    command = [sys.executable, "-u", str(ROOT / "scripts/run_ec_task.py"),
               "--viz", "none", "--seed", str(seed), "--out-dir", str(directory), "--tag", "execution"]
    example = "factory_peg_insert" if task == "factory" else TASKS[task][0]
    command += ["--no-monitor", "--no-markers", "--example", example]
    for key, value in offsets.items():
        command += [f"--{key}", str(value)]
    command += ["physics=newton_mjwarp", f"env.sim.device={device}"]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env.pop("DISPLAY", None)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = dict(task=task, seed=seed, perturbation=name, offsets=offsets, device=device,
                  command=command, artifacts=str(directory), success=False)
    start = time.monotonic()
    summary_path = directory / "ec_task_execution.json"
    try:
        with (directory / "run.log").open("w") as output:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, timeout=timeout)
        result["returncode"] = process.returncode
        if not summary_path.exists():
            result["failure"] = "no task summary; see run.log"
        else:
            summary = json.loads(summary_path.read_text())
            result["summary"] = summary
            result["failure"] = failure_reason(task, summary, process.returncode, device)
            result["success"] = result["failure"] is None
    except subprocess.TimeoutExpired:
        result["failure"] = f"exceeded {timeout:g}s wall time"
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failure"] = f"invalid execution evidence: {error}"
    result["wall_seconds"] = round(time.monotonic() - start, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(PERTURBATIONS), default=list(PERTURBATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/robustness" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be positive and finite")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("--out-dir must be empty to avoid mixing runs")
    cases = [(task, seed, name, offsets) for task in dict.fromkeys(args.tasks)
             for seed in dict.fromkeys(args.seeds) for name, offsets in PERTURBATIONS[task].items()]
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), device=args.device,
                  planned=len(cases), completed=0, passed=0, complete=False, cases=[])

    def save():
        temporary = out / "report.tmp"
        temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "report.json")

    save()
    for task, seed, name, offsets in cases:
        tag = f"{task}_s{seed}_{name}"
        print(f"[{report['completed'] + 1}/{len(cases)}] {tag}", flush=True)
        result = run_case(task, seed, name, offsets, out / tag, args.device, args.timeout)
        report["cases"].append(result)
        report["completed"] += 1
        report["passed"] += int(result["success"])
        save()
        print(f"  {'PASS' if result['success'] else 'FAIL'} ({result['wall_seconds']}s)"
              f" {result.get('failure') or ''}", flush=True)
    report["complete"] = True
    save()
    print(f"{report['passed']}/{report['planned']} passed; {out / 'report.json'}", flush=True)
    return 0 if report["passed"] == report["planned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
