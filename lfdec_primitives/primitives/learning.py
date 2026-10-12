# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Learn EC geometry from annotated demonstration segments and compile a runnable task.

The segment labels and contact-changing events are supplied annotations. Geometry is fitted
from the recorded poses, independently of Isaac Lab, scene coordinates and robot adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from .base import Demo, fit_line_direction, fit_plane, fit_revolute_axis, rotvec_from_matrix, tangent_basis
from .graph import load_task_spec, parse_task_spec, read_yaml, validate_configuration

__all__ = ["LearnedEC", "learn_ec", "compile_demonstration"]

STATES = {"free": "freespace", "prismatic": "translation", "revolute": "rotation",
          "screw": "rotation", "plane": "plane"}


@dataclass(frozen=True)
class LearnedEC:
    """Fitted geometry and its measured residual; pitch is expressed in metres per radian."""

    parameters: dict
    diagnostics: dict


def _screw_fit(poses: np.ndarray) -> LearnedEC:
    increments = np.array([rotvec_from_matrix(b[:3, :3] @ a[:3, :3].T)
                           for a, b in zip(poses[:-1], poses[1:])])
    net = increments.sum(axis=0)
    angle = np.linalg.norm(net)
    if angle < np.deg2rad(1):
        raise ValueError("rotation learning needs at least one degree of demonstrated turn")
    _, _, vt = np.linalg.svd(increments, full_matrices=False)
    axis = vt[0]
    if axis @ net < 0:
        axis = -axis
    theta = np.r_[0., np.cumsum(increments @ axis)]
    if np.any(np.diff(theta) < -np.deg2rad(0.5)):
        raise ValueError("a rotation segment must turn consistently in one direction")
    P = poses[:, :3, 3]
    centroid = P.mean(axis=0)
    basis = np.column_stack(tangent_basis(axis))
    relative_R = poses[1:, :3, :3] @ poses[0, :3, :3].T
    I_minus_R = np.eye(3) - relative_R
    A = I_minus_R @ basis
    A = np.concatenate([A, theta[1:, None, None] * axis[None, :, None]], axis=2)
    y = P[1:] - relative_R @ P[0] - I_minus_R @ centroid
    solution, _, rank, _ = np.linalg.lstsq(A.reshape(-1, A.shape[-1]), y.ravel(), rcond=None)
    if rank != A.shape[-1]:
        raise ValueError("demonstration does not identify a screw axis and point")
    point = centroid + basis @ solution[:2]
    pitch = float(solution[2])
    predicted = relative_R @ P[0] + I_minus_R @ point + theta[1:, None] * pitch * axis
    residual = float(np.sqrt(np.mean(np.sum((P[1:] - predicted) ** 2, axis=1))))
    angular_residual = float(np.sqrt(np.mean(np.linalg.norm(
        increments - (increments @ axis)[:, None] * axis, axis=1) ** 2)))
    return LearnedEC(dict(axis=axis.tolist(), point=point.tolist(), pitch=pitch),
                     dict(samples=len(poses), turn_rad=float(theta[-1]),
                          position_rmse_m=residual, rotation_rmse_rad=angular_residual))


def learn_ec(demo: Demo, state: str, *, revolute_axis_source: str = "position") -> LearnedEC:
    """Fit one labelled segment. Stationary and geometrically degenerate segments are rejected.

    Plane normals use recorded contact force when available, otherwise the Panda hand's
    negative local z convention. Revolute fitting defaults to the reference's position circle;
    an explicit orientation-axis option is available for rigid grasps with noisy short arcs.
    """
    if state not in STATES:
        raise ValueError(f"unknown EC state {state!r}; expected one of {sorted(STATES)}")
    if state == "free":
        return LearnedEC({}, dict(samples=len(demo.poses)))
    if len(demo.poses) < 3:
        raise ValueError(f"{state} learning needs at least three poses")
    P = demo.poses[:, :3, 3]
    centered = P - P.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if state == "screw":
        return _screw_fit(demo.poses)
    if state == "revolute":
        # The knob grasp can slip in orientation; fit its position arc rather than
        # assume every measured hand rotation is a rigid rotation of the door.
        axis, point = fit_revolute_axis(demo.poses, axis_source=revolute_axis_source)
        centered_axis = P - point
        axial = centered_axis @ axis
        transverse = centered_axis - axial[:, None] * axis
        radii = np.linalg.norm(transverse, axis=1)
        return LearnedEC(dict(axis=axis.tolist(), point=point.tolist(), pitch=0.),
                         dict(samples=len(P), radius_m=float(radii.mean()),
                              position_rmse_m=float(np.sqrt(np.mean((radii - radii.mean()) ** 2 + axial ** 2)))))
    if singular[0] < 1e-5:
        raise ValueError(f"{state} learning needs measurable translation")
    if state == "prismatic":
        axis = fit_line_direction(P)
        if abs(axis @ (P[-1] - P[0])) < 1e-5:
            raise ValueError("a prismatic segment needs a signed net displacement")
        residual = centered - (centered @ axis)[:, None] * axis
        return LearnedEC(dict(axis=axis.tolist()),
                         dict(samples=len(P), travel_m=float(axis @ (P[-1] - P[0])),
                              position_rmse_m=float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))))
    if singular[1] < max(1e-6, 1e-3 * singular[0]):
        raise ValueError("plane learning needs two independent surface directions, not a line")
    outward = -demo.poses[:, :3, 2].mean(axis=0)
    init_cond = None
    if demo.ee_ft_robot is not None:
        base_force = np.einsum("nij,nj->ni", demo.poses[:, :3, :3], demo.ee_ft_robot[:, :3]).mean(axis=0)
        if np.linalg.norm(base_force) >= 1.0:
            outward = base_force
            init_cond = np.r_[base_force / np.linalg.norm(base_force), 0.0]
    normal, point = fit_plane(P, init_cond=init_cond)
    if abs(normal @ outward) < 1e-3:
        raise ValueError("the demonstrated hand orientation does not determine the plane normal's sign")
    if normal @ outward < 0:
        normal = -normal
    return LearnedEC(dict(normal=normal.tolist()),
                     dict(samples=len(P), point=point.tolist(),
                          position_rmse_m=float(np.sqrt(np.mean((centered @ normal) ** 2)))))


def _keys(doc, allowed, location):
    if not isinstance(doc, dict):
        raise ValueError(f"{location}: expected a mapping")
    unknown = set(doc) - set(allowed)
    if unknown:
        raise ValueError(f"{location}: unexpected keys {sorted(unknown)}")


def _read_manifest(manifest: Path) -> dict:
    """Validate the supplied segmentation and task fields."""
    doc = read_yaml(manifest)
    _keys(doc, ("schema_version", "task", "start", "segments", "edges", "execution"), str(manifest))
    if type(doc.get("schema_version")) is not int or doc["schema_version"] != 1:
        raise ValueError("demonstration manifest requires schema_version: 1")
    for key in ("task", "start", "segments", "edges"):
        if key not in doc:
            raise ValueError(f"demonstration manifest missing {key!r}")
    if not isinstance(doc["segments"], dict) or not doc["segments"]:
        raise ValueError("segments must be a nonempty mapping in demonstration order")
    return doc


def _compile_segment(name: str, segment: dict, directory: Path) -> tuple[dict, dict, Path]:
    """Fit a segment's contact samples and retain its complete replay trajectory."""
    _keys(segment, ("state", "demo", "fit_samples", "cfg", "estimator", "servo"), f"segment {name!r}")
    state = segment.get("state")
    if not isinstance(state, str) or state not in STATES:
        raise ValueError(f"segment {name!r}: invalid state {state!r}")
    source = segment.get("demo")
    if not isinstance(source, str) or not source:
        raise ValueError(f"segment {name!r}: demo must be a trajectory path")
    source = (directory / source).resolve()
    demo = Demo.load(source)
    selected = segment.get("fit_samples", [0, len(demo.poses)])
    if (not isinstance(selected, list) or len(selected) != 2
            or any(type(i) is not int for i in selected)
            or not 0 <= selected[0] < selected[1] <= len(demo.poses)):
        raise ValueError(f"segment {name!r}: fit_samples must be a valid [start, stop] range")
    samples = slice(*selected)
    fit = learn_ec(Demo(demo.poses[samples], demo.grip_closed[samples],
                        ee_ft_robot=None if demo.ee_ft_robot is None else demo.ee_ft_robot[samples],
                        expl_direction=demo.expl_direction), state)
    estimator = segment.get("estimator", {})
    if not isinstance(estimator, dict):
        raise ValueError(f"segment {name!r}: estimator must be a mapping")
    if "pitch" in estimator:
        raise ValueError(f"segment {name!r}: pitch must be learned, not supplied")
    if estimator and state in ("free", "plane"):
        raise ValueError(f"segment {name!r}: {state} has no estimator")
    ec = dict(fit.parameters)
    if state in ("screw", "revolute"):
        ec["ekf"] = {**estimator, "pitch": ec.pop("pitch")}
    elif estimator:
        ec["pf"] = dict(estimator)
    node = dict(type=STATES[state], demo=source.name)
    if ec:
        node["ec"] = ec
    node.update({key: segment[key] for key in ("cfg", "servo") if key in segment})
    report = dict(state=state, source=segment["demo"],
                  sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  fit_samples=selected, parameters=fit.parameters, diagnostics=fit.diagnostics)
    return node, report, source


def _write_bundle(output: Path, manifest: Path, bundled_manifest: Path, sources: dict,
                  portable_manifest: dict, runnable: dict, reports: dict) -> None:
    """Write the validated bundle without overwriting source demonstrations or annotations."""
    report_path = output.with_suffix(".learning.json")
    metadata = {path.resolve() for path in (output, report_path, bundled_manifest)}
    inputs = set(sources.values()) | {manifest}
    written_metadata = {output.resolve(), report_path.resolve()}
    if bundled_manifest != manifest:
        written_metadata.add(bundled_manifest.resolve())
    if len(metadata) != 3 or written_metadata & inputs:
        raise ValueError("output files must not overwrite input demonstrations")
    copies = {}
    for filename, source in sources.items():
        destination = (output.parent / filename).resolve()
        if destination in metadata or destination == manifest:
            raise ValueError("a bundled demonstration must not overwrite a manifest or output file")
        if destination in inputs and destination != source:
            raise ValueError("the bundle would overwrite another input demonstration")
        if destination in copies and copies[destination] != source:
            raise ValueError("bundled demonstrations must have distinct output paths")
        copies[destination] = source
    report = dict(schema_version=1, segmentation="provided EC labels and contact-sample annotations",
                  frame="robot base", manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  segments=reports)
    # Serialize before creating files so invalid data cannot leave a partial bundle.
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    manifest_text = yaml.safe_dump(portable_manifest, sort_keys=False)
    task_text = yaml.safe_dump(runnable, sort_keys=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    for destination, source in copies.items():
        if source != destination:
            shutil.copyfile(source, destination)
    if bundled_manifest != manifest:
        bundled_manifest.write_text(manifest_text)
    output.with_suffix(".learning.json").write_text(report_text)
    output.write_text("# Generated from demonstration segments; see the companion learning report.\n" + task_text)


def compile_demonstration(manifest: str | Path, output: str | Path) -> Path:
    """Learn EC parameters from annotated segments and write a portable YAML/NPZ bundle.

    ``fit_samples`` selects contact samples for fitting; replay keeps the full trajectory.
    The companion learning report contains input hashes, sample ranges and fit residuals.
    """
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if manifest in (output, output.with_suffix(".learning.json")):
        raise ValueError("the demonstration manifest and runnable YAML must be different files")
    if output.suffix not in (".yaml", ".yml"):
        raise ValueError("the runnable output must be a YAML file")
    doc = _read_manifest(manifest)
    nodes, reports, sources = {}, {}, {}
    for name, segment in doc["segments"].items():
        node, report, source = _compile_segment(name, segment, manifest.parent)
        if source.name in sources and sources[source.name] != source:
            raise ValueError(f"two source demonstrations have the same filename {source.name!r}")
        nodes[name], reports[name], sources[source.name] = node, report, source
    portable_manifest = dict(doc, segments={name: dict(segment, demo=nodes[name]["demo"])
                                            for name, segment in doc["segments"].items()})
    bundled_manifest = output.with_suffix(".demo.yaml")
    if bundled_manifest == manifest and portable_manifest != doc:
        bundled_manifest = output.with_suffix(".segments.yaml")
    runnable = dict(schema_version=1, task=doc["task"], start=doc["start"], primitives=nodes,
                    edges=doc["edges"], learning=dict(manifest=bundled_manifest.name,
                                                     report=output.with_suffix(".learning.json").name))
    if "execution" in doc:
        runnable["execution"] = doc["execution"]
    validate_configuration(parse_task_spec(runnable, output))
    _write_bundle(output, manifest, bundled_manifest, sources, portable_manifest, runnable, reports)
    load_task_spec(output)
    return output
