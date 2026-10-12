# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""A manipulation as a YAML file: the EC graph, its demonstrations and its EC parameters.

:mod:`.planner` runs a graph of primitives that a script has built. This turns the script into a
file, so ``grasp -> open_drawer`` is data rather than code -- which is what the reference does
with its ``policy_graph.yaml`` plus a merged demonstration pickle.

The reference splits the description three ways: ``policy_graph.yaml`` holds an edge list of
``{node, successor, EC_type, event}``; ``config_<experiment>.yaml`` holds control and estimator
parameters *per EC type*; and the merged demonstration pickle holds, per segment, its trajectory,
its ``EC_type``, its ``EC_parameters`` (the geometry of the constraint) and its ``cg``. Here it is
one file plus the demonstration ``.npz`` files it names, because a node *is* a primitive: the EC
type, the demonstration, the EC parameters and the tuning all belong to it.

.. code-block:: yaml

    task: LfDEC-Drawer-EC-Franka     # the scene; the run script's business, not the planner's
    start: grasp
    primitives:
      grasp:
        type: freespace
        demo: drawer_approach.npz    # relative to the YAML file
        cfg: {max_approach: 0.10}
      pull:
        type: translation
        ec: {axis: [-1.0, 0.0, 0.0]} # the EC parameters: the prismatic axis, from the demo
        cfg: {delta_p: 0.10, terminate_extra: [drawer_q, 0.25]}
    edges:
      - {node: grasp, successor: pull, event: NonC}

**The four constructors do not have the same shape**, which is why this is a registry of builders
rather than one ``cls(**cfg)``: a free-space node is built from a *demonstration*, a plane node
from a demonstration and its normal, a translation or rotation node from an *estimator seeded
with the demonstration's prior*. :data:`BUILDERS` maps the ``type`` string to a function, and
adding a type is adding one entry. What each node's ``ec`` block carries:

==============  ===================================================================================
type            ``ec``
==============  ===================================================================================
``freespace``   nothing; ``demo`` is required
``translation`` ``axis`` (prismatic direction, signed the way the demonstration moved); ``pf``
``rotation``    ``axis`` and ``point`` (the hinge, the axis signed so the demonstrated turn is
                positive about it -- :func:`~.base.fit_revolute_axis`); ``ekf``
``plane``       ``normal`` (out of the surface, toward the hand); ``demo`` optional; servoes on
                the view its *successor's* ``demo`` starts with
==============  ===================================================================================

A plane node's visual correction is toward the view the next segment starts from -- the
reference names it ``D1S{k+1}_start`` -- so every builder is handed the node's nominal successor
(the first edge out of it, the one the planner links as ``next_EC_primitive``) and the plane
builder takes the view from that node's ``demo``. That is why the segment after a plane carries
a ``demo`` even when its own primitive, a translation, replays nothing of it.

Anything the file cannot carry -- a live camera, a random generator, the servo -- arrives through
:class:`BuildContext`, which the run script fills in.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .base import Demo, Primitive, RobotLike
from .freespace import FreeCfg, FreeSpacePrimitive
from .plane import PlaneCfg, PlanePrimitive
from .planner import ECPlanner, Edge
from .rotation import RotationPrimitive, RotCfg, ScrewCfg, ScrewEKF
from .servo import ServoCfg
from .translation import PFCfg, PrismaticPF, TransCfg, TranslationPrimitive

__all__ = ["BUILDERS", "BuildContext", "NodeSpec", "TaskSpec", "build_planner", "load_task_spec"]


@dataclass(frozen=True)
class NodeSpec:
    """One node of the graph: which primitive, built from what."""

    name: str
    type: str
    demo: Path | None = None
    ec: dict[str, Any] = field(default_factory=dict)
    """The constraint's own geometry -- the reference's ``EC_parameters``. A prismatic axis, a
    plane normal, a revolute axis and point. Task knowledge, measured from the demonstration."""
    cfg: dict[str, Any] = field(default_factory=dict)
    """Overrides for the primitive's ``*Cfg`` dataclass."""
    servo: dict[str, Any] = field(default_factory=dict)
    """Visual-servo configuration for this node's reference view."""


@dataclass(frozen=True)
class TaskSpec:
    """A whole manipulation: the scene it runs in, the graph, and where it starts."""

    task: str
    start: str
    nodes: dict[str, NodeSpec]
    edges: list[Edge]
    path: Path | None = None
    execution: dict[str, Any] = field(default_factory=dict)
    learning: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildContext:
    """What the YAML cannot carry, supplied by the run script.

    Args:
        servo_factory: called with a :class:`~.base.View` and keyword overrides for the
            :class:`~.servo.ServoCfg` (a plane node passes its ``servo_z_range``), returns a
            :class:`~.servo.VisualServo` on that view; or ``None`` to run every node without a
            servo, the free-space ones open-loop from wherever the hand is.
        rng: the estimators' random generator, so a run is reproducible from a seed.
    """

    servo_factory: Callable[..., Any] | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


def _cfg(cls, overrides: dict[str, Any], node: str):
    """Build a ``*Cfg`` dataclass, rejecting keys it does not have.

    A typo in a YAML key is otherwise silently ignored and the run quietly uses a default, which
    is exactly the kind of thing that costs an afternoon.
    """
    if not isinstance(overrides, dict):
        raise ValueError(f"node {node!r}: {cls.__name__} overrides must be a mapping")
    definitions = {f.name: f for f in fields(cls)}
    known = set(definitions)
    unknown = set(overrides) - known
    if unknown:
        raise KeyError(f"node {node!r}: {cls.__name__} has no {sorted(unknown)}; "
                       f"it has {sorted(known)}")
    for key, value in overrides.items():
        if key == "servo_mode" and value not in ("ibvs", "pbvs"):
            raise ValueError(f"node {node!r}: servo_mode must be 'ibvs' or 'pbvs'")
        if key == "approach_dir" and value is not None:
            direction = np.asarray(value, dtype=float)
            if (direction.shape != (3,) or not np.isfinite(direction).all()
                    or np.linalg.norm(direction) < 1e-9):
                raise ValueError(f"node {node!r}: approach_dir must be a finite nonzero 3-vector")
        default = definitions[key].default
        if type(default) in (int, float, bool):
            valid = (type(value) is bool if type(default) is bool else
                     type(value) is int if type(default) is int else type(value) in (int, float))
            if not valid or (type(default) is not bool and not np.isfinite(value)):
                raise ValueError(f"node {node!r}: {key} has an invalid scalar value")
            if type(default) is not bool and key != "pitch" and value < 0:
                raise ValueError(f"node {node!r}: {key} must not be negative")
        if key in {"period", "stall_period", "delta_p", "delta_theta", "approach_step", "spiral_step",
                   "spiral_pitch", "n_particles", "waypoint_attempts", "bce_window", "error_window",
                   "max_features", "min_points", "lost_after", "r_trans", "r_rot", "meas_std"}:
            if value <= 0:
                raise ValueError(f"node {node!r}: {key} must be positive")
        if key in {"z_range", "servo_z_range"}:
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in value)
                    or not 0 < value[0] < value[1]):
                raise ValueError(f"node {node!r}: {key} needs two increasing positive depths")
        if key in {"terminate_p", "terminate_theta", "tukey_c"} and value is not None:
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"node {node!r}: {key} must be positive and finite")
    return cls(**overrides)


def _build_freespace(spec: NodeSpec, successor: NodeSpec | None, robot: RobotLike, ctx: BuildContext) -> Primitive:
    if spec.demo is None:
        raise ValueError(f"node {spec.name!r}: a freespace node needs a 'demo'")
    demo = Demo.load(spec.demo)
    overrides = dict(spec.cfg)
    if overrides.get("approach_dir") is not None:
        overrides["approach_dir"] = np.asarray(overrides["approach_dir"], dtype=float)
    servo = None
    _cfg(ServoCfg, spec.servo, spec.name)
    if ctx.servo_factory is not None and demo.view is not None:
        servo = ctx.servo_factory(demo.view, **spec.servo)
    return FreeSpacePrimitive(robot, demo, _cfg(FreeCfg, overrides, spec.name), servo=servo)


def _overrides(spec: NodeSpec) -> dict[str, Any]:
    """The node's ``cfg`` block, with YAML's ``terminate_extra: [key, threshold]`` list as the
    tuple the ``*Cfg`` dataclasses take."""
    overrides = dict(spec.cfg)
    stop = overrides.get("terminate_extra")
    if stop is not None:
        if (not isinstance(stop, (list, tuple)) or len(stop) != 2
                or not isinstance(stop[0], str) or not stop[0]
                or type(stop[1]) not in (int, float) or not np.isfinite(stop[1])):
            raise ValueError(f"node {spec.name!r}: terminate_extra needs [measurement, finite target]")
        overrides["terminate_extra"] = (stop[0], float(stop[1]))
    return overrides


def _ec(spec: NodeSpec, *keys: str) -> list[np.ndarray]:
    missing = [k for k in keys if k not in spec.ec]
    if missing:
        raise ValueError(f"node {spec.name!r}: a {spec.type} node needs ec.{' and ec.'.join(keys)}, "
                         f"measured from the demonstration; missing {missing}")
    result = [np.asarray(spec.ec[k], dtype=float) for k in keys]
    for key, value in zip(keys, result):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"node {spec.name!r}: ec.{key} must be a finite three-vector")
        if key in ("axis", "normal") and np.linalg.norm(value) < 1e-9:
            raise ValueError(f"node {spec.name!r}: ec.{key} cannot be zero")
    return result


def _build_translation(spec: NodeSpec, successor: NodeSpec | None, robot: RobotLike, ctx: BuildContext) -> Primitive:
    (u_init,) = _ec(spec, "axis")
    pf = PrismaticPF(u_init, _cfg(PFCfg, dict(spec.ec.get("pf", {})), spec.name), ctx.rng)
    demo = None if spec.demo is None else Demo.load(spec.demo)
    return TranslationPrimitive(robot, pf, u_init, _cfg(TransCfg, _overrides(spec), spec.name), demo=demo)


def _build_rotation(spec: NodeSpec, successor: NodeSpec | None, robot: RobotLike, ctx: BuildContext) -> Primitive:
    axis, point = _ec(spec, "axis", "point")
    ekf = ScrewEKF(axis, point, _cfg(ScrewCfg, dict(spec.ec.get("ekf", {})), spec.name), ctx.rng)
    # direction stays +1: the demonstration signed the axis so its own turn is positive about it.
    demo = None if spec.demo is None else Demo.load(spec.demo)
    return RotationPrimitive(robot, ekf, axis, _cfg(RotCfg, _overrides(spec), spec.name), demo=demo)


def _build_plane(spec: NodeSpec, successor: NodeSpec | None, robot: RobotLike, ctx: BuildContext) -> Primitive:
    (normal,) = _ec(spec, "normal")
    demo = None if spec.demo is None else Demo.load(spec.demo)
    cfg = _cfg(PlaneCfg, _overrides(spec), spec.name)
    # The correction on the surface servoes to the view the next segment starts from: a Demo
    # carries the view at its start, so it is the successor's demo that has it.
    servo = None
    if ctx.servo_factory is not None and successor is not None and successor.demo is not None:
        view = Demo.load(successor.demo).view
        if view is not None:
            _cfg(ServoCfg, spec.servo, spec.name)
            servo = ctx.servo_factory(view, **({"z_range": tuple(cfg.servo_z_range), "servo_mode": "ibvs"} | spec.servo))
    return PlanePrimitive(robot, normal, demo, cfg, servo=servo)


BUILDERS: dict[str, Callable[[NodeSpec, NodeSpec | None, RobotLike, BuildContext], Primitive]] = {
    "freespace": _build_freespace,
    "translation": _build_translation,
    "rotation": _build_rotation,
    "plane": _build_plane,
}


def read_yaml(path: Path | str) -> dict:
    """Read a mapping and reject duplicate keys instead of silently losing an EC segment."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"{path}: duplicate YAML key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    doc = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return doc


def load_task_spec(path: Path | str) -> TaskSpec:
    """Read a task YAML. Demonstration paths resolve relative to the file."""
    return parse_task_spec(read_yaml(path), Path(path))


def parse_task_spec(doc: dict, path: Path) -> TaskSpec:
    """Validate the task's graph and structure before constructing simulator objects."""
    path = Path(path)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    unknown = set(doc) - {"schema_version", "task", "start", "primitives", "edges", "execution", "learning"}
    if unknown:
        raise KeyError(f"{path}: unexpected top-level keys {sorted(unknown)}")
    if "schema_version" in doc and (type(doc["schema_version"]) is not int or doc["schema_version"] != 1):
        raise ValueError(f"{path}: unsupported schema_version")
    for key in ("task", "start", "primitives", "edges"):
        if key not in doc:
            raise KeyError(f"{path}: missing top-level {key!r}")

    if not isinstance(doc["task"], str) or not doc["task"]:
        raise ValueError(f"{path}: task must be a nonempty scene id")
    if not isinstance(doc["primitives"], dict) or not doc["primitives"]:
        raise ValueError(f"{path}: primitives must be a nonempty mapping")
    nodes: dict[str, NodeSpec] = {}
    for name, body in doc["primitives"].items():
        if not isinstance(name, str) or not name or not isinstance(body, dict):
            raise ValueError(f"{path}: each named primitive must be a mapping")
        body = dict(body)
        kind = body.pop("type", None)
        if kind not in BUILDERS:
            raise KeyError(f"{path}: node {name!r} has type {kind!r}; known types are "
                           f"{sorted(BUILDERS)}")
        demo = body.pop("demo", None)
        if demo is not None and (not isinstance(demo, str) or not demo):
            raise ValueError(f"{path}: node {name!r}: demo must be a trajectory path")
        for key in ("ec", "cfg", "servo"):
            if body.get(key) is not None and not isinstance(body[key], dict):
                raise ValueError(f"{path}: node {name!r}: {key} must be a mapping")
        nodes[name] = NodeSpec(name=name, type=kind,
                               demo=None if demo is None else (path.parent / demo).resolve(),
                               ec=dict(body.pop("ec", {}) or {}),
                               cfg=dict(body.pop("cfg", {}) or {}),
                               servo=dict(body.pop("servo", {}) or {}))
        if body:
            raise KeyError(f"{path}: node {name!r} has unexpected keys {sorted(body)}")

    if not isinstance(doc["edges"], list):
        raise ValueError(f"{path}: edges must be a list")
    edges, seen = [], set()
    for e in doc["edges"]:
        if not isinstance(e, dict) or set(e) != {"node", "successor", "event"}:
            raise ValueError(f"{path}: each edge requires node, successor and event")
        if e["node"] not in nodes or e["successor"] not in nodes:
            raise KeyError(f"{path}: edge refers to an unknown primitive")
        if e["event"] not in {"MC", "BC", "NonC"}:
            raise ValueError(f"{path}: edge event must be MC, BC or NonC")
        key = (e["node"], e["event"])
        if key in seen:
            raise ValueError(f"{path}: ambiguous edges from {key}")
        seen.add(key)
        edges.append(Edge(src=e["node"], dst=e["successor"], event=e["event"]))
    if doc["start"] not in nodes:
        raise KeyError(f"{path}: start {doc['start']!r} is not one of {sorted(nodes)}")
    reachable = {doc["start"]}
    while True:
        expanded = reachable | {e.dst for e in edges if e.src in reachable}
        if expanded == reachable:
            break
        reachable = expanded
    if reachable != set(nodes):
        raise ValueError(f"{path}: unreachable primitives {sorted(set(nodes) - reachable)}")
    execution = doc.get("execution", {})
    allowed = {"max_iters", "hand_dx", "hand_dy", "hand_dz", "hand_dyaw_deg"}
    if not isinstance(execution, dict) or set(execution) - allowed:
        raise ValueError(f"{path}: invalid execution settings; expected {sorted(allowed)}")
    if "max_iters" in execution and (type(execution["max_iters"]) is not int or execution["max_iters"] < 1):
        raise ValueError(f"{path}: max_iters must be a positive integer")
    for key in allowed - {"max_iters"}:
        if key in execution and (type(execution[key]) not in (int, float) or not np.isfinite(execution[key])):
            raise ValueError(f"{path}: {key} must be finite")
    learning = doc.get("learning", {})
    if not isinstance(learning, dict):
        raise ValueError(f"{path}: learning provenance must be a mapping")
    return TaskSpec(task=doc["task"], start=doc["start"], nodes=nodes, edges=edges,
                    path=path, execution=dict(execution), learning=dict(learning))


def validate_configuration(spec: TaskSpec) -> None:
    """Check compiler configuration without needing a robot or initializing a simulator."""
    classes = {"freespace": FreeCfg, "translation": TransCfg, "rotation": RotCfg, "plane": PlaneCfg}
    for node in spec.nodes.values():
        allowed_ec = {"freespace": set(), "translation": {"axis", "pf"},
                      "rotation": {"axis", "point", "ekf"}, "plane": {"normal"}}[node.type]
        if set(node.ec) - allowed_ec:
            raise ValueError(f"node {node.name!r}: unexpected EC keys {sorted(set(node.ec) - allowed_ec)}")
        required = {"freespace": (), "translation": ("axis",),
                    "rotation": ("axis", "point"), "plane": ("normal",)}[node.type]
        _ec(node, *required)
        _cfg(classes[node.type], _overrides(node), node.name)
        if node.servo:
            if node.type not in ("freespace", "plane"):
                raise ValueError(f"node {node.name!r}: this primitive has no visual-servo phase")
            _cfg(ServoCfg, node.servo, node.name)
        if node.type == "translation":
            _cfg(PFCfg, node.ec.get("pf", {}), node.name)
        if node.type == "rotation":
            _cfg(ScrewCfg, node.ec.get("ekf", {}), node.name)


def build_planner(spec: TaskSpec, robot: RobotLike, ctx: BuildContext | None = None,
                  verify_transitions: bool = False,
                  on_node: Callable[[str, Primitive], None] | None = None) -> ECPlanner:
    """Instantiate every node and wire them into an :class:`~.planner.ECPlanner`. Each builder is
    handed the node's nominal successor, the first edge out of it."""
    validate_configuration(spec)
    ctx = ctx if ctx is not None else BuildContext()

    def successor(name: str) -> NodeSpec | None:
        nxt = next((e.dst for e in spec.edges if e.src == name), None)
        return None if nxt is None else spec.nodes[nxt]

    primitives = {name: BUILDERS[node.type](node, successor(name), robot, ctx) for name, node in spec.nodes.items()}
    return ECPlanner(primitives, spec.edges, verify_transitions=verify_transitions, on_node=on_node)
