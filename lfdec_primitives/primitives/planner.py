# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Execute an EC graph using observed contact-changing events.

Successful completion and matching contact events advance the graph. Primitive failures,
unmatched transitions and failed successor verification stop execution."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import Primitive

__all__ = ["ECPlanner", "Edge"]

_EVENTS = ("MC", "BC", "NonC")


def _completed(event: str) -> bool:
    """Only explicit completion and contact outcomes may advance the graph.

    Contact labels alone cannot distinguish Done from a limit, exhausted search, or failed
    servo. Accept the result families emitted by the primitives, including parenthesized
    detail, so an unknown result also stops rather than silently following a NonC edge.
    """
    return any(event == name or (event.startswith(name + "(") and event.endswith(")"))
               for name in ("Done", "MakingContact", "BreakingContact"))


@dataclass(frozen=True)
class Edge:
    """``src`` hands over to ``dst`` on ``event`` (``"MC"``, ``"BC"`` or ``"NonC"``).

    Only completed primitives take edges: ``NonC`` matches ``Done`` outcomes, never failures.
    """

    src: str
    dst: str
    event: str


class ECPlanner:
    """Run a graph of primitives, following the edge that matches each observed event.

    Args:
        primitives: the nodes, by name.
        edges: the transitions. The *first* edge out of a node names its nominal successor -- the
            demonstration's -- which is linked as ``primitive.next_EC_primitive`` before anything runs, so a
            free-space primitive can ask the plane after it which direction to seek and a
            primitive can verify its handover (the reference's ``funnel_sequence_ec_primitives``).
        verify_transitions: after a matched event, run :meth:`Primitive.check_transition_to_the_next_EC`
            against the successor actually taken and stop with ``TransitionFailed`` if it says
            the constraint is not there.
        on_node: called with ``(name, primitive)`` before each node runs. The planner is the only
            thing that knows which primitive is live, and the visualisation is the only thing
            that needs to be told -- :meth:`~lfdec_primitives.vis.monitor.ECMonitor.begin` has this signature.
    """

    def __init__(self, primitives: dict[str, Primitive], edges: list[Edge], verify_transitions: bool = False,
                 on_node: Callable[[str, Primitive], None] | None = None):
        for e in edges:
            for name in (e.src, e.dst):
                if name not in primitives:
                    raise KeyError(f"edge {e} refers to unknown primitive {name!r}")
            if e.event not in _EVENTS:
                raise ValueError(f"edge {e}: event must be one of {_EVENTS}")
        self.primitives, self.edges, self.verify = primitives, list(edges), verify_transitions
        self.on_node = on_node
        self.funnel_sequence_ec_primitives()
        self.trace: list[dict] = []
        self.status = "Ready"

    def funnel_sequence_ec_primitives(self, node: str | None = None) -> None:
        """Restore the demonstration's successor/event pairing, including after branch verification."""
        for name, prim in self.primitives.items():
            if node is not None and name != node:
                continue
            outgoing = [e for e in self.edges if e.src == name]
            nominal = outgoing[0] if outgoing else None
            prim.set_next_EC_primitive(None if nominal is None else self.primitives[nominal.dst])
            prim.d_cg_event = "NonC" if nominal is None else nominal.event
            prim.transition_successors = {e.event: self.primitives[e.dst] for e in outgoing}

    def successor(self, node: str, event: str) -> str | None:
        """Node reached from ``node`` on contact-changing ``event``, or ``None``."""
        return next((e.dst for e in self.edges if e.src == node and e.event == event), None)

    def execution(self, start: str, max_iters: int = 300, max_nodes: int = 32, verbose: bool = True) -> str:
        """Execute from ``start``, recording every visited node and its detailed result in trace.

        ``Done`` and contact outcomes may follow edges or finish a terminal node. Other results
        stop with ``PrimitiveFailed(node:event)`` before handover verification or another node
        runs. Unmatched contact labels give ``NoTransition(node:event)``; failed verification
        gives ``TransitionFailed(src->dst)``; exhausting the node budget gives ``MaxNodes``.

        ``Finished`` means a terminal node returned a completion or contact outcome. Physical
        task success (e.g. seating rather than a rim contact) still needs task-specific checks.
        """
        self.trace = []
        self.status = "Running"
        node = start
        for _ in range(max_nodes):
            prim = self.primitives[node]
            self.funnel_sequence_ec_primitives(node)
            if self.on_node is not None:
                self.on_node(node, prim)
            if verbose:
                print(f"== [{node}] {type(prim).__name__} ==")
            event = prim.exploitation(max_iters=max_iters, verbose=verbose)
            cg = prim.cg_event
            step = dict(node=node, event=event, contact_event=cg, next=None)
            self.trace.append(step)
            if not _completed(event):
                self.status = f"PrimitiveFailed({node}:{event})"
                return self.status
            if not any(e.src == node for e in self.edges):
                self.status = "Finished"
                return self.status
            nxt = self.successor(node, cg)
            if nxt is None:
                self.status = f"NoTransition({node}:{cg})"
                return self.status
            if self.verify:
                prim.set_next_EC_primitive(self.primitives[nxt])
                if not prim.check_transition_to_the_next_EC(verbose=verbose):
                    self.status = f"TransitionFailed({node}->{nxt})"
                    return self.status
            step["next"] = nxt
            if verbose:
                print(f"   {node} --{cg}--> {nxt}")
            node = nxt
        self.status = "MaxNodes"
        return self.status
