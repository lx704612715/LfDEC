# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Read a joint reaction wrench and its child-side reporting frame.

The action adapter converts the simulator sign and transports this wrench to the TCP."""

from __future__ import annotations

import torch

__all__ = ["SensorWrench"]


class SensorWrench:
    """Joint reaction wrench from a :class:`JointWrenchSensor`. Works on every backend.

    Which bodies a sensor covers is backend-dependent, because only bodies with an *incoming
    joint* have a reaction wrench to report. PhysX keeps fixed-joint children as separate links,
    so ``panda_hand`` appears; MuJoCo-Warp merges a fixed-joint child into its parent, so under
    Newton the list stops at ``panda_link7`` and the hand is gone. Pass the preferred body first
    and its nearest actuated ancestor after it, and the first covered name wins.

    Args:
        sensor: an initialized ``JointWrenchSensor``.
        body_name: preferred body, or an ordered list of candidates (most distal first).
    """

    def __init__(self, sensor, body_name: str | list[str] | tuple[str, ...]):
        self._sensor = sensor
        names = list(sensor.body_names)
        candidates = [body_name] if isinstance(body_name, str) else list(body_name)
        for cand in candidates:
            if cand in names:
                self._idx = names.index(cand)
                self.body_name = cand
                self.fell_back = cand != candidates[0]
                return
        raise ValueError(f"none of {candidates} is covered by the sensor; it reports {names}")

    def read(self) -> torch.Tensor:
        data = self._sensor.data
        if data.force is None:
            raise RuntimeError("JointWrenchSensor has no data yet; step the sim before reading it")
        return torch.cat([data.force.torch[:, self._idx], data.torque.torch[:, self._idx]], dim=-1)

    def joint_frame(self) -> torch.Tensor | None:
        """Child-side joint frame in the reporting link frame, as px,py,pz,qx,qy,qz,qw.
        Newton exposes this transform through _sim_bind_joint_X_c. Return None
        when unavailable so the action uses its configured anchor offset."""
        xc = getattr(self._sensor, "_sim_bind_joint_X_c", None)
        if xc is None:
            return None
        import warp as wp

        return wp.to_torch(xc)[:, self._idx]
