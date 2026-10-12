# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT
"""Measured loose-peg state for the contact-only Factory grasp."""
import isaaclab.utils.math as math_utils
import numpy as np
import torch

from ..envs.factory_peg_ec_cfg import PEG_LENGTH, insertion_metrics_at_tip
from ..envs.factory_grasp_cfg import PEG_PROTRUSION
from .isaac_robot import IsaacECRobot


class FactoryGraspRobot(IsaacECRobot):
    tip_offset = np.array([0., 0., PEG_PROTRUSION])

    def set_gripper(self, closed: bool) -> None:
        super().set_gripper(closed)
        # This experiment starts with a known preloaded peg. Command state denotes an
        # assumed held payload; measured slip/release checks still decide physical success.
        mass = self.env.cfg.scene.peg.spawn.mass_props.mass if closed else 0.0
        com_tcp = self.tip_offset - np.array([0., 0., PEG_LENGTH / 2.])
        self.term.set_payload(mass, com_tcp)

    def peg_pose(self):
        """Actual peg bottom-frame pose in robot-root coordinates."""
        data = self.env.scene["peg"].data
        Rw = math_utils.matrix_from_quat(data.root_quat_w.torch)[0].cpu().numpy()
        pw = data.root_pos_w.torch[0].cpu().numpy()
        R0, p0 = self.root_pose()
        return R0.T @ Rw, R0.T @ (pw - p0)

    def extra(self):
        R, p = self.ee_pose()
        Rp, pp = self.peg_pose()
        local = R.T @ (pp - p)
        return dict(**insertion_metrics_at_tip(pp, Rp[:, 2]),
                    peg_x=float(pp[0]), peg_y=float(pp[1]), peg_z=float(pp[2]),
                    slip=float(np.linalg.norm(local - self.tip_offset)),
                    tilt=float(np.arccos(np.clip((-R[:, 2]) @ Rp[:, 2], -1, 1))),
                    finger_gap=self.gripper_gap())

    def position_peg_in_gripper(self):
        """Initialize the peg once before physics steps; no attachment or later pose correction."""
        peg = self.env.scene["peg"]
        R, p = self.ee_pose()
        R0, p0 = self.root_pose()
        pos = torch.as_tensor(p0 + R0 @ (p + R @ self.tip_offset), device=self.env.device).float()[None]
        rot = torch.as_tensor(R0 @ R @ np.diag([1., -1., -1.]), device=self.env.device).float()[None]
        peg.write_root_pose_to_sim_index(root_pose=torch.cat([pos, math_utils.quat_from_matrix(rot)], dim=-1))
        peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.env.device))
