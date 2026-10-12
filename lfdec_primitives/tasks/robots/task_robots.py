# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Robot adapters exposing measured progress for each example task."""
import numpy as np

import lfdec_primitives as lp
from ...utils.geometry import box_aperture_clearance
from .isaac_robot import IsaacECRobot


class FactoryPegRobot(IsaacECRobot):
    def extra(self):
        from ..envs.factory_peg_ec_cfg import insertion_metrics

        return dict(**super().extra(), **insertion_metrics(*self.ee_pose()), finger_gap=self.gripper_gap())


class PlatePegRobot(IsaacECRobot):
    def extra(self):
        from ..envs.peg_insert_ec_cfg import PEG_POS, PEG_PROTRUDE, PEG_SIZE, hole_axis_point, peg_yaw
        from ..envs.plate_plane_cfg import HOLE_SIDE, PLATE_THICK, plate_rotation
        from .franka import PANDA_TCP_OFFSET

        R, p = self.ee_pose()
        hole, inward = hole_axis_point()
        tip = p + R @ [0., 0., PEG_PROTRUDE]
        depth = float((tip - hole) @ inward)
        error = tip - hole - depth * inward
        plate_R = plate_rotation()
        center = p + R @ (np.asarray(PEG_POS) - PANDA_TCP_OFFSET)
        clearance = box_aperture_clearance(
            plate_R.T @ (center - hole), plate_R.T @ R @ lp.rz(peg_yaw()),
            PEG_SIZE, HOLE_SIDE, PLATE_THICK)
        return dict(**super().extra(), insertion_depth=depth,
                    hole_error=float(np.linalg.norm(error)), plate_clearance=clearance)


class BlockRobot(IsaacECRobot):
    def extra(self):
        R0, p0 = self.root_pose()
        p = self.env.scene["block"].data.root_pos_w.torch[0].cpu().numpy()
        return dict(**super().extra(), block_z=float((R0.T @ (p - p0))[2]))


class FactoryNutRobot(IsaacECRobot):
    thread_start = None
    thread_end = None
    thread_probe = None
    bolt_position = None

    def extra(self):
        from ..envs.factory_nut_cfg import NUT_OFFSET

        R, p = self.ee_pose()
        nut = p + R @ NUT_OFFSET
        return dict(**super().extra(), nut_z=float(nut[2]), nut_bottom=float(nut[2] + .01),
                    nut_overlap=float(self.bolt_position[2] + .035 - nut[2] - .01),
                    nut_axis_error=float(np.linalg.norm(nut[:2] - self.bolt_position[:2])))

    def set_active_primitive(self, name, primitive):
        super().set_active_primitive(name, primitive)
        if isinstance(primitive, lp.RotationPrimitive):
            R, p = self.ee_pose()
            self.thread_start = (R.copy(), dict(self.extra()))

    def probe_engagement(self):
        """After execution, check that real thread contact resists a pure axial load."""
        R, p = self.ee_pose()
        self.thread_end = (R.copy(), self.extra())
        R_des, p_des = self.desired_pose()
        self.set_goal(p - [0., 0., .005], R)
        self.run(1., 0., p)
        displacement = self.thread_end[1]["nut_z"] - self.extra()["nut_z"]
        force = float((self.ee_pose()[0] @ self.wrench()[:3])[2])
        self.thread_probe = dict(displacement=float(displacement), resistance=force)
        self.set_goal(p_des, R_des)
        self.run(1., 0., self.ee_pose()[1])
