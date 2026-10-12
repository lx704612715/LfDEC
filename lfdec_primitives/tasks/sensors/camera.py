# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""RGB-D wrist camera for visual servoing.

The sensor pose is updated from the measured TCP pose and mount transform before
each physics step. Images, depth and intrinsics are returned as a View."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.sensors import CameraCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import lfdec_primitives as lp

from ..robots.isaac_robot import IsaacECRobot
from ..robots.franka import PANDA_TCP_OFFSET

__all__ = ["IsaacCamera", "wrist_camera_cfg", "mount_transform"]


CAM_OUT, CAM_TOWARD = 0.06, 0.03
CAM_WIDTH, CAM_HEIGHT = 640, 480


def mount_transform(finger_axis_ee: np.ndarray) -> np.ndarray:
    """Camera pose in the TCP frame for a gripper whose fingers travel along ``finger_axis_ee``.

    The offset lies in the hand plane, perpendicular to the finger travel; the camera looks along
    the hand's z with its image x along the offset, so the block below the TCP appears displaced
    along image u.
    """
    a = lp.unit(np.asarray(finger_axis_ee, dtype=float))
    z = np.array([0.0, 0.0, 1.0])
    x = lp.unit(np.cross(a, z))
    if x[0] < 0:                       # pick the side toward +x_hand when there is a choice
        x = -x
    R = np.stack([x, np.cross(z, x), z], axis=1)
    return lp.make_transform(R, x * CAM_OUT + z * (CAM_TOWARD - float(PANDA_TCP_OFFSET[2])))

def wrist_camera_cfg(focal_length: float, width: int = 640, height: int = 480) -> CameraCfg:
    """The wrist camera's scene entry: a free prim (see the module docstring) rendering RGB and
    depth with Newton's Warp renderer, so Newton-only.

    ``focal_length`` is in mm on Isaac's 20.955 mm aperture: 15 sees ~70 deg across, which is
    the block task's view of a 5 cm block from 15 cm; 8 sees ~105 deg, which is what puts a
    cabinet handle and the sticker beside it in one frame from 15 cm away.
    """
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/WristCam",
        update_period=0.0, width=width, height=height,
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_length, clipping_range=(0.05, 5.0)),
    )


class IsaacCamera:
    """A camera sensor carried by the control point.

    Args:
        robot: the adapter whose TCP the camera rides on; the camera registers itself so
            :meth:`IsaacECRobot.run` re-poses it before every step.
        sensor_name: the ``CameraCfg`` field name in the scene.
        T_ee_cam: camera pose in the TCP frame, OpenCV convention (z forward, y down).
    """

    def __init__(self, robot: IsaacECRobot, sensor_name: str, T_ee_cam: np.ndarray):
        self.robot = robot
        self.sensor = robot.env.scene.sensors[sensor_name]
        self.T_ee_cam = np.asarray(T_ee_cam, dtype=float)
        robot.cameras.append(self)
        self.sync()

    def _T_world_ee(self) -> np.ndarray:
        # Isaac Lab 3.0 quaternions are (x, y, z, w) throughout, so they go through math_utils;
        # lp.quat_from_matrix below is (w, x, y, z) and is reordered for set_world_poses.
        data = self.robot.env.scene["robot"].data
        T_w_root = lp.make_transform(math_utils.matrix_from_quat(data.root_quat_w.torch[:1])[0].cpu().numpy(),
                                     data.root_pos_w.torch[0].cpu().numpy())
        pos, quat = self.robot.term.ee_pose
        T_root_ee = lp.make_transform(math_utils.matrix_from_quat(quat)[0].cpu().numpy(), pos[0].cpu().numpy())
        return T_w_root @ T_root_ee

    def sync(self) -> None:
        """Put the sensor where the mount says, given where the TCP is now."""
        T = self._T_world_ee() @ self.T_ee_cam
        q = lp.quat_from_matrix(T[:3, :3])   # (w, x, y, z)
        dev = self.robot.env.device
        self.sensor.set_world_poses(
            torch.as_tensor(T[:3, 3], dtype=torch.float32, device=dev).unsqueeze(0),
            torch.as_tensor([q[1], q[2], q[3], q[0]], dtype=torch.float32, device=dev).unsqueeze(0),
            convention="ros",
        )

    def capture(self) -> lp.View:
        d = self.sensor.data
        rgb = np.ascontiguousarray(d.output["rgb"].torch[0].cpu().numpy()[..., :3])
        depth = d.output["distance_to_image_plane"].torch[0].cpu().numpy().reshape(rgb.shape[0], rgb.shape[1])
        K = d.intrinsic_matrices.torch[0].cpu().numpy()
        return lp.View(rgb, depth.astype(np.float32), K)
