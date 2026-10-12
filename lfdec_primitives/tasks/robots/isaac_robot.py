# Copyright (c) 2026, Xing Li (LfDEC primitives port).
# SPDX-License-Identifier: MIT

"""Isaac Lab robot adapter for EC primitives.

Provides measured TCP poses, filtered wrench feedback, gripper commands and joint
progress. Control-rate logs feed optional markers, the monitor and video recording."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import numpy as np
import torch

import lfdec_primitives as lp

__all__ = ["IsaacECRobot"]


class IsaacECRobot:
    """Drive a compliant-action Isaac env from a primitive.

    Args:
        env: the manager-based env, already reset.
        term: its ``arm_action`` term (a
            :class:`~lfdec_primitives.tasks.actions.compliant.CompliantCartesianAction`).
        dt_step: seconds of simulated time per ``env.step``.
        marker_hz: how often to refresh the visualisation.
        markers: draws the running primitive's EC parameters in the simulator. ``None`` draws
            nothing.
        monitor: the live window. ``None`` opens none.
        video: a :class:`RunVideo` fed every control step, or ``None``.
        watch: joints to report from :meth:`extra`, as ``"<articulation>/<joint>"`` -- the keys
            a task file's ``terminate_extra`` names, e.g. ``cabinet/drawer_top_joint``.
    """

    def __init__(self, env, term, dt_step: float, marker_hz: float = 20.0, markers=None, monitor=None,
                 watch: tuple[str, ...] = (), video=None):
        self.env, self.term, self.dt_step = env, term, dt_step
        self.markers, self.monitor, self.video = markers, monitor, video
        self.node, self.primitive = "", None
        self._watch = [(key, *self._find_joint(key)) for key in watch]
        self._t = 0.0
        self._act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        self._act[:, -1] = -1.0  # gripper closed until a primitive says otherwise (set_gripper)
        self.cameras: list = []  # IsaacCamera instances re-posed before every step
        self._viewer_restored = False
        self._marker_every = max(1, round(1.0 / (marker_hz * dt_step)))
        self._tick = 0
        self.ticks: list[dict] = []  # control-rate wrench history, for visualization plots

    # ------------------------------------------------------------------ RobotLike

    @property
    def t(self) -> float:
        return self._t

    @property
    def v_max_lin(self) -> float:
        return float(self.term.controller.gains["v_max"][0, :3].min())

    @property
    def v_max_rot(self) -> float:
        return float(self.term.controller.gains["v_max"][0, 3:].min())

    @property
    def f_max_lin(self) -> float:
        return float(self.term.controller.gains["f_max"][0, :3].min())

    @property
    def f_max_rot(self) -> float:
        return float(self.term.controller.gains["f_max"][0, 3:].min())

    def ee_pose(self):
        pos, quat = self.term.ee_pose
        return math_utils.matrix_from_quat(quat)[0].cpu().numpy(), pos[0].cpu().numpy()

    def desired_pose(self):
        p, q = self.term.desired_pose   # the term converts the controller's (w, x, y, z) to Isaac's
        return math_utils.matrix_from_quat(q)[0].cpu().numpy(), p[0].cpu().numpy()

    def set_goal(self, p, R=None):
        pos = self._t3(p).unsqueeze(0).repeat(self.env.num_envs, 1)
        if R is None:
            self.term.set_goal_pose(pos)
            return
        quat = math_utils.quat_from_matrix(
            torch.as_tensor(np.asarray(R), device=self.env.device, dtype=torch.float32).unsqueeze(0)
        )
        self.term.set_goal_pose(pos, quat.repeat(self.env.num_envs, 1))

    def look_at_workspace(self) -> None:
        """Restore the configured simulation view after camera-sensor initialization."""
        cfg = self.env.cfg
        self.env.sim.set_camera_view(tuple(cfg.viewer_eye), tuple(cfg.viewer_lookat))

    def run(self, seconds: float, stop_disp: float, p_ref):
        for _ in range(max(1, round(seconds / self.dt_step))):
            for cam in self.cameras:
                cam.sync()
            self.env.step(self._act)
            if self.cameras and not self._viewer_restored:
                self.look_at_workspace()
                self._viewer_restored = True
            self._t += self.dt_step
            self._tick += 1
            self._capture()
            if self.video is not None:
                self.video.tick(self._t, self.ticks)
            if self._tick % self._marker_every == 0:
                self.draw()
            if stop_disp > 0 and np.linalg.norm(self.ee_pose()[1] - p_ref) >= stop_disp:
                break

    def wrench(self):
        """Bias-corrected, low-pass wrench used by both the controller gate and contact tests.
        Expressed in the EE frame about the TCP."""
        return self.term.controller.wrench[0].cpu().numpy()

    def reach_margin(self, twist) -> float:
        """How far the TCP can follow ``twist`` (root frame) before an arm joint reaches a limit."""
        asset = self.env.scene["robot"]
        ids = self.term._joint_ids
        _, _, r_b = self.term._ee_pose_root_frame()
        J = self.term._jacobian_root_frame(r_b)[0].cpu().numpy()
        lim = asset.data.joint_pos_limits.torch[0, ids].cpu().numpy()
        q = asset.data.joint_pos.torch[0, ids].cpu().numpy()
        return lp.reach_along(J, q, lim[:, 0], lim[:, 1], twist)

    def set_gripper(self, closed: bool) -> None:
        """Binary gripper command, applied on every step from now on."""
        self._act[:, -1] = -1.0 if closed else 1.0

    def gripper_gap(self) -> float:
        """[m] opening of one finger, from the finger joints."""
        asset = self.env.scene["robot"]
        ids, _ = asset.find_joints("panda_finger_joint.*")
        return float(asset.data.joint_pos.torch[0, ids].mean())

    def extra(self) -> dict:
        """Task diagnostics appended to every logged row: the watched joints, by their
        ``"<articulation>/<joint>"`` key. The scripted runs' subclasses add their own readings."""
        return {key: float(art.data.joint_pos.torch[0, j]) for key, art, j in self._watch}

    def _find_joint(self, key: str):
        asset, _, joint = key.partition("/")
        scene = self.env.scene.articulations
        if asset not in scene or not joint:
            raise KeyError(f"{key!r}: expected '<articulation>/<joint>'; the scene has {sorted(scene)}")
        ids, names = scene[asset].find_joints(joint)
        if len(ids) != 1:
            raise KeyError(f"{key!r} matches {names}")
        return scene[asset], ids[0]

    def finger_axis_ee(self) -> np.ndarray:
        """Unit vector from the right to the left fingertip body, TCP frame.

        Which hand axis the fingers travel along is not the same in every robot description, and
        the camera mount is defined off it (``sensors.camera.mount_transform``), so it is
        measured rather than assumed.
        """
        import lfdec_primitives as lp

        asset = self.env.scene["robot"]
        ids, _ = asset.find_bodies(["panda_leftfinger", "panda_rightfinger"])
        pos = asset.data.body_pos_w.torch[0, ids].cpu().numpy()
        R_ee, _ = self.ee_pose()
        R_root, _ = self.root_pose()
        return lp.unit((R_root @ R_ee).T @ (pos[0] - pos[1]))

    def root_pose(self):
        """(R, p) of the robot root in world."""
        data = self.env.scene["robot"].data
        return (math_utils.matrix_from_quat(data.root_quat_w.torch[:1])[0].cpu().numpy(),
                data.root_pos_w.torch[0].cpu().numpy())

    # ------------------------------------------------------------------ visualisation

    def set_active_primitive(self, node: str, primitive) -> None:
        """The node about to run -- :class:`~..primitives.planner.ECPlanner`'s ``on_node``
        signature, so ``ECPlanner(..., on_node=robot.set_active_primitive)`` is the whole
        wiring. A script that runs one primitive without a planner calls it itself."""
        if self.primitive is not None and self.primitive.cg_event == "BC":
            # The previous contact has ended. Its unfinished servo/preload goal must not
            # become a lateral observation of the next EC. Retain preload at MC handovers.
            self.term.reseat_equilibrium_pose()
            self.run(.5, 0.0, self.ee_pose()[1])
            self.term.reseat_equilibrium_pose()
        self.node, self.primitive = node, primitive
        if self.monitor is not None:
            self.monitor.begin(node, primitive)

    def draw(self) -> None:
        """Refresh the arrows and the live window, at ``marker_hz`` from inside :meth:`run`.

        The arrows come from whatever the running primitive says its constraint is, so this is
        the same code for a drawer, a door and a plane, and a primitive with nothing to estimate
        -- free space -- simply draws nothing.
        """
        if self.markers is not None and self.primitive is not None:
            state = self.primitive.ec_state()
            if state is not None:
                _, p = self.ee_pose()
                self.markers.update(
                    self.to_world(p_b=p),
                    {k: self.to_world(d_b=v) for k, v in state.directions.items()},
                    {k: self.to_world(p_b=v) for k, v in state.origins.items()},
                )
        if self.monitor is not None:
            self.monitor.update(self.ticks)

    # ------------------------------------------------------------------ helpers

    def _t3(self, v) -> torch.Tensor:
        """A (3,) tensor on the env device, from numpy or from a tensor already on the GPU."""
        if isinstance(v, torch.Tensor):
            return v.detach().to(device=self.env.device, dtype=torch.float32).reshape(3)
        return torch.as_tensor(np.asarray(v), device=self.env.device, dtype=torch.float32).reshape(3)

    def to_world(self, p_b=None, d_b=None):
        """Root-frame point and/or direction to world, for the marker helpers."""
        data = self.env.scene["robot"].data
        root_p, root_q = data.root_pos_w.torch, data.root_quat_w.torch
        out = []
        if p_b is not None:
            out.append(root_p[0] + math_utils.quat_apply(root_q, self._t3(p_b).reshape(1, 3))[0])
        if d_b is not None:
            out.append(math_utils.quat_apply(root_q, self._t3(d_b).reshape(1, 3))[0])
        return out[0] if len(out) == 1 else tuple(out)

    def _capture(self) -> None:
        """One row of the control-rate wrench history.

        ``F_*``/``T_*`` are the gate input -- bias-corrected and low-passed -- because that is the
        signal the controller acts on; the raw wrench is kept alongside so the bias and the filter
        can be seen working.
        """
        ctl = self.term.controller
        F = ctl.wrench[0].cpu().numpy()
        raw = self.term.wrench_ee[0].cpu().numpy()
        pos, _ = self.term.ee_pose
        self.ticks.append(dict(
            t=self._t,
            F_x=F[0], F_y=F[1], F_z=F[2], T_x=F[3], T_y=F[4], T_z=F[5],
            F_raw_x=raw[0], F_raw_y=raw[1], F_raw_z=raw[2],
            gap=float(torch.norm(ctl.desired_pose[0][0] - pos[0])),
            **self.extra(),
        ))
