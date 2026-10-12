# Copyright (c) 2026, Xing Li.
# SPDX-License-Identifier: MIT
"""Register Isaac Lab EC tasks and resolve their environment configurations."""
import gymnasium as gym
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.utils import resolve_task_config

__all__ = ["resolve_ec_task"]

_ENV_CONFIGS = {
    "LfDEC-Factory-Nut-Thread-EC-Franka": "factory_nut_cfg:FrankaFactoryNutECCfg",
    "LfDEC-Factory-Nut-Thread-Franka": "factory_nut_cfg:FrankaFactoryNutCfg",
    "LfDEC-Factory-Peg-Grasp-Visual-Franka": "factory_grasp_cfg:FrankaFactoryPegGraspVisualCfg",
    "LfDEC-Factory-Peg-Grasp-Franka": "factory_grasp_cfg:FrankaFactoryPegGraspCfg",
    "LfDEC-Factory-Peg-Insert-Visual-Franka": "factory_peg_ec_cfg:FrankaFactoryPegVisualECCfg",
    "LfDEC-Factory-Peg-Insert-Franka": "factory_peg_ec_cfg:FrankaFactoryPegECCfg",
    "LfDEC-Drawer-Translation-Franka": "drawer_translation_cfg:FrankaDrawerTranslationECCfg",
    "LfDEC-Plate-Plane-Franka": "plate_plane_cfg:FrankaPlatePlaneECCfg",
    "LfDEC-Door-Rotation-Franka": "door_rotation_cfg:FrankaDoorRotationECCfg",
    "LfDEC-Drawer-EC-Franka": "drawer_ec_cfg:FrankaDrawerECCfg",
    "LfDEC-Grasp-Free-Franka": "grasp_free_cfg:FrankaGraspFreeECCfg",
    "LfDEC-Door-EC-Franka": "door_ec_cfg:FrankaDoorECCfg",
    "LfDEC-Peg-Insert-Franka": "peg_insert_ec_cfg:FrankaPegInsertECCfg",
}

for name, config in _ENV_CONFIGS.items():
    gym.register(
        id=name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.{config}"},
        disable_env_checker=True,
    )


def resolve_ec_task(task: str, num_envs: int = 1):
    """Apply launcher overrides, set scene copies and configure the simulation view."""
    env_cfg, _ = resolve_task_config(task, None)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(
        eye=tuple(env_cfg.viewer_eye), lookat=tuple(env_cfg.viewer_lookat), focal_length=env_cfg.viewer_focal_length)
    return env_cfg
