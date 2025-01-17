import os

import gymnasium as gym

from . import anymal_z1_env_cfg

gym.register(
    id="Isaac-Reacher-RL",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": anymal_z1_env_cfg.ReacherRLEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/anymal_z1_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reacher-RL-Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": anymal_z1_env_cfg.ReacherRLEnvCfg_PLAY,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/anymal_z1_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reacher-RL-Flat",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": anymal_z1_env_cfg.ReacherRLFlatEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/anymal_z1_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Reacher-RL-Flat-Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": anymal_z1_env_cfg.ReacherRLFlatEnvCfg_PLAY,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/anymal_z1_cfg.yaml",
    },
)
