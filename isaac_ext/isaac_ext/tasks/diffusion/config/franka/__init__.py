import os

import gymnasium as gym

from isaac_ext.tasks.rsl_rl.config.franka import env_cfg

gym.register(
    id="Isaac-Franka-Diffusion",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaReachEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/cfg.yaml",
    },
)
