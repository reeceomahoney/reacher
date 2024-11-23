import gymnasium as gym
import os

from isaac_ext.tasks.rsl_rl.config.cartpole import cartpole_env_cfg

gym.register(
    id="Isaac-Cartpole-Diffusion",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cartpole_env_cfg.DiffusionCartpoleEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/cartpole_cfg.yaml",
    },
)
