import gymnasium as gym
import os

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

gym.register(
    id="Isaac-Diffusion-Cartpole",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/cartpole_cfg.yaml",
    },
)
