import gymnasium as gym
import os

from . import locodiff_env_cfg

gym.register(
    id="Isaac-Locodiff",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locodiff_env_cfg.LocodiffEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/locodiff_cfg.yaml",
    },
)
