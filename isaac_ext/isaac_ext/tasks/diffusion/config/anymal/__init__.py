import os

import gymnasium as gym

from . import locodiff_env_cfg

gym.register(
    id="Isaac-Locodiff",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locodiff_env_cfg.LocodiffEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/locodiff_cfg.yaml",
    },
)
