import os

import gymnasium as gym

from . import env_cfg

gym.register(
    id="Isaac-Franka-Diffusion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaDiffusionEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Classifier",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaGuidanceEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/classifier_cfg.yaml",
    },
)
