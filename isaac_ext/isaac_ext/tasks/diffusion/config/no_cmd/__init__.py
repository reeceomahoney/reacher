import gymnasium as gym
import os

from . import no_cmd_env_cfg

gym.register(
    id="Isaac-Locodiff-no-cmd",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": no_cmd_env_cfg.LocodiffEnvCfg,
        "agent_cfg_entry_point": f"{os.path.dirname(__file__)}/no_cmd_env.yaml",
    },
)
