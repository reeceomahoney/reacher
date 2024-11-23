import math

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import (
    CartpoleEnvCfg,
)


@configclass
class DiffusionCartpoleEnvCfg(CartpoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_effort.scale = 5.0
        self.events.reset_pole_position.params["position_range"] = (-math.pi, math.pi)
        self.terminations.cart_out_of_bounds.params["bounds"] = (-6.0, 6.0)
