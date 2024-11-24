import math

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import (
    CartpoleEnvCfg,
)


@configclass
class DiffusionCartpoleEnvCfg(CartpoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # robot
        self.robot.actuators["cart_actuator"].effort_limit = 100.0
        self.robot.actuators["pole_actuator"].effort_limit = 100.0
        # mdp
        self.actions.joint_effort.scale = 100.0
        self.events.reset_pole_position.params["position_range"] = (-math.pi, math.pi)
