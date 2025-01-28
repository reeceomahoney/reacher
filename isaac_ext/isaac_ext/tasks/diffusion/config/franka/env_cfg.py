from omni.isaac.lab.utils import configclass

from isaac_ext.tasks.rsl_rl.config.franka.env_cfg import FrankaReachEnvCfg


@configclass
class FrankaDiffusionEnvCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.pose_command = None
