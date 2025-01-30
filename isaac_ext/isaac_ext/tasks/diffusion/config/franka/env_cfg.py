import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from omni.isaac.lab.utils import configclass

from isaac_ext.tasks.rsl_rl.config.franka.env_cfg import FrankaReachEnvCfg


@configclass
class FrankaDiffusionEnvCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.pose_command = None


@configclass
class FrankaGuidanceEnvCfg(FrankaDiffusionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle = RigidObjectCfg(
            prim_path="/World/obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.8, 0.25),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1.0,
                    max_angular_velocity=1.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0, density=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.125)),
        )
