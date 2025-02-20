import isaaclab.sim as sim_utils
from isaac_ext.tasks.rsl_rl.config.franka.env_cfg import FrankaReachEnvCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils import configclass


@configclass
class FrankaDiffusionEnvCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.pose_command = None  # type: ignore
        self.curriculum = None  # type: ignore
        self.episode_length_s = 6.0
        self.commands.ee_pose.resampling_time_range = (6.0, 6.0)


@configclass
class FrankaGuidanceEnvCfg(FrankaDiffusionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.commands.ee_pose.ranges.pos_x = (0.8, 0.8)
        self.commands.ee_pose.ranges.pos_y = (0, 0)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.2)
        self.commands.ee_pose.ranges.roll = (0, 0)
        self.commands.ee_pose.ranges.pitch = (0, 0)
        self.commands.ee_pose.ranges.yaw = (0, 0)

        self.scene.obstacle = RigidObjectCfg(  # type: ignore
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.8, 0.6),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.3)),
        )
