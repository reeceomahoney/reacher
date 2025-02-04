import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    ReachSceneCfg,
)

from isaac_ext.tasks.rsl_rl.config.franka.env_cfg import FrankaReachEnvCfg


@configclass
class FrankaGuidanceSceneCfg(ReachSceneCfg):
    obstacle = RigidObjectCfg(
        prim_path="/World/obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.8, 0.4),
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
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.2)),
    )


@configclass
class FrankaDiffusionEnvCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.pose_command = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


@configclass
class FrankaGuidanceEnvCfg(FrankaDiffusionEnvCfg):
    scene = FrankaGuidanceSceneCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.ranges.pos_x = (0.8, 0.8)
        self.commands.ee_pose.ranges.pos_y = (0, 0)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.2)
        self.commands.ee_pose.ranges.roll = (0, 0)
        self.commands.ee_pose.ranges.pitch = (0, 0)
        self.commands.ee_pose.ranges.yaw = (0, 0)
