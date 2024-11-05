import gymnasium as gym

from latent_planning.envs.base_env import LatentPlanningEnvCfg

from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp


@configclass
class LatentFrankaEnvCfg(LatentPlanningEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )
        # observation
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "panda_joint.*"
        ]
        self.observations.policy.ee_pos.params["asset_cfg"].body_names = [
            "panda_hand"
        ]
        # rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "panda_hand"
        ]
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = ["panda_hand"]
        # command
        self.commands.ee_pose.body_name = "panda_hand"


@configclass
class LatentFrankaEnvCfg_RECORD(LatentFrankaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # increase sim frequency
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 1000.0
        # reset every step for recording
        self.episode_length_s = 1.0 / 100.0
        # disable gravity
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        # disable PD control
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0
        self.scene.robot.actuators["panda_shoulder"].damping = 0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0
        self.scene.robot.actuators["panda_forearm"].damping = 0


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Latent-Franka",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LatentFrankaEnvCfg,
        "cfg_entry_point": "source/latent_planning/cfg.yaml",
    },
)

gym.register(
    id="Isaac-Latent-Franka-Record",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LatentFrankaEnvCfg_RECORD,
        "cfg_entry_point": "source/latent_planning/cfg.yaml",
    },
)
