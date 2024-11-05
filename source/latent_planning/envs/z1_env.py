import gymnasium as gym

from latent_planning.envs.base_env import LatentPlanningEnvCfg
from latent_planning.robots import Z1_CFG

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp


@configclass
class LatentZ1EnvCfg(LatentPlanningEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # robot
        self.scene.robot = Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=1.0,
            use_default_offset=False,
        )
        # observation
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "joint.*"
        ]
        self.observations.policy.ee_state.params["asset_cfg"].body_names = [
            "gripperMover"
        ]
        # rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [
            "gripperMover"
        ]
        self.rewards.end_effector_orientation_tracking.params[
            "asset_cfg"
        ].body_names = ["gripperMover"]
        # command
        self.commands.ee_pose.body_name = "gripperMover"


@configclass
class LatentZ1EnvCfg_RECORD(LatentZ1EnvCfg):
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
        self.scene.robot.actuators["arm"].stiffness = 0
        self.scene.robot.actuators["arm"].damping = 0


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Latent-Z1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LatentZ1EnvCfg,
        "cfg_entry_point": "source/latent_planning/cfg.yaml",
    },
)

gym.register(
    id="Isaac-Latent-Z1-Record",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LatentZ1EnvCfg_RECORD,
        "cfg_entry_point": "source/latent_planning/cfg.yaml",
    },
)
