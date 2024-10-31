# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import math
import torch
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
)
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from latent_planning.robots import ANYMAL_D_Z1_CFG  # isort: skip

##
# Custom functions
##


def reset_joints_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to a random position within its full range."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state for shape
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # sample position
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = math_utils.sample_uniform(
        joint_pos_limits[..., 0],
        joint_pos_limits[..., 1],
        joint_pos.shape,
        joint_pos.device,
    )

    # set velocities to zero
    joint_vel = torch.zeros_like(joint_vel)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def ee_pos_rot(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset ee position and orientation in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze() - env.scene.env_origins
    quat = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7].squeeze()
    rot_mat = math_utils.matrix_from_quat(quat)
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([pos, ortho6d], dim=-1)


##
# Scene definition
##


@configclass
class LatentPlanningSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # NB: maybe add noise to these?
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*")},
        )
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_state = ObsTerm(
            func=ee_pos_rot,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(func=reset_joints_random, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
        },
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class LatentPlanningEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: LatentPlanningSceneCfg = LatentPlanningSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 50.0
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)


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
