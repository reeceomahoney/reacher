# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

import isaac_ext.tasks  # noqa: F401
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers.visualization_markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from isaaclab.utils.math import matrix_from_quat
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from locodiff.classifier_runner import ClassifierRunner
from locodiff.plotting import plot_3d_guided_trajectory
from locodiff.utils import dynamic_hydra_main, get_latest_run

task = "Isaac-Franka-Classifier"


def interpolate_color(t):
    start_color = (0.0, 0.0, 1.0)  # Blue
    end_color = (1.0, 0.0, 0.0)  # Red
    return tuple(
        start + (end - start) * t
        for start, end in zip(start_color, end_color, strict=False)
    )


def create_trajectory_visualizer(agent_cfg):
    trajectory_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Trajectory",
        markers={
            f"cuboid_{i}": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=interpolate_color(i / (agent_cfg.T - 1))
                ),
            )
            for i in range(agent_cfg.T)
        },
    )
    trajectory_visualizer = VisualizationMarkers(trajectory_visualizer_cfg)
    trajectory_visualizer.set_visibility(True)

    return trajectory_visualizer


@dynamic_hydra_main(task)
def main(agent_cfg: DictConfig, env_cfg: ManagerBasedRLEnvCfg):
    """Play with RSL-RL agent."""
    env_cfg.scene.num_envs = 1
    agent_cfg.num_envs = 1
    agent_cfg.dataset.task_name = task

    # create isaac environment
    env = gym.make(task, cfg=env_cfg)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore
    agent_cfg.obs_dim = env.observation_space["policy"].shape[-1]  # type: ignore
    agent_cfg.act_dim = env.action_space.shape[-1]  # type: ignore

    # load previously trained model
    runner = ClassifierRunner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath("logs/classifier/franka")
    resume_path = os.path.join(get_latest_run(log_root_path), "models", "model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # set obstacle
    obstacle = torch.tensor([[0.5, 0, 0.125, 1, 0, 0, 0]]).to(env.device)
    obstacle = obstacle.expand(env.num_envs, -1)
    # env.unwrapped.scene["obstacle"].write_root_pose_to_sim(obstacle)

    # create trajectory visualizer
    trajectory_visualizer = create_trajectory_visualizer(agent_cfg)

    # set classifier scale

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start = time.time()

        # get goal
        goal = env.unwrapped.command_manager.get_command("ee_pose")  # type: ignore
        rot_mat = matrix_from_quat(goal[:, 3:])
        ortho6d = rot_mat[..., :2].reshape(-1, 6)
        goal = torch.cat([goal[:, :3], ortho6d], dim=-1)

        # agent stepping
        runner.policy.alpha = 200
        output = policy({"obs": obs, "obstacle": obstacle[:, :3], "goal": goal})
        trajectory_visualizer.visualize(output["obs_traj"][0, :, 18:21])

        # plot trajectory
        alphas = [0, 10, 50, 100, 200]
        plot_3d_guided_trajectory(runner.policy, obs, goal, obstacle[:, :3], alphas, "alphas")
        plt.show()

        # env stepping
        for i in range(runner.policy.T_action):
            obs, _, dones, _ = env.step(output["action"][:, i])

            end = time.time()
            if end - start < 1 / 30:
                time.sleep(1 / 30 - (end - start))

            start = time.time()

        if dones.any():
            runner.policy.reset(dones)

        timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    main()  # type: ignore
    simulation_app.close()
