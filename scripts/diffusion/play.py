# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# parse args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--plot", action="store_true", default=False, help="Whether to plot guidance."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.plot:
    args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import time

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
from locodiff.plotting import plot_trajectory
from locodiff.runner import DiffusionRunner
from locodiff.utils import dynamic_hydra_main, get_latest_run

task = "Isaac-Franka-Diffusion" if not args_cli.plot else "Isaac-Franka-Classifier"


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
    # env = gym.make(task, cfg=env_cfg)
    env = None

    # wrap around environment for rsl-rl
    # env = RslRlVecEnvWrapper(env)  # type: ignore
    agent_cfg.obs_dim = 4
    agent_cfg.act_dim = 2
    agent_cfg.dataset.test = True

    # load previously trained model
    runner = DiffusionRunner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath("logs/diffusion/franka")
    resume_path = os.path.join(get_latest_run(log_root_path), "models", "model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    # policy = runner.get_inference_policy(device=env.unwrapped.device)

    # set obstacle
    obstacle = torch.tensor([[0.5, 0, 0.125, 1, 0, 0, 0]])
    # obstacle = obstacle.expand(env.num_envs, -1)
    # env.unwrapped.scene["obstacle"].write_root_pose_to_sim(obstacle)

    # create trajectory visualizer
    trajectory_visualizer = create_trajectory_visualizer(agent_cfg)

    # reset environment
    # obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        start = time.time()

        # get goal
        # goal = env.unwrapped.command_manager.get_command("ee_pose")  # type: ignore
        # rot_mat = matrix_from_quat(goal[:, 3:])
        # ortho6d = rot_mat[..., :2].reshape(-1, 6)
        # goal = torch.cat([goal[:, :3], ortho6d], dim=-1)

        obs = torch.zeros(1, agent_cfg.obs_dim).to(agent_cfg.device)
        goal = torch.zeros(1, agent_cfg.obs_dim).to(agent_cfg.device)
        goal[0, 0] = 1
        # goal[0, 1] = 1

        # plot trajectory
        if args_cli.plot:
            # lambdas = [0, 1, 2, 5, 10]
            output = runner.policy.act(
                {"obs": obs, "obstacle": obstacle[:, :3], "goal": goal}
            )
            traj = output["obs_traj"][0].detach().cpu().numpy()
            plot_trajectory(traj, traj[0], goal[0].cpu().numpy())
            plt.show()
            simulation_app.close()
            exit()

        # agent stepping
        output = runner.policy.act(
            {"obs": obs, "obstacle": obstacle[:, :3], "goal": goal}
        )
        trajectory_visualizer.visualize(output["obs_traj"][0, :, 18:21])

        # env stepping
        for i in range(runner.policy.T_action):
            obs = env.step(output["action"][:, i])[0]

            end = time.time()
            if end - start < 1 / 30:
                time.sleep(1 / 30 - (end - start))
            start = time.time()

    # close the simulator
    env.close()


if __name__ == "__main__":
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    # torch.set_printoptions(precision=1, threshold=1000000, linewidth=500)
    main()  # type: ignore
    simulation_app.close()
