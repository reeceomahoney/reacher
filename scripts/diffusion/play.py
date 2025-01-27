# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
sys.argv.append("hydra.output_subdir=null")
sys.argv.append("hydra.run.dir=.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch
from omegaconf import DictConfig
from omni.isaac.debug_draw import _debug_draw  # type: ignore
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

import isaac_ext.tasks  # noqa: F401
from locodiff.runner import DiffusionRunner
from locodiff.utils import dynamic_hydra_main
from vae.utils import get_latest_run

task = "Isaac-Franka-Diffusion"


@dynamic_hydra_main(task)
def main(agent_cfg: DictConfig, env_cfg: ManagerBasedRLEnvCfg):
    """Play with RSL-RL agent."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = agent_cfg.episode_length
    agent_cfg.dataset.task_name = task

    # create isaac environment
    env = gym.make(
        task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore
    agent_cfg.obs_dim = env.observation_space["policy"].shape[-1]  # type: ignore
    agent_cfg.act_dim = env.action_space.shape[-1]  # type: ignore

    # load previously trained model
    runner = DiffusionRunner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath("logs/diffusion/franka")
    resume_path = os.path.join(get_latest_run(log_root_path), "models", "model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obstacle = torch.tensor([[0, 0, 0]]).to(env.device)
    obstacle = obstacle.expand(env.num_envs, -1)

    goal = torch.tensor([[0.5, 0, 0.25]]).to(env.device)
    goal = goal.expand(env.num_envs, -1)

    # draw goal point
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_points([(0.5, 0, 0.25)], [(0.8, 0.2, 0, 1)], [25])

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            output = policy({"obs": obs, "obstacle": obstacle, "goal": goal})

            # env stepping
            for i in range(runner.policy.T_action):
                obs, _, dones, _ = env.step(output["action"][:, i])

        if dones.any():
            runner.policy.reset(dones)

        timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore
    # close sim app
    simulation_app.close()
