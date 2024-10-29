# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a latent planning agent.")
parser.add_argument("--video", action="store_true", default=False)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# prevent hydra directory creation
sys.argv.append("hydra.output_subdir=null")
sys.argv.append("hydra.run.dir=.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import hydra
import latent_planning.env_cfg  # noqa: F401
from latent_planning.runner import Runner
from omegaconf import DictConfig
from utils import get_latest_run

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra.main(config_path=".", config_name="cfg.yaml", version_base=None)
def main(agent_cfg: DictConfig):
    """Train latent planning agent."""
    # load env cfg
    task = "Isaac-Latent-Planning-Play"
    env_cfg = parse_env_cfg(task, device=agent_cfg.device, num_envs=agent_cfg.num_envs)

    # override env configs
    env_cfg.seed = agent_cfg.seed
    env_cfg.episode_length_s = agent_cfg.episode_length

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = Runner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "latent_planning"))
    resume_path = os.path.join(get_latest_run(log_root_path), "models", "model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # reset environment
    obs, _ = env.get_observations()
    runner.alg.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # agent stepping
        goal_ee_pos = env.unwrapped.command_manager.get_command("ee_pose")[:, :3]
        actions = runner.alg.act(obs, goal_ee_pos)
        # env stepping
        obs, _, dones, _ = env.step(actions)

        # reset prior loss weight
        runner.alg.reset(dones)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
