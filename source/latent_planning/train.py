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
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import hydra
import latent_planning.env_cfg  # noqa: F401
from hydra.core.hydra_config import HydraConfig
from latent_planning.runner import Runner
from omegaconf import DictConfig, OmegaConf
from utils import get_latest_run

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

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
    task = "Isaac-Latent-Franka"
    env_cfg = parse_env_cfg(task, device=agent_cfg.device, num_envs=agent_cfg.num_envs)

    # override env configs
    env_cfg.seed = agent_cfg.seed
    env_cfg.episode_length_s = agent_cfg.episode_length

    # specify directory for logging experiments
    log_dir = HydraConfig.get().runtime.output_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % agent_cfg.video_interval == 0,
            "video_length": agent_cfg.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = Runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)

    # load the checkpoint
    if agent_cfg.resume:
        log_root_path = os.path.abspath(os.path.join("logs", "latent_planning"))
        resume_path = get_latest_run(log_root_path, resume=True)
        resume_path = os.path.join(resume_path, "models", "model.pt")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    agent_cfg = OmegaConf.to_container(agent_cfg)  # type: ignore
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
