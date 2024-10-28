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
from datetime import datetime

import latent_planning.env_cfg  # noqa: F401
from latent_planning.runner import Runner
from omegaconf import DictConfig

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config("Isaac-Latent-Planning-Play", "cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: DictConfig):
    """Train latent planning agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = agent_cfg["num_envs"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = agent_cfg["device"]
    env_cfg.episode_length_s = agent_cfg["episode_length"]

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", "latent_planning"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        "Isaac-Latent-Planning-Play",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # save resume path before creating a new log_dir
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % agent_cfg["video_interval"] == 0,
            "video_length": agent_cfg["video_length"],
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = Runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg["device"])
    # load the checkpoint
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    goal_ee_pos = runner.get_goal_ee_pos()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = runner.alg.act(obs, goal_ee_pos)
            # env stepping
            obs, _, _, _ = env.step(actions)
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
