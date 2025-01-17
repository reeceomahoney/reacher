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
    "--num_envs", type=int, default=16, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_timesteps", type=int, default=128, help="Name of the task.")
parser.add_argument(
    "--filename", type=str, default="hdf_dataset", help="Basename of output file."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
args_cli.headless = True

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
from omegaconf import DictConfig, OmegaConf
from omni.isaac.lab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

import isaac_ext.tasks  # noqa: F401
from locodiff.utils import dynamic_hydra_main
from vae.data_collector import DataCollector
from vae.utils import get_latest_run

task = "Isaac-Cartpole-RL"


@dynamic_hydra_main(task)
def main(agent_cfg: DictConfig, env_cfg: ManagerBasedRLEnvCfg):
    """Play with RSL-RL agent."""
    # load policy
    log_root_path = os.path.join("logs/rsl_rl/cartpole")
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_latest_run(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # create log directory
    log_dir = os.path.join("logs/rsl_rl/cartpole_collect")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg.num_envs = args_cli.num_envs

    # create isaac environment
    env = gym.make(
        task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_cfg_dict = OmegaConf.to_container(agent_cfg)
    ppo_runner = OnPolicyRunner(
        env, agent_cfg_dict, log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic,
        ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )

    collector_interface = DataCollector(
        directory_path=log_dir,
        filename=args_cli.filename,
    )
    collector_interface.reset()

    # reset environment
    obs, _ = env.get_observations()
    dones = torch.zeros(env.num_envs, dtype=torch.bool)
    timestep = 0
    # simulate environment
    with tqdm(total=args_cli.num_timesteps) as pbar:
        while simulation_app.is_running():
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # collect data
                collector_interface.add("obs", obs)
                collector_interface.add("actions", actions)
                # root_pos
                # root_pos = (
                #     env.unwrapped.scene["robot"].data.root_pos_w
                #     - env.unwrapped.scene.env_origins
                # )
                # collector_interface.add("root_pos", root_pos)
                # dones indicate first step in episode to make data splitting easier
                collector_interface.add("first_steps", dones)
                # env stepping
                obs, _, dones, _ = env.step(actions)

            timestep += 1
            pbar.update(1)
            if timestep >= args_cli.num_timesteps:
                break

        # close the simulator
        collector_interface.flush()
        collector_interface.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
