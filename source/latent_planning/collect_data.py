# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect demonstrations for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=8192, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_timesteps",
    type=int,
    default=128,
    help="Number of timesteps to store in the dataset.",
)
parser.add_argument(
    "--filename", type=str, default="hdf_dataset", help="Basename of output file."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import tqdm

import source.latent_planning.env_cfg # noqa: F401
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from source.latent_planning.latent_planning_data_collector import (
    LatentPlanningDataCollector,
)

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    task = "Isaac-Latent-Planning-Record"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)

    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(task, cfg=env_cfg)

    # specify directory for logging experiments
    log_dir = "./logs/latent_planning_record"
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = LatentPlanningDataCollector(
        directory_path=log_dir,
        filename=args_cli.filename,
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    collector_interface.reset()

    # simulate environment -- run everything in inference mode
    timestep = 0
    with torch.inference_mode() and tqdm.tqdm(total=args_cli.num_timesteps) as pbar:
        while simulation_app.is_running():
            actions = torch.zeros(
                (args_cli.num_envs, 7),
                device=args_cli.device,
                dtype=torch.float,
            )

            # -- obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"obs/{key}", value)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():  # type: ignore
                break
            
            # -- dones
            collector_interface.add("dones", dones)  # type: ignore

            timestep += 1
            pbar.update(1)

            # flush data from collector
            if timestep >= args_cli.num_timesteps:
                collector_interface.flush()
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
