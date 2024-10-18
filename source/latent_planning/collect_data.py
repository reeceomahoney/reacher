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
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=1,
    help="Number of episodes to store in the dataset.",
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

from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from source.latent_planning.latent_planning_data_collector import (
    LatentPlanningDataCollector,
)

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)

    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(task, cfg=env_cfg)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/latent_planning", task)
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

    #env.scene.articulations["robot"].data.default_joint_limits

    # simulate environment -- run everything in inference mode
    timestep = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = torch.zeros(
                (env.unwrapped.num_envs, 7),
                device=env.unwrapped.device,
                dtype=torch.float,
            )

            # store signals before stepping
            # -- obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"next_obs/{key}", value)
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            print(timestep)
            timestep += 1

            # flush data from collector
            if timestep >= 100:
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
