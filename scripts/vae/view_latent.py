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

# prevent hydra directory creation
sys.argv.append("hydra.output_subdir=null")
sys.argv.append("hydra.run.dir=.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import os
import torch

import hydra
from omegaconf import DictConfig

import isaac_ext.tasks  # noqa: F401
from vae.dataset import get_dataloaders
from vae.normalizer import GaussianNormalizer
from vae.utils import get_latest_run
from vae.vae import VAE

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra.main(
    config_path="../../isaac_ext/isaac_ext/tasks/reacher/config",
    config_name="cfg.yaml",
    version_base=None,
)
def main(agent_cfg: DictConfig):
    train_loader, test_loader, all_obs = get_dataloaders(**agent_cfg.dataset)
    normalizer = GaussianNormalizer(all_obs)
    policy = VAE(normalizer, device=agent_cfg.device, **agent_cfg.policy)

    # load the checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "reacher"))
    resume_path = os.path.join(get_latest_run(log_root_path), "models", "model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    loaded_dict = torch.load(resume_path, weights_only=True)
    policy.load_state_dict(loaded_dict["model_state_dict"])
    normalizer.load_state_dict(loaded_dict["norm_state_dict"])

    _, _, all_obs = get_dataloaders(**agent_cfg.dataset)
    all_obs = all_obs[:5000].to(agent_cfg.device)
    print(all_obs.shape)

    z, mu, logvar = policy.encode(all_obs)
    std = torch.exp(0.5 * logvar).detach().cpu().numpy()
    plt.hist(std.mean(axis=0), bins=20)
    plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
