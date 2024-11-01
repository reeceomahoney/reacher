#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from tqdm import trange

import wandb
from latent_planning.dataset import get_dataloaders
from latent_planning.normalizer import GaussianNormalizer
from latent_planning.vae import VAE
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from omni.isaac.lab.utils.math import matrix_from_quat


class Runner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, agent_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = agent_cfg
        self.device = device

        # classes
        self.train_loader, self.test_loader, all_obs = get_dataloaders(
            **agent_cfg.dataset
        )
        self.normalizer = GaussianNormalizer(all_obs)
        self.policy = VAE(self.normalizer, device=self.device, **self.cfg.policy)

        # variables
        self.log_dir = log_dir
        self.num_steps_per_env = int(
            self.cfg.episode_length / (self.env.cfg.decimation * self.env.cfg.sim.dt)
        )
        self.current_learning_iteration = 0

        # logging
        if self.log_dir is not None:
            # initialize wandb
            wandb.init(project=self.cfg.wandb_project, dir=log_dir, config=self.cfg)
            # make model directory
            os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
            # save git diffs
            store_code_state(self.log_dir, [__file__])

    def learn(self):
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)
        self.policy.reset()
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque()
        lenbuffer = deque()
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = int(start_iter + self.cfg.num_learning_iterations)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter):
            start = time.time()

            # Rollout
            if it % self.cfg.sim_interval == 0:
                goal_ee_state = self.get_goal_ee_state()
                # TODO: handle this properly
                first_step = True
                for _ in range(self.num_steps_per_env):
                    actions = self.policy.act(obs, goal_ee_state, first_step)
                    obs, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )
                    # move device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    if first_step:
                        first_step = False

                    # reset prior loss weight
                    self.policy.reset(dones)

                    if self.log_dir is not None:
                        # rewards and dones
                        ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            # evaluation
            if it % self.cfg.eval_interval == 0:
                test_recon_loss = []
                for batch in self.test_loader:
                    test_recon_loss.append(self.policy.test(batch))
                test_recon_loss = statistics.mean(test_recon_loss)

            # training
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                batch = next(generator)

            loss, recon_loss, kl_loss = self.policy.update(batch)

            # logging
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.cfg.log_interval == 0:
                # timing
                stop = time.time()
                iter_time = stop - start

                self.log(locals())
                if it % self.cfg.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", "model.pt"))
                ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "models", "model.pt"))

    def log(self, locs: dict):
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                # get the mean of each ep info value
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    wandb.log({key: value}, step=locs["it"])
                else:
                    wandb.log({"Episode/" + key, value}, step=locs["it"])

        if locs["it"] % self.cfg.eval_interval == 0:
            wandb.log(
                {"Loss/test_recon_loss": locs["test_recon_loss"]}, step=locs["it"]
            )
        if locs["it"] % self.cfg.sim_interval == 0:
            wandb.log(
                {
                    "Train/mean_reward": statistics.mean(locs["rewbuffer"]),
                    "Train/mean_episode_length": statistics.mean(locs["lenbuffer"]),
                },
                step=locs["it"],
            )
        wandb.log(
            {
                "Loss/loss": locs["loss"],
                "Loss/recon_loss": locs["recon_loss"],
                "Loss/kl_loss": locs["kl_loss"],
                "Loss/beta": self.policy.beta,
                "Perf/iter_time": locs["iter_time"] / self.cfg.log_interval,
            },
            step=locs["it"],
        )

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
            "norm_state_dict": self.normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def load(self, path):
        loaded_dict = torch.load(path, weights_only=True)
        self.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.normalizer.load_state_dict(loaded_dict["norm_state_dict"])
        self.policy.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()

    def get_goal_ee_state(self):
        goal = self.env.unwrapped.command_manager.get_command("ee_pose")
        rot_mat = matrix_from_quat(goal[:, 3:7])
        ortho6d = rot_mat[..., :2].reshape(-1, 6)
        return torch.cat([goal[:, :3], ortho6d], dim=-1)
