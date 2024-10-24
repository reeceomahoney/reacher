#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from dataclasses import asdict

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

import wandb
from source.latent_planning.vae import VAE
from source.latent_planning.dataset import get_dataloaders


class Runner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env
        self.alg = VAE(device=self.device, **self.alg_cfg)
        self.num_steps_per_env = int(
            self.cfg["episode_length"] / (self.env.cfg.decimation * self.env.cfg.sim.dt)
        )
        self.save_interval = self.cfg["save_interval"]
        self.obs_normalizer = None  # TODO
        self.num_learning_iterations = float(self.cfg["num_learning_iterations"])
        self.train_loader, self.test_loader = get_dataloaders(**train_cfg["dataset"])

        # Log
        self.log_dir = log_dir
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # initialize wandb
        if self.log_dir is not None:
            wandb.init(project=self.cfg["wandb_project"])
            wandb.config.update({"runner_cfg": self.cfg})
            wandb.config.update({"env_cfg": asdict(self.env.cfg)})  # type: ignore
            wandb.config.update({"alg_cfg": self.alg_cfg})

    def learn(self):
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)
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
        tot_iter = int(start_iter + self.num_learning_iterations)
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )
                    # move device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    # obs = self.obs_normalizer(obs)

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

                # timing
                stop = time.time()
                collection_time = stop - start
                start = stop

            loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
                # if it % self.save_interval == 0:
                #     self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                ep_infos.clear()
                if it == start_iter:
                    # obtain all the diff files
                    git_file_paths = store_code_state(
                        self.log_dir, self.git_status_repos
                    )
                    # store them to wandb
                    for path in git_file_paths:
                        wandb.save(path, base_path=os.path.dirname(path))

        # if self.log_dir is not None:
        #     self.save(
        #         os.path.join(
        #             self.log_dir, f"model_{self.current_learning_iteration}.pt"
        #         )
        #     )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
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
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            else:
                wandb.log({"Episode/" + key, value}, step=locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        wandb.log(
            {
                # "Loss/learning_rate": self.alg.learning_rate,
                "Perf/collection time": locs["collection_time"],
                "Perf/learning_time": locs["learn_time"],
                "Train/mean_reward": statistics.mean(locs["rewbuffer"]),
                "Train/mean_episode_length": statistics.mean(locs["lenbuffer"]),
            },
            step=locs["it"],
        )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        )
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
        )

        eta_secs = (
            self.tot_time
            / (locs["it"] + 1)
            * (self.num_learning_iterations - locs["it"])
        )
        hours = int(eta_secs // 3600)
        minutes = int((eta_secs % 3600) // 60)
        seconds = int(eta_secs % 60)
        log_string += f"""{'ETA:':>{pad}} {hours:02d}:{minutes:02d}:{seconds:02d}\n"""

        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        wandb.save(path, step=self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if device is not None:
            self.obs_normalizer.to(device)
        policy = lambda x: self.alg.actor_critic.act_inference(
            self.obs_normalizer(x)
        )  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.train()
        # self.obs_normalizer.train()

    def eval_mode(self):
        self.alg.eval()
        # self.obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
