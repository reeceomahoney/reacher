import logging
import os
import statistics
import time
from collections import deque

import torch
import wandb
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state
from tqdm import tqdm, trange

from locodiff.dataset import get_dataloaders
from locodiff.envs import MazeEnv
from locodiff.models.unet import ConditionalUnet1D
from locodiff.models.transformer import DiffusionTransformer
from locodiff.policy import DiffusionPolicy
from locodiff.utils import ExponentialMovingAverage, InferenceContext, Normalizer

# A logger for this file
log = logging.getLogger(__name__)


class DiffusionRunner:
    def __init__(
        self, env: VecEnv | MazeEnv, agent_cfg, log_dir: str | None = None, device="cpu"
    ):
        self.env = env
        self.env.reset()
        self.cfg = agent_cfg
        self.device = device

        # classes
        self.train_loader, self.test_loader = get_dataloaders(**self.cfg.dataset)
        self.normalizer = Normalizer(self.train_loader, agent_cfg.scaling, device)
        # TODO: init model with hydra
        model = ConditionalUnet1D(**self.cfg.model)
        # model = DiffusionTransformer(**self.cfg.model)
        self.policy = DiffusionPolicy(model, self.normalizer, env, **self.cfg.policy)

        # ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.get_params(), self.cfg.ema_decay, self.cfg.device
        )
        self.use_ema = agent_cfg.use_ema

        # variables
        if isinstance(env, VecEnv):
            self.num_steps_per_env = int(
                self.cfg.episode_length
                / (self.env.cfg.decimation * self.env.cfg.sim.dt)  # type: ignore
            )
        elif isinstance(env, MazeEnv):
            self.num_steps_per_env = int(self.cfg.episode_length / 0.1)
        self.log_dir = log_dir
        self.current_learning_iteration = 0

        # logging
        if self.log_dir is not None:
            # initialize wandb
            wandb.init(project=self.cfg.wandb_project, dir=log_dir, config=self.cfg)
            # make model directory
            os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)  # type: ignore
            # save git diffs
            store_code_state(self.log_dir, [__file__])

    def learn(self):
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)
        self.policy.reset()
        self.train_mode()  # switch to train mode (for dropout for example)

        rewbuffer = deque()
        lenbuffer = deque()
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = int(start_iter + self.cfg.num_iters)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter):
            start = time.time()

            # simulation
            if it % self.cfg.sim_interval == 0:
                t = 0
                ep_infos = []
                self.env.reset()
                self.policy.set_goal(self.env.goal)
                obstacle = torch.tensor([[-1.0, -2.0]]).to(self.device)

                with InferenceContext(self) and tqdm(
                    total=self.num_steps_per_env, desc="Simulating...", leave=False
                ) as pbar:
                    while t < self.num_steps_per_env:
                        actions = self.policy.act({"obs": obs, "obstacles": obstacle})[
                            "action"
                        ]
                        for i in range(self.policy.T_action):
                            obs, rewards, dones, infos = self.env.step(actions[:, i])
                            if i < self.policy.T_action - 1:
                                self.policy.update_history({"obs": obs})

                            if t == self.num_steps_per_env - 1:
                                dones = torch.ones_like(dones)

                            # move device
                            obs, rewards, dones = (
                                obs.to(self.device),
                                rewards.to(self.device),
                                dones.to(self.device),
                            )
                            self.policy.reset(dones)

                            if self.log_dir is not None:
                                # rewards and dones
                                if "log" in infos:
                                    ep_infos.append(infos["log"])
                                cur_reward_sum += rewards
                                cur_episode_length += 1
                                new_ids = (dones > 0).nonzero(as_tuple=False)
                                rewbuffer.extend(
                                    cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                                )
                                lenbuffer.extend(
                                    cur_episode_length[new_ids][:, 0]
                                    .cpu()
                                    .numpy()
                                    .tolist()
                                )
                                cur_reward_sum[new_ids] = 0
                                cur_episode_length[new_ids] = 0

                            t += 1
                            pbar.update(1)
                            if t == self.num_steps_per_env:
                                break

            # evaluation
            if it % self.cfg.eval_interval == 0:
                with InferenceContext(self):
                    test_mse, test_obs_mse, test_act_mse = [], [], []
                    plot = True
                    for batch in tqdm(self.test_loader, desc="Testing...", leave=False):
                        mse, obs_mse, act_mse = self.policy.test(batch, plot)
                        plot = False
                        test_mse.append(mse)
                        test_obs_mse.append(obs_mse)
                        test_act_mse.append(act_mse)
                    test_mse = statistics.mean(test_mse)
                    test_obs_mse = statistics.mean(test_obs_mse)
                    test_act_mse = statistics.mean(test_act_mse)

            # training
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                batch = next(generator)

            loss = self.policy.update(batch)
            self.ema_helper.update(self.policy.parameters())

            # logging
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.cfg.log_interval == 0:
                # timing
                stop = time.time()
                iter_time = stop - start

                self.log(locals())
                if it % self.cfg.sim_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", "model.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "models", "model.pt"))

    def log(self, locs: dict):
        # training
        wandb.log(
            {
                "Loss/loss": locs["loss"],
                "Perf/iter_time": locs["iter_time"] / self.cfg.log_interval,
            },
            step=locs["it"],
        )
        # evaluation
        if locs["it"] % self.cfg.eval_interval == 0:
            wandb.log(
                {
                    "Loss/test_mse": locs["test_mse"],
                    "Loss/test_obs_mse": locs["test_obs_mse"],
                    "Loss/test_act_mse": locs["test_act_mse"],
                },
                step=locs["it"],
            )
        # simulation
        if locs["it"] % self.cfg.sim_interval == 0:
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
                        infotensor = torch.cat(
                            (infotensor, ep_info[key].to(self.device))
                        )
                    value = torch.mean(infotensor)
                    # log
                    if "/" in key:
                        wandb.log({key: value}, step=locs["it"])
                    else:
                        wandb.log({"Episode/" + key, value}, step=locs["it"])
            wandb.log(
                {
                    "Train/mean_reward": statistics.mean(locs["rewbuffer"]),
                    "Train/mean_episode_length": statistics.mean(locs["lenbuffer"]),
                },
                step=locs["it"],
            )

    def save(self, path, infos=None):
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())

        saved_dict = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
            "norm_state_dict": self.normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())

    def load(self, path):
        loaded_dict = torch.load(path)
        self.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.normalizer.load_state_dict(loaded_dict["norm_state_dict"])
        self.policy.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.policy.to(device)
        return self.policy.act

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()
