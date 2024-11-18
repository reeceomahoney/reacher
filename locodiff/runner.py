import logging
import matplotlib.pyplot as plt
import os
import statistics
import time
import torch
import torch.nn as nn
from collections import deque
from tqdm import tqdm, trange

from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

import locodiff.utils as utils
import wandb
from locodiff.policy import DiffusionPolicy
from locodiff.utils import ExponentialMovingAverage
from locodiff.wrappers import ScalingWrapper
from locodiff.dataset import get_dataloaders_and_scaler

# A logger for this file
log = logging.getLogger(__name__)


class DiffusionRunner:
    def __init__(self, env: VecEnv, agent_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = agent_cfg
        self.device = device

        # classes
        self.train_loader, self.test_loader, self.normalizer = get_dataloaders_and_scaler(
            **agent_cfg.dataset
        )
        self.policy = ScalingWrapper(
            model=DiffusionPolicy(self.normalizer, device=self.device, **self.cfg.policy),
            sigma_data=agent_cfg.sigma_data
        )

        # ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.parameters, agent_cfg.decay
        )
        self.use_ema = agent_cfg.use_ema

        # training

        # variables
        self.log_dir = log_dir
        self.num_steps_per_env = int(
            self.cfg.episode_length / (self.env.cfg.decimation * self.env.cfg.sim.dt)
        )
        self.current_learning_iteration = 0
        # flag whether to simulate or not
        self.sim = False

        # logging
        if self.log_dir is not None:
            # initialize wandb
            wandb.init(project=self.cfg.wandb_project, dir=log_dir, config=self.cfg)
            # make model directory
            os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
            # save git diffs
            store_code_state(self.log_dir, [__file__])

    ############
    # Training #
    ############

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
            if it % self.cfg.sim_interval == 0 and self.sim:
                goal_ee_state = self.get_goal_ee_state()
                for _ in range(self.num_steps_per_env):
                    actions = self.policy.act(obs, goal_ee_state)
                    leg_actions = torch.zeros(
                        (actions.shape[0], 12), device=actions.device
                    )

                    actions = torch.cat([actions, leg_actions], dim=1)
                    obs, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )
                    # move device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

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

    def train(self):
        """
        Main training loop
        """
        best_total_mse = 1e10
        generator = iter(self.train_loader)

        for step in trange(self.train_steps, desc="Training", dynamic_ncols=True):
            # evaluate
            if not step % self.eval_every:
                # reset the log_info
                log_info = {k: [] for k in self.eval_keys}

                # run evaluation
                for batch in tqdm(self.test_loader, desc="evaluating"):
                    info = self.evaluate(batch)
                    log_info = {k: v + [info[k]] for k, v in log_info.items()}

                # calculate the means and lr
                log_info = {k: sum(v) / len(v) for k, v in log_info.items()}
                log_info["lr"] = self.optimizer.param_groups[0]["lr"]

                # save model if it has improved mse
                if log_info["total_mse"] < best_total_mse:
                    best_total_mse = log_info["total_mse"]
                    self.save()
                    log.info("New best test loss. Stored weights have been updated!")

                # log to wandb
                wandb.log({k: v for k, v in log_info.items()}, step=step)

            # simulate
            if not step % self.sim_every:
                results = self.env.simulate(self)
                wandb.log(results, step=step)

            # train
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(self.train_loader)
                batch_loss = self.train_step(next(generator))
            if not step % 100:
                wandb.log({"loss": batch_loss}, step=step)

        self.save()
        log.info("Training done!")

    def train_step(self, batch: dict):
        data_dict = self.process_batch(batch)
        loss = self.agent.loss(data_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.ema_helper.update(self.agent.parameters())
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict:
        info = {}
        data_dict = self.process_batch(batch)

        if self.use_ema:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())

        if self.cond_mask_prob > 0:
            x_0, x_0_max_return = self.agent(data_dict)
            x_0 = self.scaler.clip(x_0)
            x_0 = self.scaler.inverse_scale_output(x_0)

            # divergence between normal and max_return outputs
            x_0_max_return = self.scaler.clip(x_0_max_return)
            x_0_max_return = self.scaler.inverse_scale_output(x_0_max_return)
            output_divergence = torch.abs(x_0 - x_0_max_return).mean().item()
            info["output_divergence"] = output_divergence
        else:
            x_0 = self.agent(data_dict)
            x_0 = self.scaler.clip(x_0)
            x_0 = self.scaler.inverse_scale_output(x_0)

        # calculate the MSE
        raw_action = self.scaler.inverse_scale_output(data_dict["action"])
        mse = nn.functional.mse_loss(x_0, raw_action, reduction="none")
        total_mse = mse.mean().item()
        first_mse = mse[:, 0, :].mean().item()
        last_mse = mse[:, -1, :].mean().item()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.agent.parameters())

        info["total_mse"] = total_mse
        info["first_mse"] = first_mse
        info["last_mse"] = last_mse

        return info

    ##############
    # Inference #
    ##############

    @torch.no_grad()
    def __call__(self, batch: dict, new_sampling_steps=None):
        batch = self.update_history(batch)
        data_dict = self.process_batch(batch)

        if new_sampling_steps is not None:
            self.agent.sampling_steps = new_sampling_steps

        if self.use_ema:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())

        if self.cond_mask_prob > 0:
            pred_action, _ = self.agent(data_dict)
        else:
            pred_action = self.agent(data_dict)

        pred_action = self.scaler.clip(pred_action)
        pred_action = self.scaler.inverse_scale_output(pred_action)
        pred_action = pred_action.cpu().numpy()
        pred_action = pred_action[:, : self.T_action].copy()

        if self.use_ema:
            self.ema_helper.restore(self.agent.parameters())

        return pred_action

    def reset(self, done):
        self.obs_hist[done] = 0
        self.skill_hist[done] = 0

    ######################
    # Saving and Loading #
    ######################

    def load(self, weights_path: str) -> None:
        model_dir = os.path.join(weights_path, "model")

        self.agent.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device),
            strict=False,
        )

        # Load scaler attributes
        scaler_state = torch.load(
            os.path.join(model_dir, "scaler.pt"), map_location=self.device
        )
        for attr, value in scaler_state.items():
            setattr(self.scaler, attr, value)

        log.info("Loaded pre-trained agent parameters and scaler")

    def save(self) -> None:
        model_dir = os.path.join(self.output_dir, "model")

        if self.use_ema:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())

        torch.save(self.agent.state_dict(), os.path.join(model_dir, "model.pt"))

        if self.use_ema:
            self.ema_helper.restore(self.agent.parameters())

        # Save scaler attributes
        scaler_state = ["x_max", "x_min", "y_max", "y_min", "x_mean", "x_std", "y_mean"]
        torch.save(
            {k: getattr(self.scaler, k) for k in scaler_state}, model_dir + "/scaler.pt"
        )

    ###################
    # Data Processing #
    ###################

    @torch.no_grad()
    def process_batch(self, batch: dict) -> dict:
        batch = self.dict_to_device(batch)

        raw_obs = batch["obs"]
        raw_action = batch.get("action", None)
        skill = batch["skill"]

        vel_cmd = batch.get("vel_cmd", None)
        if vel_cmd is None:
            vel_cmd = self.sample_vel_cmd(raw_obs.shape[0])

        returns = batch.get("return", None)
        if returns is None:
            returns = self.compute_returns(raw_obs, vel_cmd)

        obs = self.scaler.scale_input(raw_obs[:, : self.T_cond])

        if raw_action is None:
            action = None
        else:
            action = self.scaler.scale_output(
                torch.cat(
                    [
                        # raw_obs[:, self.T_cond - 1 : self.T_cond + self.T - 1],
                        raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1],
                    ],
                    dim=-1,
                )
            )

        processed_batch = {
            "obs": obs,
            "action": action,
            "vel_cmd": vel_cmd,
            "skill": skill,
            "return": returns,
        }

        return processed_batch

    def update_history(self, batch):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = batch["obs"]
        batch["obs"] = self.obs_hist.clone()
        return batch

    def dict_to_device(self, batch):
        return {k: v.clone().to(self.device) for k, v in batch.items()}

    def sample_vel_cmd(self, batch_size):
        # vel_cmd = torch.randint(0, 2, (batch_size, 1), device=self.device).float()
        # return vel_cmd * 2 - 1
        vel_limits = [0.8, 0.5, 1.0]
        vel_cmd = torch.rand(batch_size, 3, device=self.device)

        for i in range(3):
            vel_cmd[i] = vel_cmd[i] * 2 * vel_limits[i] - vel_limits[i]

        return vel_cmd

    def compute_returns(self, obs, vel_cmd):
        rewards = utils.reward_function(obs, vel_cmd, self.reward_fn)
        rewards = (
            rewards[:, self.T_cond - 1 : self.T_cond + self.return_horizon - 1] - 1
        )

        gammas = torch.tensor([0.99**i for i in range(self.return_horizon)]).to(
            self.device
        )
        returns = (rewards * gammas).sum(dim=-1)
        returns = torch.exp(returns / 50)
        returns = (returns - returns.min()) / (returns.max() - returns.min())

        # self.plot_returns(returns)

        return returns.unsqueeze(-1)

    def plot_returns(self, returns):
        plt.figure(figsize=(10, 5))
        plt.hist(returns.cpu().numpy(), bins=20)
        plt.show()
        exit()

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()
