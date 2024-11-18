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
from locodiff.dataset import get_dataloaders_and_scaler
from locodiff.policy import DiffusionPolicy
from locodiff.transformer import DiffusionTransformer
from locodiff.utils import ExponentialMovingAverage
from locodiff.wrappers import ScalingWrapper

# A logger for this file
log = logging.getLogger(__name__)


class DiffusionRunner:
    def __init__(
        self, env: VecEnv, agent_cfg, log_dir: str | None = None, device="cpu"
    ):
        self.env = env
        self.cfg = agent_cfg
        self.device = device

        # classes
        self.train_loader, self.test_loader, self.normalizer = (
            get_dataloaders_and_scaler(**self.cfg.dataset)
        )
        model = ScalingWrapper(
            model=DiffusionTransformer(**self.cfg.model),
            sigma_data=agent_cfg.policy.sigma_data,
        )
        self.policy = DiffusionPolicy(model, self.normalizer, **self.cfg.policy)

        # ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.get_params(), self.cfg.ema_decay, self.cfg.device
        )
        self.use_ema = agent_cfg.use_ema

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
        tot_iter = int(start_iter + self.cfg.num_iters)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter):
            start = time.time()

            # Rollout
            if it % self.cfg.sim_interval == 0:
                with torch.inference_mode():
                    self.eval_mode()
                    for _ in range(self.num_steps_per_env):
                        actions = self.policy.act({"obs": obs})
                        obs, rewards, dones, infos = self.env.step(actions)
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
                with torch.inference_mode():
                    self.eval_mode()
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

            self.train_mode()
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

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()
