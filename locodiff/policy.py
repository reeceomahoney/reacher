import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import (
    EDMDPMSolverMultistepScheduler,
)

import wandb
from locodiff.utils import CFGWrapper, apply_conditioning, rand_log_logistic


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model,
        normalizer,
        env,
        obs_dim: int,
        act_dim: int,
        T: int,
        T_cond: int,
        T_action: int,
        num_envs: int,
        sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        cond_lambda: int,
        cond_mask_prob: float,
        lr: float,
        betas: tuple,
        num_iters: int,
        inpaint: bool,
        device: str,
    ):
        super().__init__()
        # model
        if cond_mask_prob > 0:
            model = CFGWrapper(model, cond_lambda, cond_mask_prob)
        self.model = model

        # other classes
        self.env = env
        self.normalizer = normalizer
        self.obs_hist = torch.zeros((num_envs, T_cond, obs_dim), device=device)
        self.noise_scheduler = EDMDPMSolverMultistepScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            num_train_timesteps=sampling_steps,
        )

        # dims
        self.obs_dim = obs_dim
        self.input_dim = obs_dim + act_dim
        self.action_dim = act_dim
        self.input_len = T + T_cond - 1 if inpaint else T
        self.T = T
        self.T_cond = T_cond
        self.T_action = T_action
        self.num_envs = num_envs
        self.goal_dim = 4

        # diffusion
        self.sampling_steps = sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_mask_prob = cond_mask_prob
        self.inpaint = inpaint

        # optimizer and lr scheduler
        optim_groups = self.model.get_optim_groups()
        self.optimizer = AdamW(optim_groups, lr=lr, betas=betas)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

        # reward guidance
        self.gammas = torch.tensor([0.99**i for i in range(self.T)]).to(device)

        self.device = device
        self.to(device)

    ############
    # Main API #
    ############

    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, self.action_dim :]

        # extract action
        if self.inpaint:
            action = x[
                :, self.T_cond - 1 : self.T_cond + self.T_action - 1, : self.action_dim
            ]
        else:
            action = x[:, : self.T_action, : self.action_dim]

        return {"action": action, "obs_traj": obs}

    def update(self, data):
        # preprocess data
        data = self.process(data)
        cond = self.create_conditioning(data)

        # noise data
        noise = torch.randn_like(data["input"])
        sigma = self.sample_training_density(len(noise)).view(-1, 1, 1)
        x_noise = data["input"] + noise * sigma
        # scale inputs
        x_noise_in = self.noise_scheduler.precondition_inputs(x_noise, sigma)
        x_noise_in = apply_conditioning(x_noise_in, cond, self.action_dim)
        sigma_in = self.noise_scheduler.precondition_noise(sigma)

        # cfg masking
        if self.cond_mask_prob > 0:
            cond_mask = torch.rand_like(data["returns"]) < self.cond_mask_prob
            data["returns"][cond_mask] = 0

        # compute model output
        out = self.model(x_noise_in, sigma_in, data)
        out = self.noise_scheduler.precondition_outputs(x_noise, out, sigma)
        out = apply_conditioning(out, cond, self.action_dim)
        # calculate loss
        loss = torch.nn.functional.mse_loss(out, data["input"])

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def test(self, data: dict, plot) -> tuple[float, float, float]:
        data = self.process(data)
        x = self.forward(data)
        # calculate losses
        input = self.normalizer.inverse_scale_output(data["input"])
        loss = nn.functional.mse_loss(x, input, reduction="none")
        obs_loss = loss[:, :, self.action_dim :].mean()
        action_loss = loss[:, :, : self.action_dim].mean()

        if plot:
            obs_traj = x[0, :, self.action_dim :].cpu().numpy()
            fig = plt.figure()
            # maze
            plt.imshow(self.env.get_maze(), cmap="gray", extent=(-4, 4, -4, 4))
            # trajectory
            colors = plt.cm.inferno(np.linspace(0, 1, len(obs_traj)))  # type: ignore
            plt.scatter(obs_traj[:, 0], obs_traj[:, 1], c=colors)
            # obs and goal
            marker_params = {"markersize": 10, "markeredgewidth": 3}
            plt.plot(
                obs_traj[0, 0], obs_traj[0, 1], "x", color="green", **marker_params  # type: ignore
            )
            plt.plot(
                obs_traj[-1, 0], obs_traj[-1, 1], "x", color="red", **marker_params  # type: ignore
            )
            # log
            wandb.log({"Image": wandb.Image(fig)})
            plt.close(fig)

        return loss.mean().item(), obs_loss.item(), action_loss.item()

    def reset(self, dones=None):
        if dones is not None:
            self.obs_hist[dones.bool()] = 0
        else:
            self.obs_hist.zero_()

    #####################
    # Inference backend #
    #####################

    @torch.no_grad()
    def forward(self, data: dict) -> torch.Tensor:
        # sample noise
        B = data["obs"].shape[0]
        x = torch.randn((B, self.input_len, self.input_dim)).to(self.device)
        # we should need this but performance is better without it
        # x *= self.noise_scheduler.init_noise_sigma

        # create inpainting conditioning
        cond = self.create_conditioning(data)
        # this needs to called every time we do inference
        self.noise_scheduler.set_timesteps(self.sampling_steps)

        # inference loop
        for t in self.noise_scheduler.timesteps:
            x_in = self.noise_scheduler.scale_model_input(x, t)
            x_in = apply_conditioning(x_in, cond, 2)
            output = self.model(x_in, t.expand(B), data)
            x = self.noise_scheduler.step(output, t, x, return_dict=False)[0]

        # final conditioning
        x = apply_conditioning(x, cond, 2)
        # denormalize
        x = self.normalizer.clip(x)
        x = self.normalizer.inverse_scale_output(x)
        return x

    ###################
    # Data processing #
    ###################

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)
        raw_action = data.get("action", None)

        if raw_action is None:
            # sim
            data = self.update_history(data)
            raw_obs = data["obs"]
            input = None
            goal = self.normalizer.scale_input(self.goal)
            returns = torch.ones_like(raw_obs[:, 0, :1])
        else:
            # train and test
            raw_obs = data["obs"]
            if self.inpaint:
                input_obs, input_act = raw_obs, raw_action
            else:
                input_obs = raw_obs[:, self.T_cond - 1 :]
                input_act = raw_action[:, self.T_cond - 1 :]
            input = torch.cat([input_act, input_obs], dim=-1)

            returns = self.calculate_return(input)

            # find last non-zero value
            mask = input[..., self.action_dim :].sum(dim=-1) != 0
            lengths = mask.sum(dim=-1)

            input = self.normalizer.scale_output(input)
            goal = input[range(input.shape[0]), lengths - 1, self.action_dim :]

        obs = self.normalizer.scale_input(raw_obs[:, : self.T_cond])
        return {"obs": obs, "input": input, "goal": goal, "returns": returns}

    def create_conditioning(self, data: dict) -> dict:
        if self.inpaint:
            return {0: data["obs"].squeeze(1), self.T - 1: data["goal"]}
        else:
            return {}

    def calculate_return(self, input):
        pos = input[:, :, 2:4]
        rewards = pos.norm(dim=-1)

        # TODO: get the true min and max from dataset
        returns = (rewards * self.gammas).sum(dim=-1)
        returns = (returns - returns.min()) / (returns.max() - returns.min())

        return returns.unsqueeze(-1)

        # return torch.zeros_like(input[:, 0, 0:1])

    ###########
    # Helpers #
    ###########

    @torch.no_grad()
    def sample_training_density(self, size):
        """
        Generate a density function for training sigmas
        """
        loc = math.log(self.sigma_data)
        density = rand_log_logistic(
            (size,), loc, 0.5, self.sigma_min, self.sigma_max, self.device
        )
        return density

    def update_history(self, x):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = x["obs"]
        x["obs"] = self.obs_hist.clone()
        return x

    def set_goal(self, goal):
        self.goal = torch.cat([goal, torch.zeros_like(goal)], dim=-1)

    def dict_to_device(self, data):
        return {k: v.to(self.device) for k, v in data.items()}

    def get_params(self):
        return self.model.get_params()
