import math
import numpy as np
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from locodiff.helpers import Losses, apply_conditioning, cosine_beta_schedule, extract
from locodiff.samplers import (
    get_sampler,
    get_sigmas_exponential,
    get_sigmas_linear,
    rand_log_logistic,
)
from locodiff.wrappers import CFGWrapper


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
        sampler_type: str,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        cond_lambda: int,
        cond_mask_prob: float,
        lr: float,
        betas: tuple,
        num_iters: int,
        inpaint_obs: bool,
        inpaint_final_obs: bool,
        device: str,
        resampling_steps: int,
        jump_length: int,
    ):
        super().__init__()
        # model
        if cond_mask_prob > 0:
            model = CFGWrapper(model, cond_lambda, cond_mask_prob)
        self.model = model
        self.sampler = get_sampler(sampler_type)
        self.sampler_type = sampler_type
        self.sampling_steps = sampling_steps
        if sampler_type == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=sampling_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                variance_type="fixed_small",
                clip_sample=True,
                prediction_type="epsilon",
            )

        self.normalizer = normalizer
        self.obs_hist = torch.zeros((num_envs, T_cond, obs_dim), device=device)
        self.device = device
        self.env = env

        # dims
        self.obs_dim = obs_dim
        self.input_dim = obs_dim + act_dim
        self.action_dim = act_dim
        self.input_len = T + T_cond - 1 if inpaint_obs else T
        self.T = T
        self.T_cond = T_cond
        self.T_action = T_action
        self.num_envs = num_envs
        self.goal_dim = 4

        # diffusion
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_mask_prob = cond_mask_prob
        self.inference_sigmas = get_sigmas_exponential(
            sampling_steps, sigma_min, sigma_max, device
        )
        self.inpaint_obs = inpaint_obs
        self.inpaint_final_obs = inpaint_final_obs
        self.resampling_steps = resampling_steps
        self.jump_length = jump_length

        # optimizer and lr scheduler
        optim_groups = self.model.get_optim_groups()
        self.optimizer = Adam(optim_groups, lr=lr)
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)
        loss_weights = self.get_loss_weights(1.0, 1.0, None)
        self.loss_fn = Losses["l2"](loss_weights, self.action_dim)

        diff_betas = cosine_beta_schedule(sampling_steps)
        alphas = 1.0 - diff_betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.to(device)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.input_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.T, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, : self.action_dim] = action_weight
        return loss_weights

    def forward(self, data: dict) -> torch.Tensor:
        B = data["obs"].shape[0]
        # sample noise
        noise = torch.randn((B, self.input_len, self.input_dim))
        noise = noise.to(self.device)

        # create inpainting mask and target
        tgt, mask = self.create_inpainting_data(noise, data)
        kwargs = {"tgt": tgt, "mask": mask}

        # create noise
        if self.sampler_type == "ddpm":
            self.noise_scheduler.set_timesteps(self.sampling_steps)
            kwargs["noise_scheduler"] = self.noise_scheduler
        else:
            noise = noise * self.sigma_max
            inference_sigmas = get_sigmas_exponential(
                self.sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
            kwargs["sigmas"] = inference_sigmas
            kwargs["resampling_steps"] = self.resampling_steps
            kwargs["jump_length"] = self.jump_length

        # inference
        x = self.sampler(self.model, noise, data, **kwargs)
        x = self.normalizer.clip(x)
        x = self.normalizer.inverse_scale_output(x)
        return x

    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, : self.obs_dim]

        # extract action
        if self.inpaint_obs:
            action = x[
                :, self.T_cond - 1 : self.T_cond + self.T_action - 1, self.obs_dim :
            ]
        else:
            action = x[:, : self.T_action, self.obs_dim :]
        return {"action": action, "obs_traj": obs}

    def update(self, data):
        data = self.process(data)
        noise = torch.randn_like(data["input"])
        # create inpainting mask and target
        # tgt, mask = self.create_inpainting_data(noise, data)
        # kwargs = {"tgt": tgt, "mask": mask}

        #     # calculate loss
        #     if self.sampler_type == "ddpm":
        #         timesteps = torch.randint(0, self.sampling_steps, (noise.shape[0],))
        #         noise_trajectory = self.noise_scheduler.add_noise(
        #             data["input"], noise, timesteps
        #         )
        #         timesteps = timesteps.float().to(self.device)
        #         noise = tgt * mask + noise * (1 - mask)
        #         pred = self.model(noise_trajectory, timesteps, data)
        #         pred = tgt * mask + pred * (1 - mask)
        #         loss = torch.nn.functional.mse_loss(pred, noise)
        #     else:
        #         sigma = self.make_sample_density(len(noise))
        #         loss = self.model.loss(noise, sigma, data, **kwargs)

        x_start = data["input"]
        t = torch.randint(
            0, self.sampling_steps, (x_start.shape[0],), device=x_start.device
        ).long()
        cond = {
            0: x_start[:, 0, self.action_dim :],
            self.T - 1: x_start[:, -1, self.action_dim :],
        }

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        loss, info = self.loss_fn(x_recon, x_start)

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.lr_scheduler.step()

        return loss.item()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def test(self, data: dict, plot) -> tuple[float, float, float]:
        data = self.process(data)
        x = self.forward(data)
        # calculate loss
        input = self.normalizer.inverse_scale_output(data["input"])
        loss = nn.functional.mse_loss(x, input, reduction="none")
        obs_loss = loss[:, :, self.action_dim:].mean()
        action_loss = loss[:, :, :self.action_dim].mean()

        if plot:
            obs_traj = x[0, :, self.action_dim:]
            fig = plt.figure()
            maze = self.env.env.unwrapped.maze_array
            maze -= 10
            maze[np.where(maze == 2)] = 0

            plt.imshow(maze, cmap="gray", extent=(-4, 4, -4, 4))
            plt.scatter(obs_traj[:, 0], obs_traj[:, 1])
            wandb.Image(fig)
        
        return loss.mean().item(), obs_loss.item(), action_loss.item()

    def reset(self, dones=None):
        if dones is not None:
            self.obs_hist[dones.bool()] = 0
        else:
            self.obs_hist.zero_()

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)
        raw_action = data.get("action", None)

        if raw_action is None:
            # inference
            data = self.update_history(data)
            raw_obs = data["obs"]
            input = None
            goal = self.normalizer.scale_pos(self.goal)
        else:
            # training
            raw_obs = data["obs"]
            if self.inpaint_obs:
                input_obs, input_act = raw_obs, raw_action
            else:
                input_obs = raw_obs[:, self.T_cond - 1 : self.T_cond + self.T - 1]
                input_act = raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1]
            input = torch.cat([input_act, input_obs], dim=-1)
            input = self.normalizer.scale_output(input)
            goal = input[:, -1, -self.goal_dim :]

        obs = self.normalizer.scale_input(raw_obs[:, : self.T_cond])
        return {"obs": obs, "input": input, "goal": goal}

    def update_history(self, x):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = x["obs"]
        x["obs"] = self.obs_hist.clone()
        return x

    def dict_to_device(self, data):
        return {k: v.to(self.device) for k, v in data.items()}

    def create_inpainting_data(self, noise: torch.Tensor, data: dict):
        tgt = torch.zeros_like(noise)
        mask = torch.zeros_like(noise)
        if self.inpaint_obs:
            tgt[:, : self.T_cond, : self.obs_dim] = data["obs"]
            mask[:, : self.T_cond, : self.obs_dim] = 1.0
        if self.inpaint_final_obs:
            tgt[:, -1, : self.goal_dim] = data["goal"]
            mask[:, -1, : self.goal_dim] = 1.0

        return tgt, mask

    def set_goal(self, goal):
        self.goal = goal.unsqueeze(0)
        self.goal = torch.cat([self.goal, torch.zeros_like(self.goal)], dim=-1)

    @torch.no_grad()
    def make_sample_density(self, size):
        """
        Generate a density function for training sigmas
        """
        loc = math.log(self.sigma_data)
        density = rand_log_logistic(
            (size,), loc, 0.5, self.sigma_min, self.sigma_max, self.device
        )
        return density

    def get_params(self):
        return self.model.get_params()
