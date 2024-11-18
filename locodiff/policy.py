import math
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from locodiff.samplers import get_sampler, get_sigmas_exponential, rand_log_logistic
from locodiff.wrappers import CFGWrapper


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model,
        normalizer,
        obs_dim: int,
        act_dim: int,
        T: int,
        T_cond: int,
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
        device,
    ):
        super().__init__()

        # model
        if cond_mask_prob > 0:
            model = CFGWrapper(model, cond_lambda, cond_mask_prob)
        self.model = model
        self.sampler = get_sampler(sampler_type)
        self.normalizer = normalizer
        self.device = device
        self.obs_hist = torch.zeros((num_envs, T_cond, obs_dim), device=device)

        # dims
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.T = T
        self.T_cond = T_cond
        self.num_envs = num_envs

        # diffusion
        self.sampling_steps = sampling_steps
        self.sampler_type = sampler_type
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_mask_prob = cond_mask_prob

        # optimizer and lr scheduler
        optim_groups = self.model.get_optim_groups()
        self.optimizer = AdamW(optim_groups, lr=lr, betas=betas)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

    def act(self, data: dict) -> tuple:
        data = self.process(data)

        if data["action"] is None:
            batch_size = self.num_envs
        else:
            batch_size = data["action"].shape[0]

        noise = torch.randn((batch_size, self.T, self.act_dim)).to(self.device)
        noise = noise * self.sigma_max
        sigmas = get_sigmas_exponential(
            self.sampling_steps, self.sigma_min, self.sigma_max, self.device
        )
        x_0 = self.sampler(self.model, noise, data, sigmas=sigmas)

        if self.cond_mask_prob > 0:
            data["return"] = torch.ones_like(data["return"])
            x_0_max_return = self.sampler(self.model, noise, data, sigmas=sigmas)
            return x_0, x_0_max_return
        else:
            return x_0

    def update(self, data_dict) -> torch.Tensor:
        self.train()

        action = data_dict["action"]
        noise = torch.randn_like(action)
        if self.sampler_type == "ddpm":
            timesteps = torch.randint(0, self.sampling_steps, (noise.shape[0],))
            noise_trajectory = self.noise_scheduler.add_noise(action, noise, timesteps)
            timesteps = timesteps.float().to(self.device)
            pred = self.model(noise_trajectory, timesteps, data_dict)
            loss = torch.nn.functional.mse_loss(pred, noise)
        else:
            sigma = self.make_sample_density(len(noise))
            loss = self.model.loss(noise, sigma, data_dict)

        return loss

    def reset(self, dones=None):
        if dones is not None:
            self.obs_hist[dones.bool()] = 0
        else:
            self.obs_hist.zero_()

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

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)
        raw_action = data.get("action", None)

        if raw_action is None:
            # inference
            data = self.update_history(data)
            raw_obs = data["obs"]
            action = None
        else:
            # training
            raw_obs = data["obs"]
            action = self.normalizer.scale_output(
                torch.cat(
                    [
                        raw_obs[:, self.T_cond - 1 : self.T_cond + self.T - 1],
                        raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1],
                    ],
                    dim=-1,
                )
            )
        obs = self.normalizer.scale_input(raw_obs[:, : self.T_cond])

        return {"obs": obs, "action": action}

    def update_history(self, x):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = x["obs"]
        x["obs"] = self.obs_hist.clone()
        return x

    def dict_to_device(self, data):
        return {k: v.to(self.device) for k, v in data.items()}
