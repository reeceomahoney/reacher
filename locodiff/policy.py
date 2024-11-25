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
        inpaint_obs: bool,
        inpaint_final_obs: bool,
        device,
    ):
        super().__init__()
        # model
        if cond_mask_prob > 0:
            model = CFGWrapper(model, cond_lambda, cond_mask_prob)
        self.model = model
        self.sampler = get_sampler(sampler_type)
        self.normalizer = normalizer
        self.obs_hist = torch.zeros((num_envs, T_cond, obs_dim), device=device)
        self.device = device

        # dims
        self.obs_dim = obs_dim
        self.input_dim = obs_dim + act_dim
        self.input_len = T + T_cond - 1 if inpaint_obs else T
        self.T = T
        self.T_cond = T_cond
        self.num_envs = num_envs

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

        # optimizer and lr scheduler
        optim_groups = self.model.get_optim_groups()
        self.optimizer = AdamW(optim_groups, lr=lr, betas=betas)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

    def forward(self, data: dict) -> torch.Tensor:
        B = data["obs"].shape[0]
        # sample noise
        noise = torch.randn((B, self.input_len, self.input_dim))
        noise = noise.to(self.device) * self.sigma_max

        # create inpainting mask and target
        tgt, mask = self.create_inpainting_data(noise, data)

        # inference
        kwargs = {"sigmas": self.inference_sigmas, "tgt": tgt, "mask": mask}
        x = self.sampler(self.model, noise, data, **kwargs)
        x = self.normalizer.clip(x)
        x = self.normalizer.inverse_scale_output(x)
        return x

    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, :self.obs_dim]

        # extract action
        if self.inpaint_obs:
            action = x[:, self.T_cond - 1, self.obs_dim :]
        else:
            action = x[:, 0, self.obs_dim :]
        return {"action": action, "obs_traj": obs}

    def update(self, data: dict) -> torch.Tensor:
        data = self.process(data)
        noise = torch.randn_like(data["input"])
        # create inpainting mask and target
        tgt, mask = self.create_inpainting_data(noise, data)
        kwargs = {"tgt": tgt, "mask": mask}
        # calculate loss
        sigma = self.make_sample_density(len(noise))
        loss = self.model.loss(noise, sigma, data, **kwargs)

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def test(self, data: dict) -> float:
        data = self.process(data)
        x = self.forward(data)
        # calculate loss
        input = self.normalizer.inverse_scale_output(data["input"])
        loss = nn.functional.mse_loss(x, input)
        return loss.item()

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
        else:
            # training
            raw_obs = data["obs"]
            if self.inpaint_obs:
                input_obs, input_act = raw_obs, raw_action
            else:
                input_obs = raw_obs[:, self.T_cond - 1 : self.T_cond + self.T - 1]
                input_act = raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1]
            input = torch.cat([input_obs, input_act], dim=-1)
            input = self.normalizer.scale_output(input)

        obs = self.normalizer.scale_input(raw_obs[:, : self.T_cond])
        return {"obs": obs, "input": input}

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
            tgt_pos = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]).to(self.device)
            tgt_pos = self.normalizer.scale_pos(tgt_pos)
            tgt[:, -1, :5] = tgt_pos
            mask[:, -1, :5] = 1.0

        return tgt, mask

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
