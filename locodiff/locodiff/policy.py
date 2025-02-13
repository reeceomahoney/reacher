import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import (
    EDMDPMSolverMultistepScheduler,
)
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from locodiff.models.unet import ValueUnet1D
from locodiff.plotting import plot_guided_trajectory
from locodiff.utils import (
    CFGWrapper,
    Normalizer,
    apply_conditioning,
    rand_log_logistic,
)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model,
        normalizer: Normalizer,
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
        classifier: ValueUnet1D | None = None,
    ):
        super().__init__()
        # model
        if classifier is not None:
            self.classifier = classifier
            self.alpha = 0.0
        elif cond_mask_prob > 0:
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

        # diffusion
        self.sampling_steps = sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_mask_prob = cond_mask_prob
        self.inpaint = inpaint

        # optimizer and lr scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=lr, betas=betas)
        self.classifier_optimizer = AdamW(
            self.classifier.parameters(), lr=lr, betas=betas
        )
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)
        self.classifier_lr_scheduler = CosineAnnealingLR(
            self.classifier_optimizer, T_max=num_iters
        )

        # reward guidance
        self.gammas = torch.tensor([0.99**i for i in range(self.T)]).to(device)
        # self.open_squares = get_open_maze_squares(self.env.get_maze())

        self.device = device
        self.to(device)

    ############
    # Main API #
    ############

    @torch.no_grad()
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
            cond_mask = torch.rand(noise.shape[0], 1) < self.cond_mask_prob
            data["returns"][cond_mask.expand_as(data["returns"])] = 0

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

    @torch.no_grad()
    def test(self, data: dict, plot) -> tuple[float, float, float]:
        data = self.process(data)
        x = self.forward(data)
        # calculate losses
        input = self.normalizer.inverse_scale_output(data["input"])
        loss = nn.functional.mse_loss(x, input, reduction="none")
        obs_loss = loss[:, :, self.action_dim :].mean()
        action_loss = loss[:, :, : self.action_dim].mean()

        if plot:
            obs = torch.tensor([[-2.5, -0.5, 0, 0]]).to(self.device)
            goal = torch.tensor([[2.5, 2.5, 0, 0]]).to(self.device)
            obstacle = torch.tensor([[-1, 0]]).to(self.device)
            alphas = [0, 200, 300, 500, 700, 1e3]
            # Generate plots
            fig = plot_guided_trajectory(self, self.env, obs, goal, obstacle, alphas)
            # log
            wandb.log({"CFG Trajectory": wandb.Image(fig)})
            plt.close(fig)

            self.plot_collsion_rate(100)

        return loss.mean().item(), obs_loss.item(), action_loss.item()

    ##################
    # Classifier API #
    ##################

    def update_classifier(self, data):
        # preprocess data
        data = self.process(data)

        # compute partially denoised sample
        timesteps = random.randint(0, self.sampling_steps - 1)
        x, t = self.truncated_forward(data, timesteps)
        pred_value = self.classifier(x, t, data)

        # calculate loss
        loss = torch.nn.functional.mse_loss(pred_value, data["returns"])

        # update model
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        self.classifier_lr_scheduler.step()

        return loss.item()

    def test_classifier(self, data, plot) -> float:
        # preprocess data
        data = self.process(data)

        # compute partially denoised sample
        timesteps = random.randint(0, self.sampling_steps - 1)
        x, t = self.truncated_forward(data, timesteps)
        pred_value = self.classifier(x, t, data)

        # calculate loss
        loss = torch.nn.functional.mse_loss(pred_value, data["returns"])

        if plot:
            obs = torch.tensor([[-2.5, -0.5, 0, 0]]).to(self.device)
            goal = torch.tensor([[2.5, 2.5, 0, 0]]).to(self.device)
            obstacle = torch.tensor([[-1, 0]]).to(self.device)
            alphas = [0, 200, 300, 500, 700, 1e3]
            # Generate plots
            fig = plot_guided_trajectory(self, self.env, obs, goal, obstacle, alphas)
            # log
            wandb.log({"Guided Trajectory": wandb.Image(fig)})
            plt.close(fig)

        return loss.item()

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
        x *= self.noise_scheduler.init_noise_sigma

        # create inpainting conditioning
        cond = self.create_conditioning(data)
        # this needs to called every time we do inference
        self.noise_scheduler.set_timesteps(self.sampling_steps)

        # inference loop
        for t in self.noise_scheduler.timesteps:
            x_in = self.noise_scheduler.scale_model_input(x, t)
            x_in = apply_conditioning(x_in, cond, self.action_dim)
            output = self.model(x_in, t.expand(B), data)
            x = self.noise_scheduler.step(output, t, x, return_dict=False)[0]

            # guidance
            if self.alpha > 0:
                with torch.enable_grad():
                    x_grad = x.detach().clone().requires_grad_(True)
                    y = self.classifier(x_grad, t, data)
                    grad = torch.autograd.grad(y, x_grad, create_graph=True)[0]
                    x = x_grad + self.alpha * torch.exp(4 * t) * grad.detach()

        # final conditioning
        x = apply_conditioning(x, cond, self.action_dim)
        # denormalize
        x = self.normalizer.clip(x)
        x = self.normalizer.inverse_scale_output(x)
        return x

    @torch.no_grad()
    def truncated_forward(
        self, data: dict, timesteps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sample noise
        B = data["obs"].shape[0]
        x = torch.randn((B, self.input_len, self.input_dim)).to(self.device)
        # we should need this but performance is better without it
        x *= self.noise_scheduler.init_noise_sigma

        # create inpainting conditioning
        cond = self.create_conditioning(data)
        # this needs to called every time we do inference
        self.noise_scheduler.set_timesteps(self.sampling_steps)

        # inference loop
        for i, t in enumerate(self.noise_scheduler.timesteps):
            x_in = self.noise_scheduler.scale_model_input(x, t)
            x_in = apply_conditioning(x_in, cond, self.action_dim)
            output = self.model(x_in, t.expand(B), data)
            x = self.noise_scheduler.step(output, t, x, return_dict=False)[0]

            if i >= timesteps:
                return x, t.expand(B)

        return x, t.expand(B)

    ###################
    # Data processing #
    ###################

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)
        raw_action = data.get("action", None)

        if raw_action is None:
            # sim
            # data = self.update_history(data)
            input, returns = None, None
            raw_obs = data["obs"].unsqueeze(1)
            obstacle = self.normalizer.scale_3d_pos(data["obstacle"])
            goal = self.normalizer.scale_9d_pos(data["goal"])
        else:
            # train and test
            raw_obs = data["obs"]
            if self.inpaint:
                input_obs, input_act = raw_obs, raw_action
            else:
                input_obs = raw_obs[:, self.T_cond - 1 :]
                input_act = raw_action[:, self.T_cond - 1 :]
            input = torch.cat([input_act, input_obs], dim=-1)

            obstacle = torch.zeros((input.shape[0], 3)).to(self.device)
            returns = self.calculate_return(input, data["mask"])

            obstacle = self.normalizer.scale_3d_pos(obstacle)
            input = self.normalizer.scale_output(input)

            lengths = data["mask"].sum(dim=-1).int()
            goal = input[
                range(input.shape[0]),
                lengths - 1,
                self.action_dim + 18 : self.action_dim + 27,
            ]

        obs = self.normalizer.scale_input(raw_obs[:, : self.T_cond])
        return {
            "obs": obs,
            "input": input,
            "goal": goal,
            "returns": returns,
            "obstacle": obstacle,
        }

    def create_conditioning(self, data: dict) -> dict:
        if self.inpaint:
            return {0: data["obs"].squeeze(1)}
        else:
            return {}

    def calculate_return(self, input, mask):
        reward = self.check_collisions(input[..., 25:28])
        reward = ((~reward) * mask).float()
        # average reward for valid timesteps
        returns = (reward * self.gammas).sum(dim=-1) / mask.sum(dim=-1)
        returns = (returns - returns.min()) / (returns.max() - returns.min())
        return returns.unsqueeze(-1)

    def calculate_obstacles(self, size: int) -> torch.Tensor:
        # Sample random coordinates within the maze (bottom left corner)
        idx = torch.randint(0, len(self.open_squares), (size,))
        samples = self.open_squares[idx].to(self.device)
        return samples.to(self.device)

    def check_collisions(self, traj: torch.Tensor) -> torch.Tensor:
        x_mask = (traj[..., 0] >= 0.45) & (traj[..., 0] <= 0.55)
        y_mask = (traj[..., 1] >= -0.8) & (traj[..., 1] <= 0.8)
        z_mask = (traj[..., 2] >= 0.0) & (traj[..., 2] <= 0.8)
        return x_mask & y_mask & z_mask

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

    def dict_to_device(self, data):
        return {k: v.to(self.device) for k, v in data.items()}

    def plot_collsion_rate(self, batch_size):
        cond_lambda = [0, 1, 2, 3, 5, 10]

        obs = self.open_squares[torch.randint(0, len(self.open_squares), (batch_size,))]
        obs = torch.cat([obs, torch.zeros(batch_size, 2)], dim=1).to(self.device)
        goal = self.open_squares[
            torch.randint(0, len(self.open_squares), (batch_size,))
        ].to(self.device)
        obstacle = self.open_squares[
            torch.randint(0, len(self.open_squares), (batch_size,))
        ].to(self.device)

        total_collisions = []
        for lam in cond_lambda:
            self.model.cond_lambda = lam
            obs_traj = self.act({"obs": obs, "obstacle": obstacle, "goal": goal})
            collisions = self.check_collisions(obs_traj["obs_traj"][..., :2], obstacle)
            total_collisions.append(collisions.sum().item())

        plt.plot(cond_lambda, total_collisions)
        wandb.log({"Collision Rate": wandb.Image(plt)})
        plt.close()
