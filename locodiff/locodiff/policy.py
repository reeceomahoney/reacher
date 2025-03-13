import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from isaaclab.utils.math import matrix_from_quat
from locodiff.models.transformer import DiffusionTransformer
from locodiff.plotting import plot_3d_guided_trajectory
from locodiff.utils import (
    Normalizer,
)


def expand_t(tensor: Tensor, bsz: int) -> Tensor:
    return tensor.view(1, -1, 1).expand(bsz, -1, 1)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model: DiffusionTransformer,
        classifier: DiffusionTransformer | None,
        normalizer: Normalizer,
        env,
        obs_dim: int,
        act_dim: int,
        T: int,
        T_action: int,
        sampling_steps: int,
        cond_lambda: int,
        cond_mask_prob: float,
        lr: float,
        betas: tuple,
        num_iters: int,
        device: str,
        algo: str,
    ):
        super().__init__()
        # model
        if classifier is not None:
            self.classifier = classifier
            self.alpha = 0.0
        self.model = model
        # other classes
        self.env = env
        self.normalizer = normalizer
        self.device = device

        # dims
        self.input_dim = act_dim + obs_dim
        self.action_dim = act_dim
        self.T = T
        self.T_action = T_action

        # diffusion / flow matching
        self.sampling_steps = sampling_steps
        self.beta_dist = torch.distributions.beta.Beta(1.5, 1.0)
        self.scheduler = DDPMScheduler(self.sampling_steps)
        self.algo = algo  # ddpm or flow
        self.scheduling_matrix = self._generate_scheduling_matrix("autoregressive")

        # optimizer and lr scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=2e-4)
        self.classifier_optimizer = AdamW(
            self.classifier.get_optim_groups(), lr=lr, betas=betas
        )
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)
        self.classifier_lr_scheduler = CosineAnnealingLR(
            self.classifier_optimizer, T_max=num_iters
        )

        # guidance
        self.gammas = torch.tensor([0.99**i for i in range(self.T)]).to(device)
        self.cond_mask_prob = cond_mask_prob
        self.cond_lambda = cond_lambda

        self.to(device)

    ############
    # Main API #
    ############

    @torch.no_grad()
    def act(self, data: dict) -> dict[str, torch.Tensor]:
        data = self.process(data)
        x = self.forward(data)
        obs = x[:, :, self.action_dim :]
        action = x[:, : self.T_action, : self.action_dim]
        return {"action": action, "obs_traj": obs}

    def update(self, data):
        data = self.process(data)

        # sample noise and timestep
        x_1 = data["input"]
        x_0 = torch.randn_like(x_1)

        if self.algo == "flow":
            samples = self.beta_dist.sample((len(x_1), 1, 1)).to(self.device)
            t = 0.999 * (1 - samples)
            # compute target
            x_t = (1 - t) * x_0 + t * x_1
            target = x_1 - x_0

        elif self.algo == "ddpm":
            t = torch.randint(0, self.sampling_steps, (x_1.shape[0], 1)).to(self.device)
            x_t = self.scheduler.add_noise(x_1, x_0, t)  # type: ignore
            target = x_0

        # inpaint
        x_t[:, 0, self.action_dim :] = data["obs"][:, 0]
        x_t[:, -1, self.action_dim :] = data["goal"]

        # cfg masking
        if self.cond_mask_prob > 0:
            cond_mask = torch.rand(x_1.shape[0], 1) < self.cond_mask_prob
            data["returns"][cond_mask] = 0

        # compute model output
        out = self.model(x_t, t.float(), data)
        out[:, 0, self.action_dim :] = data["obs"][:, 0]
        out[:, -1, self.action_dim :] = data["goal"]

        loss = F.mse_loss(out, target)
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    @torch.no_grad()
    def test(self, data: dict) -> tuple[float, float, float]:
        data = self.process(data)
        x = self.forward(data)

        # calculate losses
        input = self.normalizer.inverse_scale_output(data["input"])
        loss = F.mse_loss(x, input, reduction="none")
        obs_loss = loss[:, :, self.action_dim :].mean().item()
        action_loss = loss[:, :, : self.action_dim].mean().item()

        return loss.mean().item(), obs_loss, action_loss

    ##################
    # Classifier API #
    ##################

    def update_classifier(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised sample
        n = random.randint(0, self.sampling_steps)
        x, t = self.truncated_forward(data, n)

        # compute model output
        pred_value = self.classifier(x, t, data)
        loss = F.mse_loss(pred_value, data["returns"])
        # update model
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        self.classifier_lr_scheduler.step()

        return loss.item()

    def test_classifier(self, data: dict) -> float:
        data = self.process(data)

        # compute partially denoised sample
        n = random.randint(0, self.sampling_steps - 1)
        x, t = self.truncated_forward(data, n)
        pred_value = self.classifier(x, t, data)

        return F.mse_loss(pred_value, data["returns"]).item()

    #####################
    # Inference backend #
    #####################

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, data: dict) -> Tensor:
        t_start = expand_t(t_start, x_t.shape[0])
        t_end = expand_t(t_end, x_t.shape[0])
        return x_t + (t_end - t_start) * self.model(x_t, t_start, data)

        # return x_t + (t_end - t_start) * self.model(
        #     x_t + self.model(x_t, t_start, data) * (t_end - t_start) / 2,
        #     t_start + (t_end - t_start) / 2,
        #     data,
        # )

    @torch.no_grad()
    def forward(self, data: dict) -> torch.Tensor:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)

        if self.algo == "flow":
            timesteps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)
        elif self.algo == "ddpm":
            self.scheduler.set_timesteps(self.sampling_steps)
            timesteps = self.scheduler.timesteps

        if self.cond_lambda > 0:
            data = {
                k: torch.cat([v] * 2) if v is not None else None
                for k, v in data.items()
            }
            data["returns"][bsz:] = 0

        # inpaint
        x[:, 0, self.action_dim :] = data["obs"][:, 0]
        x[:, -1, self.action_dim :] = data["goal"]

        # inference
        for i in range(self.sampling_steps):
            x = torch.cat([x] * 2) if self.cond_lambda > 0 else x
            if self.algo == "flow":
                x = self.step(x, timesteps[i], timesteps[i + 1], data)
            elif self.algo == "ddpm":
                t = timesteps[i].view(-1, 1).expand(bsz, 1).float()
                out = self.model(x, t, data)
                x = self.scheduler.step(out, timesteps[i], x).prev_sample  # type: ignore

            # guidance
            if self.alpha > 0:
                with torch.enable_grad():
                    x_grad = x.detach().clone().requires_grad_(True)
                    y = self.classifier(x_grad, expand_t(timesteps[i + 1], bsz), data)
                    grad = torch.autograd.grad(y.sum(), x_grad, create_graph=True)[0]
                    x = x_grad + self.alpha * (1 - timesteps[i + 1]) * grad.detach()
            elif self.cond_lambda > 0:
                x_cond, x_uncond = x.chunk(2)
                x = x_uncond + self.cond_lambda * (x_cond - x_uncond)

            # inpaint
            x[:, 0, self.action_dim :] = data["obs"][:, 0]
            x[:, -1, self.action_dim :] = data["goal"]

        # denormalize
        x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_output(x)

    @torch.no_grad()
    def truncated_forward(
        self, data: dict, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sample noise
        bsz = data["obs"].shape[0]
        x = torch.randn((bsz, self.T, self.input_dim)).to(self.device)
        time_steps = torch.linspace(0, 1.0, self.sampling_steps + 1).to(self.device)
        time_steps = time_steps[: n + 1]

        # inference
        # TODO: change this to batch samples from every step
        for i in range(n):
            x = self.step(x, time_steps[i], time_steps[i + 1], data)

        return x, expand_t(time_steps[-1], bsz)

    ###################
    # Data processing #
    ###################

    @torch.no_grad()
    def process(self, data: dict) -> dict:
        data = self.dict_to_device(data)
        raw_action = data.get("action", None)

        if raw_action is None:
            # sim
            input = None
            returns = torch.ones((data["obs"].shape[0], 1)).to(self.device)
            raw_obs = data["obs"].unsqueeze(1)
            obstacle = torch.zeros((data["obs"].shape[0], 3)).to(self.device)
            # obstacle = self.normalizer.scale_3d_pos(data["obstacle"])
            goal = self.normalizer.scale_input(data["goal"])
        else:
            # train and test
            raw_obs = data["obs"]
            input = torch.cat([raw_action, raw_obs], dim=-1)
            # input = raw_action
            # goal = sample_goal_poses_from_list(bsz, self.device)
            goal = data["goal"]

            obstacle = torch.zeros((input.shape[0], 3)).to(self.device)
            # returns = calculate_return(
            #     input[..., 25:28], raw_obs, goal, data["mask"], self.gammas
            # )
            returns = torch.ones((data["obs"].shape[0], 1)).to(self.device)
            # returns = self.normalizer.scale_return(returns)

            # obstacle = self.normalizer.scale_3d_pos(obstacle)
            input = self.normalizer.scale_output(input)
            goal = self.normalizer.scale_input(goal)

        obs = self.normalizer.scale_input(raw_obs[:, :1])
        return {
            "obs": obs,
            "input": input,
            "goal": goal,
            "returns": returns,
            "obstacle": obstacle,
        }

    def calculate_obstacles(self, size: int) -> torch.Tensor:
        # Sample random coordinates within the maze (bottom left corner)
        idx = torch.randint(0, len(self.open_squares), (size,))
        samples = self.open_squares[idx].to(self.device)
        return samples.to(self.device)

    ###########
    # Helpers #
    ###########

    def _generate_scheduling_matrix(self, schedule_type: str):
        match schedule_type:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(self.T, 1.0)
            case "autoregressive":
                return self._generate_autoregressive_scheduling_matrix(self.T)
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(self.T, 1.0)
            case "full":
                return self._generate_full_sequence_scheduling_matrix(self.T)
        raise ValueError(f"Invalid scheduling matrix type: {schedule_type}")

    def _generate_pyramid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float
    ):
        height = self.sampling_steps + int((horizon - 1) * uncertainty_scale) + 1
        self.global_timesteps = height - 1
        row_indices = torch.arange(height, dtype=torch.float32).view(-1, 1)
        col_indices = torch.arange(horizon, dtype=torch.float32)
        t_scaled = col_indices * uncertainty_scale
        normalized_matrix = (row_indices - t_scaled) / self.sampling_steps
        return torch.clamp(normalized_matrix, min=0.0, max=1.0).to(self.device)

    def _generate_autoregressive_scheduling_matrix(self, horizon: int):
        matrix = self._generate_pyramid_scheduling_matrix(horizon, self.sampling_steps)
        for i in range(matrix.shape[1]):
            matrix[20 + 10 * i :, i] = 0
        return matrix

    def _generate_trapezoid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float
    ):
        height = self.sampling_steps + int((horizon + 1) // 2 * uncertainty_scale)
        self.global_timesteps = height - 1
        scheduling_matrix = torch.zeros((height, horizon))
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = m - int(t * uncertainty_scale)
                scheduling_matrix[m, -(t + 1)] = m - int(t * uncertainty_scale)
        scheduling_matrix /= self.sampling_steps
        return torch.clamp(scheduling_matrix, min=0, max=1).to(self.device)

    def _generate_full_sequence_scheduling_matrix(self, horizon: int):
        self.global_timesteps = self.sampling_steps
        return (
            torch.linspace(0, 1, self.sampling_steps + 1)
            .repeat(horizon, 1)
            .to(self.device)
            .T
        )

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
            self.cond_lambda = lam
            obs_traj = self.act({"obs": obs, "obstacle": obstacle, "goal": goal})
            collisions = self.check_collisions(obs_traj["obs_traj"][..., :2], obstacle)
            total_collisions.append(collisions.sum().item())

        plt.plot(cond_lambda, total_collisions)
        wandb.log({"Collision Rate": wandb.Image(plt)})
        plt.close()

    def plot_guided_trajectory(self, it: int, scales: list, alphas_or_lambdas: str):
        # get obs
        obs, _ = self.env.get_observations()
        obs = obs[0].unsqueeze(0)
        # get goal
        goal = self.env.unwrapped.command_manager.get_command("ee_pose")  # type: ignore
        rot_mat = matrix_from_quat(goal[:, 3:])
        ortho6d = rot_mat[..., :2].reshape(-1, 6)
        goal = torch.cat([goal[:, :3], ortho6d], dim=-1)[0].unsqueeze(0)

        # plot trajectory
        obstacle = torch.zeros_like(goal)
        fig = plot_3d_guided_trajectory(
            self, obs, goal, obstacle[:, :3], scales, alphas_or_lambdas
        )

        # log
        wandb.log({"Guided Trajectory": wandb.Image(fig)}, step=it)
        plt.close(fig)
