import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import (
    EDMDPMSolverMultistepScheduler,
)
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from locodiff.models.unet import ValueUnet1D
from locodiff.plotting import plot_cfg_analysis
from locodiff.utils import (
    CFGWrapper,
    Normalizer,
    apply_conditioning,
    get_open_maze_squares,
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
            self.alpha = 1e-3
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
        self.classifier_optimizer = AdamW(
            self.classifier.parameters(), lr=lr, betas=betas
        )
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_iters)

        # reward guidance
        self.gammas = torch.tensor([0.99**i for i in range(self.T)]).to(device)
        self.open_squares = get_open_maze_squares(self.env.get_maze())

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
            goal = torch.tensor([[2.5, 2.5]]).to(self.device)
            obstacle = torch.tensor([[-1, 0]]).to(self.device)
            cond_lambda = [0, 1, 2, 3, 5, 10]
            # Generate plots
            fig = plot_cfg_analysis(self, self.env, obs, goal, obstacle, cond_lambda)
            # log
            wandb.log({"CFG Trajectory": wandb.Image(fig)})
            plt.close(fig)

            self.plot_collsion_rate(100)

        return loss.mean().item(), obs_loss.item(), action_loss.item()

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
        self.lr_scheduler.step()

        return loss.item()

    def test_classifier(self, data):
        # preprocess data
        data = self.process(data)

        # compute partially denoised sample
        timesteps = random.randint(0, self.sampling_steps - 1)
        x, t = self.truncated_forward(data, timesteps)
        pred_value = self.classifier(x, t, data)

        # calculate loss
        loss = torch.nn.functional.mse_loss(pred_value, data["returns"])

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

            # if hasattr(self, "classifier"):
            #     x_grad = x.detach().clone().requires_grad_(True)
            #     y = self.classifier(x_grad, t, data)
            #     grad = torch.autograd.grad(y, x_grad, create_graph=True)[0]
            #     x = x_grad + self.alpha * grad.detach()

        # final conditioning
        x = apply_conditioning(x, cond, 2)
        # denormalize
        x = self.normalizer.clip(x)
        x = self.normalizer.inverse_scale_output(x)
        return x

    def truncated_forward(
        self, data: dict, timesteps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        for i, t in enumerate(self.noise_scheduler.timesteps):
            x_in = self.noise_scheduler.scale_model_input(x, t)
            x_in = apply_conditioning(x_in, cond, 2)
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
            input = None
            raw_obs = data["obs"].unsqueeze(1)
            obstacles = self.normalizer.scale_pos(data["obstacles"])
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

            obstacles = self.calculate_obstacles(input.shape[0])
            returns = self.calculate_return(input, data["mask"], obstacles)

            obstacles = self.normalizer.scale_pos(obstacles)
            input = self.normalizer.scale_output(input)

            lengths = data["mask"].sum(dim=-1).int()
            goal = input[range(input.shape[0]), lengths - 1, self.action_dim :]

        # returns = torch.cat([returns, obstacles], dim=-1)
        obs = self.normalizer.scale_input(raw_obs[:, : self.T_cond])
        return {
            "obs": obs,
            "input": input,
            "goal": goal,
            "returns": returns,
            "obstacles": obstacles,
        }

    def create_conditioning(self, data: dict) -> dict:
        if self.inpaint:
            return {0: data["obs"].squeeze(1), self.T - 1: data["goal"]}
        else:
            return {}

    def calculate_return(self, input, mask, obstacles):
        # collision reward
        reward = self.check_collisions(input[:, :, 2:4], obstacles)
        reward = ((~reward) * mask).float()

        # distance reward
        # reward = self.calculate_distances(input[:, :, 2:4], obstacles)
        # reward *= mask

        lengths = mask.sum(dim=-1)
        returns = reward.sum(dim=-1) / lengths
        returns = (returns - returns.min()) / (returns.max() - returns.min())
        return returns.unsqueeze(-1)

    def calculate_obstacles(self, size: int) -> torch.Tensor:
        # Sample random coordinates within the maze (bottom left corner)
        # random
        samples = self.open_squares[
            torch.randint(0, len(self.open_squares), (size,))
        ].to(self.device)

        # fixed
        # x_vals = -1 * torch.ones(size, dtype=torch.float32)
        # y_vals = torch.zeros(size, dtype=torch.float32)
        # samples = torch.stack((x_vals, y_vals), dim=1)

        # 2 random
        # points = torch.tensor([[0, -1], [-1, 0]])
        # indices = torch.randint(0, 2, (size,))
        # samples = points[indices]

        return samples.to(self.device)

    def check_collisions(
        self, trajectories: torch.Tensor, box_corners: torch.Tensor
    ) -> torch.Tensor:
        # Expand box corners to match trajectory timesteps
        box_corners_tr = (box_corners + 1.0).unsqueeze(1)
        box_corners_bl = box_corners.unsqueeze(1)
        # Check if any point in each trajectory is inside its box
        inside_box = (trajectories > box_corners_bl) & (trajectories < box_corners_tr)
        # Both x and y coordinates must be inside for a collision
        return inside_box.all(dim=2)

    def calculate_distances(
        self, trajectories: torch.Tensor, box_corners: torch.Tensor
    ) -> torch.Tensor:
        box_centers = (box_corners + 0.5).unsqueeze(1)
        distances = torch.norm(trajectories - box_centers, dim=2)
        return distances

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
        # obstacle = torch.zeros(batch_size, 2).to(self.device)
        # obstacle[:, 0] = -1

        self.set_goal(goal)

        total_collisions = []
        for lam in cond_lambda:
            self.model.cond_lambda = lam
            obs_traj = self.act({"obs": obs, "obstacles": obstacle})
            collisions = self.check_collisions(obs_traj["obs_traj"][..., :2], obstacle)
            total_collisions.append(collisions.sum().item())

        plt.plot(cond_lambda, total_collisions)
        wandb.log({"Collision Rate": wandb.Image(plt)})
        plt.close()
