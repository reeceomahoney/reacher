import math
import re
from datetime import datetime
from functools import wraps
from pathlib import Path

import gymnasium as gym
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from isaaclab.utils.math import matrix_from_euler


def dynamic_hydra_main(task_name: str):
    """
    Custom decorator to dynamically set Hydra's config_path based on the task name
    """
    # this is here to stop isaac errors
    from isaaclab_tasks.utils import parse_env_cfg

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # get agent config
            agent_cfg_entry_point = gym.spec(task_name).kwargs.get(
                "agent_cfg_entry_point"
            )
            config_path, config_name = agent_cfg_entry_point.rsplit("/", 1)
            print(f"[INFO]: Parsing configuration from: {config_path}")

            @hydra.main(
                config_path=config_path, config_name=config_name, version_base=None
            )
            def hydra_wrapper(agent_cfg: DictConfig):
                env_cfg = parse_env_cfg(
                    task_name,
                    device=agent_cfg.device,
                    num_envs=agent_cfg.num_envs,
                )

                return func(agent_cfg, env_cfg, *args, **kwargs)

            return hydra_wrapper()

        return wrapper

    return decorator


def get_open_maze_squares(maze):
    coordinates = []

    for y in range(8):
        for x in range(8):
            if maze[y][x] == 1:
                coord_x = x - 4
                coord_y = (7 - y) - 4
                coordinates.append((coord_x, coord_y))

    return torch.tensor(coordinates, dtype=torch.float)


def get_latest_run(base_path, resume=False):
    """
    Find the most recent directory in a nested structure like Oct-29/13-01-34/
    Returns the full path to the most recent time directory
    """

    def extract_model_number(file_path):
        match = re.search(r"model_(\d+)\.pt", file_path.name)
        return int(match.group(1)) if match else -1

    all_dirs = []
    base_path = Path(base_path)

    # find all dates
    for date_dir in base_path.iterdir():
        if not date_dir.is_dir():
            continue
        # find all times
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            try:
                dir_datetime = datetime.strptime(
                    f"{date_dir.name}/{time_dir.name}", "%b-%d/%H-%M-%S"
                )
                all_dirs.append((time_dir, dir_datetime))
            except ValueError:
                continue

    # sort
    sorted_directories = sorted(all_dirs, key=lambda x: x[1], reverse=True)
    target_dir = sorted_directories[1][0] if resume else sorted_directories[0][0]

    # get latest model
    model_files = list(target_dir.glob("model_*.pt"))
    if model_files:
        latest_model_file = max(model_files, key=extract_model_number)
        return latest_model_file
    else:
        return target_dir


def sample_goal_poses(bsz, device):
    goal_ranges = [
        (0.35, 1),
        (-0.5, 0.5),
        (0.15, 0.9),
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
    ]

    # goal position
    goal_pos = torch.zeros((bsz, 3), device=device)
    goal_pos[:, 0].uniform_(*goal_ranges[0])
    goal_pos[:, 1].uniform_(*goal_ranges[1])
    goal_pos[:, 2].uniform_(*goal_ranges[2])
    # goal orientation
    goal_euler = torch.zeros((bsz, 3), device=device)
    goal_euler[:, 0].uniform_(*goal_ranges[3])
    goal_euler[:, 1].uniform_(*goal_ranges[4])
    goal_euler[:, 2].uniform_(*goal_ranges[5])

    # convert to ortho6d and concatenate
    rot_mat = matrix_from_euler(goal_euler, "XYZ")
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([goal_pos, ortho6d], dim=-1)


def sample_goal_poses_from_list(bsz, device):
    goal_list = [
        [0.8, 0, 0.6, 0, 1, 0, 0],
        [0.8, 0, 0.2, 0, 1, 0, 0],
    ]
    goal_tensor = torch.tensor(goal_list, device=device, dtype=torch.float)
    random_indices = torch.randint(0, len(goal_list), (bsz,), device=device)
    return goal_tensor[random_indices]


def check_collisions(traj: Tensor) -> Tensor:
    """0 if in collision, 1 otherwise"""
    x_mask = (traj[..., 0] >= 0.55) & (traj[..., 0] <= 0.65)
    y_mask = (traj[..., 1] >= -0.8) & (traj[..., 1] <= 0.8)
    z_mask = (traj[..., 2] >= 0.0) & (traj[..., 2] <= 0.6)
    return ~(x_mask & y_mask & z_mask)


def calculate_return(
    traj: Tensor, goal: Tensor, mask: Tensor, gammas: Tensor
) -> Tensor:
    # reward = check_collisions(traj).float()
    # goal_reward = 1 - torch.exp(-torch.norm(traj - goal[:, :3].unsqueeze(1), dim=-1))
    goal_reward = 0
    # reward[:, -1] += goal_reward
    return ((goal_reward * mask) * gammas).sum(dim=-1, keepdim=True)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# copied from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, device: str = "cuda", use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self._device = device
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

        self.steps = 0

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters, strict=False):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters, strict=False):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_shadow_params(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters, strict=False):
            if param.requires_grad:
                s_param.data.copy_(param.data)

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]


class Normalizer(nn.Module):
    def __init__(self, data_loader, scaling: str, device: str):
        super().__init__()
        dl = data_loader.dataset.dataset.dataset

        # linear scaling
        self.register_buffer("x_max", dl.x_max)
        self.register_buffer("x_min", dl.x_min)
        self.register_buffer("y_max", dl.y_max)
        self.register_buffer("y_min", dl.y_min)
        self.register_buffer("r_max", dl.r_max)
        self.register_buffer("r_min", dl.r_min)

        # gaussian scaling
        self.register_buffer("x_mean", dl.x_mean)
        self.register_buffer("x_std", dl.x_std)
        self.register_buffer("y_mean", dl.y_mean)
        self.register_buffer("y_std", dl.y_std)

        # bounds
        y_bounds = torch.zeros((2, self.y_mean.shape[-1]))
        self.register_buffer("y_bounds", y_bounds)
        if scaling == "linear":
            self.y_bounds[0, :] = -1 - 1e-4
            self.y_bounds[1, :] = 1 + 1e-4
        elif scaling == "gaussian":
            self.y_bounds[0, :] = -5
            self.y_bounds[1, :] = 5

        self.scaling = scaling
        self.to(device)

    def scale_input(self, x) -> torch.Tensor:
        if self.scaling == "linear":
            return (x - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        elif self.scaling == "gaussian":
            return (x - self.x_mean) / self.x_std
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_3d_pos(self, pos) -> torch.Tensor:
        if self.scaling == "linear":
            return (pos - self.x_min[18:21]) / (
                self.x_max[18:21] - self.x_min[18:21]
            ) * 2 - 1
        elif self.scaling == "gaussian":
            return (pos - self.x_mean[18:21]) / self.x_std[18:21]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_9d_pos(self, pos) -> torch.Tensor:
        if self.scaling == "linear":
            return (pos - self.x_min[18:27]) / (
                self.x_max[18:27] - self.x_min[18:27]
            ) * 2 - 1
        elif self.scaling == "gaussian":
            return (pos - self.x_mean[18:27]) / self.x_std[18:27]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_output(self, y) -> torch.Tensor:
        if self.scaling == "linear":
            return (y - self.y_min) / (self.y_max - self.y_min) * 2 - 1
        elif self.scaling == "gaussian":
            return (y - self.y_mean) / self.y_std
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def inverse_scale_output(self, y) -> torch.Tensor:
        if self.scaling == "linear":
            return (y + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        elif self.scaling == "gaussian":
            return y * self.y_std + self.y_mean
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

    def scale_return(self, r) -> torch.Tensor:
        return (r - self.r_min) / (self.r_max - self.r_min)

    def clip(self, y):
        return torch.clamp(y, self.y_bounds[0, :], self.y_bounds[1, :])


class InferenceContext:
    """
    Context manager for inference mode
    """

    def __init__(self, runner):
        self.runner = runner
        self.policy = runner.policy
        self.ema_helper = runner.ema_helper
        self.use_ema = runner.use_ema

    def __enter__(self):
        self.inference_mode_context = torch.inference_mode()
        self.inference_mode_context.__enter__()
        self.runner.eval_mode()
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.runner.train_mode()
        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())
        self.inference_mode_context.__exit__(exc_type, exc_value, traceback)
