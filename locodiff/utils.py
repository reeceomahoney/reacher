import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, device):
        x = x.to(device)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.unsqueeze(0)


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
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
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
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_shadow_params(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.data.copy_(param.data)

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]


class Normalizer(nn.Module):
    def __init__(
        self, x_data: torch.Tensor, y_data: torch.Tensor, scaling: str, device: str
    ):
        super().__init__()
        x_data = x_data.detach()
        y_data = y_data.detach()

        # linear scaling
        self.register_buffer("x_max", x_data.max(0).values)
        self.register_buffer("x_min", x_data.min(0).values)
        self.register_buffer("y_max", y_data.max(0).values)
        self.register_buffer("y_min", y_data.min(0).values)

        # gaussian scaling
        self.register_buffer("x_mean", x_data.mean(0))
        self.register_buffer("x_std", x_data.std(0))
        self.register_buffer("y_mean", y_data.mean(0))
        self.register_buffer("y_std", y_data.std(0))

        # bounds
        y_bounds = torch.zeros((2, y_data.shape[-1]))
        self.register_buffer("y_bounds", y_bounds)
        if scaling == "linear":
            self.y_bounds[0, :] = -1.1
            self.y_bounds[1, :] = 1.1
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
