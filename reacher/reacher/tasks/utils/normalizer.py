import torch
from torch import nn


class GaussianNormalizer(nn.Module):
    """Normalize mean and variance of values based on batch statistics."""

    def __init__(self, obs):
        super().__init__()
        mean = torch.mean(obs, dim=0)
        std = torch.std(obs, dim=0)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x):
        return (x - self._mean) / (self._std + 1e-6)

    def inverse(self, y):
        return y * (self._std + 1e-6) + self._mean

    def normalize_goal(self, goal):
        return (goal - self._mean[7:]) / self._std[7:]

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()
