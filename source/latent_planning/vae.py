import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, device):
        super().__init__()
        self.net = nn.Linear(input_dim, latent_dim)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

    def act(self, x):
        return torch.zeros((x.shape[0], 7), device=self.device)

    def update(self):
        return 0
