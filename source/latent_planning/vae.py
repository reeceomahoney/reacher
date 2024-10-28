import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW


class VAE(nn.Module):
    def __init__(
        self,
        normalizer,
        input_dim,
        latent_dim,
        hidden_dims,
        learning_rate,
        goal=0.1,
        beta_init=1.0,
        beta_min=1e-6,
        beta_max=10,
        alpha=0.99,
        geco_lr=1e-5,
        am_lr=0.03,
        am_prior_weight=1.0,
        speedup=None,
        device="cpu",
    ):
        super().__init__()
        # encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.ELU())

        for i in range(1, len(hidden_dims)):
            encoder_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            encoder_layers.append(nn.ELU())

        encoder_layers.append(nn.Linear(hidden_dims[-1], 2 * latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ELU())

        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            decoder_layers.append(nn.ELU())

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.optimizer = AdamW(self.parameters(), lr=float(learning_rate))
        self.normalizer = normalizer
        self.latent_dim = latent_dim
        self.goal = float(goal)
        self.beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.alpha = alpha
        self.geco_lr = float(geco_lr)
        # self.am_lr = float(am_lr)
        # self.am_prior_weight = am_prior_weight
        self.speedup = speedup
        self.err_ema = None

        # planning
        self.am_lr = 0.03
        self.prior_weight = 1.0
        self.prior_goal = 0.9
        self.prior_ema = None
        self.prior_geco_lr = 0.01

        self.device = device
        self.to(device)

    ##
    # Inference
    ##

    def act(self, x, goal_ee_pos):
        x = self.normalizer(x)
        z, mu, logvar = self.encode(x)

        z = z.detach().requires_grad_(True)
        z.retain_grad()
        x_hat = self.decoder(z)

        # calculate losses
        mse = torch.mean((x_hat[:, -3:] - goal_ee_pos) ** 2, dim=-1)
        d = z.shape[-1]
        prior_loss = 0.5 * (
            d * np.log(2 * math.pi)
            + logvar.sum(dim=-1)
            + ((z - mu) ** 2 / logvar.exp()).sum(dim=-1)
        )  # -log p(z)

        # update prior weight with geco
        with torch.no_grad():
            # update ema
            if self.prior_ema is None:
                self.prior_ema = prior_loss
            else:
                self.prior_ema = 0.05 * prior_loss + 0.95 * self.prior_ema

            # update beta
            constraint = self.prior_ema - self.prior_goal
            factor = torch.exp(self.prior_geco_lr * constraint)
            self.prior_weight = factor * self.prior_weight

        loss = mse + self.prior_weight * prior_loss

        # print("mse: ", mse.mean())
        # print("prior_loss: ", prior_loss.mean())

        loss = loss.sum()  # torch can only store retain graph for scalars
        loss.backward(retain_graph=True)

        with torch.inference_mode():
            z = z - self.am_lr * z.grad
            x_hat = self.decoder(z)

        x_hat = self.normalizer.inverse(x_hat)
        joint_pos_target = x_hat[:, :7]
        return joint_pos_target

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = torch.split(x, self.latent_dim, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(mu)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def test(self, batch):
        x_unnorm = batch[0].to(self.device)
        x = self.normalizer(x_unnorm)
        x_hat = self(x)[0]
        x_hat = self.normalizer.inverse(x_hat)
        recon_loss = torch.mean((x_unnorm - x_hat) ** 2)
        return recon_loss.item()

    def reset(self):
        self.prior_weight = 1.0

    ##
    # Training
    ##

    def update(self, batch):
        x = batch[0].to(self.device)
        x = self.normalizer(x)

        x_hat, mu, logvar = self(x)

        recon_loss = torch.mean((x - x_hat) ** 2, dim=-1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp(), dim=-1)

        self.geco_step(recon_loss)

        loss = recon_loss + self.beta * kl_loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, recon_loss.mean(), kl_loss.mean()

    def geco_step(self, err):
        err = err.mean()
        with torch.no_grad():
            # update ema
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0 - self.alpha) * err + self.alpha * self.err_ema

            # update beta
            constraint = self.goal - self.err_ema
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.geco_lr * constraint)
            else:
                factor = torch.exp(self.geco_lr * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
