import torch
import torch.nn as nn
from torch.optim.adamw import AdamW


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, learning_rate, device):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 2 * latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.ReLU(),
        )
        self.optimizer = AdamW(self.parameters(), lr=float(learning_rate))
        self.latent_dim = latent_dim
        self.goal = 0.1
        self.beta = 1.0
        self.beta_min = 1e-6
        self.beta_max = 10
        self.alpha = 0.99
        self.step_size = 1e-5
        self.err_ema = None
        self.speedup = None

        self.device = device
        self.to(device)

    ##
    # Inference
    ##

    def act(self, x, goal):
        z = self.encode(x)[0]

        z = z.detach().requires_grad_(True)
        z.retain_grad()
        x_hat = self.decoder(z)

        loss = torch.mean((x_hat - goal) ** 2)
        loss.backward(retain_graph=True)

        with torch.inference_mode():
            z = z - 0.1 * z.grad
            x_hat = self.decoder(z)

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

    ##
    # Training
    ##

    def update(self, batch):
        obs = batch[0].to(self.device)
        x_hat, mu, logvar = self(obs)

        recon_loss = torch.mean((obs - x_hat) ** 2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())

        loss = self.geco_loss(recon_loss, kl_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, recon_loss, kl_loss

    def geco_loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0 - self.alpha) * err + self.alpha * self.err_ema
            constraint = self.goal - self.err_ema
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        # Return loss
        return loss
