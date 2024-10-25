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
        device="cpu",
    ):
        super().__init__()
        self.normalizer = normalizer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], input_dim),
        )
        self.beta = nn.Parameter(torch.tensor(beta_init))

        self.optimizer = AdamW(self.parameters(), lr=float(learning_rate))
        self.beta_optimizer = AdamW([self.beta], lr=float(learning_rate))

        self.latent_dim = latent_dim
        self.goal = float(goal)
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.device = device
        self.to(device)

    ##
    # Inference
    ##

    def act(self, x, goal):
        x = self.normalizer(x)
        z = self.encode(x)[0]

        z = z.detach().requires_grad_(True)
        z.retain_grad()
        x_hat = self.decoder(z)

        loss = torch.mean((x_hat - goal) ** 2)
        loss.backward(retain_graph=True)

        with torch.inference_mode():
            z = z - z.grad
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

    ##
    # Training
    ##

    def update(self, batch):
        x = batch[0].to(self.device)
        x = self.normalizer(x)

        # optimize network
        x_hat, mu, logvar = self(x)
        recon_loss = torch.mean((x - x_hat) ** 2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        loss = recon_loss + self.beta.detach() * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # optimize beta
        with torch.inference_mode():
            x_hat = self(x)[0]
            recon_loss = torch.mean((x - x_hat) ** 2)
        beta_loss = self.beta * torch.max(recon_loss - self.goal, torch.tensor(0.0))

        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()

        with torch.inference_mode():
            self.beta.clamp_(self.beta_min, self.beta_max)

        return loss, recon_loss, kl_loss
