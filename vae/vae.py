import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim.adamw import AdamW


class VAE(nn.Module):
    def __init__(
        self,
        normalizer,
        input_dim,
        latent_dim,
        hidden_dims,
        learning_rate,
        num_envs,
        goal=0.1,
        beta_min=1e-6,
        beta_max=10,
        alpha=0.95,
        geco_lr=1e-5,
        am_lr=0.03,
        prior_goal=0.9,
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

        self.optimizer = AdamW(self.parameters(), lr=learning_rate)
        self.normalizer = normalizer
        self.latent_dim = latent_dim
        self.goal = goal
        self.beta = 1.0
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.alpha = alpha
        self.geco_lr = geco_lr
        self.speedup = speedup
        self.err_ema = None

        # planning
        self.am_lr = am_lr
        self.prior_weight = torch.ones(num_envs, device=device)
        self.prior_geco_lr = 0.01
        self.prior_goal = prior_goal
        self.prior_ema = None
        self.prior_dist = Normal(0, 1)
        # weights for position and orientation losses
        self.mse_weights = torch.tensor(3 * [1.0] + 6 * [0.2], device=device)

        self.device = device
        self.to(device)

    ##
    # Inference
    ##

    def act(self, x: torch.Tensor, goal_ee_state: torch.Tensor, first_step: bool):
        if first_step:
            # encode initial observation
            x = self.normalizer(x)
            with torch.no_grad():
                z = self.encode(x)[0]
            self.z = z.detach().requires_grad_(True)
            # create optimizer
            self.optimizer_am = AdamW([self.z], lr=self.am_lr, weight_decay=0.0)
            # create goal state
            self.goal_ee_state = self.normalizer.normalize_goal(goal_ee_state)

        # calculate losses
        x_hat = self.decoder(self.z)
        mse = self.mse_weights * ((x_hat[:, 7:] - self.goal_ee_state) ** 2)
        mse = torch.mean(mse, dim=-1)
        prior_loss = (-self.prior_dist.log_prob(self.z)).mean(dim=-1)

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
            self.prior_weight = (factor * self.prior_weight).clamp(
                self.beta_min, self.beta_max
            )

        loss = mse + self.prior_weight * prior_loss
        loss = loss.sum()  # torch can only store retain graph for scalars

        self.optimizer_am.zero_grad()
        loss.backward()
        self.optimizer_am.step()

        with torch.no_grad():
            x_hat = self.decoder(self.z)

        x_hat = self.normalizer.inverse(x_hat)
        joint_pos_target = x_hat[:, :7]

        # self.log(locals())

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

    @torch.no_grad()
    def test(self, batch):
        x_unnorm = batch[0].to(self.device)
        x = self.normalizer(x_unnorm)
        x_hat = self(x)[0]
        x_hat = self.normalizer.inverse(x_hat)
        recon_loss = torch.mean((x_unnorm - x_hat) ** 2)
        return recon_loss.item()

    def reset(self, dones=None):
        if dones is not None:
            self.prior_weight[dones.bool()] = 1.0
        else:
            self.prior_weight = torch.ones_like(self.prior_weight)

    ##
    # Training
    ##

    def update(self, batch):
        x = batch[0].to(self.device)
        x = self.normalizer(x)

        x_hat, mu, logvar = self(x)

        recon_loss = torch.mean((x - x_hat) ** 2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())

        self.geco_step(recon_loss)

        loss = recon_loss + self.beta * kl_loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, recon_loss, kl_loss

    def geco_step(self, err):
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

    ##
    # Utils
    ##

    def log(self, locs: dict):
        print(
            f"""
            mse: {locs["mse"].mean().item():.2f} 
            prior_loss: {locs["prior_loss"].mean().item():.2f}
            """
        )
