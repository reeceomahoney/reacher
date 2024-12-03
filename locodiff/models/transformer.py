import logging
import torch
import torch.nn as nn

from locodiff.utils import SinusoidalPosEmb

log = logging.getLogger(__name__)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        T: int,
        T_cond: int,
        cond_mask_prob: float,
        emb_dropout: float,
        attn_dropout: float,
        weight_decay: float,
        inpaint_obs: bool,
        device: str,
    ):
        super().__init__()
        # variables
        input_dim = obs_dim + act_dim
        self.cond_mask_prob = cond_mask_prob
        self.weight_decay = weight_decay
        self.inpaint_obs = inpaint_obs
        self.device = device

        # embeddings
        self.input_emb = nn.Linear(input_dim, d_model)
        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.sigma_emb = nn.Linear(1, d_model)

        # change dims depending on if we're inpainting the obs
        input_len = T + T_cond - 1 if inpaint_obs else T
        cond_len = 1 if inpaint_obs else T_cond + 2

        # dropout and position encoding
        self.pos_emb = SinusoidalPosEmb(d_model, device)(torch.arange(input_len))
        self.cond_pos_emb = SinusoidalPosEmb(d_model, device)(torch.arange(cond_len))
        self.drop = nn.Dropout(emb_dropout)

        # transformer
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.register_buffer("mask", self.generate_mask(input_len))
        # output
        self.ln_f = nn.LayerNorm(d_model)
        self.output_pred = nn.Linear(d_model, input_dim)

        self.apply(self._init_weights)
        self.to(device)

        total_params = sum(p.numel() for p in self.get_params())
        log.info(f"Total parameters: {total_params:e}")

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerDecoderLayer,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Sequential,
            DiffusionTransformer,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, x, sigma, data_dict, uncond=False):
        # embeddings
        sigma = sigma.log() / 4
        sigma_emb = self.sigma_emb(sigma.view(-1, 1, 1))
        x_emb = self.input_emb(x)

        # create conditioning
        if self.inpaint_obs:
            cond_emb = sigma_emb
        else:
            obs_emb = self.obs_emb(data_dict["obs"])
            goal_emb = self.obs_emb(data_dict["goal"]).unsqueeze(1)
            cond_emb = torch.cat([sigma_emb, obs_emb, goal_emb], dim=1)
        # add position encoding and dropout
        cond_emb = self.drop(cond_emb + self.cond_pos_emb)
        x_emb = self.drop(x_emb + self.pos_emb)

        # output
        x = self.decoder(tgt=x_emb, memory=cond_emb, tgt_mask=self.mask)
        x = self.ln_f(x)
        return self.output_pred(x)

    def generate_mask(self, x):
        mask = (torch.triu(torch.ones(x, x)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def mask_cond(self, cond, force_mask=False):
        cond = cond.clone()
        if force_mask:
            cond[...] = 0
            return cond
        elif self.training and self.cond_mask_prob > 0:
            mask = (torch.rand(cond.shape[0], 1) > self.cond_mask_prob).float()
            mask = mask.expand_as(cond)
            cond[mask == 0] = 0
            return cond
        else:
            return cond

    def get_params(self):
        return self.parameters()

    def detach_all(self):
        for _, param in self.named_parameters():
            param.detach_()
