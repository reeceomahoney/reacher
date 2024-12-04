import math
import torch
from typing import Callable


def get_sampler(sampler_type: str) -> Callable:
    if sampler_type == "ddim":
        return sample_ddim
    if sampler_type == "ddim_resample":
        return sample_resample_ddim
    elif sampler_type == "euler_ancestral":
        return sample_euler_ancestral
    elif sampler_type == "ddpm":
        return sample_ddpm
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def get_resampling_sequence(T, r, j):
    """
    Generate a sequence of "up" and "down" steps for diffusion resampling

    Parameters:
        T (int): Total diffusion steps.
        r (int): Resampling steps per diffusion step
        j (int): Jump length of resampling steps

    Returns:
        list: The generated sequence of "up" and "down".
    """
    # Start with T initial "down" steps
    sequence = ["down"] * j
    resampling_sequence = ["down"] * j + (["up"] * j + ["down"] * j) * (r - 1)
    sequence += resampling_sequence * ((T - j) // j)
    sequence += ["down"] * ((T - j) % j)

    return sequence


@torch.no_grad()
def sample_ddim(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """
    Perform inference using the DDIM sampler
    """
    sigmas = kwargs["sigmas"]
    x_t = noise
    s_in = x_t.new_ones([x_t.shape[0]])

    num_steps = kwargs.get("num_steps", len(sigmas) - 1)
    # inpainting data
    tgt = kwargs["tgt"]
    mask = kwargs["mask"]

    for i in range(num_steps):
        # x_t = tgt * mask + x_t * (1 - mask)
        denoised = model(x_t, sigmas[i] * s_in, data_dict, **kwargs)
        t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
        h = t_next - t
        x_t = ((-t_next).exp() / (-t).exp()) * x_t - (-h).expm1() * denoised

    # x_t = tgt * mask + x_t * (1 - mask)

    return x_t


@torch.no_grad()
def sample_resample_ddim(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    sigmas = kwargs["sigmas"]
    x_t = noise
    s_in = x_t.new_ones([x_t.shape[0]])

    # inpainting data
    # tgt = kwargs["tgt"]
    # mask = kwargs["mask"]

    # resampling sequence
    resampling_sequence = get_resampling_sequence(
        len(sigmas) - 1, kwargs["resampling_steps"], kwargs["jump_length"]
    )

    i = 0
    for step in resampling_sequence:
        if step == "down":
            # denoising step
            denoised = model(x_t, sigmas[i] * s_in, data_dict, **kwargs)
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            x_t = ((-t_next).exp() / (-t).exp()) * x_t - (-h).expm1() * denoised
            # inpaint
            # noised_tgt = tgt + torch.randn_like(tgt) * sigmas[i + 1]
            # x_t = noised_tgt * mask + x_t * (1 - mask)
            i += 1

        if step == "up":
            # noise resampling step
            d_sigma = torch.sqrt(sigmas[i - 1] ** 2 - sigmas[i] ** 2)
            x_t = x_t + torch.randn_like(x_t) * d_sigma
            i -= 1

    return x_t


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler_ancestral(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """
    Ancestral sampling with Euler method steps.

    1. compute dx_{i}/dt at the current timestep
    2. get sigma_{up} and sigma_{down} from ancestral method
    3. compute x_{t-1} = x_{t} + dx_{t}/dt * sigma_{down}
    4. Add additional noise after the update step x_{t-1} =x_{t-1} + z * sigma_{up}
    """
    sigmas = kwargs["sigmas"]
    x_t = noise
    s_in = x_t.new_ones([x_t.shape[0]])

    # inpainting mask
    tgt = kwargs["tgt"]
    mask = kwargs["mask"]
    # unsure if this is necessary
    # x_t = tgt * mask + x_t * (1 - mask)

    for i in range(len(sigmas) - 1):
        # compute x_{t-1}
        denoised = model(x_t, sigmas[i] * s_in, data_dict)
        # get ancestral steps
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        # compute dx/dt
        d = (x_t - denoised) / sigmas[i]
        # compute dt based on sigma_down value
        dt = sigma_down - sigmas[i]
        # update current action
        x_t = x_t + d * dt
        if sigma_down > 0:
            x_t = x_t + torch.randn_like(x_t) * sigma_up
        # inpaint
        noised_tgt = tgt + torch.randn_like(tgt) * sigmas[i]
        x_t = noised_tgt * mask + x_t * (1 - mask)

    return x_t


@torch.no_grad()
def sample_ddpm(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """
    Perform inference using the DDPM sampler
    """
    noise_scheduler = kwargs["noise_scheduler"]
    x_t = noise
    # inpainting mask
    tgt = kwargs["tgt"]
    mask = kwargs["mask"]

    for idx, t in enumerate(noise_scheduler.timesteps):
        x_t = tgt * mask + x_t * (1 - mask)
        t_pt = t.float().to(noise.device)
        output = model(x_t, t_pt.expand(x_t.shape[0]), data_dict)
        x_t = noise_scheduler.step(output, t, x_t).prev_sample

    x_t = tgt * mask + x_t * (1 - mask)
    return x_t


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_linear(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an linear noise schedule."""
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def rand_log_logistic(
    shape,
    loc=0.0,
    scale=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)
