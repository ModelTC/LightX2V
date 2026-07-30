import torch

from ..dmd.math import expand_sigma


def raw_timesteps_to_sigmas(
    raw_timesteps,
    *,
    device,
    dtype,
    warp_denoising_step,
    denoising_scheduler,
    num_train_timestep,
):
    """Convert raw diffusion timesteps to the active warped sigma space."""
    raw_timesteps = raw_timesteps.to(
        device=device,
        dtype=torch.long,
    )
    if warp_denoising_step:
        timesteps = denoising_scheduler.timesteps.to(
            device=device,
            dtype=torch.float32,
        )
        indices = denoising_scheduler.num_train_timesteps - raw_timesteps
        warped_timesteps = timesteps[indices]
    else:
        warped_timesteps = raw_timesteps.float()
    return (warped_timesteps / num_train_timestep).to(dtype=dtype)


def phase_sigma(
    batch_size,
    *,
    match_timestep,
    device,
    dtype,
    warp_denoising_step,
    denoising_scheduler,
    num_train_timestep,
):
    raw_timestep = torch.full(
        (int(batch_size),),
        match_timestep,
        device=device,
        dtype=torch.long,
    )
    return raw_timesteps_to_sigmas(
        raw_timestep,
        device=device,
        dtype=dtype,
        warp_denoising_step=warp_denoising_step,
        denoising_scheduler=denoising_scheduler,
        num_train_timestep=num_train_timestep,
    )


def sample_score_sigma_range(
    batch_size,
    raw_min,
    raw_max,
    *,
    device,
    dtype,
    score_timestep_min,
    score_timestep_max,
    num_train_timestep,
    convert_timesteps,
    broadcast_value,
):
    raw_min = max(
        1,
        int(raw_min),
        score_timestep_min,
    )
    raw_max = min(
        num_train_timestep,
        int(raw_max),
        score_timestep_max + 1,
    )
    raw_candidates = torch.arange(
        raw_min,
        raw_max,
        device=device,
        dtype=torch.long,
    )
    candidate_sigmas = convert_timesteps(
        raw_candidates,
        dtype=torch.float32,
    )
    if candidate_sigmas.numel() == 0:
        raise RuntimeError(f"No valid score timesteps remain in raw range [{raw_min}, {raw_max}).")
    candidate_indices = torch.randint(
        0,
        candidate_sigmas.numel(),
        (int(batch_size),),
        device=device,
        dtype=torch.long,
    )
    sigma = candidate_sigmas[candidate_indices]
    return broadcast_value(sigma.to(dtype=dtype))


def phased_coefficients(
    sigma_s,
    sigma_t,
    ndim,
    eps,
):
    sigma_s = expand_sigma(sigma_s.float(), ndim)
    sigma_t = expand_sigma(sigma_t.float(), ndim)
    one_minus_s = (1.0 - sigma_s).clamp_min(eps)
    alpha = (1.0 - sigma_t) / one_minus_s
    beta_squared = (sigma_t.square() - alpha.square() * sigma_s.square()).clamp_min(eps)
    return (
        sigma_s,
        sigma_t,
        one_minus_s,
        alpha,
        beta_squared.sqrt(),
    )


def phased_forward(
    xs,
    noise,
    sigma_s,
    sigma_t,
    eps,
):
    _, _, _, alpha, beta = phased_coefficients(
        sigma_s,
        sigma_t,
        xs.ndim,
        eps,
    )
    return (alpha * xs.float() + beta * noise.float()).to(dtype=xs.dtype)


def phased_velocity_target(
    xs,
    noise,
    sigma_s,
    sigma_t,
    eps,
):
    sigma_s, sigma_t, one_minus_s, _, beta = phased_coefficients(
        sigma_s,
        sigma_t,
        xs.ndim,
        eps,
    )
    coefficient_xs = -1.0 / one_minus_s
    coefficient_noise = (sigma_t + (1.0 - sigma_t) * sigma_s.square() / one_minus_s.square()) / beta
    return (coefficient_xs * xs.float() + coefficient_noise * noise.float()).to(dtype=xs.dtype)
