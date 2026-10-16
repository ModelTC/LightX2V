"""Pure PyTorch Geo-FSRL, CPS and Flow-GRPO transition kernels.

Adapted from FewStepRL's ``wan_distill_pipeline_with_logprob.py`` at
3dd8e422 (Apache-2.0). All public times are normalized sigmas, not the
0--1000 model timesteps. Density reduction intentionally uses the mean,
matching FewStepRL, rather than a joint-density sum over latent elements.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .config import FewStepConfig


@dataclass(frozen=True)
class BridgeStepOutput:
    next_sample: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    sigma: torch.Tensor
    next_sigma: torch.Tensor
    sqrt_dt: torch.Tensor
    std_dev_t: torch.Tensor


def _validate_sigmas(sigma: torch.Tensor, next_sigma: torch.Tensor, *, terminal: bool = False) -> None:
    if not torch.isfinite(sigma).all() or not torch.isfinite(next_sigma).all():
        raise ValueError("Transition sigmas must be finite.")
    invalid_target = next_sigma < 0 if terminal else next_sigma <= 0
    if torch.any((sigma <= 0) | (sigma > 1) | invalid_target | (next_sigma >= sigma)):
        interval = "[0, sigma)" if terminal else "(0, sigma)"
        raise ValueError(f"Expected sigma in (0, 1] and next_sigma in {interval}.")


def _batch_sigma(value: float | torch.Tensor, sample: torch.Tensor, name: str) -> torch.Tensor:
    values = torch.as_tensor(value, device=sample.device, dtype=torch.float32).reshape(-1)
    batch_size = sample.shape[0]
    if values.numel() == 1:
        values = values.expand(batch_size)
    if values.numel() != batch_size:
        raise ValueError(f"{name} must be scalar or contain one value per sample ({batch_size}).")
    return values.reshape((batch_size,) + (1,) * (sample.ndim - 1))


def _mean_nonbatch(value: torch.Tensor) -> torch.Tensor:
    return value.mean(dim=tuple(range(1, value.ndim))) if value.ndim > 1 else value


def geo_coefficients(sigma: torch.Tensor, next_sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Geo's independent-noise coefficient ``a`` and retained-noise ``b``."""
    sigma, next_sigma = torch.broadcast_tensors(sigma.float(), next_sigma.float())
    _validate_sigmas(sigma, next_sigma)
    b = next_sigma * (1 - sigma) / (sigma * (1 - next_sigma))
    a_sq = 1 - b.square()
    if torch.any(a_sq < -1e-6):
        raise ValueError("Geo coefficients produced a^2 < 0; check timestep ordering.")
    return torch.sqrt(a_sq.clamp_min(0.0)), b


def geo_reweight_scale(sigma: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
    """Geometry factor used for Geo's matched policy and KL reweighting."""
    sigma, next_sigma = torch.broadcast_tensors(sigma.float(), next_sigma.float())
    a, b = geo_coefficients(sigma, next_sigma)
    denominator = sigma * (1 - next_sigma) - next_sigma * b * (1 - sigma)
    return next_sigma * a / denominator


def _validate_latents(model_output: torch.Tensor, sample: torch.Tensor) -> None:
    if sample.ndim < 1 or sample.shape[0] == 0 or sample.numel() == 0:
        raise ValueError("sample must have a nonempty leading batch dimension.")
    if model_output.shape != sample.shape or model_output.device != sample.device:
        raise ValueError("model_output and sample must have the same shape and device.")
    if not sample.is_floating_point() or not model_output.is_floating_point():
        raise ValueError("model_output and sample must be floating-point tensors.")


def _randn_like(sample: torch.Tensor, generator: torch.Generator | Sequence[torch.Generator] | None) -> torch.Tensor:
    def draw(shape: tuple[int, ...], rng: torch.Generator | None) -> torch.Tensor:
        # A CPU generator also works for a CUDA sample without changing its RNG stream.
        rng_device = sample.device if rng is None else rng.device
        return torch.randn(shape, generator=rng, device=rng_device, dtype=torch.float32).to(sample.device)

    if isinstance(generator, Sequence):
        if len(generator) != sample.shape[0] or not all(isinstance(g, torch.Generator) for g in generator):
            raise ValueError("generator sequence must contain one torch.Generator per sample.")
        return torch.cat([draw((1, *sample.shape[1:]), rng) for rng in generator], dim=0)
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or a sequence of generators.")
    return draw(tuple(sample.shape), generator)


def euler_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Deterministic transition used at all unselected steps, including sigma=0."""
    _validate_latents(model_output, sample)
    sigma = _batch_sigma(sigma, sample, "sigma")
    next_sigma = _batch_sigma(next_sigma, sample, "next_sigma")
    _validate_sigmas(sigma, next_sigma, terminal=True)
    return sample.float() + (next_sigma - sigma) * model_output.float()


def bridge_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
    *,
    config: FewStepConfig,
    next_sample: torch.Tensor | None = None,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
    noise: torch.Tensor | None = None,
) -> BridgeStepOutput:
    """Sample a transition or recompute the likelihood of a fixed observed action.

    ``next_sample``, ``noise`` and ``generator`` are mutually exclusive. All
    transition math is float32. The observed action is detached when evaluating
    its likelihood so that the score gradient goes only through the mean.
    ``std_dev_t`` retains FewStepRL's convention: std/sqrt(dt) for Geo and
    Flow-GRPO, but the actual std for CPS. Prefer ``std`` for KL calculations.
    """
    _validate_latents(model_output, sample)
    if sum(value is not None for value in (next_sample, noise, generator)) > 1:
        raise ValueError("next_sample, noise and generator are mutually exclusive.")
    for name, value in (("next_sample", next_sample), ("noise", noise)):
        if value is not None and (value.shape != sample.shape or value.device != sample.device):
            raise ValueError(f"{name} must have the same shape and device as sample.")
    sample, model_output = sample.float(), model_output.float()
    sigma = _batch_sigma(sigma, sample, "sigma")
    next_sigma = _batch_sigma(next_sigma, sample, "next_sigma")
    _validate_sigmas(sigma, next_sigma)
    dt = next_sigma - sigma
    sqrt_dt = torch.sqrt(-dt)

    if config.bridge == "geo":
        x0 = sample - sigma * model_output
        x1 = sample + (1 - sigma) * model_output
        a, b = geo_coefficients(sigma, next_sigma)
        mean = (1 - next_sigma) * x0 + next_sigma * b * x1
        std = next_sigma * a
        std_dev_t = std / sqrt_dt
    elif config.bridge == "flow_grpo":
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, next_sigma, sigma))) * config.noise_level
        if config.diffusion_clip:
            std_dev_t = torch.minimum(std_dev_t, config.diffusion_clip_value / sqrt_dt)
        mean = sample * (1 + std_dev_t.square() / (2 * sigma) * dt) + model_output * (1 + std_dev_t.square() * (1 - sigma) / (2 * sigma)) * dt
        std = std_dev_t * sqrt_dt
    else:  # CPS: intentionally matches the Wan bridge, including its noise convention.
        std = next_sigma * math.sin(config.noise_level * math.pi / 2)
        std_dev_t = std
        x0 = sample - sigma * model_output
        x1 = sample + (1 - sigma) * model_output
        mean = (1 - next_sigma) * x0 + torch.sqrt((next_sigma.square() - std.square()).clamp_min(0)) * x1

    if torch.any(std <= 0) or not torch.isfinite(std).all():
        raise ValueError("Stochastic bridge requires a finite positive transition std.")
    if next_sample is None:
        variance_noise = _randn_like(sample, generator) if noise is None else noise.float()
        next_sample = mean + std * variance_noise
    else:
        next_sample = next_sample.float()
    log_prob = -(next_sample.detach() - mean).square() / (2 * std.square()) - torch.log(std) - math.log(math.sqrt(2 * math.pi))
    return BridgeStepOutput(
        next_sample=next_sample,
        log_prob=_mean_nonbatch(log_prob),
        mean=mean,
        std=std,
        sigma=sigma,
        next_sigma=next_sigma,
        sqrt_dt=sqrt_dt,
        std_dev_t=std_dev_t,
    )
