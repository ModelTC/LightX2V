"""CPU-testable helpers for MiniMax-H3 suffix regeneration and prefix drafts.

The resample_strength is the *video sigma*, not a fraction of the inference steps.
Audio uses the sigma at the same index of its own shifted schedule.
"""

import math
from numbers import Integral

import torch


def validate_draft_steps(draft_steps: int | None, infer_steps: int) -> int | None:
    """Return a shorter prefix, or None to use the original sampling path."""
    if draft_steps is None:
        return None
    if isinstance(draft_steps, bool) or not isinstance(draft_steps, Integral) or draft_steps < 1:
        raise ValueError(f"MiniMax-H3 draft_steps must be a positive integer, got {draft_steps!r}")
    return int(draft_steps) if draft_steps < infer_steps else None


def flow_predict_clean(sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Estimate x0 from the *pre-update* sample and H3's data-ward velocity.

    For x_sigma = (1-sigma)*x0 + sigma*epsilon, the training target is
    v = x0-epsilon; therefore x0_hat = x_sigma + sigma*v_hat. Video and
    audio must use their respective sigmas at the evaluated step.
    """
    if sample.shape != velocity.shape:
        raise ValueError(f"Sample/velocity shape mismatch: {tuple(sample.shape)} vs {tuple(velocity.shape)}")
    sigma = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    return sample.float() + sigma * velocity.to(device=sample.device, dtype=torch.float32)


def select_suffix_index(video_sigmas: torch.Tensor, resample_strength: float) -> int:
    """Snap to the nearest original sigma knot, including the clean endpoint.

    Equidistant nearest knots prefer less noise. A small tolerance accounts for
    the original FP32 grid, e.g. the representation of the exact knot 2/3.
    """
    resample_strength = float(resample_strength)
    if not math.isfinite(resample_strength) or not 0.0 <= resample_strength <= 1.0:
        raise ValueError(f"MiniMax-H3 resample_strength must be finite and in [0, 1], got {resample_strength}")
    sigmas = video_sigmas.detach().float().cpu()
    if sigmas.ndim != 1 or sigmas.numel() < 2 or not bool(torch.isfinite(sigmas).all()):
        raise ValueError("Expected a finite one-dimensional sigma schedule")
    if sigmas[0] != 1 or sigmas[-1] != 0 or not bool((sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError("Expected a strictly decreasing sigma schedule from 1 to 0")
    distances = (sigmas.double() - resample_strength).abs()
    exact = torch.nonzero(distances <= 1e-7).flatten()
    if exact.numel():
        return int(exact[-1])
    return int(torch.nonzero(distances <= distances.min() + 1e-7)[-1, 0])


def validate_source_latents(video, audio, *, video_shape, audio_shape):
    for name, latent, expected in (("video", video, video_shape), ("audio", audio, audio_shape)):
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != tuple(expected):
            actual = tuple(latent.shape) if isinstance(latent, torch.Tensor) else type(latent).__name__
            raise ValueError(f"Source {name} latent shape must be {tuple(expected)}, got {actual}; changing duration is not supported")
        if not latent.is_floating_point() or not bool(torch.isfinite(latent).all()):
            raise ValueError(f"Source {name} latents must contain finite floating point values")


def flow_renoise(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """H3's data-ward convention: x = (1-sigma)*clean + sigma*noise."""
    if clean.shape != noise.shape:
        raise ValueError(f"Source/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}")
    clean = clean.to(device=noise.device, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, device=noise.device, dtype=torch.float32)
    return (1.0 - sigma) * clean + sigma * noise.float()
