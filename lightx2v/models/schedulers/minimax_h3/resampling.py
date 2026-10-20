"""CPU-testable helpers for MiniMax-H3 suffix regeneration and prefix drafts.

The resample_start_step selects an exact index of the original schedule.
Video and audio use that index with their respective shifted sigmas.
"""

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


def validate_resample_start_step(resample_start_step: int, infer_steps: int) -> int:
    """Select a suffix by zero-based index; infer_steps is the clean endpoint."""
    if not 0 <= resample_start_step <= infer_steps:
        raise ValueError(f"MiniMax-H3 resample_start_step must be an integer in [0, {infer_steps}], got {resample_start_step!r}")
    return int(resample_start_step)


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
