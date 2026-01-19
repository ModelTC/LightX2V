"""
LTX2 scheduler components for LightX2V.

This module contains components needed for LTX2 scheduler, migrated from ltx_core
to avoid external dependencies.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import torch

# ============================================================================
# Utility Functions
# ============================================================================


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.

    Args:
        sample: Noisy sample
        sigma: Noise level (sigma)
        denoised_sample: Denoised sample
        calc_dtype: Data type for calculation

    Returns:
        Velocity tensor
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.

    Args:
        sample: Noisy sample
        velocity: Velocity tensor
        sigma: Noise level (sigma)
        calc_dtype: Data type for calculation

    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


# ============================================================================
# Diffusion Step Protocol and Implementation
# ============================================================================


class DiffusionStepProtocol(Protocol):
    """Protocol for diffusion step implementations."""

    def step(self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
        """
        Take a single diffusion step.

        Args:
            sample: Current noisy sample
            denoised_sample: Denoised prediction
            sigmas: Sigma schedule
            step_index: Current step index

        Returns:
            Updated sample
        """
        ...


class EulerDiffusionStep:
    """
    First-order Euler method for diffusion sampling.
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """

    def step(self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
        """
        Take a single Euler step.

        Args:
            sample: Current noisy sample
            denoised_sample: Denoised prediction
            sigmas: Sigma schedule
            step_index: Current step index

        Returns:
            Updated sample
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)


# ============================================================================
# Guider Protocol and Implementation
# ============================================================================


class GuiderProtocol(Protocol):
    """Protocol for guidance implementations."""

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        """
        Compute guidance delta.

        Args:
            cond: Conditioned prediction
            uncond: Unconditioned prediction

        Returns:
            Guidance delta
        """
        ...

    def enabled(self) -> bool:
        """Check if guidance is enabled."""
        ...


@dataclass(frozen=True)
class CFGGuider:
    """
    Classifier-free guidance (CFG) guider.
    Computes the guidance delta as (scale - 1) * (cond - uncond), steering the
    denoising process toward the conditioned prediction.

    Attributes:
        scale: Guidance strength. 1.0 means no guidance, higher values increase
            adherence to the conditioning.
    """

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        """Compute CFG guidance delta."""
        return (self.scale - 1) * (cond - uncond)

    def enabled(self) -> bool:
        """Check if CFG is enabled."""
        return self.scale != 1.0


# ============================================================================
# Noiser Protocol and Implementation
# ============================================================================


class Noiser(Protocol):
    """Protocol for adding noise to latents during diffusion."""

    def __call__(self, latent: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Add noise to a latent tensor.

        Args:
            latent: Input latent tensor
            noise_scale: Scale factor for noise

        Returns:
            Noisy latent tensor
        """
        ...


class GaussianNoiser:
    """Adds Gaussian noise to a latent tensor."""

    def __init__(self, generator: torch.Generator):
        """
        Initialize Gaussian noiser.

        Args:
            generator: Random number generator
        """
        self.generator = generator

    def __call__(self, latent: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Add Gaussian noise to latent.

        Args:
            latent: Input latent tensor
            noise_scale: Scale factor for noise

        Returns:
            Noisy latent tensor
        """
        noise = torch.randn(
            *latent.shape,
            device=latent.device,
            dtype=latent.dtype,
            generator=self.generator,
        )
        return latent + noise * noise_scale


# ============================================================================
# Sigma Scheduler
# ============================================================================


BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2SigmaScheduler:
    """
    Default scheduler for LTX-2 diffusion sampling.
    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.
    """

    def execute(
        self,
        steps: int,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> torch.FloatTensor:
        """
        Generate sigma schedule.

        Args:
            steps: Number of diffusion steps
            latent: Optional latent tensor for token count calculation
            max_shift: Maximum shift parameter
            base_shift: Base shift parameter
            stretch: Whether to stretch sigmas to terminal value
            terminal: Terminal sigma value

        Returns:
            Tensor of sigmas
        """
        tokens = math.prod(latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR
        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)


# ============================================================================
# Modality (for transformer input)
# ============================================================================


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: torch.Tensor  # Shape: (B, T, D) where B is batch, T is tokens, D is dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is number of timesteps
    positions: torch.Tensor | None  # Shape: (B, 3, T) for video, where 3 is dimensions
    context: torch.Tensor | None
    enabled: bool = True
    context_mask: torch.Tensor | None = None


# ============================================================================
# X0Model Interface (simplified)
# ============================================================================


class X0ModelInterface:
    """
    Interface for X0 model (denoising model).
    This is a simplified interface that can be implemented by actual transformer models.
    """

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise video and audio according to sigma.

        Args:
            video: Video modality input
            audio: Audio modality input
            perturbations: Optional perturbations (not used in simplified version)

        Returns:
            Tuple of (denoised_video, denoised_audio)
        """
        raise NotImplementedError("Subclasses must implement __call__")
