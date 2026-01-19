"""
Post-inference module for LTX2 transformer model.

This module handles output processing including:
- Scale-shift modulation
- Output normalization
- Output projection
"""

import torch


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


class LTX2PostInfer:
    """
    Post-inference module for LTX2 transformer.

    Handles all output processing after transformer blocks.
    """

    def __init__(self, config):
        """
        Initialize post-inference module.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)

    def set_scheduler(self, scheduler):
        """Set the scheduler for inference."""
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(
        self,
        weights,
        vx: torch.Tensor,
        ax: torch.Tensor,
        video_embedded_timestep: torch.Tensor,
        audio_embedded_timestep: torch.Tensor,
    ) -> None:
        """
        Perform post-inference processing.

        Args:
            weights: LTX2PostWeights instance
            video_x: Video tensor after transformer blocks, shape [seq_len, hidden_dim]
            audio_x: Audio tensor after transformer blocks, shape [seq_len, hidden_dim]

        Returns:
            Tuple of (processed_video_x, processed_audio_x)
        """
        vx = self._process_output(
            weights.scale_shift_table.tensor,
            weights.proj_out,
            vx,
            video_embedded_timestep,
        )

        ax = self._process_output(
            weights.audio_scale_shift_table.tensor,
            weights.audio_proj_out,
            ax,
            audio_embedded_timestep,
        )
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return to_denoised(
            self.scheduler.video_latent_state.latent,
            vx,
            self.scheduler.video_timesteps_from_mask(),
        ), to_denoised(
            self.scheduler.audio_latent_state.latent,
            ax,
            self.scheduler.audio_timesteps_from_mask(),
        )

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        proj_out,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process output (no batch dimension).

        Args:
            scale_shift_table: Scale-shift table, shape [2, hidden_dim]
            proj_out: Output projection layer
            x: Input tensor, shape [seq_len, hidden_dim]
            embedded_timestep: Embedded timestep, shape [seq_len, hidden_dim]

        Returns:
            Processed output tensor, shape [seq_len, output_dim]
        """
        # Apply scale-shift modulation (no batch dimension)
        # scale_shift_table shape: [2, hidden_dim]
        # embedded_timestep shape: [seq_len, hidden_dim]
        # Result shape: [seq_len, 2, hidden_dim]
        scale_shift_values = scale_shift_table[None, :, :].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, None, :]
        shift, scale = scale_shift_values[:, 0], scale_shift_values[:, 1]

        # Output normalization (RMSNorm, no learnable params)
        x = rms_norm(x)

        # Apply scale-shift
        x = x * (1 + scale) + shift

        # Output projection
        x = proj_out.apply(x)

        return x
