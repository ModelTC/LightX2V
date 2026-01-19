"""
LTX2 Scheduler for LightX2V.

This scheduler integrates LTX-2's diffusion sampling logic including:
- Sigma schedule generation with token-count-dependent shifting
- Euler diffusion stepping
- Classifier-free guidance (CFG)
- Audio-video joint denoising
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: Tuple[int, int, int],
    causal_fix: bool = False,
) -> torch.Tensor:
    """
    Map latent-space [start, end) coordinates to pixel-space equivalents.
    Args:
        latent_coords: Tensor of latent bounds shaped [3, num_patches, 2]
        scale_factors: Tuple (temporal, height, width) with integer scale factors
        causal_fix: When True, rewrites temporal axis of first frame
    Returns:
        Pixel coordinates with shape [3, num_patches, 2]
    """
    scale_tensor = torch.tensor(scale_factors, device=latent_coords.device).view(-1, 1, 1)
    pixel_coords = latent_coords * scale_tensor

    if causal_fix:
        pixel_coords[0, ...] = (pixel_coords[0, ...] + 1 - scale_factors[0]).clamp(min=0)

    return pixel_coords


@dataclass
class LatentState:
    """
    State of latents during the diffusion denoising process.

    Attributes:
        latent: The current noisy latent tensor being denoised (patchified, shape: [T, D])
        denoise_mask: Mask encoding the denoising strength for each token (patchified, shape: [T, 1])
        positions: Positional indices for each latent element (shape: [3, T] for video, [2, T] for audio)
        clean_latent: Initial state of the latent before denoising (patchified, shape: [T, D])
    """

    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor


class VideoLatentPatchifier:
    """Video latent patchifier (adapted from ltx_core, without batch dimension)."""

    def __init__(self, patch_size: int):
        self._patch_size = (
            1,  # temporal dimension
            patch_size,  # height dimension
            patch_size,  # width dimension
        )

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify video latents from [C, F, H, W] to [T, D]."""
        return einops.rearrange(
            latents,
            "c (f p1) (h p2) (w p3) -> (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )

    def unpatchify(self, latents: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
        """Unpatchify video latents from [T, D] to [C, F, H, W]."""
        patch_grid_frames = frames // self._patch_size[0]
        patch_grid_height = height // self._patch_size[1]
        patch_grid_width = width // self._patch_size[2]

        return einops.rearrange(
            latents,
            "(f h w) (c p1 p2 p3) -> c (f p1) (h p2) (w p3)",
            f=patch_grid_frames,
            h=patch_grid_height,
            w=patch_grid_width,
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )

    def get_patch_grid_bounds(
        self,
        frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the per-dimension bounds [inclusive start, exclusive end) for every patch.
        Returns shape [3, num_patches, 2] where 3 is (frame, height, width) and 2 is [start, end).
        """
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=frames, step=self._patch_size[0], device=device),
            torch.arange(start=0, end=height, step=self._patch_size[1], device=device),
            torch.arange(start=0, end=width, step=self._patch_size[2], device=device),
            indexing="ij",
        )

        patch_starts = torch.stack(grid_coords, dim=0)

        patch_size_delta = torch.tensor(
            self._patch_size,
            device=patch_starts.device,
            dtype=patch_starts.dtype,
        ).view(3, 1, 1, 1)

        patch_ends = patch_starts + patch_size_delta

        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)

        # Flatten to [3, num_patches, 2]
        latent_coords = einops.rearrange(
            latent_coords,
            "c f h w bounds -> c (f h w) bounds",
            bounds=2,
        )

        return latent_coords


class AudioPatchifier:
    """Audio latent patchifier (adapted from ltx_core, without batch dimension)."""

    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """Patchify audio latents from [C, T, F] to [T, D]."""
        if len(audio_latents.shape) == 4:
            return einops.rearrange(
                audio_latents,
                "b c t f -> b t (c f)",
            )
        return einops.rearrange(
            audio_latents,
            "c t f -> t (c f)",
        )

    def unpatchify(self, audio_latents: torch.Tensor, channels: int, mel_bins: int) -> torch.Tensor:
        """Unpatchify audio latents from [T, D] to [C, T, F]."""
        if len(audio_latents.shape) == 3:
            return einops.rearrange(
                audio_latents,
                "b t (c f) ->b c t f",
                c=channels,
                f=mel_bins,
            )
        return einops.rearrange(
            audio_latents,
            "t (c f) -> c t f",
            c=channels,
            f=mel_bins,
        )

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convert latent indices into real-time seconds."""
        if device is None:
            device = torch.device("cpu")

        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor

        if self.is_causal:
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.audio_latent_downsample_factor).clip(min=0)

        return audio_mel_frame * self.hop_length / self.sample_rate

    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Builds a [B, 1, T, 2] tensor containing timestamps for each latent frame.
        This helper method underpins get_patch_grid_bounds for the audio patchifier.
        Args:
            batch_size: Number of sequences to broadcast the timings over (we use 1 for no batch).
            num_steps: Number of latent frames (time steps) to convert into timestamps.
            device: Device on which the resulting tensor should reside.
        """
        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cpu")

        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            num_steps + self.shift,
            torch.float32,
            resolved_device,
        )
        start_timings = start_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            num_steps + self.shift + 1,
            torch.float32,
            resolved_device,
        )
        end_timings = end_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        return torch.stack([start_timings, end_timings], dim=-1)

    def get_patch_grid_bounds(
        self,
        frames: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the temporal bounds [inclusive start, exclusive end) for every patch.
        Returns shape [1, time_steps, 2] where 1 is temporal dimension and 2 is [start, end) in seconds.
        """
        # Use batch_size=1 since we removed batch dimension
        return self._compute_audio_timings(batch_size=1, num_steps=frames, device=device)


class LTX2Scheduler(BaseScheduler):
    """
    Scheduler for LTX-2 diffusion model.

    Handles sigma schedule generation, Euler stepping, and CFG guidance
    for joint audio-video denoising.
    """

    def __init__(self, config):
        """
        Initialize LTX2 scheduler.

        Args:
            config: Configuration dictionary containing:
                - infer_steps: Number of inference steps
                - cfg_guidance_scale: CFG guidance scale (default: 1.0)
                - scheduler_type: Type of sigma scheduler (default: "ltx2")
                - max_shift: Maximum shift for LTX2 scheduler (default: 2.05)
                - base_shift: Base shift for LTX2 scheduler (default: 0.95)
                - stretch: Whether to stretch sigmas (default: True)
                - terminal: Terminal sigma value (default: 0.1)
        """
        super().__init__(config)
        self.infer_steps = config.get("infer_steps", 50)
        self.cfg_guidance_scale = config.get("cfg_guidance_scale", 1.0)

        # Sigma scheduler configuration
        self.max_shift = config.get("max_shift", 2.05)
        self.base_shift = config.get("base_shift", 0.95)
        self.stretch = config.get("stretch", True)
        self.terminal = config.get("terminal", 0.1)

        # Constants for sigma scheduling
        self.base_shift_anchor = 1024
        self.max_shift_anchor = 4096

        # Patchifier configuration
        self.video_patch_size = config.get("video_patch_size", 1)
        self.fps = config.get("fps", 24)  # Frames per second for position calculation
        self.video_scale_factors = config.get("video_scale_factors", (8, 32, 32))  # (time, height, width)

        # Initialize patchifiers
        self.video_patchifier = VideoLatentPatchifier(patch_size=self.video_patch_size)
        self.audio_patchifier = AudioPatchifier(
            patch_size=config.get("audio_patch_size", 1),
            sample_rate=config.get("audio_sampling_rate", 16000),
            hop_length=config.get("audio_hop_length", 160),
            audio_latent_downsample_factor=config.get("audio_scale_factor", 4),
            is_causal=config.get("audio_is_causal", True),
            shift=config.get("audio_shift", 0),
        )

        # Diffusion components
        # self.stepper = EulerDiffusionStep()
        self.sample_guide_scale = self.config["sample_guide_scale"]

        # State
        self.sigmas = None
        self.video_latent_state = None
        self.audio_latent_state = None

        # Step state
        self.v_noise_pred = None
        self.a_noise_pred = None
        self.keep_latents_dtype_in_scheduler = config.get("keep_latents_dtype_in_scheduler", False)

    def step_pre(self, step_index):
        self.step_index = step_index

    def prepare(
        self,
        seed: int,
        video_latent_shape: tuple,
        audio_latent_shape: tuple,
        initial_video_latent: Optional[torch.Tensor] = None,
        initial_audio_latent: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
    ):
        """
        Prepare scheduler for inference.

        Args:
            seed: Random seed for noise generation
            video_latent_shape: Shape of video latents
            audio_latent_shape: Shape of audio latents
            video_context_p: Positive video context embeddings
            video_context_n: Negative video context embeddings
            audio_context_p: Positive audio context embeddings
            audio_context_n: Negative audio context embeddings
            transformer: X0Model transformer instance
            initial_video_latent: Optional initial video latent (for conditioning)
            initial_audio_latent: Optional initial audio latent (for conditioning)
            noise_scale: Scale factor for noise
            dtype: Data type for latents
        """
        # Initialize generator
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)

        # Prepare latents
        self.prepare_latents(
            video_latent_shape=video_latent_shape,
            audio_latent_shape=audio_latent_shape,
            initial_video_latent=initial_video_latent,
            initial_audio_latent=initial_audio_latent,
            noise_scale=noise_scale,
        )

        # Generate sigma schedule (needs unpatchified latent for token count calculation)
        self.set_timesteps(infer_steps=self.infer_steps)

    def prepare_latents(
        self,
        video_latent_shape: tuple,
        audio_latent_shape: tuple,
        initial_video_latent: Optional[torch.Tensor] = None,
        initial_audio_latent: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        video_denoise_mask: Optional[torch.Tensor] = None,
        audio_denoise_mask: Optional[torch.Tensor] = None,
    ):
        """
        Prepare initial latents for denoising and patchify them.

        This method follows LTX2's create_initial_state logic:
        1. Create unpatchified latents [C, F, H, W] for video, [C, F, M] for audio
        2. Add noise following GaussianNoiser formula
        3. Patchify to [T, D] format
        4. Create positions and denoise_mask in patchified format
        5. Return LatentState objects

        Args:
            video_latent_shape: Shape of video latents (C, F, H, W) - batch dimension removed
            audio_latent_shape: Shape of audio latents (C, F, M) - batch dimension removed
            initial_video_latent: Optional initial video latent (for conditioning)
            initial_audio_latent: Optional initial audio latent (for conditioning)
            noise_scale: Scale factor for noise
            video_denoise_mask: Optional denoise mask for video (unpatchified)
            audio_denoise_mask: Optional denoise mask for audio (unpatchified)
            dtype: Data type for latents (defaults to GET_DTYPE())
        """

        # Prepare video latents
        self._prepare_video_latents(
            video_latent_shape=video_latent_shape,
            initial_video_latent=initial_video_latent,
            noise_scale=noise_scale,
            video_denoise_mask=video_denoise_mask,
            dtype=GET_DTYPE(),
        )

        # Prepare audio latents
        self._prepare_audio_latents(
            audio_latent_shape=audio_latent_shape,
            initial_audio_latent=initial_audio_latent,
            noise_scale=noise_scale,
            audio_denoise_mask=audio_denoise_mask,
            dtype=GET_DTYPE(),
        )

    def _prepare_video_latents(
        self,
        video_latent_shape: tuple,
        initial_video_latent: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        video_denoise_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = None,
    ):
        """
        Prepare video latents for denoising.

        Args:
            video_latent_shape: Shape of video latents (C, F, H, W) - batch dimension removed
            initial_video_latent: Optional initial video latent (for conditioning)
            noise_scale: Scale factor for noise
            video_denoise_mask: Optional denoise mask for video (unpatchified)
            dtype: Data type for latents
        """
        _, frames_v, height_v, width_v = video_latent_shape

        # Save shape information for unpatchify
        self.video_latent_shape_orig = video_latent_shape

        # Initialize video latents (unpatchified)
        if initial_video_latent is not None:
            video_latent = initial_video_latent.to(dtype=dtype, device=AI_DEVICE)
        else:
            video_latent = torch.zeros(
                *video_latent_shape,
                dtype=dtype,
                device=AI_DEVICE,
            )

        clean_video_latent = video_latent.clone()

        # Create denoise mask (unpatchified)
        if video_denoise_mask is None:
            video_denoise_mask = torch.ones(
                1,
                frames_v,
                height_v,
                width_v,
                dtype=torch.float32,
                device=AI_DEVICE,
            )
        else:
            video_denoise_mask = video_denoise_mask.to(dtype=torch.float32, device=AI_DEVICE)
            if video_denoise_mask.shape[0] != 1:
                video_denoise_mask = video_denoise_mask[:1, ...]  # Take first channel

        # Patchify video latents first (aligned with source code: create_initial_state -> patchify)
        patchified_video_latent = self.video_patchifier.patchify(video_latent)
        patchified_clean_video_latent = self.video_patchifier.patchify(clean_video_latent)
        patchified_video_mask = self.video_patchifier.patchify(video_denoise_mask)

        # Process denoise_mask: ensure float32 and reduce to [T, 1] (aligned with source code)
        patchified_video_mask = patchified_video_mask.to(torch.float32)
        # Reduce patch dimension to 1 (aligned with source code behavior)
        if patchified_video_mask.shape[-1] > 1:
            # Take mean across patch dimension to get [T, 1]
            patchified_video_mask = patchified_video_mask.mean(dim=-1, keepdim=True)
        # Ensure shape is [T, 1]
        if patchified_video_mask.ndim == 1:
            patchified_video_mask = patchified_video_mask.unsqueeze(-1)

        # Add noise after patchify (aligned with source code: GaussianNoiser operates on patchified latent)
        noise_video = torch.randn(
            *patchified_video_latent.shape,
            dtype=patchified_video_latent.dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )

        scaled_mask_video = patchified_video_mask * noise_scale
        patchified_video_latent = (noise_video * scaled_mask_video + patchified_video_latent * (1 - scaled_mask_video)).to(patchified_video_latent.dtype)

        # Get positions for video
        latent_coords_video = self.video_patchifier.get_patch_grid_bounds(frames_v, height_v, width_v, AI_DEVICE)
        positions_video = get_pixel_coords(latent_coords_video, self.video_scale_factors, causal_fix=True)
        # Convert to float first, then divide by fps, then convert to dtype (aligned with source code)
        positions_video = positions_video.float()
        positions_video[0, ...] = positions_video[0, ...] / self.fps
        positions_video = positions_video.to(dtype)

        # Create video LatentState
        self.video_latent_state = LatentState(
            latent=patchified_video_latent,
            denoise_mask=patchified_video_mask,
            positions=positions_video,
            clean_latent=patchified_clean_video_latent,
        )

    def _prepare_audio_latents(
        self,
        audio_latent_shape: tuple,
        initial_audio_latent: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        audio_denoise_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = None,
    ):
        """
        Prepare audio latents for denoising.

        Args:
            audio_latent_shape: Shape of audio latents (C, F, M) - batch dimension removed
            initial_audio_latent: Optional initial audio latent (for conditioning)
            noise_scale: Scale factor for noise
            audio_denoise_mask: Optional denoise mask for audio (unpatchified)
            dtype: Data type for latents
        """
        _, frames_a, mel_bins_a = audio_latent_shape

        # Save shape information for unpatchify
        self.audio_latent_shape_orig = audio_latent_shape

        # Initialize audio latents (unpatchified)
        if initial_audio_latent is not None:
            audio_latent = initial_audio_latent.to(dtype=dtype, device=AI_DEVICE)
        else:
            audio_latent = torch.zeros(
                *audio_latent_shape,
                dtype=dtype,
                device=AI_DEVICE,
            )

        clean_audio_latent = audio_latent.clone()

        # Create denoise mask (unpatchified)
        if audio_denoise_mask is None:
            audio_denoise_mask = torch.ones(
                1,
                frames_a,
                mel_bins_a,
                dtype=torch.float32,
                device=AI_DEVICE,
            )
        else:
            audio_denoise_mask = audio_denoise_mask.to(dtype=torch.float32, device=AI_DEVICE)
            if audio_denoise_mask.shape[0] != 1:
                audio_denoise_mask = audio_denoise_mask[:1, ...]

        # Patchify audio latents first (aligned with source code: create_initial_state -> patchify)
        patchified_audio_latent = self.audio_patchifier.patchify(audio_latent)
        patchified_clean_audio_latent = self.audio_patchifier.patchify(clean_audio_latent)
        patchified_audio_mask = self.audio_patchifier.patchify(audio_denoise_mask)

        # Process denoise_mask: ensure float32 and reduce to [T, 1] (aligned with source code)
        patchified_audio_mask = patchified_audio_mask.to(torch.float32)
        # Reduce patch dimension to 1 (aligned with source code behavior)
        if patchified_audio_mask.shape[-1] > 1:
            # Take mean across patch dimension to get [T, 1]
            patchified_audio_mask = patchified_audio_mask.mean(dim=-1, keepdim=True)
        # Ensure shape is [T, 1]
        if patchified_audio_mask.ndim == 1:
            patchified_audio_mask = patchified_audio_mask.unsqueeze(-1)

        # Add noise after patchify (aligned with source code: GaussianNoiser operates on patchified latent)
        noise_audio = torch.randn(
            *patchified_audio_latent.shape,
            dtype=patchified_audio_latent.dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )
        scaled_mask_audio = patchified_audio_mask * noise_scale
        patchified_audio_latent = (noise_audio * scaled_mask_audio + patchified_audio_latent * (1 - scaled_mask_audio)).to(patchified_audio_latent.dtype)

        # Get positions for audio (just time coordinates)
        positions_audio = self.audio_patchifier.get_patch_grid_bounds(frames_a, AI_DEVICE)

        # Create audio LatentState
        self.audio_latent_state = LatentState(
            latent=patchified_audio_latent,
            denoise_mask=patchified_audio_mask,
            positions=positions_audio,
            clean_latent=patchified_clean_audio_latent,
        )

    def set_timesteps(
        self,
        infer_steps: Optional[int] = None,
        latent: Optional[torch.Tensor] = None,
    ):
        """
        Set timesteps and generate sigma schedule.

        This method generates a sigma schedule with token-count-dependent shifting,
        similar to WAN scheduler's set_timesteps method.

        Args:
            infer_steps: Number of inference steps (defaults to self.infer_steps)
            device: Device for timesteps tensor
            latent: Optional latent tensor for token count calculation
        """
        if infer_steps is None:
            infer_steps = self.infer_steps

        # Calculate token count for shift adjustment
        if latent is not None:
            tokens = math.prod(latent.shape[1:])
        else:
            tokens = self.max_shift_anchor

        # Generate linear sigma schedule (aligned with source code)
        sigmas = torch.linspace(1.0, 0.0, infer_steps + 1)

        # Apply token-count-dependent shift
        x1 = self.base_shift_anchor
        x2 = self.max_shift_anchor
        mm = (self.max_shift - self.base_shift) / (x2 - x1)
        b = self.base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        # Apply shift transformation (aligned with source code)
        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        if self.stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - self.terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        self.sigmas = sigmas.to(torch.float32).to(AI_DEVICE)

    def post_process_latent(self, denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Blend denoised output with clean state based on mask."""
        return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)

    def to_velocity(
        self,
        sample: torch.Tensor,
        sigma: float | torch.Tensor,
        denoised_sample: torch.Tensor,
        calc_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Convert the sample and its denoised version to velocity.
        Returns:
            Velocity
        """
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(calc_dtype).item()
        if sigma == 0:
            raise ValueError("Sigma can't be 0.0")
        return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)

    def step_post(self):
        self.v_noise_pred = self.post_process_latent(self.v_noise_pred, self.video_latent_state.denoise_mask, self.video_latent_state.clean_latent)
        self.a_noise_pred = self.post_process_latent(self.a_noise_pred, self.audio_latent_state.denoise_mask, self.audio_latent_state.clean_latent)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = sigma_next - sigma

        v_velocity = self.to_velocity(self.video_latent_state.latent, sigma, self.v_noise_pred)
        a_velocity = self.to_velocity(self.audio_latent_state.latent, sigma, self.a_noise_pred)

        v_latent = self.video_latent_state.latent.to(torch.float32) + v_velocity.to(torch.float32) * dt
        a_latent = self.audio_latent_state.latent.to(torch.float32) + a_velocity.to(torch.float32) * dt
        self.video_latent_state.latent = v_latent.to(self.video_latent_state.latent.dtype)
        self.audio_latent_state.latent = a_latent.to(self.audio_latent_state.latent.dtype)

        # Unpatchify latents on the final step (aligned with source code)
        if self.step_index == self.infer_steps - 1:
            channels_v, frames_v, height_v, width_v = self.video_latent_shape_orig
            channels_a, frames_a, mel_bins_a = self.audio_latent_shape_orig

            self.video_latent_state.latent = self.video_patchifier.unpatchify(self.video_latent_state.latent, frames_v, height_v, width_v)
            self.audio_latent_state.latent = self.audio_patchifier.unpatchify(self.audio_latent_state.latent, channels=channels_a, mel_bins=mel_bins_a)

    def clear(self):
        """Clear scheduler state."""
        self.audio_latents = None
        self.video_latent_state = None
        self.audio_latent_state = None
        self.v_noise_pred = None
        self.a_noise_pred = None
        self.sigmas = None

    def video_timesteps_from_mask(self) -> torch.Tensor:
        """Compute timesteps from a denoise mask and sigma value.
        Multiplies the denoise mask by sigma to produce timesteps for each position
        in the latent state. Areas where the mask is 0 will have zero timesteps.
        """
        return self.video_latent_state.denoise_mask * self.sigmas[self.step_index]

    def audio_timesteps_from_mask(self) -> torch.Tensor:
        """Compute timesteps from a denoise mask and sigma value.
        Multiplies the denoise mask by sigma to produce timesteps for each position
        in the latent state. Areas where the mask is 0 will have zero timesteps.
        """
        return self.audio_latent_state.denoise_mask * self.sigmas[self.step_index]

    def get_final_latents(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get final latents after unpatchifying (aligned with source code).

        Returns:
            Tuple of (video_latent, audio_latent) in unpatchified form
        """
        channels_v, frames_v, height_v, width_v = self.video_latent_shape_orig
        channels_a, frames_a, mel_bins_a = self.audio_latent_shape_orig

        video_latent = self.video_patchifier.unpatchify(self.video_latent_state.latent, frames_v, height_v, width_v)
        audio_latent = self.audio_patchifier.unpatchify(self.audio_latent_state.latent, channels=channels_a, mel_bins=mel_bins_a)

        return video_latent, audio_latent
