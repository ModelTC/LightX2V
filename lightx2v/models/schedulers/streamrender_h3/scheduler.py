"""Native two-step StreamRender-H3 AV window scheduler.

The scheduler owns the fixed-window plan, per-stage prefix banks, and x0/DDIM
updates. Input encoding and transformer execution belong to the runner/model.
"""

import torch

from lightx2v.models.networks.minimax_h3.packing import patchify_video_latents
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.models.schedulers.streamrender_h3.geometry import (
    build_window_layout,
    build_window_plan,
    frame_boundary,
    window_starts,
)


def _shifted_timesteps(shift: float) -> torch.Tensor:
    base = torch.tensor([1.0, 0.5], dtype=torch.float32)
    shifted = shift * base / (1 + (shift - 1) * base)
    # Released T=1.0 is continuous, so both shifted stages retain their fractions.
    return shifted


def _ddim_x0_step(latents: torch.Tensor, prediction: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor) -> torch.Tensor:
    """Match the release's linear-interpolation x0 sampler operation order."""
    current = latents.float()
    estimate = prediction.float()
    noise = (current - (1 - t) * estimate) / t
    return ((1 - next_t) * estimate + next_t * noise).to(latents.dtype)


class StreamRenderH3Scheduler(BaseScheduler):
    """Stage and window state consumed by a native StreamRender-H3 model."""

    def __init__(self, config):
        super().__init__(config)
        if self.infer_steps != 2:
            raise ValueError("released StreamRender-H3 uses exactly two denoising steps")
        self.video_timesteps = _shifted_timesteps(6)
        self.audio_timesteps = _shifted_timesteps(1.5)
        self.prefix_banks = {}
        self.prefix_bank = None
        self.prefix_bank_key = None
        self.window_plans = ()
        self.prefix_cache_capacity = 0
        self.plan = None
        self.layout = None
        self.video_latents = None
        self.audio_latents = None
        self.picture_latents = None
        self.reference_latents = None
        self.selected_video_latents = None
        self.selected_audio_latents = None

    def prepare_request(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        reference_height: int,
        reference_width: int,
        text_token_tags: torch.Tensor,
        picture_latent_size: tuple[int, int] | None = None,
    ) -> None:
        if any(dim % 32 for dim in (height, width, reference_height, reference_width)):
            raise ValueError("StreamRender-H3 canvas dimensions must be multiples of 32")
        starts = window_starts(num_frames)
        latent_count = next(index for index in range(1, num_frames + 1) if frame_boundary(index) == num_frames)
        target_size = (height // 16, width // 16)
        reference_size = (reference_height // 16, reference_width // 16)
        self.text_token_tags = text_token_tags.cpu().to(torch.long)
        self.picture_latent_size = picture_latent_size or target_size
        self.window_plans = tuple(
            build_window_plan(
                start,
                latent_count if index + 1 == len(starts) else starts[index + 1],
                target_latent_size=target_size,
                reference_latent_size=reference_size,
                audio_length=round(num_frames * 40 / 24),
            )
            for index, start in enumerate(starts)
        )
        continuation_layouts = (build_window_layout(plan, self.text_token_tags, picture_latent_size=self.picture_latent_size, prefix_cached=True) for plan in self.window_plans[1:])
        self.prefix_cache_capacity = max((layout.prefix_length + layout.valid_sequence_length for layout in continuation_layouts), default=0)
        self.prefix_banks.clear()
        self.prefix_bank = None
        self.prefix_bank_key = None
        self.plan = None
        self.layout = None
        self.video_latents = None
        self.audio_latents = None
        self.picture_latents = None
        self.reference_latents = None
        self.selected_video_latents = None
        self.selected_audio_latents = None

    def prepare_window(
        self,
        index: int,
        *,
        picture_latents: torch.Tensor,
        reference_latents: torch.Tensor,
        selected_video_latents: torch.Tensor,
        selected_audio_latents: torch.Tensor,
    ) -> None:
        """Bind one selected window, including immutable sink/history media."""
        self.plan = self.window_plans[index]
        self.window_index = index
        self.picture_latents = picture_latents
        self.reference_latents = reference_latents
        self.selected_video_latents = selected_video_latents
        self.selected_audio_latents = selected_audio_latents
        self.video_latents = selected_video_latents[:, self.plan.video_noisy_mask].clone()
        self.audio_latents = selected_audio_latents[:, :, self.plan.audio_noisy_mask].clone()
        self.video_noise_pred = None
        self.audio_noise_pred = None

    def _pack_window_tokens(self) -> None:
        """Use original native row order; the model selects layout.source_rows."""
        plan = self.plan
        video = self.selected_video_latents.clone()
        audio = self.selected_audio_latents.clone()
        video[:, plan.video_noisy_mask] = self.video_latents
        audio[:, :, plan.audio_noisy_mask] = self.audio_latents
        picture_rows = patchify_video_latents(self.picture_latents.unsqueeze(0))
        reference_rows = patchify_video_latents(self.reference_latents.unsqueeze(0))
        video_rows = patchify_video_latents(video.unsqueeze(0))
        audio_rows = audio.permute(0, 2, 1).reshape(-1, audio.shape[1])
        text_count = self.text_token_tags.numel()
        image_count = picture_rows.shape[0] + reference_rows.shape[0]
        sound_count = audio_rows.shape[0]
        self.video_tokens = torch.cat(
            (
                video_rows.new_zeros((text_count, video_rows.shape[1])),
                picture_rows,
                reference_rows,
                video_rows.new_zeros((sound_count, video_rows.shape[1])),
                video_rows,
            )
        )
        self.audio_tokens = torch.cat(
            (
                audio_rows.new_zeros((text_count + image_count, audio_rows.shape[1])),
                audio_rows,
                audio_rows.new_zeros((video_rows.shape[0], audio_rows.shape[1])),
            )
        )
        clean_t = self.video_timestep.new_tensor(0.001)
        video_times = torch.where(plan.video_noisy_mask, self.video_timestep, clean_t)
        audio_times = torch.where(plan.audio_noisy_mask, self.audio_timestep, clean_t)
        full_timesteps = torch.cat(
            (
                self.video_timestep.expand(text_count),
                clean_t.expand(image_count),
                audio_times.repeat(2),
                video_times.repeat_interleave(plan.target_frame_rows),
            )
        )
        self.row_timesteps = full_timesteps[self.layout.source_rows]
        self.timestep_conditioning_mask = torch.cat(
            (
                torch.ones(text_count, dtype=torch.bool),
                torch.zeros(image_count, dtype=torch.bool),
                plan.audio_noisy_mask.repeat(2),
                plan.video_noisy_mask.repeat_interleave(plan.target_frame_rows),
            )
        )[self.layout.source_rows]

    def step_pre(self, step_index: int, branch: int = 0) -> None:
        self.step_index = int(step_index)
        self.branch = int(branch)
        video_t = float(self.video_timesteps[self.step_index])
        audio_t = float(self.audio_timesteps[self.step_index])
        self.video_timestep = self.video_timesteps[self.step_index]
        self.audio_timestep = self.audio_timesteps[self.step_index]
        self.prefix_bank_key = (self.branch, video_t, audio_t)
        self.prefix_bank = None if self.plan.is_bootstrap else self.prefix_banks.get(self.prefix_bank_key)
        self.layout = build_window_layout(
            self.plan,
            self.text_token_tags,
            picture_latent_size=self.picture_latent_size,
            prefix_cached=self.prefix_bank is not None,
        )
        self.capture_prefix = self.layout.capture_prefix
        self.prefix_length = self.layout.prefix_length
        self.valid_sequence_length = self.layout.valid_sequence_length
        self._pack_window_tokens()

    def set_prefix_bank(self, bank) -> None:
        """Retain one read-only sink bank for this branch and paired timestep."""
        self.prefix_banks[self.prefix_bank_key] = bank
        self.prefix_bank = bank

    def step_post(self) -> None:
        video_next = self.video_timesteps[self.step_index + 1] if self.step_index + 1 < self.infer_steps else self.video_timesteps.new_zeros(())
        audio_next = self.audio_timesteps[self.step_index + 1] if self.step_index + 1 < self.infer_steps else self.audio_timesteps.new_zeros(())
        self.video_latents = _ddim_x0_step(self.video_latents, self.video_noise_pred, self.video_timestep, video_next)
        self.audio_latents = _ddim_x0_step(self.audio_latents, self.audio_noise_pred, self.audio_timestep, audio_next)

    def finish_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_latents, self.audio_latents

    def clear(self) -> None:
        self.prefix_banks.clear()
        self.prefix_bank = None
        self.prefix_bank_key = None
        self.window_plans = ()
        self.prefix_cache_capacity = 0
        self.plan = None
        self.layout = None
        self.video_latents = None
        self.audio_latents = None
        self.picture_latents = None
        self.reference_latents = None
        self.selected_video_latents = None
        self.selected_audio_latents = None
        self.video_tokens = None
        self.audio_tokens = None
        self.row_timesteps = None


__all__ = ["StreamRenderH3Scheduler"]
