from dataclasses import replace

import torch

from lightx2v.models.networks.minimax_h3.packing import (
    AUDIO_CHANNELS,
    KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedSequence,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    validate_t2av_geometry,
    video_latent_num_frames,
)
from lightx2v.models.networks.minimax_h3.packing_ref2av import build_ref2av_packed_sequence
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


def _make_schedule(num_grid_points: int, shift: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    if num_grid_points < 2:
        raise ValueError(f"MiniMax-H3 infer_steps must be at least 2, got {num_grid_points}")
    base = torch.linspace(1.0, 0.0, num_grid_points, dtype=torch.float32, device="cpu")
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = torch.unique_consecutive(sigmas).to(device)
    return sigmas, 1.0 - sigmas[:-1]


def _layout_to_device(layout: MiniMaxH3PackedSequence, device) -> MiniMaxH3PackedSequence:
    return replace(
        layout,
        position_ids=layout.position_ids.to(device),
        token_tags=layout.token_tags.to(device),
        video_indices=layout.video_indices.to(device),
        audio_indices=layout.audio_indices.to(device),
        text_indices=layout.text_indices.to(device),
    )


class MiniMaxH3Scheduler(BaseScheduler):
    """Two synchronized MiniMax-H3 data-ward rectified-flow schedules."""

    def __init__(self, config):
        super().__init__(config)
        self.num_grid_points = int(config["infer_steps"])
        self.video_shift = float(config.get("video_flow_shift", 12.0))
        self.audio_shift = float(config.get("audio_flow_shift", 3.0))
        if self.video_shift <= 0 or self.audio_shift <= 0:
            raise ValueError("MiniMax-H3 flow shifts must be positive")
        self.video_sigmas, self.video_timesteps = _make_schedule(self.num_grid_points, self.video_shift, AI_DEVICE)
        self.audio_sigmas, self.audio_timesteps = _make_schedule(self.num_grid_points, self.audio_shift, AI_DEVICE)
        if self.video_timesteps.numel() != self.audio_timesteps.numel():
            raise ValueError("video and audio schedules collapsed to different step counts")
        # The user-facing value counts sigma grid points including terminal 0.
        # LightX2V's loop count is the number of model evaluations.
        self.infer_steps = int(self.video_timesteps.numel())
        self.video_latents = None
        self.audio_latents = None
        self.video_noise_pred = None
        self.audio_noise_pred = None
        self.layout = None
        self.layout_cpu = None

    def prepare(
        self,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        text_token_tags: torch.Tensor,
        *,
        keyframe_anchors: tuple[str, ...] = (),
        condition_video_latents: list[torch.Tensor] | None = None,
        condition_audio_latents: list[torch.Tensor] | None = None,
        references=None,
    ):
        validate_t2av_geometry(num_frames, height, width)
        latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // int(self.config.get("vae_spatial_scale_factor", 16))
        latent_width = width // int(self.config.get("vae_spatial_scale_factor", 16))
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = tuple(self.config.get("patch_size", (1, 2, 2)))

        # The released pipeline uses one CPU random stream even when inference
        # runs on CUDA: float32 video noise first, then channel-major audio.
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        condition_video_latents = condition_video_latents or []
        condition_audio_latents = condition_audio_latents or []
        condition_video_rows = []
        for clean in condition_video_latents:
            noise = torch.randn(clean.shape, generator=self.generator, device="cpu", dtype=torch.float32)
            clean_rows = patchify_video_latents(clean.float(), patch_size).to(AI_DEVICE)
            noise_rows = patchify_video_latents(noise.to(AI_DEVICE), patch_size)
            # Match Diffusers' ``scheduler.scale_noise`` exactly: the
            # conditioning VAE rows are moved first and mixed on the execution
            # device, with the scalar represented in the sample dtype.  Doing
            # this FP32 operation on CPU differs by an ulp on CUDA and that
            # perturbation is amplified by the 48-layer denoiser.
            timestep = torch.tensor(KEYFRAME_NOISE_AUG, dtype=clean_rows.dtype, device=clean_rows.device)
            condition_video_rows.append(timestep * clean_rows + (1.0 - timestep) * noise_rows)

        video_noise = torch.randn(
            (1, int(self.config.get("in_channels", 24)), latent_frames, latent_height, latent_width),
            generator=self.generator,
            device="cpu",
            dtype=torch.float32,
        )
        target_video_rows = patchify_video_latents(video_noise, patch_size)
        self.video_latents = torch.cat(condition_video_rows + [target_video_rows.to(AI_DEVICE)])
        target_audio_rows = torch.randn(
            (
                num_audio_latents * AUDIO_CHANNELS,
                int(self.config.get("audio_in_channels", 32)),
            ),
            generator=self.generator,
            device="cpu",
            dtype=torch.float32,
        )
        condition_audio_rows = [latent.transpose(1, 2).reshape(-1, latent.shape[1]).float() for latent in condition_audio_latents]
        self.audio_latents = torch.cat(condition_audio_rows + [target_audio_rows]).to(AI_DEVICE)

        if references is None:
            self.layout_cpu = build_packed_sequence(text_token_tags.cpu(), latent_frames, latent_height, latent_width, num_audio_latents, patch_size, keyframe_anchors)
        else:
            self.layout_cpu = build_ref2av_packed_sequence(text_token_tags.cpu(), references, latent_frames, latent_height, latent_width, num_audio_latents, patch_size)
        self.layout = _layout_to_device(self.layout_cpu, AI_DEVICE)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.num_audio_latents = num_audio_latents
        self.num_condition_video_rows = self.layout_cpu.num_condition_video_rows
        self.num_condition_audio_rows = self.layout_cpu.num_condition_audio_rows
        self.step_index = 0
        self.video_noise_pred = None
        self.audio_noise_pred = None

    def step_pre(self, step_index):
        self.step_index = int(step_index)
        video_timestep = float(self.video_timesteps[self.step_index])
        audio_timestep = float(self.audio_timesteps[self.step_index])
        unique, inverse = build_row_timesteps(self.layout_cpu, video_timestep, audio_timestep)
        self.unique_timesteps = unique.to(AI_DEVICE)
        self.timestep_indices = inverse.to(AI_DEVICE)

    @staticmethod
    def _step(sample, model_output, timestep, sigmas, step_index):
        # H3 predicts a data-ward velocity.  Keep the round trip through
        # timestep separate from the stored sigma grid to match the reference.
        sigma_from_timestep = 1.0 - timestep.to(device=sample.device, dtype=sample.dtype)
        denoised = sample + sigma_from_timestep * model_output
        sigma = sigmas[step_index].to(device=sample.device, dtype=torch.float32)
        sigma_next = sigmas[step_index + 1].to(device=sample.device, dtype=torch.float32)
        ratio = sigma_next / sigma
        return ratio * sample.float() + (1.0 - ratio) * denoised.float()

    def step_post(self):
        if self.video_noise_pred is None or self.audio_noise_pred is None:
            raise RuntimeError("MiniMax-H3 transformer did not populate both velocity predictions")
        condition_video_rows = self.layout.num_condition_video_rows
        condition_audio_rows = self.layout.num_condition_audio_rows
        self.video_latents[condition_video_rows:] = self._step(
            self.video_latents[condition_video_rows:],
            self.video_noise_pred[condition_video_rows:].float(),
            self.video_timesteps[self.step_index],
            self.video_sigmas,
            self.step_index,
        )
        self.audio_latents[condition_audio_rows:] = self._step(
            self.audio_latents[condition_audio_rows:],
            self.audio_noise_pred[condition_audio_rows:].float(),
            self.audio_timesteps[self.step_index],
            self.audio_sigmas,
            self.step_index,
        )

    def clear(self):
        for name in (
            "video_latents",
            "audio_latents",
            "video_noise_pred",
            "audio_noise_pred",
            "layout",
            "layout_cpu",
            "unique_timesteps",
            "timestep_indices",
        ):
            setattr(self, name, None)
        self.generator = None
