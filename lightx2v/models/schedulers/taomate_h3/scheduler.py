"""Native Stage3 phase and three-step flow state for TaoMate-H3."""

from dataclasses import replace

import torch

from lightx2v.models.networks.minimax_h3.packing import (
    build_packed_sequence,
    patchify_video_latents,
)
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device
from lightx2v.models.schedulers.taomate_h3.geometry import request_plan, video_time
from lightx2v_platform.base.global_var import AI_DEVICE

_DISTILLED_STATE_INDICES = (0, 16, 33, 49)
_FULL_VIDEO_LATENTS = 37
_FULL_AUDIO_LATENTS = 207


def distilled_sigmas(shift):
    base = torch.linspace(1.0, 0.0, 50, dtype=torch.float32)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    return shifted[list(_DISTILLED_STATE_INDICES)].contiguous()


class TaoMateH3Scheduler(MiniMaxH3Scheduler):
    """Own request noise, per-phase packed layout and Base10 audio guides."""

    def __init__(self, config):
        super().__init__(config)
        if int(config["infer_steps"]) != 3:
            raise ValueError("TaoMate-H3 Stage3 uses exactly three denoise forwards")
        self.video_sigmas = distilled_sigmas(self.video_shift).to(AI_DEVICE)
        self.audio_sigmas = distilled_sigmas(self.audio_shift).to(AI_DEVICE)
        self.video_timesteps = 1.0 - self.video_sigmas[:-1]
        self.audio_timesteps = 1.0 - self.audio_sigmas[:-1]
        self.infer_steps = 3
        self.media_time_origin = None
        self.video_offset = 0
        self.audio_offset = 0
        self.video_chunks = []
        self.audio_chunks = []
        self.clean_commit = False
        self.video_anchor = None

    def prepare_request(self, *, request_index, video_noise_seed, audio_noise_seed, height, width, text_token_tags, teacher_milestones):
        plan = request_plan(request_index)
        if height % 32 or width % 32 or min(height, width) not in (480, 768, 1088):
            raise ValueError("TaoMate-H3 canvas needs 32 alignment and a supported short edge")
        latent_height, latent_width = height // 16, width // 16
        video_shape = (1, 24, _FULL_VIDEO_LATENTS, latent_height, latent_width)
        video_generator = torch.Generator(device="cpu").manual_seed(int(video_noise_seed))
        audio_generator = torch.Generator(device="cpu").manual_seed(int(audio_noise_seed))
        video_noise = torch.randn(video_shape, generator=video_generator, dtype=torch.float32)
        torch.randn(video_shape, generator=audio_generator, dtype=torch.float32)
        audio_noise = torch.randn((2 * _FULL_AUDIO_LATENTS, 32), generator=audio_generator, dtype=torch.float32)
        video_rows = patchify_video_latents(video_noise)
        prefix_video = _FULL_VIDEO_LATENTS - plan.video_latents
        prefix_audio = _FULL_AUDIO_LATENTS - plan.audio_latents
        frame_rows = (latent_height // 2) * (latent_width // 2)
        self.request_video_noise = video_rows[prefix_video * frame_rows :]
        self.request_audio_noise = audio_noise.reshape(2, _FULL_AUDIO_LATENTS, 32)[:, prefix_audio:].reshape(-1, 32)
        self.plan = plan
        self.request_index = request_index
        self.text_token_tags = text_token_tags.cpu()
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.frame_rows = frame_rows
        self.num_latent_frames = plan.video_latents
        self.num_audio_latents = plan.audio_latents
        self.num_condition_video_rows = 0
        self.num_condition_audio_rows = 0
        self.teacher_milestones = tuple(tensor.to(AI_DEVICE) for tensor in teacher_milestones)
        if len(self.teacher_milestones) != 3 or any(tuple(tensor.shape) != (2 * plan.audio_latents, 32) for tensor in self.teacher_milestones):
            raise ValueError("Base10 teacher must provide three complete Stage3 audio milestones")
        if self.media_time_origin is None:
            self.media_time_origin = int(self.text_token_tags.numel())
        self.video_chunks.clear()
        self.audio_chunks.clear()

    def _phase_layout(self, phase):
        text_count = int(self.text_token_tags.numel())
        layout = build_packed_sequence(
            self.text_token_tags,
            phase.video_latents,
            self.latent_height,
            self.latent_width,
            phase.audio_latents,
        )
        positions = layout.position_ids.clone()
        text_start = self.media_time_origin + float(video_time(self.video_offset)) - text_count
        positions[layout.text_indices, 0] = text_start + torch.arange(text_count, dtype=torch.float64)
        temporal = torch.tensor(
            [self.media_time_origin + float(video_time(self.video_offset + phase.video_start + index)) for index in range(phase.video_latents)],
            dtype=torch.float64,
        )
        positions[layout.video_indices, 0] = temporal.repeat_interleave(self.frame_rows)
        audio_clock = self.media_time_origin + self.audio_offset + phase.audio_start + torch.arange(phase.audio_latents, dtype=torch.float64)
        positions[layout.audio_indices, 0] = audio_clock.repeat(2)
        used = layout.sequence_length
        padding = (-used) % 64
        token_tags = torch.cat((layout.token_tags, torch.full((padding,), -1, dtype=torch.long)))
        positions = torch.cat((positions, torch.zeros(padding, 3, dtype=torch.float64)))
        layout = replace(layout, sequence_length=used + padding, token_tags=token_tags, position_ids=positions)
        return layout, (0, used, used + padding)

    def prepare_phase(self, phase_index):
        phase = self.plan.phases[phase_index]
        self.phase = phase
        self.layout_cpu, self.cu_seqlens_host = self._phase_layout(phase)
        self.layout = _layout_to_device(self.layout_cpu, AI_DEVICE)
        video_start = phase.video_start * self.frame_rows
        video_stop = phase.video_stop * self.frame_rows
        self.video_latents = self.request_video_noise[video_start:video_stop].to(AI_DEVICE).clone()
        audio = self.request_audio_noise.reshape(2, self.plan.audio_latents, 32)
        self.audio_latents = audio[:, phase.audio_start : phase.audio_stop].reshape(-1, 32).to(AI_DEVICE).clone()
        self.commit_mask = torch.zeros(self.layout.sequence_length, dtype=torch.bool, device=AI_DEVICE)
        self.commit_mask[self.layout.video_indices] = True
        self.commit_mask[self.layout.audio_indices] = True
        self.token_tags = self.layout.token_tags
        self.clean_commit = False

    def step_post(self):
        super().step_post()
        guide = self.teacher_milestones[self.step_index].reshape(2, self.plan.audio_latents, 32)
        self.audio_latents.copy_(guide[:, self.phase.audio_start : self.phase.audio_stop].reshape(-1, 32))

    def prepare_clean_commit(self):
        current = self.video_latents.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.video_anchor is None:
            self.video_anchor = (mean, std)
        else:
            anchor_mean, anchor_std = self.video_anchor
            self.video_latents.copy_(((current - mean) / std * anchor_std + anchor_mean).to(self.video_latents.dtype))
        self.clean_commit = True
        self.unique_timesteps_cpu = torch.tensor([1.0], dtype=torch.float32)
        self.unique_timesteps = self.unique_timesteps_cpu.to(AI_DEVICE)
        self.timestep_indices = torch.zeros(self.layout.sequence_length, dtype=torch.long, device=AI_DEVICE)
        self.step_index = self.infer_steps

    def finish_phase(self):
        self.video_chunks.append(self.video_latents.detach())
        self.audio_chunks.append(self.audio_latents.detach().reshape(2, self.phase.audio_latents, 32))

    def finish_request(self):
        video = torch.cat(self.video_chunks, dim=0)
        audio = torch.cat(self.audio_chunks, dim=1).reshape(-1, 32)
        self.video_offset += self.plan.video_latents
        self.audio_offset += self.plan.audio_latents
        return video, audio

    def clear(self):
        super().clear()
        self.video_chunks.clear()
        self.audio_chunks.clear()
        self.request_video_noise = None
        self.request_audio_noise = None
        self.teacher_milestones = None
        self.media_time_origin = None
        self.video_offset = 0
        self.audio_offset = 0
        self.video_anchor = None


__all__ = ["TaoMateH3Scheduler", "distilled_sigmas"]
