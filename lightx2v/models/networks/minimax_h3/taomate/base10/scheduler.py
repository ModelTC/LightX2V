"""Request-local nine-forward Base10 audio denoising state."""

import torch

from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device
from lightx2v.models.schedulers.taomate_h3.geometry import request_plan
from lightx2v_platform.base.global_var import AI_DEVICE

from .packing import build_audio_only_layout

_FULL_VIDEO_LATENTS = 37
_FULL_AUDIO_LATENTS = 207
_REFERENCE_LATENTS = 40
_MILESTONE_STATES = (3, 6, 9)


class TaoMateBase10Scheduler(MiniMaxH3Scheduler):
    """Match the official FL2VA noise order, clocks, and frozen audio tail."""

    def __init__(self, config):
        native_config = dict(config)
        native_config.update(infer_steps=9, video_flow_shift=12.0, audio_flow_shift=3.0)
        super().__init__(native_config)
        self.milestones = ()
        self.refined_text = None

    def prepare_request(self, *, text_token_tags, audio_seed, request_index, previous_clean, height, width):
        plan = request_plan(request_index)
        latent_height, latent_width = int(height) // 16, int(width) // 16
        if height % 32 or width % 32:
            raise ValueError("Base10 height and width must be divisible by 32")
        previous_count = None if previous_clean is None else int(previous_clean.shape[0]) // 2
        if previous_clean is not None and tuple(previous_clean.shape) != (2 * previous_count, 32):
            raise ValueError("Previous Base10 clean audio must have shape [2 * audio_latents, 32]")
        if request_index == 0 and previous_clean is not None or request_index > 0 and previous_clean is None:
            raise ValueError("Only continuation requests have a previous clean audio tail")

        generator = torch.Generator(device="cpu").manual_seed(int(audio_seed))
        torch.randn((1, 24, _FULL_VIDEO_LATENTS, latent_height, latent_width), generator=generator, dtype=torch.float32)
        full_audio = torch.randn((2 * _FULL_AUDIO_LATENTS, 32), generator=generator, dtype=torch.float32)
        active_count = plan.audio_latents
        target = full_audio.view(2, _FULL_AUDIO_LATENTS, 32)[:, _FULL_AUDIO_LATENTS - active_count :].contiguous().view(-1, 32)
        if previous_clean is None:
            initial = target
        else:
            reference = previous_clean.cpu().float().view(2, previous_count, 32)[:, -_REFERENCE_LATENTS:].contiguous().view(-1, 32)
            initial = torch.cat((reference, target), dim=0)

        self.layout_cpu = build_audio_only_layout(
            text_token_tags,
            active_count,
            latent_height,
            latent_width,
            previous_audio_latents=previous_count,
            reference_latents=_REFERENCE_LATENTS,
        )
        self.layout = _layout_to_device(self.layout_cpu, AI_DEVICE)
        self.audio_latents = initial.to(AI_DEVICE, dtype=torch.float32, copy=True)
        self.video_latents = torch.empty((0, 96), device=AI_DEVICE, dtype=torch.float32)
        self.audio_target_start = self.layout.num_condition_audio_rows
        self.audio_scratch = torch.empty_like(self.audio_latents[self.audio_target_start :])
        self.num_audio_latents = active_count
        self.num_condition_audio_rows = self.audio_target_start
        self.num_condition_video_rows = 0
        self.step_index = 0
        self.video_noise_pred = None
        self.audio_noise_pred = None
        self.refined_text = None
        self.milestones = []

    def step_post(self):
        """Apply the official data-ward Euler update to target rows only."""
        index = self.step_index
        state = self.audio_latents[self.audio_target_start :]
        velocity = self.audio_noise_pred[self.audio_target_start :].float()
        sigma_t = 1.0 - self.audio_timesteps[index]
        ratio = self.audio_sigmas[index + 1] / self.audio_sigmas[index]
        scratch = self.audio_scratch
        torch.mul(sigma_t, velocity, out=scratch)
        torch.add(state, scratch, out=scratch)
        torch.mul(1.0 - ratio, scratch, out=velocity)
        torch.mul(ratio, state, out=state)
        torch.add(state, velocity, out=state)
        if index + 1 in _MILESTONE_STATES:
            self.milestones.append(state.detach().to(device="cpu", dtype=torch.float32).contiguous())

    def finish_request(self):
        if len(self.milestones) != len(_MILESTONE_STATES):
            raise RuntimeError("Base10 did not produce all three audio milestones")
        return tuple(self.milestones)

    def clear(self):
        super().clear()
        self.refined_text = None
        self.audio_scratch = None
        self.milestones = ()
