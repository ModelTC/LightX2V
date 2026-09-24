"""H3-World's BF16 sampling contract for image-and-action-to-audio/video."""

import torch

from lightx2v.models.networks.minimax_h3.packing import (
    AUDIO_CHANNELS,
    audio_latent_num_frames,
    build_packed_sequence,
    patchify_video_latents,
    validate_t2av_geometry,
    video_latent_num_frames,
)
from lightx2v.models.networks.minimax_h3.packing_ref2av import build_ref2av_packed_sequence
from lightx2v.models.networks.minimax_h3_world.action import prepare_h3_world_layout
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device, _make_schedule
from lightx2v_platform.base.global_var import AI_DEVICE


def _make_world_schedule(infer_steps: int, shift: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Share H3-World's exact timestep rounding with offline AdaLN caching."""
    sigmas, _ = _make_schedule(infer_steps, shift, "cpu")
    # Keep FP32 multiplication, Python float division/subtraction, then FP32
    # storage in this order to match the training fork's timestep conversion.
    timesteps = torch.tensor([1.0 - float(sigma * 1000) / 1000 for sigma in sigmas[:-1]], dtype=torch.float32)
    return sigmas.to(device), timesteps.to(device)


class MiniMaxH3WorldScheduler(MiniMaxH3Scheduler):
    """Reuse native flow schedules with the action checkpoint's noise and Euler steps."""

    def __init__(self, config):
        super().__init__(config)
        infer_steps = int(config["infer_steps"])
        self.video_sigmas, self.video_timesteps = _make_world_schedule(infer_steps, self.video_shift, AI_DEVICE)
        self.audio_sigmas, self.audio_timesteps = _make_world_schedule(infer_steps, self.audio_shift, AI_DEVICE)

    def prepare(
        self,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        text_token_tags: torch.Tensor,
        *,
        action_token_lengths: list[int],
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
        condition_video_latents = condition_video_latents or []
        condition_audio_latents = condition_audio_latents or []

        # The action checkpoint's training fork draws each modality from a
        # separate CPU generator initialized with the same seed, in BF16.
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        video_noise = torch.randn(
            (1, int(self.config.get("in_channels", 24)), latent_frames, latent_height, latent_width),
            generator=self.generator,
            device="cpu",
            dtype=torch.bfloat16,
        )
        # The fork's BF16 keyframe augmentation scalar rounds 0.999 to 1.0.
        condition_video_rows = [patchify_video_latents(clean.to(torch.bfloat16), patch_size).to(AI_DEVICE) for clean in condition_video_latents]
        target_video_rows = patchify_video_latents(video_noise, patch_size).to(AI_DEVICE)
        self.video_latents = torch.cat(condition_video_rows + [target_video_rows])
        audio_noise = torch.randn(
            (AUDIO_CHANNELS, int(self.config.get("audio_in_channels", 32)), num_audio_latents),
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            device="cpu",
            dtype=torch.bfloat16,
        )
        audio_latents = [*condition_audio_latents, audio_noise]
        self.audio_latents = torch.cat([latent.transpose(1, 2).reshape(-1, latent.shape[1]).to(device=AI_DEVICE, dtype=torch.bfloat16) for latent in audio_latents])

        if references is None:
            self.layout_cpu = build_packed_sequence(text_token_tags.cpu(), latent_frames, latent_height, latent_width, num_audio_latents, patch_size, keyframe_anchors)
        else:
            self.layout_cpu = build_ref2av_packed_sequence(text_token_tags.cpu(), references, latent_frames, latent_height, latent_width, num_audio_latents, patch_size)
        self.layout = prepare_h3_world_layout(
            _layout_to_device(self.layout_cpu, AI_DEVICE),
            int(text_token_tags.numel()) - sum(action_token_lengths),
            action_token_lengths,
            latent_frames,
            (latent_height // patch_size[1]) * (latent_width // patch_size[2]),
        )
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

    @staticmethod
    def _step_h3_world(sample, model_output, sigmas, step_index):
        # Preserve the final integration's CPU-scalar BF16 Euler behavior.
        # Moving this scalar to CUDA changes promotion/rounding.
        delta = (sigmas[step_index + 1] - sigmas[step_index]).cpu()
        return sample.to(torch.bfloat16) + (-model_output.to(torch.bfloat16)) * delta

    def step_post(self):
        if self.video_noise_pred is None or self.audio_noise_pred is None:
            raise RuntimeError("MiniMax-H3-World transformer did not populate both velocity predictions")
        condition_video_rows = self.layout.num_condition_video_rows
        condition_audio_rows = self.layout.num_condition_audio_rows
        self.video_latents[condition_video_rows:] = self._step_h3_world(self.video_latents[condition_video_rows:], self.video_noise_pred[condition_video_rows:], self.video_sigmas, self.step_index)
        self.audio_latents[condition_audio_rows:] = self._step_h3_world(self.audio_latents[condition_audio_rows:], self.audio_noise_pred[condition_audio_rows:], self.audio_sigmas, self.step_index)
