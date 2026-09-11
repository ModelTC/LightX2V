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


def _make_schedule(infer_steps: int, shift: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    if infer_steps < 1:
        raise ValueError(f"MiniMax-H3 infer_steps must be at least 1, got {infer_steps}")
    base = torch.linspace(1.0, 0.0, infer_steps + 1, dtype=torch.float32, device="cpu")
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = torch.unique_consecutive(sigmas).to(device)
    return sigmas, 1.0 - sigmas[:-1]


def _time_shift_sigma(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Map a sigma from one shifted flow schedule onto another."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _audio_from_carry(
    carry: torch.Tensor,
    video_sigma: torch.Tensor,
    audio_sigma: torch.Tensor,
    audio_scale: float,
) -> torch.Tensor:
    """Convert ComfyUI's video-schedule audio carry to the model's audio stream."""
    video_sigma = video_sigma.to(device=carry.device, dtype=torch.float32)
    if bool(video_sigma == 0):
        return carry.float() / audio_scale
    audio_sigma = audio_sigma.to(device=carry.device, dtype=torch.float32)
    return carry.float() * (audio_sigma / video_sigma)


def _comfy_res_multistep_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    old_denoised: torch.Tensor | None,
    old_sigma_down: torch.Tensor | None,
) -> torch.Tensor:
    """Apply ComfyUI's deterministic ``res_multistep`` update (eta=0)."""
    sample = sample.float()
    denoised = denoised.float()
    sigma = sigmas[step_index].to(device=sample.device, dtype=torch.float32)
    sigma_down = sigmas[step_index + 1].to(device=sample.device, dtype=torch.float32)

    if old_denoised is None or bool(sigma_down == 0):
        derivative = (sample - denoised) / sigma
        return sample + derivative * (sigma_down - sigma)

    old_sigma_down = old_sigma_down.to(device=sample.device, dtype=torch.float32)
    t = -sigma.log()
    t_old = -old_sigma_down.log()
    t_next = -sigma_down.log()
    t_prev = -sigmas[step_index - 1].to(device=sample.device, dtype=torch.float32).log()
    h = t_next - t
    c2 = (t_prev - t_old) / h

    neg_h = -h
    phi1 = torch.expm1(neg_h) / neg_h
    phi2 = (phi1 - 1.0) / neg_h
    b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
    b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
    return torch.exp(-h) * sample + h * (b1 * denoised + b2 * old_denoised.float())


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
        infer_steps = int(config["infer_steps"])
        self.video_shift = float(config.get("video_flow_shift", 12.0))
        self.audio_shift = float(config.get("audio_flow_shift", 3.0))
        self.step_update = config.get("h3_step_update", "reference_blend")
        valid_step_updates = {"reference_blend", "training_euler", "comfyui_res_multistep"}
        if self.step_update not in valid_step_updates:
            raise ValueError(f"MiniMax-H3 h3_step_update must be one of {sorted(valid_step_updates)}, got {self.step_update!r}")
        if self.video_shift <= 0 or self.audio_shift <= 0:
            raise ValueError("MiniMax-H3 flow shifts must be positive")
        self.audio_scale = self.video_shift / self.audio_shift
        self.video_sigmas, self.video_timesteps = _make_schedule(infer_steps, self.video_shift, AI_DEVICE)
        if self.step_update == "comfyui_res_multistep":
            # ComfyUI carries both streams on the video grid and derives audio
            # sigma from that exact float32 value inside the model.
            self.audio_sigmas = _time_shift_sigma(self.video_sigmas, self.video_shift, self.audio_shift)
            self.audio_timesteps = 1.0 - self.audio_sigmas[:-1]
        else:
            self.audio_sigmas, self.audio_timesteps = _make_schedule(infer_steps, self.audio_shift, AI_DEVICE)
        if self.video_timesteps.numel() != self.audio_timesteps.numel():
            raise ValueError("video and audio schedules collapsed to different step counts")
        self.infer_steps = int(self.video_timesteps.numel())
        self.video_latents = None
        self.audio_latents = None
        self.video_noise_pred = None
        self.audio_noise_pred = None
        self.audio_carry_latents = None
        self.old_video_denoised = None
        self.old_audio_carry_denoised = None
        self.old_sigma_down = None
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

        comfyui_mode = self.step_update == "comfyui_res_multistep"
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        condition_video_latents = condition_video_latents or []
        condition_audio_latents = condition_audio_latents or []
        condition_video_rows = []
        for clean in condition_video_latents:
            clean_rows_cpu = patchify_video_latents(clean.float(), patch_size)
            if comfyui_mode:
                # ComfyUI restarts the same CPU stream for every visual
                # condition and draws in patch-row order. These draws are
                # independent from the target video/audio noise stream. Its
                # VAE returns conditions on the intermediate (CPU) device, so
                # preserve the CPU mix before moving the completed rows.
                condition_generator = torch.Generator(device="cpu").manual_seed(int(seed))
                noise_rows = torch.randn(clean_rows_cpu.shape, generator=condition_generator, device="cpu", dtype=torch.float32)
                mixed_rows = KEYFRAME_NOISE_AUG * clean_rows_cpu + (1.0 - KEYFRAME_NOISE_AUG) * noise_rows
                condition_video_rows.append(mixed_rows.to(AI_DEVICE))
            else:
                clean_rows = clean_rows_cpu.to(AI_DEVICE)
                noise = torch.randn(clean.shape, generator=self.generator, device="cpu", dtype=torch.float32)
                noise_rows = patchify_video_latents(noise.to(AI_DEVICE), patch_size)
                # Match Diffusers' ``scheduler.scale_noise`` exactly: the
                # conditioning VAE rows are moved first and mixed on the
                # execution device, with the scalar represented in the sample
                # dtype.
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
        audio_in_channels = int(self.config.get("audio_in_channels", 32))
        if comfyui_mode:
            # ComfyUI samples the nested audio tensor after video as
            # [1, 32, 2, T], then MiniMax packs it channel-major.
            target_audio = torch.randn(
                (1, audio_in_channels, AUDIO_CHANNELS, num_audio_latents),
                generator=self.generator,
                device="cpu",
                dtype=torch.float32,
            )
            target_audio_rows = target_audio[0].permute(1, 2, 0).reshape(num_audio_latents * AUDIO_CHANNELS, audio_in_channels).contiguous()
        else:
            target_audio_rows = torch.randn(
                (num_audio_latents * AUDIO_CHANNELS, audio_in_channels),
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
        if comfyui_mode:
            self.audio_carry_latents = self.audio_latents[self.num_condition_audio_rows :].clone()
        self.step_index = 0
        self.video_noise_pred = None
        self.audio_noise_pred = None
        self.old_video_denoised = None
        self.old_audio_carry_denoised = None
        self.old_sigma_down = None

    def step_pre(self, step_index):
        self.step_index = int(step_index)
        if self.step_update == "comfyui_res_multistep":
            self.audio_latents[self.num_condition_audio_rows :] = _audio_from_carry(
                self.audio_carry_latents,
                self.video_sigmas[self.step_index],
                self.audio_sigmas[self.step_index],
                self.audio_scale,
            )
        video_timestep = float(self.video_timesteps[self.step_index])
        audio_timestep = float(self.audio_timesteps[self.step_index])
        unique, inverse = build_row_timesteps(self.layout_cpu, video_timestep, audio_timestep)
        self.unique_timesteps_cpu = unique
        self.unique_timesteps = unique.to(AI_DEVICE)
        self.timestep_indices = inverse.to(AI_DEVICE)

    @staticmethod
    def _step(sample, model_output, timestep, sigmas, step_index, step_update):
        # H3 predicts a data-ward velocity.  Keep the round trip through
        # timestep separate from the stored sigma grid to match the reference.
        sigma_from_timestep = 1.0 - timestep.to(device=sample.device, dtype=sample.dtype)
        denoised = sample + sigma_from_timestep * model_output
        sigma = sigmas[step_index].to(device=sample.device, dtype=torch.float32)
        sigma_next = sigmas[step_index + 1].to(device=sample.device, dtype=torch.float32)
        if step_update == "training_euler":
            # Match MiniMaxH3T2AVDmdTrainer.run_back_simulation exactly.
            return sample.float() + (sigma - sigma_next) * model_output.float()
        ratio = sigma_next / sigma
        return ratio * sample.float() + (1.0 - ratio) * denoised.float()

    def _step_comfyui_res_multistep(self, condition_video_rows: int, condition_audio_rows: int) -> None:
        video_sigma = self.video_sigmas[self.step_index].to(device=self.video_latents.device, dtype=torch.float32)
        audio_sigma = self.audio_sigmas[self.step_index].to(device=self.audio_latents.device, dtype=torch.float32)
        target_video = self.video_latents[condition_video_rows:].float()
        target_audio = self.audio_latents[condition_audio_rows:].float()
        video_velocity = self.video_noise_pred[condition_video_rows:].float()
        audio_velocity = self.audio_noise_pred[condition_audio_rows:].float()

        video_denoised = target_video + video_sigma * video_velocity
        audio_carry_velocity = (self.audio_scale - 1.0) * target_audio
        audio_carry_velocity = audio_carry_velocity + (1.0 + (self.audio_scale - 1.0) * audio_sigma) * audio_velocity
        audio_carry_denoised = self.audio_carry_latents.float() + video_sigma * audio_carry_velocity

        next_video = _comfy_res_multistep_step(
            target_video,
            video_denoised,
            self.video_sigmas,
            self.step_index,
            self.old_video_denoised,
            self.old_sigma_down,
        )
        next_audio_carry = _comfy_res_multistep_step(
            self.audio_carry_latents,
            audio_carry_denoised,
            self.video_sigmas,
            self.step_index,
            self.old_audio_carry_denoised,
            self.old_sigma_down,
        )

        self.video_latents[condition_video_rows:] = next_video
        self.audio_carry_latents = next_audio_carry
        video_sigma_next = self.video_sigmas[self.step_index + 1]
        audio_sigma_next = self.audio_sigmas[self.step_index + 1]
        self.audio_latents[condition_audio_rows:] = _audio_from_carry(
            next_audio_carry,
            video_sigma_next,
            audio_sigma_next,
            self.audio_scale,
        )
        self.old_video_denoised = video_denoised
        self.old_audio_carry_denoised = audio_carry_denoised
        self.old_sigma_down = video_sigma_next.to(device=next_video.device, dtype=torch.float32)

    def step_post(self):
        if self.video_noise_pred is None or self.audio_noise_pred is None:
            raise RuntimeError("MiniMax-H3 transformer did not populate both velocity predictions")
        condition_video_rows = self.layout.num_condition_video_rows
        condition_audio_rows = self.layout.num_condition_audio_rows
        if self.step_update == "comfyui_res_multistep":
            self._step_comfyui_res_multistep(condition_video_rows, condition_audio_rows)
            return
        self.video_latents[condition_video_rows:] = self._step(
            self.video_latents[condition_video_rows:],
            self.video_noise_pred[condition_video_rows:].float(),
            self.video_timesteps[self.step_index],
            self.video_sigmas,
            self.step_index,
            self.step_update,
        )
        self.audio_latents[condition_audio_rows:] = self._step(
            self.audio_latents[condition_audio_rows:],
            self.audio_noise_pred[condition_audio_rows:].float(),
            self.audio_timesteps[self.step_index],
            self.audio_sigmas,
            self.step_index,
            self.step_update,
        )

    def clear(self):
        for name in (
            "video_latents",
            "audio_latents",
            "video_noise_pred",
            "audio_noise_pred",
            "audio_carry_latents",
            "old_video_denoised",
            "old_audio_carry_denoised",
            "old_sigma_down",
            "layout",
            "layout_cpu",
            "unique_timesteps_cpu",
            "unique_timesteps",
            "timestep_indices",
        ):
            setattr(self, name, None)
        self.generator = None
