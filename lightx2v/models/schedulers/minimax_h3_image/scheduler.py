"""The six-step, data-ward PDD Euler schedule of MiniMax-H3-Image."""

import json
from dataclasses import replace

import torch

from lightx2v.models.networks.minimax_h3_image.packing import build_image_packed_sequence, patchify_video_latents, unpatchify_video_tokens
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE

REFERENCE_TIMESTEP = 0.999


class MiniMaxH3ImageScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(AI_DEVICE)
        with open(config["pdd_config_path"], encoding="utf-8") as handle:
            pdd_config = json.load(handle)
        if self.infer_steps != 6 or pdd_config["num_inference_steps"] != 6:
            raise ValueError("MiniMax-H3-Image PDD requires exactly six inference steps")
        sample_sigmas = pdd_config["sigmas"]
        # Preserve the reference sampler's round trip through float32
        # timesteps before calculating the Euler deltas.
        self.timesteps_cpu = torch.tensor([1.0 - sigma for sigma in sample_sigmas[:-1]], dtype=torch.float32)
        self.sigmas = torch.cat((1.0 - self.timesteps_cpu, torch.zeros(1))).to(self.device)
        self.timesteps = self.timesteps_cpu.to(self.device)
        self.clear()

    @property
    def target_latents(self):
        return self.video_latents[self.num_condition_video_rows :]

    def unpack_target_latents(self):
        return unpatchify_video_tokens(self.target_latents, 1, self.latent_height, self.latent_width, channels=int(self.config.get("in_channels", 24)), patch_size=self.patch_size)

    def prepare(self, height, width, text_token_tags, reference_latents, target_noise, reference_noises):
        if min(height, width) <= 0 or height % 32 or width % 32:
            raise ValueError("MiniMax-H3-Image canvas height and width must be positive multiples of 32")
        self.clear()
        self.latent_height, self.latent_width = height // 16, width // 16
        self.patch_size = tuple(self.config.get("patch_size", (1, 2, 2)))
        reference_rows, reference_grids = [], []
        for clean, noise in zip(reference_latents, reference_noises):
            reference_grids.append(tuple(clean.shape[-2:]))
            clean_rows = patchify_video_latents(clean.float(), self.patch_size).to(self.device)
            noise_rows = patchify_video_latents(noise.float(), self.patch_size).to(self.device)
            timestep = torch.tensor(REFERENCE_TIMESTEP, dtype=torch.float32, device=self.device)
            reference_rows.append(timestep * clean_rows + (1.0 - timestep) * noise_rows)
        self.layout_cpu = build_image_packed_sequence(text_token_tags.cpu(), reference_grids, self.latent_height, self.latent_width, self.patch_size)
        self.layout = replace(self.layout_cpu, **{name: getattr(self.layout_cpu, name).to(self.device) for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")})
        self.num_condition_video_rows = self.layout.num_condition_video_rows
        target_rows = patchify_video_latents(target_noise.float(), self.patch_size).to(self.device)
        self.video_latents = torch.cat([*reference_rows, target_rows])
        self.latents = self.video_latents
        self.reference_anchors = self.video_latents[: self.num_condition_video_rows].clone()

    def step_pre(self, step_index):
        self.step_index = step_index
        if self.num_condition_video_rows:
            self.video_latents[: self.num_condition_video_rows].copy_(self.reference_anchors)
        row_timesteps = torch.full((self.layout.sequence_length,), float(self.timesteps_cpu[self.step_index]), dtype=torch.float32)
        row_timesteps[self.layout_cpu.video_indices[: self.num_condition_video_rows]] = REFERENCE_TIMESTEP
        unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        self.unique_timesteps_cpu = unique
        self.unique_timesteps = unique.to(self.device)
        self.timestep_indices = inverse.to(self.device)
        self.video_noise_pred = None

    def step_post(self):
        start = self.num_condition_video_rows
        delta = self.sigmas[self.step_index] - self.sigmas[self.step_index + 1]
        self.video_latents[start:] = self.target_latents + delta * self.video_noise_pred[start:].float()

    def clear(self):
        self.generator = None
        self.latents = self.video_latents = self.video_noise_pred = None
        self.layout = self.layout_cpu = self.reference_anchors = None
        self.unique_timesteps = self.unique_timesteps_cpu = self.timestep_indices = None
        self.num_condition_video_rows = 0
        self.step_index = 0
        self.latent_height = self.latent_width = self.patch_size = None
