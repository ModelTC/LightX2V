"""LingBot Fast's FP32 latent/RNG contract around BF16 model execution."""

import torch

from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class WanLingbotFastScheduler(WanSFScheduler):
    def clear(self):
        super().clear()
        self.sample_input = None

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        if self._mode != "index":
            raise ValueError("Released LingBot Fast uses the timesteps_index schedule")
        self.generator = torch.Generator(AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(latent_shape, device=AI_DEVICE, dtype=torch.float32, generator=self.generator)
        self.noise_pred = torch.zeros_like(self.latents)
        self.stream_output = None
        self._prepare_index_schedule()

    def step_pre(self, seg_index, step_index, is_rerun=False):
        self.step_index, self.seg_index, self.is_rerun = step_index, seg_index, is_rerun
        start = seg_index * self.num_frame_per_chunk
        end = min(start + self.num_frame_per_chunk, self.num_output_frames)
        self.sample_input = self.latents[:, start:end]
        # Equivalent to the source's BF16 autocast at the patch projection,
        # while retaining FP32 samples for the x0/scheduler calculation.
        self.latents_input = self.sample_input.to(GET_DTYPE())
        timestep = self.context_noise if is_rerun else self.selected_timesteps[step_index]
        self.timestep_input = torch.full((1, end - start), timestep, dtype=torch.long, device=AI_DEVICE)

    def step_post(self):
        start = self.seg_index * self.num_frame_per_chunk
        end = min(start + self.num_frame_per_chunk, self.num_output_frames)
        flow = self.noise_pred[:, start:end]
        clean = self._convert_flow_pred_to_x0(flow, self.sample_input, self.selected_timesteps[self.step_index])
        if self.step_index + 1 < self.infer_steps:
            noise = torch.randn(clean.shape, generator=self.generator, device=clean.device, dtype=torch.float32)
            clean = self._add_noise(clean, noise, self.selected_timesteps[self.step_index + 1])
        self.latents[:, start:end] = clean
        if self.step_index + 1 == self.infer_steps:
            self.stream_output = clean
