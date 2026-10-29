"""Krea Realtime's request RNG layout, warped schedule and context refill."""

import torch

from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


def realtime_schedule(steps, shift=5.0):
    if steps < 2:
        raise ValueError("Krea Realtime requires at least two denoising steps")
    sigma = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    zero_padded = torch.cat((sigma * 1000, sigma.new_zeros(1)))
    indices = 1000 - torch.linspace(1000, 0, steps, dtype=torch.float32).to(torch.long)
    return sigma, zero_padded[indices]


class WanRealtimeVideoScheduler(WanSFScheduler):
    def __init__(self, config):
        ar = config["ar_config"]
        self.context_frames = int(ar.get("context_frames", 3))
        if self.context_frames != 3 or ar.get("num_frame_per_chunk", 3) != 3:
            raise ValueError("The released Krea Realtime policy uses context_frames=3 and num_frame_per_chunk=3")
        if ar.get("local_attn_size", 6) != 6 or ar.get("sink_size", 0) != 0:
            raise ValueError("Krea Realtime requires local_attn_size=6 and sink_size=0")
        ar.update(dict(num_frame_per_chunk=3, local_attn_size=6, sink_size=0))
        _, timesteps = realtime_schedule(int(config.get("infer_steps", 4)), float(config.get("sample_shift", 5.0)))
        ar["denoising_step_list"] = timesteps.tolist()
        super().__init__(config)
        self.clear()

    def clear(self):
        super().clear()
        self.is_context_prefill = False
        self.local_frame_start = 0

    def _prepare_denoise_schedule(self):
        sigma, steps = realtime_schedule(self.infer_steps, self.sample_shift)
        self.sigmas_sf = sigma.to(AI_DEVICE)
        self.timesteps_sf = (sigma * 1000).to(AI_DEVICE)
        self.denoising_step_list = steps.tolist()

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.generator = torch.Generator(AI_DEVICE).manual_seed(seed)
        channels, frames, height, width = latent_shape
        # Official GenerationSession samples contiguous [B,T,C,H,W], then
        # permutes at the model boundary; C,T ordering changes seeded noise.
        self.latents = torch.randn((1, frames, channels, height, width), generator=self.generator, device=AI_DEVICE, dtype=torch.bfloat16)[0].transpose(0, 1).contiguous()
        self.noise_pred = torch.zeros_like(self.latents)
        self.stream_output = None
        self.is_context_prefill = False
        self.local_frame_start = 0
        self._prepare_denoise_schedule()

    def step_pre(self, seg_index, step_index, is_rerun=False):
        super().step_pre(seg_index, step_index, is_rerun)
        self.local_frame_start = 0 if seg_index == 0 else self.context_frames

    def prepare_context(self, context, seg_index):
        self.seg_index = seg_index
        self.step_index = 0
        self.is_rerun = True
        self.is_context_prefill = True
        self.local_frame_start = 0
        self.latents_input = context
        self.timestep_input = torch.zeros((1, context.shape[1]), dtype=torch.float32, device=context.device)

    def _step_post_denoise(self, seg_start, seg_end):
        flow = self.noise_pred[:, seg_start:seg_end].transpose(0, 1)
        sample = self.latents_input.transpose(0, 1)
        timestep = self.timestep_input.flatten().double()
        sigma = self._sigma_at_timestep(timestep).reshape(-1, 1, 1, 1).double()
        clean = (sample.double() - sigma * flow.double()).to(flow.dtype)
        if self.step_index + 1 < self.infer_steps:
            next_t = torch.full((clean.shape[0],), self.denoising_step_list[self.step_index + 1], device=clean.device, dtype=torch.float32)
            index = (self.timesteps_sf[None] - next_t[:, None]).abs().argmin(dim=1)
            next_sigma = self.sigmas_sf[index].reshape(-1, 1, 1, 1)
            noise = torch.randn(clean.shape, generator=self.generator, device=clean.device, dtype=torch.bfloat16)
            clean = ((1 - next_sigma) * clean + next_sigma * noise).to(noise.dtype)
        self.latents[:, seg_start:seg_end] = clean.transpose(0, 1)
        if self.step_index + 1 == self.infer_steps:
            self.stream_output = clean.transpose(0, 1)
