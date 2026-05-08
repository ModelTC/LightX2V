import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

# class WanSFScheduler(WanScheduler):
#     def __init__(self, config):
#         super().__init__(config)
#         self.dtype = torch.bfloat16
#         self.num_frame_per_block = self.config["sf_config"]["num_frame_per_block"]
#         self.num_output_frames = self.config["sf_config"]["num_output_frames"]
#         self.num_blocks = self.num_output_frames // self.num_frame_per_block
#         self.denoising_step_list = self.config["sf_config"]["denoising_step_list"]
#         self.infer_steps = len(self.denoising_step_list)
#         self.all_num_frames = [self.num_frame_per_block] * self.num_blocks
#         self.num_input_frames = 0
#         self.denoising_strength = 1.0
#         self.sigma_max = 1.0
#         self.sigma_min = 0
#         self.sf_shift = self.config["sf_config"]["shift"]
#         self.inverse_timesteps = False
#         self.extra_one_step = True
#         self.reverse_sigmas = False
#         self.num_inference_steps = self.config["sf_config"]["num_inference_steps"]
#         self.context_noise = 0

#     def prepare(self, seed, latent_shape, image_encoder_output=None):
#         self.latents = torch.randn(latent_shape, device=AI_DEVICE, dtype=self.dtype)

#         timesteps = []
#         for frame_block_idx, current_num_frames in enumerate(self.all_num_frames):
#             frame_steps = []

#             for step_index, current_timestep in enumerate(self.denoising_step_list):
#                 timestep = torch.ones([self.num_frame_per_block], device=AI_DEVICE, dtype=torch.int64) * current_timestep
#                 frame_steps.append(timestep)

#             timesteps.append(frame_steps)
#         self.timesteps = timesteps

#         self.noise_pred = torch.zeros(latent_shape, device=AI_DEVICE, dtype=self.dtype)

#         sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * self.denoising_strength
#         if self.extra_one_step:
#             self.sigmas_sf = torch.linspace(sigma_start, self.sigma_min, self.num_inference_steps + 1)[:-1]
#         else:
#             self.sigmas_sf = torch.linspace(sigma_start, self.sigma_min, self.num_inference_steps)
#         if self.inverse_timesteps:
#             self.sigmas_sf = torch.flip(self.sigmas_sf, dims=[0])
#         self.sigmas_sf = self.sf_shift * self.sigmas_sf / (1 + (self.sf_shift - 1) * self.sigmas_sf)
#         if self.reverse_sigmas:
#             self.sigmas_sf = 1 - self.sigmas_sf
#         self.sigmas_sf = self.sigmas_sf.to(AI_DEVICE)

#         self.timesteps_sf = self.sigmas_sf * self.num_train_timesteps
#         self.timesteps_sf = self.timesteps_sf.to(AI_DEVICE)

#         self.stream_output = None

#     def step_pre(self, seg_index, step_index, is_rerun=False):
#         self.step_index = step_index
#         self.seg_index = seg_index

#         if not GET_DTYPE() == GET_SENSITIVE_DTYPE():
#             self.latents = self.latents.to(GET_DTYPE())

#         if not is_rerun:
#             self.timestep_input = torch.stack([self.timesteps[self.seg_index][self.step_index]])
#             self.latents_input = self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)]
#         else:
#             # rerun with timestep zero to update KV cache using clean context
#             self.timestep_input = torch.ones_like(torch.stack([self.timesteps[self.seg_index][self.step_index]])) * self.context_noise
#             self.latents_input = self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)]

#     def step_post(self):
#         # convert model outputs
#         current_start_frame = self.seg_index * self.num_frame_per_block
#         current_end_frame = (self.seg_index + 1) * self.num_frame_per_block

#         flow_pred = self.noise_pred[:, current_start_frame:current_end_frame].transpose(0, 1)
#         xt = self.latents_input.transpose(0, 1)
#         timestep = self.timestep_input.squeeze(0)

#         original_dtype = flow_pred.dtype

#         flow_pred, xt, sigmas, timesteps = map(lambda x: x.double().to(flow_pred.device), [flow_pred, xt, self.sigmas_sf, self.timesteps_sf])
#         timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
#         sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
#         x0_pred = xt - sigma_t * flow_pred
#         x0_pred = x0_pred.to(original_dtype)

#         # add noise
#         if self.step_index < self.infer_steps - 1:
#             timestep_next = self.timesteps[self.seg_index][self.step_index + 1] * torch.ones(self.num_frame_per_block, device=AI_DEVICE, dtype=torch.long)
#             timestep_id_next = torch.argmin((self.timesteps_sf.unsqueeze(0) - timestep_next.unsqueeze(1)).abs(), dim=1)
#             sigma_next = self.sigmas_sf[timestep_id_next].reshape(-1, 1, 1, 1)
#             noise_next = torch.randn_like(x0_pred)
#             sample_next = (1 - sigma_next) * x0_pred + sigma_next * noise_next
#             sample_next = sample_next.type_as(noise_next)
#             self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)] = sample_next.transpose(0, 1)
#         else:
#             self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)] = x0_pred.transpose(0, 1)
#             self.stream_output = x0_pred.transpose(0, 1)


class WanSFScheduler(WanScheduler):
    """Scheduler for self-forcing inference.

    Aligns with the source code in image2video_fast.py:
        scheduler.set_timesteps(num_train_timesteps, shift=shift)
        timesteps = scheduler.timesteps[timesteps_index].tolist()

        for timestep_idx in range(len(timesteps)):
            ...
            x0 = _convert_flow_pred_to_x0(flow_pred, xt, timestep, scheduler)
            if timestep_idx < len(timesteps) - 1:
                current_latent = scheduler.add_noise(x0, noise, next_timestep)
    """

    def __init__(self, config):
        super().__init__(config)
        self.dtype = torch.bfloat16
        self.num_frame_per_chunk = self.config["ar_config"]["num_frame_per_chunk"]
        self.timesteps_index = self.config["ar_config"]["timesteps_index"]
        self.infer_steps = len(self.timesteps_index)
        self.context_noise = 0

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.latents = torch.randn(latent_shape, device=AI_DEVICE, dtype=self.dtype)
        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.set_timesteps(self.num_train_timesteps, device=AI_DEVICE, shift=self.sample_shift)
        self.selected_timesteps = self.timesteps[self.timesteps_index].tolist()
        self.noise_pred = torch.zeros(latent_shape, device=AI_DEVICE, dtype=self.dtype)
        self.stream_output = None

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        """Align with source _convert_flow_pred_to_x0.

        x0_pred = xt - sigma_t * flow_pred
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.sigmas, self.timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _add_noise(self, original_samples, noise, timestep):
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        step_index = self.index_for_timestep(timestep, schedule_timesteps)
        sigma = sigmas[step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        while len(sigma_t.shape) < len(original_samples.shape):
            sigma_t = sigma_t.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def step_pre(self, seg_index, step_index, is_rerun=False):
        self.step_index = step_index
        self.seg_index = seg_index
        self.is_rerun = is_rerun

        if not GET_DTYPE() == GET_SENSITIVE_DTYPE():
            self.latents = self.latents.to(GET_DTYPE())

        seg_start = self.seg_index * self.num_frame_per_chunk
        seg_end = min((self.seg_index + 1) * self.num_frame_per_chunk, self.num_output_frames)
        self.latents_input = self.latents[:, seg_start:seg_end]

        if not is_rerun:
            t_val = self.selected_timesteps[self.step_index]
        else:
            t_val = self.context_noise
        self.timestep_input = torch.full([1, self.num_frame_per_chunk], t_val, device=AI_DEVICE, dtype=torch.long)

    def step_post(self):
        seg_start = self.seg_index * self.num_frame_per_chunk
        seg_end = min((self.seg_index + 1) * self.num_frame_per_chunk, self.num_output_frames)

        flow_pred = self.noise_pred[:, seg_start:seg_end]
        xt = self.latents_input
        timestep = self.selected_timesteps[self.step_index]

        x0 = self._convert_flow_pred_to_x0(flow_pred, xt, timestep)

        if self.step_index < self.infer_steps - 1:
            next_timestep = self.selected_timesteps[self.step_index + 1]
            noise = torch.randn_like(x0)
            self.latents[:, seg_start:seg_end] = self._add_noise(x0, noise, next_timestep)
        else:
            self.latents[:, seg_start:seg_end] = x0
            self.stream_output = x0
