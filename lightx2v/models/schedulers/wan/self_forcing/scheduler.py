import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


class WanSFScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.dtype = torch.bfloat16
        self.num_frame_per_block = self.config["sf_config"]["num_frame_per_block"]
        self.num_output_frames = self.config["sf_config"]["num_output_frames"]
        self.num_blocks = self.num_output_frames // self.num_frame_per_block
        self.denoising_step_list = self.config["sf_config"]["denoising_step_list"]
        self.infer_steps = len(self.denoising_step_list)
        self.all_num_frames = [self.num_frame_per_block] * self.num_blocks
        self.num_input_frames = 0
        self.denoising_strength = 1.0
        self.sigma_max = 1.0
        self.sigma_min = 0
        self.sf_shift = self.config["sf_config"]["shift"]
        self.inverse_timesteps = False
        self.extra_one_step = True
        self.reverse_sigmas = False
        self.num_inference_steps = self.config["sf_config"]["num_inference_steps"]
        self.context_noise = 0

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.latents = torch.randn(latent_shape, device=AI_DEVICE, dtype=self.dtype)

        timesteps = []
        for frame_block_idx, current_num_frames in enumerate(self.all_num_frames):
            frame_steps = []

            for step_index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([self.num_frame_per_block], device=AI_DEVICE, dtype=torch.int64) * current_timestep
                frame_steps.append(timestep)

            timesteps.append(frame_steps)
        self.timesteps = timesteps

        self.noise_pred = torch.zeros(latent_shape, device=AI_DEVICE, dtype=self.dtype)

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * self.denoising_strength
        if self.extra_one_step:
            self.sigmas_sf = torch.linspace(sigma_start, self.sigma_min, self.num_inference_steps + 1)[:-1]
        else:
            self.sigmas_sf = torch.linspace(sigma_start, self.sigma_min, self.num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas_sf = torch.flip(self.sigmas_sf, dims=[0])
        self.sigmas_sf = self.sf_shift * self.sigmas_sf / (1 + (self.sf_shift - 1) * self.sigmas_sf)
        if self.reverse_sigmas:
            self.sigmas_sf = 1 - self.sigmas_sf
        self.sigmas_sf = self.sigmas_sf.to(AI_DEVICE)

        self.timesteps_sf = self.sigmas_sf * self.num_train_timesteps
        self.timesteps_sf = self.timesteps_sf.to(AI_DEVICE)

        self.stream_output = None

    def step_pre(self, seg_index, step_index, is_rerun=False):
        self.step_index = step_index
        self.seg_index = seg_index

        if not GET_DTYPE() == GET_SENSITIVE_DTYPE():
            self.latents = self.latents.to(GET_DTYPE())

        if not is_rerun:
            self.timestep_input = torch.stack([self.timesteps[self.seg_index][self.step_index]])
            self.latents_input = self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)]
        else:
            # rerun with timestep zero to update KV cache using clean context
            self.timestep_input = torch.ones_like(torch.stack([self.timesteps[self.seg_index][self.step_index]])) * self.context_noise
            self.latents_input = self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)]

    def step_post(self):
        # convert model outputs
        current_start_frame = self.seg_index * self.num_frame_per_block
        current_end_frame = (self.seg_index + 1) * self.num_frame_per_block

        flow_pred = self.noise_pred[:, current_start_frame:current_end_frame].transpose(0, 1)
        xt = self.latents_input.transpose(0, 1)
        timestep = self.timestep_input.squeeze(0)

        original_dtype = flow_pred.dtype

        flow_pred, xt, sigmas, timesteps = map(lambda x: x.double().to(flow_pred.device), [flow_pred, xt, self.sigmas_sf, self.timesteps_sf])
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        x0_pred = x0_pred.to(original_dtype)

        # add noise
        if self.step_index < self.infer_steps - 1:
            timestep_next = self.timesteps[self.seg_index][self.step_index + 1] * torch.ones(self.num_frame_per_block, device=AI_DEVICE, dtype=torch.long)
            timestep_id_next = torch.argmin((self.timesteps_sf.unsqueeze(0) - timestep_next.unsqueeze(1)).abs(), dim=1)
            sigma_next = self.sigmas_sf[timestep_id_next].reshape(-1, 1, 1, 1)
            noise_next = torch.randn_like(x0_pred)
            sample_next = (1 - sigma_next) * x0_pred + sigma_next * noise_next
            sample_next = sample_next.type_as(noise_next)
            self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)] = sample_next.transpose(0, 1)
        else:
            self.latents[:, self.seg_index * self.num_frame_per_block : min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)] = x0_pred.transpose(0, 1)
            self.stream_output = x0_pred.transpose(0, 1)
