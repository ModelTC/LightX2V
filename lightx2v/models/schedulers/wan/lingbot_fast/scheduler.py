import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


class LingbotFastScheduler(WanScheduler):
    """Scheduler for lingbot-fast autoregressive inference.

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
        self.num_frame_per_block = self.config["sf_config"]["num_frame_per_block"]
        self.num_output_frames = self.config["sf_config"]["num_output_frames"]
        self.num_blocks = self.num_output_frames // self.num_frame_per_block
        self.timesteps_index = self.config["sf_config"]["timesteps_index"]
        self.infer_steps = len(self.timesteps_index)
        self.context_noise = 0

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.latents = torch.randn(latent_shape, device=AI_DEVICE, dtype=self.dtype)

        # --- Align with FlowUniPCMultistepScheduler.__init__ ---
        # Build base sigma schedule (shift=1) to establish sigma_min / sigma_max
        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        # --- Align with: scheduler.set_timesteps(num_train_timesteps, shift=shift) ---
        self.set_timesteps(self.num_train_timesteps, device=AI_DEVICE, shift=self.sample_shift)

        # --- Align with: timesteps = scheduler.timesteps[timesteps_index].tolist() ---
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

        if not GET_DTYPE() == GET_SENSITIVE_DTYPE():
            self.latents = self.latents.to(GET_DTYPE())

        seg_start = self.seg_index * self.num_frame_per_block
        seg_end = min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)
        self.latents_input = self.latents[:, seg_start:seg_end]

        if not is_rerun:
            # Align with: timestep = torch.stack([timesteps[timestep_idx]]).to(device)
            t_val = self.selected_timesteps[self.step_index]
        else:
            # Align with: context_timestep = [timesteps[-1] * 0.0]
            t_val = self.context_noise

        # Shape [1, num_frame_per_block] required by infer_non_blocks
        # (unflatten uses t.shape to reshape embed for head modulation broadcast)
        self.timestep_input = torch.full([1, self.num_frame_per_block], t_val, device=AI_DEVICE, dtype=torch.long)

    def step_post(self):
        """Align with source denoising loop:

        x0 = _convert_flow_pred_to_x0(flow_pred, xt, timestep, scheduler)
        if timestep_idx < len(timesteps) - 1:
            next_timestep = timesteps[timestep_idx + 1]
            current_latent = scheduler.add_noise(x0, noise, next_timestep)
        """
        seg_start = self.seg_index * self.num_frame_per_block
        seg_end = min((self.seg_index + 1) * self.num_frame_per_block, self.num_output_frames)

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
