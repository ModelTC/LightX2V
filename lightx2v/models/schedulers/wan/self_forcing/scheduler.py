import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


class WanSFScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.dtype = torch.bfloat16
        ar = config.get("ar_config", {})
        self.num_frame_per_chunk = int(ar.get("num_frame_per_chunk", 3))
        self.num_output_frames = int(config.get("num_output_frames", config.get("target_video_length", 81)))
        self.context_noise = 0

        if "denoising_step_list" in ar:
            self._mode = "denoise"
            self.denoising_step_list = [float(t) for t in ar["denoising_step_list"]]
            self.infer_steps = len(self.denoising_step_list)
            self.denoising_strength = float(ar.get("denoising_strength", 1.0))
            self.extra_one_step = bool(ar.get("extra_one_step", True))
            self.inverse_timesteps = bool(ar.get("inverse_timesteps", False))
            self.reverse_sigmas = bool(ar.get("reverse_sigmas", False))
        else:
            self._mode = "index"
            self.timesteps_index = ar["timesteps_index"]
            self.infer_steps = len(self.timesteps_index)

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.latents = torch.randn(latent_shape, device=AI_DEVICE, dtype=self.dtype)
        self.noise_pred = torch.zeros(latent_shape, device=AI_DEVICE, dtype=self.dtype)
        self.stream_output = None
        if self._mode == "denoise":
            self._prepare_denoise_schedule()
        else:
            self._prepare_index_schedule()

    def _prepare_index_schedule(self):
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

    def _prepare_denoise_schedule(self):
        sigma_start = self.denoising_strength
        if self.extra_one_step:
            s = torch.linspace(sigma_start, 0.0, self.num_train_timesteps + 1, device=AI_DEVICE, dtype=torch.float32)[:-1]
        else:
            s = torch.linspace(sigma_start, 0.0, self.num_train_timesteps, device=AI_DEVICE, dtype=torch.float32)
        if self.inverse_timesteps:
            s = torch.flip(s, dims=[0])
        s = self.sample_shift * s / (1 + (self.sample_shift - 1) * s)
        if self.reverse_sigmas:
            s = 1 - s
        self.sigmas_sf = s
        self.timesteps_sf = s * self.num_train_timesteps

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        """x0_pred = xt - sigma_t * flow_pred (index schedule: scalar timestep)."""
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

        if self._mode == "denoise":
            t = float(self.context_noise if is_rerun else self.denoising_step_list[self.step_index])
            self.timestep_input = torch.full((1, self.num_frame_per_chunk), t, device=AI_DEVICE, dtype=torch.float32)
        else:
            t_val = self.context_noise if is_rerun else self.selected_timesteps[self.step_index]
            self.timestep_input = torch.full((1, self.num_frame_per_chunk), t_val, device=AI_DEVICE, dtype=torch.long)

    def step_post(self):
        seg_start = self.seg_index * self.num_frame_per_chunk
        seg_end = min((self.seg_index + 1) * self.num_frame_per_chunk, self.num_output_frames)
        if self._mode == "denoise":
            self._step_post_denoise(seg_start, seg_end)
            return

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

    def _sigma_at_timestep(self, timestep_1d):
        """Match legacy: argmin in float64 on ``timesteps_sf`` / ``sigmas_sf``."""
        ts = self.timesteps_sf.unsqueeze(0).double()
        tid = torch.argmin((ts - timestep_1d.unsqueeze(1).double()).abs(), dim=1)
        return self.sigmas_sf[tid]

    def _step_post_denoise(self, seg_start, seg_end):
        original_dtype = self.noise_pred.dtype
        flow_pred = self.noise_pred[:, seg_start:seg_end].transpose(0, 1).double()
        xt = self.latents_input.transpose(0, 1).double()
        timestep = self.timestep_input.squeeze(0).to(device=flow_pred.device, dtype=torch.float64)
        sigma_t = self._sigma_at_timestep(timestep).reshape(-1, 1, 1, 1).double()
        x0 = (xt - sigma_t * flow_pred).to(original_dtype)
        if self.step_index < self.infer_steps - 1:
            next_t = torch.full(
                (self.num_frame_per_chunk,),
                float(self.denoising_step_list[self.step_index + 1]),
                device=flow_pred.device,
                dtype=torch.float64,
            )
            sigma_next = self._sigma_at_timestep(next_t).reshape(-1, 1, 1, 1)
            noise_next = torch.randn_like(x0)
            sample_next = (1 - sigma_next) * x0 + sigma_next * noise_next
            sample_next = sample_next.type_as(noise_next)
            self.latents[:, seg_start:seg_end] = sample_next.transpose(0, 1)
        else:
            self.latents[:, seg_start:seg_end] = x0.transpose(0, 1)
            self.stream_output = x0.transpose(0, 1)
