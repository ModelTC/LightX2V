import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class EdenLiveScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.is_wan21 = config["model_cls"] in ["wan21_edenlive", "wan21_edenlive_ar"]

    def clear(self):
        super().clear()
        self.condition_frame = None
        self.latents_input = None
        self.chunk_start = 0
        self.cache_update = False

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(latent_shape, generator=self.generator, device=AI_DEVICE, dtype=GET_DTYPE())
        self.condition_frame = image_encoder_output["clean_first_frame"].to(self.latents)
        self.chunk_start = 0
        self.cache_update = False
        self._prepare_timesteps()

    def _prepare_step(self, step_index, chunk_start, frames, cache_update):
        self.step_index = step_index
        self.chunk_start = chunk_start
        self.cache_update = cache_update
        self.latents_input = self.latents[:, chunk_start : chunk_start + frames]
        self.timestep_input = torch.zeros(1, device=AI_DEVICE) if cache_update else self.timesteps[step_index : step_index + 1].float()

    def _pin_first_frame(self):
        self.latents_input[:, :1] = self.condition_frame
        if not self.cache_update:
            self.timestep_input = self.timestep_input.expand(self.latents_input.shape[1]).clone()
            self.timestep_input[0] = 0

    def step_pre(self, step_index, chunk_start=0, cache_update=False):
        self._prepare_step(step_index, chunk_start, self.latents.shape[1], cache_update)
        if not self.is_wan21 and chunk_start == 0:
            self._pin_first_frame()

    def _prepare_timesteps(self):
        if self.is_wan21:
            sigmas = torch.linspace(1, 1 / 1000, self.infer_steps + 1, dtype=torch.float32)[:-1]
            sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
            sigmas[0] -= 1e-6
            self.sigmas = torch.cat((sigmas, sigmas.new_zeros(1))).to(AI_DEVICE)
            self.timesteps = self.sigmas[:-1] * 1000
            return
        # Diffusers UniPC flow schedule: linspace(1, 1/N, steps+1), shifted once.
        sigmas = np.linspace(1, 1 / 1000, self.infer_steps + 1)[:-1]
        sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
        sigmas[0] -= 1e-6
        self.sigmas = torch.from_numpy(np.concatenate((sigmas, [0])).astype(np.float32)).to(AI_DEVICE)
        self.timesteps = torch.from_numpy((sigmas * 1000).astype(np.int64)).to(AI_DEVICE)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None

    def step_post(self):
        if self.is_wan21:
            delta = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
            self.latents = self.latents.float().add_(self.noise_pred.float().mul_(delta)).to(self.latents.dtype)
            return
        # Keep the upstream BF16 solver arithmetic; Wan's default solver upcasts.
        sample = self.latents
        prediction = self.noise_pred.to(sample.dtype)
        converted = self.convert_model_output(prediction, sample=sample)
        if self.step_index > 0 and self.last_sample is not None:
            sample = self.multistep_uni_c_bh_update(converted, last_sample=self.last_sample, this_sample=sample, order=self.this_order)
        self.model_outputs = self.model_outputs[1:] + [converted]
        self.timestep_list = self.timestep_list[1:] + [self.timesteps[self.step_index]]
        self.this_order = min(self.solver_order, len(self.timesteps) - self.step_index, self.lower_order_nums + 1)
        self.last_sample = sample
        self.latents = self.multistep_uni_p_bh_update(prediction, sample=sample, order=self.this_order)
        self.lower_order_nums = min(self.lower_order_nums + 1, self.solver_order)
        self.latents[:, :1] = self.condition_frame


class EdenLiveARScheduler(EdenLiveScheduler):
    def _prepare_timesteps(self):
        self.timesteps = torch.tensor(self.config["ar_config"]["denoising_step_list"], device=AI_DEVICE, dtype=torch.float32)
        self.sigmas = self.timesteps / 1000
        self.timesteps = self.sigmas * 1000

    def step_pre(self, step_index, chunk_start=0, cache_update=False):
        self._prepare_step(step_index, chunk_start, self.config["ar_config"]["num_frame_per_chunk"], cache_update)
        if chunk_start == 0:
            self._pin_first_frame()

    def step_post(self):
        sigma = self.sigmas[self.step_index].to(self.latents_input.dtype)
        x0 = self.noise_pred.to(self.latents_input.dtype).mul(sigma).neg_().add_(self.latents_input)
        if self.step_index + 1 < self.infer_steps:
            noise = torch.randn(x0.shape, generator=self.generator, device=x0.device, dtype=x0.dtype)
            sigma_next = self.sigmas[self.step_index + 1].to(x0.dtype)
            x0.mul_(1 - sigma_next).add_(noise.mul_(sigma_next))
        if self.chunk_start == 0:
            x0[:, :1] = self.condition_frame
        self.latents[:, self.chunk_start : self.chunk_start + x0.shape[1]] = x0
