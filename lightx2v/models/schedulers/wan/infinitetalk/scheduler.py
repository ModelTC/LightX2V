import random

import numpy as np
import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


def timestep_transform(t, shift=5.0, num_timesteps=1000):
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    return new_t * num_timesteps


class InfiniteTalkScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.sample_shift = float(config["sample_shift"])
        self.sample_text_guide_scale = float(config.get("sample_text_guide_scale", 5.0))
        self.sample_audio_guide_scale = float(config.get("sample_audio_guide_scale", 4.0))
        self.keep_latents_dtype_in_scheduler = True
        self.timesteps = None
        self.latent_motion_frames = None
        self.cur_motion_frames_latent_num = 0
        self.is_first_clip = True

    def seed_everything(self, seed):
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        return seed

    def prepare(self, seed, latent_shape, latent_motion_frames=None, is_first_clip=True, cur_motion_frames_latent_num=1):
        self.latents = torch.randn(*latent_shape, dtype=torch.float32, device=AI_DEVICE)
        self.latent_motion_frames = latent_motion_frames
        self.is_first_clip = is_first_clip
        self.cur_motion_frames_latent_num = cur_motion_frames_latent_num
        self.noise_pred = None
        self.step_index = 0

        timesteps = list(np.linspace(self.num_train_timesteps, 1, self.infer_steps, dtype=np.float32))
        timesteps.append(0.0)
        self.timesteps = [torch.tensor([t], device=AI_DEVICE) for t in timesteps]
        if self.config.get("use_timestep_transform", True):
            self.timesteps = [timestep_transform(t, shift=self.sample_shift, num_timesteps=self.num_train_timesteps) for t in self.timesteps]

        if not self.is_first_clip and self.latent_motion_frames is not None:
            self._consume_motion_noise(self.timesteps[0])
            self._apply_clean_motion_prefix()

    def _apply_clean_motion_prefix(self):
        if self.latent_motion_frames is None:
            return
        self.latents[:, : self.cur_motion_frames_latent_num] = self.latent_motion_frames.to(self.latents.dtype).to(self.latents.device)

    def _consume_motion_noise(self, timestep):
        if self.latent_motion_frames is None:
            return
        motion = self.latent_motion_frames.to(dtype=self.latents.dtype, device=self.latents.device)
        motion_noise = torch.randn_like(motion).contiguous()
        _ = self.add_noise(motion, motion_noise, timestep)

    def add_noise(self, original_samples, noise, timesteps):
        timesteps = timesteps.float() / self.num_train_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))
        return (1 - timesteps) * original_samples + timesteps * noise

    def step_pre(self, step_index):
        self.step_index = step_index
        self.timestep_input = self.timesteps[step_index]
        self._apply_clean_motion_prefix()

    def step_post(self):
        if self.noise_pred is None:
            return
        dt = (self.timesteps[self.step_index] - self.timesteps[self.step_index + 1]) / self.num_train_timesteps
        self.latents = self.latents + self.noise_pred.to(torch.float32) * dt[:, None, None, None]
        if not self.is_first_clip and self.latent_motion_frames is not None:
            self._consume_motion_noise(self.timesteps[self.step_index + 1])
        self._apply_clean_motion_prefix()
