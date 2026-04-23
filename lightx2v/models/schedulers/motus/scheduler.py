import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class MotusScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.video_latents = None
        self.action_latents = None
        self.action_noise_pred = None
        self.condition_frame_latent = None

    def prepare(self, seed, latent_shape, image_encoder_output, action_shape):
        self.vae_encoder_out = image_encoder_output["vae_encoder_out"]
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)

        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.set_timesteps(self.infer_steps, device=self.latents.device, shift=self.sample_shift)
        self.video_latents = self.latents.unsqueeze(0)
        self.condition_frame_latent = self.vae_encoder_out.unsqueeze(0)
        self.action_latents = torch.randn(
            action_shape,
            device=self.video_latents.device,
            dtype=self.video_latents.dtype,
            generator=self.generator,
        )
        self.latents = self.video_latents.squeeze(0)

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        self.generator = torch.Generator(device=self.config.get("device", self.vae_encoder_out.device)).manual_seed(seed)
        self.latents = torch.randn(
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device=self.vae_encoder_out.device,
            generator=self.generator,
        )
        mask = torch.ones_like(self.latents)
        mask[:, :1] = 0
        self.latents = (1.0 - mask) * self.vae_encoder_out + mask * self.latents

    def step_pre(self, step_index):
        super().step_pre(step_index)
        if self.latents.dim() == 4:
            self.video_latents = self.latents.unsqueeze(0)
        else:
            self.video_latents = self.latents
        timestep = self.sigmas[step_index].to(device=self.video_latents.device, dtype=self.video_latents.dtype)
        self.timestep_input = timestep.unsqueeze(0)

    def step_post(self):
        super().step_post()
        self.video_latents = self.latents if self.latents.dim() == 5 else self.latents.unsqueeze(0)
        if self.action_noise_pred is None:
            raise RuntimeError("MotusScheduler requires action_noise_pred before step_post().")

        dt = self.sigmas[self.step_index + 1].to(device=self.action_latents.device, dtype=self.action_latents.dtype) - self.sigmas[self.step_index].to(device=self.action_latents.device, dtype=self.action_latents.dtype)
        self.action_latents = self.action_latents + self.action_noise_pred.to(self.action_latents.dtype) * dt
        self.video_latents[:, :, 0:1] = self.condition_frame_latent.to(device=self.video_latents.device, dtype=self.video_latents.dtype)
        self.latents = self.video_latents.squeeze(0)

    def clear(self):
        self.video_latents = None
        self.action_latents = None
        self.action_noise_pred = None
        self.condition_frame_latent = None
        self.vae_encoder_out = None
        self.latents = None
