import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WanSchedulerCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerTaylorCaching(WanSchedulerCaching):
    def __init__(self, config):
        super().__init__(config)

    def prepare(self, seed, latent_shape, infer_steps, image_encoder_output=None):
        self.infer_steps = infer_steps
        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((self.config.infer_steps + 3) // 4))[: self.config.infer_steps]
        self.caching_records_2 = (pattern * ((self.config.infer_steps + 3) // 4))[: self.config.infer_steps]

        if self.config["model_cls"] == "wan2.2" and self.config["task"] in ["i2v", "s2v", "rs2v"]:
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

        self.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.sample_shift)
