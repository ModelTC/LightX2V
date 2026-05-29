import torch

from lightx2v.models.schedulers.helios.helios_dmd import HeliosDMDScheduler
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class HeliosDistilledScheduler(BaseScheduler):
    def __init__(self, config):
        infer_steps = sum(config.get("pyramid_num_inference_steps_list", [2, 2, 2]))
        config["infer_steps"] = infer_steps
        super().__init__(config)
        self.inner = HeliosDMDScheduler.from_pretrained(config["scheduler_path"])
        self.sample_guide_scale = config.get("sample_guide_scale", 1.0)
        self.generator = None
        self.latents = None
        self.latent_shape = None
        self.image_encoder_output = None
        self.timesteps = None
        self.sigmas = None

    def prepare(self, seed, latent_shape, image_encoder_output=None, generator=None):
        self.generator = generator if generator is not None else torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latent_shape = latent_shape
        self.image_encoder_output = image_encoder_output

    def set_timesteps(self, num_inference_steps, stage_idx, device, mu, is_amplify_first_chunk):
        self.inner.set_timesteps(
            num_inference_steps,
            stage_idx,
            device=device,
            mu=mu,
            is_amplify_first_chunk=is_amplify_first_chunk,
        )
        self.timesteps = self.inner.timesteps
        self.sigmas = self.inner.sigmas
        self.infer_steps = len(self.timesteps)
        self.step_index = 0

    def step(self, *args, **kwargs):
        return self.inner.step(*args, **kwargs)

    def step_pre(self, step_index):
        self.step_index = step_index

    def clear(self):
        self.generator = None
        self.latents = None
        self.latent_shape = None
        self.image_encoder_output = None
        self.timesteps = None
        self.sigmas = None
