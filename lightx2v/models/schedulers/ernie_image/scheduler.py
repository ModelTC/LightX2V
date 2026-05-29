import torch


class ErnieImageScheduler:
    def __init__(self, config, scheduler):
        self.config = config
        self.scheduler = scheduler
        self.generator = None
        self.latents = None
        self.noise_pred = None
        self.step_index = None
        self.timesteps = None
        self.infer_steps = None

    @staticmethod
    def make_generator(seed):
        if seed is None:
            return None
        return torch.Generator(device="cpu").manual_seed(int(seed))

    def prepare_latents(self, latent_shape, device, dtype, seed=None, generator=None):
        try:
            from diffusers.utils.torch_utils import randn_tensor
        except ImportError as exc:
            raise ImportError("ERNIE-Image scheduler requires diffusers.utils.torch_utils.randn_tensor.") from exc

        self.generator = generator if generator is not None else self.make_generator(seed)
        self.latents = randn_tensor(
            latent_shape,
            generator=self.generator,
            device=device,
            dtype=dtype,
        )
        return self.latents

    def prepare(self, num_inference_steps, device, latent_shape=None, dtype=None, seed=None, generator=None):
        self.infer_steps = int(num_inference_steps)
        sigmas = torch.linspace(1.0, 0.0, self.infer_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)
        self.timesteps = self.scheduler.timesteps
        if latent_shape is not None:
            if dtype is None:
                raise ValueError("ERNIE-Image scheduler requires dtype when preparing latents.")
            self.prepare_latents(latent_shape, device, dtype, seed=seed, generator=generator)
        return self.timesteps

    def step_pre(self, step_index):
        self.step_index = int(step_index)

    def step(self, pred, timestep, latents=None):
        if latents is None:
            if self.latents is None:
                raise ValueError("ERNIE-Image scheduler latents are not prepared.")
            latents = self.latents
        self.noise_pred = pred
        self.latents = self.scheduler.step(pred, timestep, latents).prev_sample
        return self.latents
