import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


class FastWAMActionScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__({"infer_steps": int(config.get("action_infer_steps", 20))})
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.action_sample_shift = float(config.get("action_sample_shift", 5.0))
        self.noise_pred = None

    @staticmethod
    def phi(u, shift):
        return shift * u / (1.0 + (shift - 1.0) * u)

    def prepare_loop(self, action_shape, *, seed, device, dtype, infer_steps=None):
        self.infer_steps = int(infer_steps or self.infer_steps)
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        self.latents = torch.randn(
            action_shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        u_steps = torch.linspace(1.0, 0.0, self.infer_steps + 1, device=device, dtype=torch.float32)
        sigmas = self.phi(u_steps, self.action_sample_shift)
        self.timesteps = (sigmas[:-1] * float(self.num_train_timesteps)).to(dtype=dtype)
        self.deltas = (sigmas[1:] - sigmas[:-1]).to(dtype=dtype)
        self.noise_pred = None

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current_timestep = self.timesteps[step_index]
        self.current_delta = self.deltas[step_index]

    def step_post(self):
        if self.noise_pred is None:
            raise RuntimeError("FastWAMActionScheduler requires noise_pred before step_post().")
        delta = self.current_delta.to(device=self.latents.device, dtype=self.latents.dtype)
        self.latents = self.latents + self.noise_pred * delta
        self.noise_pred = None

    def clear(self):
        self.latents = None
        self.noise_pred = None
