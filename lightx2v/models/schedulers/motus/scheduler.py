import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


class MotusScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.video_latents = None
        self.action_latents = None
        self.timesteps = None

    def prepare(self, seed, condition_frame_latent, action_shape, dtype, device):
        batch, channels, _, latent_h, latent_w = condition_frame_latent.shape
        total_latent_frames = 1 + self.config["num_video_frames"] // 4
        generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

        self.video_latents = torch.randn(
            (batch, channels, total_latent_frames, latent_h, latent_w),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.video_latents[:, :, 0:1] = condition_frame_latent
        self.action_latents = torch.randn(action_shape, device=device, dtype=dtype, generator=generator)
        self.timesteps = torch.linspace(1.0, 0.0, self.infer_steps + 1, device=device, dtype=dtype)
        self.latents = self.video_latents

    def iter_steps(self):
        for step_index in range(self.infer_steps):
            t = self.timesteps[step_index]
            t_next = self.timesteps[step_index + 1]
            yield step_index, t, t_next, t_next - t

    def step(self, video_velocity, action_velocity, dt, condition_frame_latent):
        self.video_latents = self.video_latents + video_velocity * dt
        self.action_latents = self.action_latents + action_velocity * dt
        self.video_latents[:, :, 0:1] = condition_frame_latent
        self.latents = self.video_latents
