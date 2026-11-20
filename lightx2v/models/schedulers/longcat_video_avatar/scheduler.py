import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


class LongCatVideoAvatarScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.keep_latents_dtype_in_scheduler = True
        self.mix_steps = config["mix_steps"]
        self.mix_alpha = config["mix_alpha"]
        train_steps = 1000
        stride = train_steps // self.infer_steps
        raw_sigmas = torch.linspace(1, 0, train_steps, dtype=torch.float32)[::stride]
        shift = config["sample_shift"]
        sigmas = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)
        self.sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        self.timesteps = sigmas * train_steps
        self.noise_pred = None

    def prepare(self, info, reference, generator, anchor=None):
        device = generator.device
        latent_frames = (info.num_frames - 1) // self.config["vae_scale"] + 1
        if reference is None:
            height, width = (side // self.config["vae_scale_factor"] for side in info.size)
            self.num_cond_latents = 0
        else:
            self.num_cond_latents, height, width = reference.shape[-3:]
        shape = (1, self.config["in_channels"], latent_frames, height, width)
        # The release samples continuation noise directly in BF16, then prepends the FP32 anchor.
        dtype = torch.bfloat16 if anchor is not None else torch.float32
        self.latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        if reference is not None:
            self.latents[:, :, : self.num_cond_latents] = reference
        if anchor is not None:
            self.latents = torch.cat((anchor, self.latents), dim=2)
            self.num_cond_latents += 1
        self.frame_timesteps = torch.zeros(self.latents.shape[2], device=device, dtype=torch.bfloat16)
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.frame_timesteps[self.num_cond_latents :] = self.timesteps[step_index]
        dmd = self.mix_alpha if step_index < self.mix_steps else 0.0
        self.lora_strengths = (dmd, 1.0 - dmd)

    def step_post(self):
        # The upstream pipeline negates the DiT prediction before its Euler step.
        sigma_delta = self.sigmas[self.step_index] - self.sigmas[self.step_index + 1]
        self.latents[:, :, self.num_cond_latents :] += sigma_delta * self.noise_pred[:, :, self.num_cond_latents :]
        self.noise_pred = None

    def clear(self):
        self.latents = None
        self.noise_pred = None
        self.frame_timesteps = None
