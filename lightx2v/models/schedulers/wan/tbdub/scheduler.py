import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


def replace_latent_borders(latents, reference, start_frame=0, width=1):
    """Replace only spatial borders, leaving an optional motion prefix intact."""
    if width <= 0:
        return latents
    result = latents.clone()
    result[:, start_frame:, :width, :] = reference[:, start_frame:, :width, :]
    result[:, start_frame:, -width:, :] = reference[:, start_frame:, -width:, :]
    result[:, start_frame:, :, :width] = reference[:, start_frame:, :, :width]
    result[:, start_frame:, :, -width:] = reference[:, start_frame:, :, -width:]
    return result


class TBDubScheduler(BaseScheduler):
    """Wan flow-match Euler with TBDub student's motion and border constraints."""

    def __init__(self, config):
        super().__init__(config)
        self.num_train_timesteps = 1000
        self.sample_shift = float(config["sample_shift"])
        self.replace_borders = config.get("replace_border_latents", True)
        self.border_width = int(config.get("border_width", 1))
        self.keep_latents_dtype_in_scheduler = True
        self.set_timesteps(self.infer_steps)

    def set_timesteps(self, infer_steps):
        if infer_steps < 1 or self.sample_shift <= 0:
            raise ValueError("TBDub infer_steps and sample_shift must be positive")
        sigmas = torch.linspace(1.0, 0.0, int(infer_steps) + 1)[:-1]
        self.sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.infer_steps = len(self.timesteps)

    def prepare_loop(self, ref_latents, seed, motion_latents=None, noise=None):
        if ref_latents.ndim != 4:
            raise ValueError("TBDub reference latents must have shape [C,F,H,W]")
        self.ref_latents = ref_latents
        self.motion_latents = motion_latents
        self.motion_frames = 0 if motion_latents is None else motion_latents.shape[1]
        if motion_latents is not None and (
            motion_latents.ndim != 4 or motion_latents.shape[0] != ref_latents.shape[0] or motion_latents.shape[2:] != ref_latents.shape[2:] or self.motion_frames > ref_latents.shape[1]
        ):
            raise ValueError("TBDub motion latents must match reference channels/spatial dimensions")
        self.set_timesteps(self.config["infer_steps"])
        self.sigmas = self.sigmas.to(ref_latents.device)
        self.timesteps = self.timesteps.to(ref_latents.device)
        if noise is None:
            self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
            noise = torch.randn(ref_latents.shape, generator=self.generator, dtype=torch.float32, device="cpu")
        if noise.shape != ref_latents.shape:
            raise ValueError("TBDub noise and reference shapes must match")
        self.latents = noise.to(ref_latents).clone()
        self._restore_motion()
        self.noise_pred = None
        self.step_index = 0

    def _restore_motion(self):
        if self.motion_latents is not None:
            self.latents[:, : self.motion_frames] = self.motion_latents

    def step_pre(self, step_index):
        super().step_pre(int(step_index))
        self.timestep_input = self.timesteps[step_index : step_index + 1]

    def step_post(self):
        if self.noise_pred is None:
            raise RuntimeError("TBDubScheduler.step_post requires model noise_pred")
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1] if self.step_index + 1 < self.infer_steps else sigma.new_zeros(())
        self.latents = self.latents + self.noise_pred * (sigma_next - sigma)
        self._restore_motion()
        if self.step_index + 1 == self.infer_steps and self.replace_borders:
            self.latents = replace_latent_borders(self.latents, self.ref_latents, self.motion_frames, self.border_width)
            self._restore_motion()
        self.noise_pred = None

    def clear(self):
        self.generator = None
        self.latents = None
        self.ref_latents = None
        self.motion_latents = None
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None
        self.noise_pred_guided = None
        self.timestep_input = None
