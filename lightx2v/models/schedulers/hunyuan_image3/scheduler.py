import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class HunyuanImage3Scheduler(BaseScheduler):
    def __init__(self, config):
        scheduler_config = dict(config)
        scheduler_config["infer_steps"] = int(config.get("infer_steps") or config.get("diff_infer_steps", 50))
        super().__init__(scheduler_config)
        self.sample_guide_scale = config.get("sample_guide_scale", config.get("diff_guidance_scale", 1.0))
        self.flow_shift = config.get("sample_shift", config.get("flow_shift", 1.0))
        self.timesteps = None
        self.sigmas = None
        self.noise_pred = None

    def prepare(self, input_info):
        seed = getattr(input_info, "seed", None) or self.config.get("seed", 42)
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)

    def set_timesteps(self, num_inference_steps=None, device=None):
        num_inference_steps = num_inference_steps or self.infer_steps
        device = device or AI_DEVICE
        sigmas = torch.linspace(1, 0, num_inference_steps + 1, device=device)
        if self.flow_shift != 1.0:
            sigmas = (self.flow_shift * sigmas) / (1 + (self.flow_shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32)

    def step_pre(self, step_index):
        super().step_pre(step_index)
        if self.timesteps is None:
            self.set_timesteps(self.infer_steps, device=self.latents.device if self.latents is not None else AI_DEVICE)
        self.timestep_input = self.timesteps[step_index : step_index + 1]

    def step_post(self):
        if self.latents is None or self.noise_pred is None:
            return
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        self.latents = self.latents.float() + self.noise_pred.float() * (sigma_next - sigma)
