import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class HunyuanImage3Scheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.sample_guide_scale = config.get("sample_guide_scale", config.get("diff_guidance_scale", 1.0))
        self.flow_shift = config.get("sample_shift", config.get("flow_shift", 1.0))
        self.timesteps = None
        self.timesteps_full = None
        self.sigmas = None
        self.noise_pred = None
        self._step_index = None

    def prepare(self, input_info):
        seed = getattr(input_info, "seed", None) or self.config.get("seed", 42)
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
        num_inference_steps = num_inference_steps or self.infer_steps
        device = device or AI_DEVICE
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of timesteps or sigmas can be passed.")
        if timesteps is not None:
            timesteps = torch.as_tensor(timesteps, dtype=torch.float32, device=device)
            sigmas = timesteps / 1000.0
            if sigmas[-1] != 0:
                sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)
        elif sigmas is not None:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
        else:
            sigmas = torch.linspace(1, 0, num_inference_steps + 1, device=device)
        if self.flow_shift != 1.0:
            sigmas = (self.flow_shift * sigmas) / (1 + (self.flow_shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
        self.timesteps_full = (sigmas * 1000).to(dtype=torch.float32, device=device)
        self._step_index = None

    def index_for_timestep(self, timestep):
        if self.timesteps is None:
            raise ValueError("set_timesteps must be called before index_for_timestep.")
        timestep = timestep.to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        indices = (self.timesteps == timestep).nonzero()
        if indices.numel() == 0:
            return int(torch.argmin(torch.abs(self.timesteps - timestep)).item())
        return int(indices[0].item())

    def get_timestep_r(self, timestep):
        if self.timesteps_full is None:
            raise ValueError("set_timesteps must be called before get_timestep_r.")
        step_index = self.index_for_timestep(timestep)
        return self.timesteps_full[step_index + 1]

    def scale_model_input(self, sample, *args, **kwargs):
        return sample

    def step(self, model_output, timestep, sample, return_dict=True, **kwargs):
        if self.sigmas is None:
            raise ValueError("set_timesteps must be called before step.")
        self._step_index = self.index_for_timestep(timestep)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        prev_sample = sample.float() + model_output.float() * (sigma_next - sigma)
        if not return_dict:
            return (prev_sample,)
        return {"prev_sample": prev_sample}

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
