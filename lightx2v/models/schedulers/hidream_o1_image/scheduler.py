import einops
import numpy as np
import torch
from PIL import Image
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from lightx2v.models.networks.hidream_o1_image.utils import PATCH_SIZE
from lightx2v.models.schedulers.hidream_o1_image.fm_solvers_unipc import FlowUniPCMultistepScheduler
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class FlashFlowMatchEulerDiscreteScheduler:
    """Minimal HiDream flash scheduler, matching the official inference recipe."""

    def __init__(self, num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False):
        if use_dynamic_shifting:
            raise ValueError("HiDream flash scheduler does not use dynamic shifting in LightX2V.")
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None
        self._step_index = None

    def set_timesteps(self, num_inference_steps, device=None):
        timesteps = np.linspace(self.num_train_timesteps, 1, num_inference_steps, dtype=np.float32)
        sigmas = timesteps / self.num_train_timesteps
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.num_inference_steps = num_inference_steps
        self._step_index = None

    def index_for_timestep(self, timestep):
        indices = (self.timesteps == timestep.to(self.timesteps.device)).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        self._step_index = self.index_for_timestep(timestep)

    def step(self, model_output, timestep, sample, s_noise=1.0, noise_clip_std=0.0, generator=None, return_dict=True):
        if self._step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self._step_index]
        sample = sample.to(torch.float32)
        denoised = sample - model_output * sigma

        if self._step_index < self.num_inference_steps:
            sigma_next = self.sigmas[self._step_index + 1]
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=denoised.dtype,
            )
            if noise_clip_std > 0:
                clip_val = noise_clip_std * noise.std().item()
                noise = noise.clamp(min=-clip_val, max=clip_val)
            sample = sigma_next * noise * s_noise + (1.0 - sigma_next) * denoised

        self._step_index += 1
        sample = sample.to(model_output.dtype)
        if not return_dict:
            return (sample,)
        return {"prev_sample": sample}


def build_scheduler(num_inference_steps, timesteps_list, shift, device, scheduler_name="default"):
    if scheduler_name == "flash":
        sched = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    elif scheduler_name == "flow_match":
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    elif scheduler_name == "default":
        sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
    else:
        raise ValueError(f"Unknown HiDream-O1-Image scheduler_name={scheduler_name!r}")
    sched.set_timesteps(num_inference_steps, device=device)
    if timesteps_list is not None:
        sched.timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in sched.timesteps]
        sigmas.append(0.0)
        sched.sigmas = torch.tensor(sigmas, device=device)
    return sched


class HidreamO1ImageScheduler(BaseScheduler):
    def __init__(self, config, torch_dtype=torch.bfloat16):
        super().__init__(config)
        self.dtype = torch_dtype
        self.sched = None
        self.timesteps = None
        self.noise_scale_schedule = None
        self.current_timestep = None
        self.current_t_pixeldit = None
        self.current_sigma = None
        self.noise_pred = None
        self.height = None
        self.width = None
        self.h_patches = None
        self.w_patches = None
        self.generation_config = None

    def set_dtype(self, torch_dtype):
        self.dtype = torch_dtype

    def prepare(self, inputs):
        cfg = inputs["generation_config"]
        self.generation_config = cfg
        self.height = inputs["height"]
        self.width = inputs["width"]
        self.h_patches = inputs["h_patches"]
        self.w_patches = inputs["w_patches"]
        seed = inputs["seed"]

        noise = cfg["noise_scale_start"] * torch.randn(
            (1, 3, self.height, self.width),
            generator=torch.Generator("cpu").manual_seed(seed + 1),
        ).to(AI_DEVICE, self.dtype)
        self.latents = einops.rearrange(
            noise,
            "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )
        self.sched = build_scheduler(
            cfg["num_inference_steps"],
            cfg["timesteps_list"],
            cfg["shift"],
            AI_DEVICE,
            cfg["scheduler_name"],
        )
        self.timesteps = self.sched.timesteps
        self.infer_steps = len(self.timesteps)
        self.noise_scale_schedule = self._build_noise_scale_schedule(cfg, self.infer_steps)
        self.noise_pred = None

        torch.manual_seed(seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1)

    def _build_noise_scale_schedule(self, cfg, num_steps):
        if num_steps > 1:
            return [cfg["noise_scale_start"] + (cfg["noise_scale_end"] - cfg["noise_scale_start"]) * i / (num_steps - 1) for i in range(num_steps)]
        return [cfg["noise_scale_start"]]

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current_timestep = self.timesteps[self.step_index]
        self.current_t_pixeldit = 1.0 - self.current_timestep.float() / 1000.0
        self.current_sigma = (self.current_timestep.float() / 1000.0).to(dtype=torch.float32).clamp_min(0.001)
        self.noise_pred = None

    def step_post(self):
        if self.generation_config["scheduler_name"] == "flash":
            self.latents = self.sched.step(
                self.noise_pred.float(),
                self.current_timestep.to(dtype=torch.float32),
                self.latents.float(),
                s_noise=self.noise_scale_schedule[self.step_index],
                noise_clip_std=self.generation_config["noise_clip_std"],
                return_dict=False,
            )[0].to(self.dtype)
        else:
            self.latents = self.sched.step(
                self.noise_pred.float(),
                self.current_timestep.to(dtype=torch.float32),
                self.latents.float(),
                return_dict=False,
            )[0].to(self.dtype)

    def decode(self):
        img = (self.latents + 1) / 2
        img = einops.rearrange(
            img.cpu().float(),
            "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
            H=self.h_patches,
            W=self.w_patches,
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )
        arr = np.round(np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def clear(self):
        self.latents = None
        self.sched = None
        self.timesteps = None
        self.noise_scale_schedule = None
        self.current_timestep = None
        self.current_t_pixeldit = None
        self.current_sigma = None
        self.noise_pred = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
