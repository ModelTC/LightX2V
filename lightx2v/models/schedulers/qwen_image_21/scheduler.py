import json
import math
from pathlib import Path

import numpy as np
import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class QwenImage21Scheduler(BaseScheduler):
    """Flow Euler with the released resolution-dependent exponential schedule."""

    def __init__(self, config):
        super().__init__(config)
        path = Path(config["model_path"]) / "scheduler" / "scheduler_config.json"
        self.scheduler_config = json.loads(path.read_text())
        # Preserve resolution shifting; this switch only controls the optional
        # stretch to the checkpoint's terminal sigma (disabled for Viggle).
        self.do_time_shift = config.get("do_time_shift", True)
        if not isinstance(self.do_time_shift, bool):
            raise ValueError("Qwen-Image-2.1 do_time_shift must be a boolean")
        sc = self.scheduler_config
        if (
            sc["time_shift_type"] != "exponential"
            or not sc["use_dynamic_shifting"]
            or any(sc.get(k, False) for k in ("invert_sigmas", "stochastic_sampling", "use_beta_sigmas", "use_exponential_sigmas", "use_karras_sigmas"))
        ):
            raise ValueError("Qwen-Image-2.1 expects its released exponential FlowMatchEuler schedule")
        self.sample_guide_scale = config["sample_guide_scale"]
        self.raw_sigmas = None
        if config.get("sigmas") is not None:
            sigmas = np.asarray(config["sigmas"], dtype=np.float32)
            if sigmas.shape != (self.infer_steps,) or not np.all(np.isfinite(sigmas)) or not np.all((sigmas > 0) & (sigmas <= 1)) or not np.all(np.diff(sigmas) < 0):
                raise ValueError("Qwen-Image-2.1 sigmas must contain infer_steps finite, strictly decreasing values in (0, 1]")
            self.raw_sigmas = sigmas

    def prepare(self, input_info):
        self.generator = torch.Generator(device="cpu").manual_seed(input_info.seed)
        noise = torch.randn(input_info.latent_shape, generator=self.generator, dtype=GET_DTYPE())
        self.latents = noise.flatten(3).squeeze(1).transpose(1, 2).to(AI_DEVICE)
        sc = self.scheduler_config
        slope = (sc["max_shift"] - sc["base_shift"]) / (sc["max_image_seq_len"] - sc["base_image_seq_len"])
        h, w = input_info.latent_shape[-2:]
        mu = slope * h * w + sc["base_shift"] - slope * sc["base_image_seq_len"]
        # Custom nodes are raw noise levels; apply the same resolution shift.
        sigmas = self.raw_sigmas if self.raw_sigmas is not None else np.linspace(1.0, 1 / self.infer_steps, self.infer_steps).astype(np.float32)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        if self.do_time_shift and sc.get("shift_terminal") and self.infer_steps > 1:
            one_minus = 1 - sigmas
            sigmas = 1 - one_minus / (one_minus[-1] / (1 - sc["shift_terminal"]))
        self.sigmas = torch.from_numpy(sigmas).to(device=AI_DEVICE, dtype=torch.float32)
        self.timesteps = self.sigmas * sc["num_train_timesteps"]
        self.sigmas = torch.cat((self.sigmas, self.sigmas.new_zeros(1)))
        self.step_index = 0

    def step_post(self):
        delta = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self.latents = (self.latents.float() + delta * self.noise_pred).to(self.noise_pred.dtype)

    def clear(self):
        self.latents = self.noise_pred = self.timesteps = self.sigmas = self.generator = None
        self.step_index = 0
