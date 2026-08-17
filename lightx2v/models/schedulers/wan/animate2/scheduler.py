from __future__ import annotations

import numpy as np
import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WanAnimate2Scheduler(BaseScheduler):
    """Wan-Animate-2 distilled shifted-flow DPM-Solver++ scheduler."""

    solver_order = 2
    num_train_timesteps = 1000

    def __init__(self, config):
        super().__init__(config)
        self.sample_shift = float(config["sample_shift"])
        self.sample_guide_scale = float(config["sample_guide_scale"])
        self.keep_latents_dtype_in_scheduler = True
        self.caching_records_2 = [True] * self.infer_steps
        self.noise_pred = None
        self.timesteps = None
        self.sigmas = None
        self.current_timestep = None
        self.timestep_input = None
        self.latents_input = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    def refresh_from_config(self, config) -> None:
        """Refresh request-scoped values used by Wan disaggregated inference."""
        self.config = config
        self.infer_steps = int(config["infer_steps"])
        self.sample_shift = float(config["sample_shift"])
        self.sample_guide_scale = float(config["sample_guide_scale"])
        self.caching_records = [True] * self.infer_steps
        self.caching_records_2 = [True] * self.infer_steps
        self.step_index = 0

    def set_timesteps(self, infer_steps: int | None = None, device=AI_DEVICE) -> None:
        """Build the exact upstream custom shifted-sigma schedule."""
        if infer_steps is not None:
            self.infer_steps = int(infer_steps)
        if self.infer_steps <= 0:
            raise ValueError(f"infer_steps must be positive, got {self.infer_steps}")

        # Upstream get_sampling_sigmas uses NumPy float64, then the solver casts
        # sigmas to float32 only after deriving integer timesteps. Keep that
        # ordering because float rounding can change a truncated timestep by 1.
        sigmas = np.linspace(1.0, 0.0, self.infer_steps + 1)[: self.infer_steps]
        sigmas = self.sample_shift * sigmas / (1.0 + (self.sample_shift - 1.0) * sigmas)
        timestep_values = sigmas * self.num_train_timesteps

        self.timesteps = torch.from_numpy(timestep_values).to(device=device, dtype=torch.int64)
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.caching_records = [True] * self.infer_steps
        self.caching_records_2 = [True] * self.infer_steps

    def _prepare_latents(self, seed, latent_shape) -> None:
        # The upstream pipeline creates one device generator per request and
        # keeps consuming it across video clips. Reusing an existing generator
        # in reset() preserves that behavior.
        if self.generator is None:
            self.generator = torch.Generator(device=AI_DEVICE).manual_seed(int(seed))
        self.latents = torch.randn(
            latent_shape,
            generator=self.generator,
            dtype=torch.float32,
            device=AI_DEVICE,
        )

    def prepare(self, seed, latent_shape, image_encoder_output=None) -> None:
        del image_encoder_output
        self._prepare_latents(seed, latent_shape)
        self.set_timesteps(device=AI_DEVICE)
        self.step_index = 0
        self.noise_pred = None
        self.latents_input = self.latents
        self.timestep_input = None

    def reset(self, seed, latent_shape, step_index=None) -> None:
        """Start another source-compatible video clip within the same request."""
        self._prepare_latents(seed, latent_shape)
        self.set_timesteps(device=AI_DEVICE)
        self.step_index = 0 if step_index is None else int(step_index)
        self.noise_pred = None
        self.latents_input = self.latents
        self.timestep_input = None

    def step_pre(self, step_index) -> None:
        step_index = int(step_index)
        if not 0 <= step_index < self.infer_steps:
            raise IndexError(f"step_index={step_index} is outside [0, {self.infer_steps})")
        self.step_index = step_index
        self.latents_input = self.latents
        self.current_timestep = self.timesteps[step_index]
        self.timestep_input = self.current_timestep.reshape(1)

    @staticmethod
    def _alpha(sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 - sigma

    def _solver_scalars(self, sample: torch.Tensor, *indices: int) -> tuple[torch.Tensor, ...]:
        del sample
        # Upstream intentionally keeps scheduler sigmas as CPU scalar tensors;
        # PyTorch accepts those in CUDA expressions.  Preserving that detail
        # avoids a small ULP drift in the 10/40-step source-parity profiles.
        return tuple(self.sigmas[index].to(dtype=torch.float32) for index in indices)

    def _convert_flow_to_x0(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        (sigma_s,) = self._solver_scalars(sample, self.step_index)
        return sample - sigma_s * model_output

    def _first_order_update(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigma_t, sigma_s = self._solver_scalars(sample, self.step_index + 1, self.step_index)
        alpha_t, alpha_s = self._alpha(sigma_t), self._alpha(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        exp_neg_h_minus_one = torch.exp(-h) - 1.0
        return (sigma_t / sigma_s) * sample - alpha_t * exp_neg_h_minus_one * model_output

    def _second_order_midpoint_update(self, sample: torch.Tensor) -> torch.Tensor:
        sigma_t, sigma_s0, sigma_s1 = self._solver_scalars(
            sample,
            self.step_index + 1,
            self.step_index,
            self.step_index - 1,
        )
        alpha_t = self._alpha(sigma_t)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(self._alpha(sigma_s0)) - torch.log(sigma_s0)
        lambda_s1 = torch.log(self._alpha(sigma_s1)) - torch.log(sigma_s1)

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r_0 = h_0 / h
        model_s0, model_s1 = self.model_outputs[-1], self.model_outputs[-2]
        # Match the source expression order exactly (it is observably different
        # from division after BF16 DiT outputs have been promoted to FP32).
        d_1 = (1.0 / r_0) * (model_s0 - model_s1)
        exp_neg_h_minus_one = torch.exp(-h) - 1.0
        return (sigma_t / sigma_s0) * sample - alpha_t * exp_neg_h_minus_one * model_s0 - 0.5 * alpha_t * exp_neg_h_minus_one * d_1

    @torch.no_grad()
    def step(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Advance one sequential denoising step."""
        if not 0 <= self.step_index < self.infer_steps:
            raise IndexError(f"step_index={self.step_index} is outside [0, {self.infer_steps})")

        # Preserve the BF16 dtype produced at the source model's outer FSDP
        # boundary.  In particular, ``sigma * model_output`` must round as a
        # BF16 tensor before it is subtracted from the FP32 latent.  Casting it
        # to FP32 early is numerically invisible in step 0 (sigma == 1), but
        # changes every later multistep update.
        sample = sample.to(dtype=torch.float32)
        converted = self._convert_flow_to_x0(model_output, sample)
        self.model_outputs[0] = self.model_outputs[1]
        self.model_outputs[1] = converted

        # Upstream lower_order_final=True and final_sigmas_type="zero", so the
        # final update is first order. The first update is also first order while
        # the multistep history warms up; all intervening updates use midpoint.
        use_first_order = self.lower_order_nums < 1 or self.step_index == self.infer_steps - 1
        if use_first_order:
            previous_sample = self._first_order_update(converted, sample)
        else:
            previous_sample = self._second_order_midpoint_update(sample)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        return previous_sample.to(dtype=converted.dtype)

    @torch.no_grad()
    def step_post(self) -> None:
        if self.noise_pred is None:
            raise RuntimeError("WanAnimate2Scheduler requires noise_pred before step_post().")
        self.latents = self.step(self.noise_pred, self.latents)
        self.latents_input = self.latents
        self.noise_pred = None

    def clear(self) -> None:
        self.generator = None
        self.latents = None
        self.latents_input = None
        self.noise_pred = None
        self.timesteps = None
        self.sigmas = None
        self.timestep_input = None
        self.current_timestep = None
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.step_index = 0
