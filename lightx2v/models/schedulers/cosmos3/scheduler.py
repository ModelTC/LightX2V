import copy
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class Cosmos3UniPCMultistepScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        trained_betas=None,
        solver_order=2,
        prediction_type="flow_prediction",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value=1.0,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        disable_corrector=None,
        solver_p=None,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
        use_flow_sigmas=False,
        flow_shift=1.0,
        timestep_spacing="linspace",
        steps_offset=0,
        final_sigmas_type="zero",
        rescale_betas_zero_snr=False,
        use_dynamic_shifting=False,
        time_shift_type="exponential",
        sigma_min=None,
        sigma_max=None,
        shift_terminal=None,
        **_,
    ):
        disable_corrector = [] if disable_corrector is None else disable_corrector
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            solver_order=solver_order,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            predict_x0=predict_x0,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            disable_corrector=disable_corrector,
            solver_p=solver_p,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            use_flow_sigmas=use_flow_sigmas,
            flow_shift=flow_shift,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            final_sigmas_type=final_sigmas_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            shift_terminal=shift_terminal,
        )
        if use_beta_sigmas:
            raise NotImplementedError("Cosmos3 scheduler does not support beta sigmas without scipy.")
        if sum([use_beta_sigmas, use_exponential_sigmas, use_karras_sigmas]) > 1:
            raise ValueError("Only one of beta, exponential, or Karras sigmas can be enabled.")
        if rescale_betas_zero_snr:
            raise NotImplementedError("Cosmos3 scheduler does not implement zero-terminal-SNR beta rescaling.")
        if shift_terminal is not None and not use_flow_sigmas:
            raise ValueError("shift_terminal is only supported when use_flow_sigmas=True.")
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.config.solver_type = "bh2"
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.init_noise_sigma = 1.0
        self.predict_x0 = predict_x0
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self.this_order = None
        self.num_inference_steps = None
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index=0):
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("mu must be passed when use_dynamic_shifting=True.")
        if sigmas is not None:
            if not self.config.use_flow_sigmas:
                raise ValueError("Custom sigmas are only supported when use_flow_sigmas=True.")
            num_inference_steps = len(sigmas)

        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // (num_inference_steps + 1)
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.config.timestep_spacing}")

        if self.config.use_karras_sigmas:
            if sigmas is None:
                sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(sigmas, num_inference_steps)
            if self.config.use_flow_sigmas:
                sigmas = sigmas / (sigmas + 1)
                timesteps = (sigmas * self.config.num_train_timesteps).copy()
            else:
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigma_last = self._get_last_sigma(sigmas)
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        elif self.config.use_flow_sigmas:
            if sigmas is None:
                sigmas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)[:-1]
            if self.config.use_dynamic_shifting:
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                sigmas = self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas)
            if self.config.shift_terminal:
                sigmas = self.stretch_shift_to_terminal(sigmas)
            if np.fabs(sigmas[0] - 1) < 1e-6:
                sigmas[0] -= 1e-6
            timesteps = (sigmas * self.config.num_train_timesteps).copy()
            sigma_last = self._get_last_sigma(sigmas)
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        else:
            if sigmas is None:
                sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_last = self._get_last_sigma(sigmas)
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to("cpu")
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)
        self._step_index = None
        self._begin_index = None

    def _get_last_sigma(self, sigmas):
        if self.config.final_sigmas_type == "sigma_min":
            return sigmas[-1]
        if self.config.final_sigmas_type == "zero":
            return 0
        raise ValueError(f"Unsupported final_sigmas_type: {self.config.final_sigmas_type}")

    def time_shift(self, mu, sigma, t):
        if self.config.time_shift_type == "exponential":
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        if self.config.time_shift_type == "linear":
            return mu / (mu + (1 / t - 1) ** sigma)
        raise ValueError(f"Unsupported time_shift_type: {self.config.time_shift_type}")

    def stretch_shift_to_terminal(self, t):
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        return 1 - (one_minus_z / scale_factor)

    def _threshold_sample(self, sample):
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value).unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample.to(dtype)

    @staticmethod
    def _sigma_to_t(sigma, log_sigmas):
        log_sigma = np.log(np.maximum(sigma, 1e-10))
        dists = log_sigma - log_sigmas[:, np.newaxis]
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.reshape(sigma.shape)

    def _sigma_to_alpha_sigma_t(self, sigma):
        if self.config.use_flow_sigmas:
            return 1 - sigma, sigma
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        return alpha_t, sigma * alpha_t

    def _convert_to_karras(self, in_sigmas, num_inference_steps):
        sigma_min = self.config.sigma_min if self.config.sigma_min is not None else in_sigmas[-1].item()
        sigma_max = self.config.sigma_max if self.config.sigma_max is not None else in_sigmas[0].item()
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def convert_model_output(self, model_output, sample):
        sigma = self.sigmas[self.step_index].to(sample.device)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.config.prediction_type == "flow_prediction":
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")
            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred

        if self.config.prediction_type == "epsilon":
            return model_output
        if self.config.prediction_type == "sample":
            return (sample - alpha_t * model_output) / sigma_t
        if self.config.prediction_type == "v_prediction":
            return alpha_t * model_output + sigma_t * sample
        raise ValueError(f"Unsupported prediction_type for predict_x0=False: {self.config.prediction_type}")

    def multistep_uni_p_bh_update(self, model_output, sample, order):
        model_output_list = self.model_outputs
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            return self.solver_p.step(model_output, s0, x).prev_sample

        device = sample.device
        sigma_t = self.sigmas[self.step_index + 1].to(device)
        sigma_s0 = self.sigmas[self.step_index].to(device)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device))
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.ones((), device=device))
        rks = torch.stack(rks)
        matrix_r = []
        vector_b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            matrix_r.append(torch.pow(rks, i - 1))
            vector_b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        matrix_r = torch.stack(matrix_r)
        vector_b = torch.stack(vector_b) if len(vector_b) > 0 else torch.tensor(vector_b, device=device)

        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.ones(1, dtype=x.dtype, device=device) * 0.5
            else:
                rhos_p = torch.linalg.solve(matrix_r[:-1, :-1], vector_b[:-1]).to(device).to(x.dtype)
        else:
            d1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t_ - alpha_t * b_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t_ - sigma_t * b_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        device = this_sample.device
        sigma_t = self.sigmas[self.step_index].to(device)
        sigma_s0 = self.sigmas[self.step_index - 1].to(device)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device))
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.ones((), device=device))
        rks = torch.stack(rks)
        matrix_r = []
        vector_b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            matrix_r.append(torch.pow(rks, i - 1))
            vector_b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        matrix_r = torch.stack(matrix_r)
        vector_b = torch.stack(vector_b) if len(vector_b) > 0 else torch.tensor(vector_b, device=device)
        d1s = torch.stack(d1s, dim=1) if len(d1s) > 0 else None
        if order == 1:
            rhos_c = torch.ones(1, dtype=x.dtype, device=device) * 0.5
        else:
            rhos_c = torch.linalg.solve(matrix_r, vector_b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * (model_t - m0))
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            x_t = x_t_ - sigma_t * b_h * (corr_res + rhos_c[-1] * (model_t - m0))
        return x_t.to(x.dtype)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        index_candidates = (schedule_timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            return len(self.timesteps) - 1
        if len(index_candidates) > 1:
            return index_candidates[1].item()
        return index_candidates[0].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output, timestep, sample, return_dict=True):
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps must be called before step.")
        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SimpleNamespace(prev_sample=prev_sample)


class Cosmos3Scheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.sample_guide_scale = float(config.get("sample_guide_scale", 4.0))
        scheduler_path = config.get("scheduler_path", os.path.join(config["model_path"], "scheduler"))
        with open(os.path.join(scheduler_path, "scheduler_config.json"), "r") as f:
            self.scheduler_config = json.load(f)
        sample_shift = config.get("sample_shift", 3.0)
        if sample_shift is not None:
            self.scheduler_config["flow_shift"] = float(sample_shift)
        self.unipc = None
        self.sound_unipc = None
        self.action_unipc = None
        self.timesteps = None
        self.noise_pred = None
        self.noise_pred_sound = None
        self.noise_pred_action = None
        self.keep_latents_dtype_in_scheduler = True

    def _build_unipc(self):
        kwargs = dict(self.scheduler_config)
        kwargs.pop("_class_name", None)
        kwargs.pop("_diffusers_version", None)
        return Cosmos3UniPCMultistepScheduler(**kwargs)

    def prepare_latents(self, input_info):
        shape = tuple(input_info.target_shape)
        condition_latents = getattr(input_info, "vision_condition_latents", None)
        condition_frame_indexes = getattr(input_info, "vision_condition_frame_indexes", None)
        if condition_latents is not None:
            shape = tuple(condition_latents.shape)
            input_info.target_shape = shape
        if not shape:
            height = int(self.config.get("target_height", 1024))
            width = int(self.config.get("target_width", 1024))
            scale = int(self.config.get("vae_scale_factor_spatial", self.config.get("vae_scale_factor", 16)))
            channels = int(self.config.get("latent_channel", 48))
            temporal_scale = int(self.config.get("vae_scale_factor_temporal", 4))
            frames = (int(self.config.get("target_video_length", 1)) - 1) // temporal_scale + 1
            shape = (1, channels, frames, height // scale, width // scale)
            input_info.target_shape = shape
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(int(input_info.seed))
        noise = torch.randn(shape, generator=self.generator, device=AI_DEVICE, dtype=GET_DTYPE())
        self.vision_condition_frame_indexes = None
        self.vision_condition_mask = None
        if condition_latents is not None:
            condition_latents = condition_latents.to(device=AI_DEVICE, dtype=GET_DTYPE())
            latent_t = int(shape[2])
            if condition_frame_indexes is None:
                condition_frame_indexes = [0]
            condition_frame_indexes = [int(idx) for idx in condition_frame_indexes if 0 <= int(idx) < latent_t]
            mask = torch.zeros((latent_t,), device=AI_DEVICE, dtype=GET_DTYPE())
            if condition_frame_indexes:
                mask[condition_frame_indexes] = 1.0
            mask = mask.view(1, 1, latent_t, 1, 1)
            self.latents = mask * condition_latents + (1.0 - mask) * noise
            self.vision_condition_frame_indexes = condition_frame_indexes
            self.vision_condition_mask = mask
        else:
            self.latents = noise
        self.sound_latents = None
        if self.config.get("enable_sound", False) or self.config.get("task") in ("t2av", "i2av"):
            sound_shape = getattr(input_info, "sound_latent_shape", None) or getattr(input_info, "audio_latent_shape", None)
            if not sound_shape:
                sound_dim = int(self.config.get("sound_dim", 64))
                sound_len = int(self.config.get("sound_latent_length", 0))
                if sound_len <= 0:
                    num_frames = int(self.config.get("target_video_length", 189))
                    fps = float(self.config.get("target_fps", 24.0))
                    sampling_rate = int(self.config.get("sound_sampling_rate", 48000))
                    hop_size = int(self.config.get("sound_hop_size", 1920))
                    sound_len = (int(num_frames / fps * sampling_rate) + hop_size - 1) // hop_size
                sound_shape = (sound_dim, sound_len)
            self.sound_latents = torch.randn(tuple(sound_shape), generator=self.generator, device=AI_DEVICE, dtype=GET_DTYPE())

        self.action_latents = None
        self.action_domain_id = getattr(input_info, "action_domain_id", None)
        self.action_condition_frame_indexes = getattr(input_info, "action_condition_frame_indexes", None)
        self.raw_action_dim = getattr(input_info, "raw_action_dim", None)
        action_latents = getattr(input_info, "action_latents", None)
        if action_latents is not None:
            self.action_latents = action_latents.to(device=AI_DEVICE, dtype=GET_DTYPE())
            if self.raw_action_dim is not None:
                self.action_latents[:, int(self.raw_action_dim) :] = 0
        else:
            action_shape = getattr(input_info, "action_latent_shape", None)
            if action_shape is not None:
                self.action_latents = torch.randn(tuple(action_shape), generator=self.generator, device=AI_DEVICE, dtype=GET_DTYPE())
                if self.raw_action_dim is not None:
                    self.action_latents[:, int(self.raw_action_dim) :] = 0
        self.noise_pred = None
        self.noise_pred_sound = None
        self.noise_pred_action = None

    def prepare(self, input_info):
        self.prepare_latents(input_info)
        self.unipc = self._build_unipc()
        self.unipc.set_timesteps(int(self.config["infer_steps"]), device=AI_DEVICE)
        self.timesteps = self.unipc.timesteps
        self.sound_unipc = copy.deepcopy(self.unipc) if self.sound_latents is not None else None
        self.action_unipc = copy.deepcopy(self.unipc) if self.action_latents is not None else None
        self.infer_steps = len(self.timesteps)
        self.step_index = 0

    def step_pre(self, step_index):
        self.step_index = int(step_index)
        if GET_DTYPE() == GET_SENSITIVE_DTYPE() and not self.keep_latents_dtype_in_scheduler:
            self.latents = self.latents.to(GET_DTYPE())
        self.current_timestep = self.timesteps[self.step_index]

    def step_post(self):
        if self.noise_pred is None:
            raise RuntimeError("Cosmos3Scheduler requires noise_pred before step_post().")
        t = self.timesteps[self.step_index]
        self.latents = self.unipc.step(
            self.noise_pred.unsqueeze(0),
            t,
            self.latents.unsqueeze(0),
            return_dict=False,
        )[0].squeeze(0)
        if self.sound_latents is not None:
            if self.noise_pred_sound is None:
                raise RuntimeError("Cosmos3Scheduler requires noise_pred_sound before sound step_post().")
            self.sound_latents = self.sound_unipc.step(
                self.noise_pred_sound.unsqueeze(0),
                t,
                self.sound_latents.unsqueeze(0),
                return_dict=False,
            )[0].squeeze(0)
        if self.action_latents is not None and self.noise_pred_action is not None:
            self.action_latents = self.action_unipc.step(
                self.noise_pred_action.unsqueeze(0),
                t,
                self.action_latents.unsqueeze(0),
                return_dict=False,
            )[0].squeeze(0)
            if self.raw_action_dim is not None:
                self.action_latents[:, int(self.raw_action_dim) :] = 0
        self.noise_pred = None
        self.noise_pred_sound = None
        self.noise_pred_action = None

    def clear(self):
        self.generator = None
        self.latents = None
        self.sound_latents = None
        self.action_latents = None
        self.timesteps = None
        self.unipc = None
        self.sound_unipc = None
        self.action_unipc = None
        self.noise_pred = None
        self.noise_pred_sound = None
        self.noise_pred_action = None
        self.vision_condition_frame_indexes = None
        self.vision_condition_mask = None
        self.action_domain_id = None
        self.action_condition_frame_indexes = None
        self.raw_action_dim = None
