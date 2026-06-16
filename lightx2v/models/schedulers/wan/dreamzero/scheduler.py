import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class DreamZeroFlowUniPCScheduler(WanScheduler):
    def __init__(self, config, *, shift_key="sample_shift", infer_steps_key="infer_steps"):
        scheduler_config = dict(config)
        scheduler_config["infer_steps"] = int(config[infer_steps_key])
        scheduler_config["sample_shift"] = float(config[shift_key])
        scheduler_config.setdefault("target_video_length", config.get("target_video_length", 1))
        scheduler_config.setdefault("sample_guide_scale", config.get("sample_guide_scale", 1.0))
        scheduler_config.setdefault("seq_parallel", False)
        super().__init__(scheduler_config)
        self.prediction_type = scheduler_config.get("prediction_type", "flow_prediction")
        self.use_dynamic_shifting = bool(scheduler_config.get("use_dynamic_shifting", False))
        self.thresholding = bool(scheduler_config.get("thresholding", False))
        self.dynamic_thresholding_ratio = float(scheduler_config.get("dynamic_thresholding_ratio", 0.995))
        self.sample_max_value = float(scheduler_config.get("sample_max_value", 1.0))
        self.predict_x0 = bool(scheduler_config.get("predict_x0", True))
        self.solver_type = scheduler_config.get("solver_type", "bh2")
        if self.solver_type not in ["bh1", "bh2"]:
            if self.solver_type in ["midpoint", "heun", "logrho"]:
                self.solver_type = "bh2"
            else:
                raise NotImplementedError(f"{self.solver_type} is not implemented for {self.__class__}")
        self.lower_order_final = bool(scheduler_config.get("lower_order_final", True))
        self.disable_corrector = list(scheduler_config.get("disable_corrector", []))
        self.final_sigmas_type = scheduler_config.get("final_sigmas_type", "zero")
        self.solver_p = None
        self.num_inference_steps = None
        self.loop_inputs = None
        self.step_input_builder = None
        self.noise_pred_processor = None
        self._prepare_base_sigmas()

    def _prepare_base_sigmas(self):
        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.lower_order_nums = 0
        self.this_order = None

    def set_timesteps(self, infer_steps=None, device=None, sigmas=None, mu=None, shift=None):
        self.num_inference_steps = int(infer_steps)

        if self.use_dynamic_shifting and mu is None:
            raise ValueError("DreamZeroFlowUniPCScheduler requires `mu` when dynamic shifting is enabled.")

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, self.num_inference_steps + 1).copy()[:-1]

        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.final_sigmas_type == "zero":
            sigma_last = 0
        elif self.final_sigmas_type == "sigma_min":
            sigma_last = self.sigma_min
        else:
            raise ValueError(f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {self.final_sigmas_type}.")

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None

    @staticmethod
    def time_shift(mu, sigma, t):
        return np.exp(mu) / (np.exp(mu) + (1 / t - 1) ** sigma)

    def _threshold_sample(self, sample):
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.sample_max_value).unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample.to(dtype)

    def convert_model_output(self, model_output, sample, step_index=None):
        if step_index is None:
            step_index = self.step_index
        sigma_t = self.sigmas[step_index]

        if self.predict_x0:
            if self.prediction_type != "flow_prediction":
                raise ValueError(f"Unsupported DreamZero prediction_type: {self.prediction_type}")
            x0_pred = sample - sigma_t * model_output
            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred

        if self.prediction_type != "flow_prediction":
            raise ValueError(f"Unsupported DreamZero prediction_type: {self.prediction_type}")
        epsilon = sample - (1 - sigma_t) * model_output
        if self.thresholding:
            x0_pred = sample - sigma_t * model_output
            x0_pred = self._threshold_sample(x0_pred)
            epsilon = model_output + x0_pred
        return epsilon

    def _solver_bh_terms(self, h, order, dtype, device):
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        if self.solver_type == "bh1":
            b_h = hh
        elif self.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        return hh, h_phi_1, h_phi_k, b_h

    def multistep_uni_p_bh_update(self, model_output, sample, order, step_index):
        model_output_list = self.model_outputs
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            return self.solver_p.step(model_output, s0, x).prev_sample

        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        _, h_phi_1, h_phi_k, b_h = self._solver_bh_terms(h, order, x.dtype, x.device)
        matrix_r = []
        vector_b = []
        factorial_i = 1
        hh = -h if self.predict_x0 else h
        for i in range(1, order + 1):
            matrix_r.append(torch.pow(rks, i - 1))
            vector_b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        matrix_r = torch.stack(matrix_r, dim=0)
        vector_b = torch.stack(vector_b, dim=0)

        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device)
            else:
                rhos_p = torch.linalg.solve_ex(matrix_r[:-1, :-1], vector_b[:-1])[0].to(x.dtype)
        else:
            d1s = None
            rhos_p = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t_ - alpha_t * b_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t_ - sigma_t * b_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order, step_index):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        _, h_phi_1, h_phi_k, b_h = self._solver_bh_terms(h, order, x.dtype, x.device)
        matrix_r = []
        vector_b = []
        factorial_i = 1
        hh = -h if self.predict_x0 else h
        for i in range(1, order + 1):
            matrix_r.append(torch.pow(rks, i - 1))
            vector_b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        matrix_r = torch.stack(matrix_r, dim=0)
        vector_b = torch.stack(vector_b, dim=0)
        d1s = torch.stack(d1s, dim=1) if len(d1s) > 0 else None

        if order == 1:
            rhos_c = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device)
        else:
            rhos_c = torch.linalg.solve_ex(matrix_r, vector_b)[0].to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            d1_t = model_t - m0
            x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            d1_t = model_t - m0
            x_t = x_t_ - sigma_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(x.dtype)

    def prepare_loop(self, *, infer_steps, device, latent_shape=None, latents=None, seed=None, dtype=None):
        if latents is None:
            if latent_shape is None:
                raise ValueError("DreamZeroFlowUniPCScheduler.prepare_loop requires latents or latent_shape.")
            if self.generator is None:
                self.generator = torch.Generator(device=device).manual_seed(int(seed))
            latents = torch.randn(*latent_shape, generator=self.generator, device=device, dtype=dtype)
        self.latents = latents
        self.infer_steps = int(infer_steps)
        self.set_timesteps(self.infer_steps, device=device, shift=self.sample_shift)
        self.noise_pred = None

    def bind_step_inputs(self, inputs, input_builder):
        self.loop_inputs = inputs
        self.step_input_builder = input_builder

    def bind_noise_pred_processor(self, noise_pred_processor):
        self.noise_pred_processor = noise_pred_processor

    def _match_noise_pred_to_latents(self):
        if self.noise_pred is None or self.latents is None or self.noise_pred.shape == self.latents.shape:
            return
        matched = torch.zeros_like(self.latents, dtype=self.noise_pred.dtype)
        common_slices = tuple(slice(0, min(pred_dim, latent_dim)) for pred_dim, latent_dim in zip(self.noise_pred.shape, self.latents.shape))
        matched[common_slices] = self.noise_pred[common_slices]
        self.noise_pred = matched

    def step_pre(self, step_index):
        self.step_index = int(step_index)
        if GET_DTYPE() == GET_SENSITIVE_DTYPE() and not self.keep_latents_dtype_in_scheduler:
            self.latents = self.latents.to(GET_DTYPE())
        self.current_timestep = self.timesteps[self.step_index]
        if self.loop_inputs is not None and self.step_input_builder is not None:
            self.loop_inputs.clear()
            self.loop_inputs.update(self.step_input_builder(self))

    def step_post(self):
        if self.noise_pred is None:
            raise RuntimeError("DreamZeroFlowUniPCScheduler requires noise_pred before step_post().")
        if self.noise_pred_processor is not None:
            self.noise_pred = self.noise_pred_processor(self.noise_pred)
        self._match_noise_pred_to_latents()
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is None; call set_timesteps before stepping.")

        model_output = self.noise_pred.to(torch.float32)
        timestep = self.timesteps[self.step_index]
        sample = self.latents.to(torch.float32)

        use_corrector = self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None

        model_output_convert = self.convert_model_output(model_output, sample=sample, step_index=self.step_index)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                step_index=self.step_index,
            ).clone()

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
            step_index=self.step_index,
        ).clone()

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self.latents = prev_sample

    def clear(self):
        self.latents = None
        self.noise_pred = None
        self.loop_inputs = None
        self.step_input_builder = None
        self.noise_pred_processor = None
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.lower_order_nums = 0
        self.this_order = None
