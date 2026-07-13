import math

import torch

from lightx2v_train.runtime.distributed import get_device
from lightx2v_train.schedulers.time_shift import build_time_shift_mu
from lightx2v_train.utils.utils import get_running_dtype


class RectifiedFlowMatchingScheduler:
    def __init__(self, config):
        self.config = config
        self.device = get_device()

        scheduler_config = config["scheduler"]
        self.num_train_timesteps = scheduler_config.get("num_train_timesteps", 1000)
        self.timestep_distribution = scheduler_config.get("timestep_distribution", "logitnormal")

        self.logitnormal_mean = scheduler_config.get("logitnormal_mean", 0.0)
        self.logitnormal_std = scheduler_config.get("logitnormal_std", 1.0)

        self.min_t = scheduler_config.get("min_t", 0.001)
        self.max_t = scheduler_config.get("max_t", 1.0)

        time_shift_settings = scheduler_config["time_shift_settings"]
        self.do_time_shift = time_shift_settings.get("do_time_shift", False)
        self.time_shift_power = time_shift_settings.get("time_shift_power", 1.0)
        self.shift_type = time_shift_settings.get("shift_type", "linear")
        self.time_shift_mu = build_time_shift_mu(time_shift_settings)

        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])

        # ==============================
        # The following attributes are for inference only
        # ==============================
        self.infer_sigmas = None
        self.infer_timesteps = None
        self.num_inference_steps = None

    def sample_timestep_or_sigma(self, num_samples, latent_hw=None):
        if self.timestep_distribution == "logitnormal":
            timestep_or_sigma = torch.randn((num_samples,), device=self.device, dtype=torch.float32) * self.logitnormal_std + self.logitnormal_mean
            timestep_or_sigma = torch.sigmoid(timestep_or_sigma)
            timestep_or_sigma = timestep_or_sigma * (self.max_t - self.min_t) + self.min_t  # [0, 1] -> [min_t, max_t]
        elif self.timestep_distribution == "uniform":
            timestep_or_sigma = torch.rand((num_samples,), device=self.device)
            timestep_or_sigma = timestep_or_sigma * (self.max_t - self.min_t) + self.min_t  # [0, 1] -> [min_t, max_t]
        else:
            raise ValueError(f"Unsupported timestep distribution: {self.timestep_distribution}")
        if self.do_time_shift:
            timestep_or_sigma = self.time_shift(timestep_or_sigma, latent_hw=latent_hw)
        return timestep_or_sigma.to(self.running_dtype)

    def time_shift(self, t, latent_hw=None, num_steps=None):
        mu = self.time_shift_mu(latent_hw=latent_hw, num_steps=num_steps)
        if self.shift_type == "exponential":
            mu = math.exp(mu)
        return mu / (mu + (1 / t - 1) ** self.time_shift_power)

    def add_noise(self, latent, noise, sigmas):
        sigmas = self._expand_to_ndim(sigmas, latent.ndim)
        return (1.0 - sigmas) * latent + sigmas * noise

    def build_train_gt(self, latent, noise):
        return noise - latent

    def _expand_to_ndim(self, values, ndim):
        if values.ndim == 0:
            values = values.reshape(1)
        return values.reshape(values.shape[0], *([1] * (ndim - 1)))

    # ==============================
    # The following methods are for inference only
    # ==============================
    def set_timesteps(self, num_inference_steps, sigmas=None, latent_hw=None):
        self.num_inference_steps = num_inference_steps

        if sigmas is None:
            sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            if self.do_time_shift:
                sigmas = self.time_shift(sigmas, latent_hw=latent_hw, num_steps=num_inference_steps)
        else:
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.infer_sigmas = torch.cat([sigmas, torch.zeros(1)]).to(self.device)
        self.infer_timesteps = (sigmas * self.num_train_timesteps).to(self.device)

    def step(self, model_output, step_index, latent):
        f"""
        ADD NOISE:
            x_t = (1 - sigma_t) * x_0 + sigma_t * N  ------ self.add_noise(...)
            =>  x_t = sigma_t * (N - x_0) + x_0
            =>  x_t = sigma_t * v + x_0
        REMOVE NOISE:
            x_t = sigma_t * v + x_0
            x_t-1 = sigma_t-1 * v + x_0
            =>  x_t - x_t-1 = (sigma_t - sigma_t-1) * v
            =>  x_t-1 = x_t + (sigma_t-1 - sigma_t) * v
            =>  x_t-1 = x_t + (sigma_next - sigma) * model_output  ------------------------ (*)
        """
        sigma = self.infer_sigmas[step_index]
        sigma_next = self.infer_sigmas[step_index + 1]
        prev_sample = latent + (sigma_next - sigma) * model_output  # --------------------- (*) from above
        return prev_sample


class CausalForcingFlowMatchScheduler:
    def __init__(self, num_train_timesteps=1000, time_shift_settings=None, shift=None, sigma_min=0.0, extra_one_step=True):
        self.num_train_timesteps = int(num_train_timesteps)
        if time_shift_settings is None:
            time_shift_settings = {}

        if shift is None:
            if time_shift_settings.get("do_time_shift", True):
                shift = float(time_shift_settings.get("time_shift_mu", 5.0))
                shift_type = time_shift_settings.get("shift_type", "linear")
                if shift_type == "exponential":
                    shift = math.exp(shift)
                elif shift_type != "linear":
                    raise ValueError(f"Unsupported shift_type for CausalForcingFlowMatchScheduler: {shift_type}")
            else:
                shift = 1.0

        self.shift = float(shift)
        self.time_shift_power = float(time_shift_settings.get("time_shift_power", 1.0))
        sigma_min = time_shift_settings.get("sigma_min", sigma_min)
        extra_one_step = time_shift_settings.get("extra_one_step", extra_one_step)
        self.sigma_min = float(sigma_min)
        self.extra_one_step = bool(extra_one_step)
        self.set_timesteps(self.num_train_timesteps, training=True)

    def set_timesteps(self, num_inference_steps=1000, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (1.0 - self.sigma_min) * denoising_strength
        num_steps = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        if self.time_shift_power == 1.0:
            self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        else:
            self.sigmas = self.shift / (self.shift + (1 / sigmas - 1) ** self.time_shift_power)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())

    def sample_chunkwise(self, batch_size, num_frames, num_frame_per_chunk, device, dtype):
        index = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size, num_frames),
            device=device,
            dtype=torch.long,
        )
        index = index.reshape(batch_size, -1, num_frame_per_chunk)
        index[:, :, 1:] = index[:, :, 0:1]
        index = index.reshape(batch_size, num_frames)

        sigmas = self.sigmas.to(device=device, dtype=dtype)[index]
        weights = self.linear_timesteps_weights.to(device=device, dtype=torch.float32)[index]
        return sigmas, weights

    def sample_clean_augmentation(self, batch_size, num_frames, num_frame_per_chunk, max_timestep, device, dtype):
        index = torch.randint(
            int(max_timestep),
            self.num_train_timesteps,
            (batch_size, num_frames),
            device=device,
            dtype=torch.long,
        )
        index = index.reshape(batch_size, -1, num_frame_per_chunk)
        index[:, :, 1:] = index[:, :, 0:1]
        index = index.reshape(batch_size, num_frames)
        return self.sigmas.to(device=device, dtype=dtype)[index]

    def add_noise(self, latent, noise, sigmas):
        sigmas = sigmas.reshape(sigmas.shape[0], 1, sigmas.shape[1], 1, 1)
        return (1.0 - sigmas) * latent + sigmas * noise


class WanContinuousFlowMatchScheduler:
    """Continuous flow matching used by FastWAM video and action experts."""

    def __init__(self, num_train_timesteps=1000, shift=5.0, eps=1e-10):
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        if shift <= 0:
            raise ValueError("shift must be positive.")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(value, shift):
        return shift * value / (1.0 + (shift - 1.0) * value)

    def _precompute_training_weight_stats(self):
        steps = self.num_train_timesteps
        grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        timestep = self._phi(grid, self.shift) * float(steps)
        weights = torch.exp(-2.0 * ((timestep - steps / 2.0) / steps) ** 2)
        minimum = float(weights.min().item())
        return minimum, float((weights - minimum).mean().item())

    def sample_training_t(self, batch_size, device, dtype):
        value = torch.rand((batch_size,), device=device, dtype=torch.float32)
        return (self._phi(value, self.shift) * self.num_train_timesteps).to(dtype=dtype)

    def training_weight(self, timestep):
        timestep = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        weight = torch.exp(-2.0 * ((timestep - steps / 2.0) / steps) ** 2)
        weight = (weight - self._y_min) / (self._weight_norm_const + self.eps)
        return weight.reshape(()) if weight.numel() == 1 else weight

    def add_noise(self, original_samples, noise, timestep):
        sigma = (timestep / self.num_train_timesteps).to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim:
            sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample, noise, timestep):
        del timestep
        return noise - sample

    def build_inference_schedule(self, num_inference_steps, device, dtype, shift_override=None):
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        shift = self.shift if shift_override is None else float(shift_override)
        values = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        sigmas = self._phi(values, shift)
        timesteps = sigmas[:-1] * self.num_train_timesteps
        return timesteps.to(dtype=dtype), (sigmas[1:] - sigmas[:-1]).to(dtype=dtype)

    @staticmethod
    def step(model_output, delta, sample):
        delta = delta.to(device=sample.device, dtype=sample.dtype)
        if delta.ndim:
            delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
