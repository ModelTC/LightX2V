import math
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .flow_match import FlowMatchScheduler

class FlowMatchPairScheduler(FlowMatchScheduler, SchedulerMixin, ConfigMixin):
    """
    On the basis of FlowMatchScheduler, provide a convenient pairing interface:
    - Default return tensor with shape [num_timesteps, 2], each row is (t, t)
    - Allow setting a post-processing function through set_pair_postprocess(fn) to modify pairing behavior
    
    Usage example:
        scheduler = FlowMatchPairScheduler(num_inference_steps=10)
        pairs = scheduler.get_pairs()  # Pairing based on timesteps (t, t)
        sigma_pairs = scheduler.get_pairs(source="sigmas")  # Pairing based on sigmas (t, t)
    """

    @register_to_config
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self._pair_postprocess_fn = None
        self._pair_postprocess_requires_source = False
        self.pair_timesteps: torch.Tensor | None = None
        self.pair_sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )

    def set_pair_postprocess(self, fn):
        """
        Set a post-processing function, used to customize the pairing behavior after the default pairing generation.
        Requirements:
        - fn(pairs: torch.Tensor) -> torch.Tensor
        - The returned tensor must have the same shape as the input pairs, otherwise raise directly.
        """
        if fn is not None and not callable(fn):
            raise TypeError("pair_postprocess must be a callable object or None")
        self._pair_postprocess_fn = fn
        self._pair_postprocess_requires_source = False if fn is None else bool(getattr(fn, "_requires_source", False))
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        self._refresh_pair_cache()

    def set_pair_postprocess_by_name(self, name: str | None, **kwargs):
        """
        Quickly configure the post-processing function by name.
        Supports:
        - None/"none"/"off"/"false"/"no"
        - "dual_sigma_shift"
        """
        if name is None or str(name).lower() in ("none", "off", "false", "no"):
            self.set_pair_postprocess(None)
            return
        if name == "dual_sigma_shift":
            visual_shift = float(kwargs.get("visual_shift", self.shift))
            audio_shift = float(kwargs.get("audio_shift", self.shift))
            visual_denoising_strength = float(kwargs.get("visual_denoising_strength", 1.0))
            audio_denoising_strength = float(kwargs.get("audio_denoising_strength", 1.0))
            visual_mu = kwargs.get("visual_exponential_shift_mu", self.exponential_shift_mu)
            audio_mu = kwargs.get("audio_exponential_shift_mu", self.exponential_shift_mu)

            def _dual_sigma_shift(pairs: torch.Tensor, *, source: str):
                if not isinstance(pairs, torch.Tensor):
                    raise TypeError("pairs must be a torch.Tensor")
                if pairs.ndim != 2 or pairs.shape[1] != 2:
                    raise ValueError("pairs must be a torch.Tensor with shape [N, 2]")
                if pairs.shape[0] == 0:
                    raise ValueError("pairs length must be greater than 0")
                if source not in ("timesteps", "sigmas"):
                    raise ValueError("source only supports 'timesteps' or 'sigmas'")

                num_steps = pairs.shape[0]
                device = pairs.device
                dtype = pairs.dtype

                def _build_column(shift_value: float, denoising_strength: float, mu_override):
                    if shift_value <= 0:
                        raise ValueError("shift must be a positive number")
                    if denoising_strength <= 0:
                        raise ValueError("denoising_strength must be a positive number")

                    sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
                    if self.extra_one_step:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps + 1, device=device, dtype=dtype)[:-1]
                    else:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps, device=device, dtype=dtype)

                    if self.inverse_timesteps:
                        base = torch.flip(base, dims=[0])

                    if self.exponential_shift:
                        mu_value = mu_override
                        if mu_value is None:
                            raise RuntimeError("exponential_shift is enabled but exponential_shift_mu is not provided")
                        exp_mu = math.exp(float(mu_value))
                        base = exp_mu / (exp_mu + (1 / base - 1))
                    else:
                        base = shift_value * base / (1 + (shift_value - 1) * base)

                    if self.shift_terminal is not None:
                        one_minus_z = 1 - base
                        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
                        base = 1 - (one_minus_z / scale_factor)

                    if self.reverse_sigmas:
                        base = 1 - base

                    if source == "timesteps":
                        return base * self.num_train_timesteps
                    return base

                visual_column = _build_column(visual_shift, visual_denoising_strength, visual_mu)
                audio_column = _build_column(audio_shift, audio_denoising_strength, audio_mu)
                return torch.stack([visual_column, audio_column], dim=1)

            _dual_sigma_shift._requires_source = True
            self.set_pair_postprocess(_dual_sigma_shift)
            return
        raise ValueError(f"Unsupported pair postprocessing name: {name}")

    def _make_pairs_from_vector(self, vec: torch.Tensor) -> torch.Tensor:
        if not isinstance(vec, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if vec.ndim != 1:
            raise ValueError("input vector must be a one-dimensional tensor")
        if vec.numel() == 0:
            raise ValueError("input vector length must be greater than 0")

        # Default each row is (t, t), shape [N, 2]
        pairs = torch.stack([vec, vec], dim=1)
        return pairs

    def get_pairs(self, source: str = "timesteps") -> torch.Tensor:
        """
        Generate pairing results with shape [num_timesteps, 2].
        - source="timesteps": Use self.timesteps
        - source="sigmas": Use self.sigmas
        Default each row has two columns equal; if you need to modify, you can use set_pair_postprocess(fn).
        """
        if source == "timesteps":
            pairs = self.pair_timesteps
        elif source == "sigmas":
            pairs = self.pair_sigmas
        else:
            raise ValueError("source only supports 'timesteps' or 'sigmas'")

        if pairs is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")

        return pairs

    @property
    def visual_timesteps(self) -> torch.Tensor:
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        return self.pair_timesteps[:, 0]

    @property
    def audio_timesteps(self) -> torch.Tensor:
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        return self.pair_timesteps[:, 1]

    def set_timesteps(self, *args, **kwargs):
        super().set_timesteps(*args, **kwargs)
        self._refresh_pair_cache()

    def timestep_to_sigma(self, timestep: torch.Tensor | float) -> torch.Tensor:
        """Return the corresponding sigma for the given timestep (scalar) by nearest neighbor search in self.timesteps."""
        t = timestep
        # if torch.is_tensor(t):
        #     t_tensor = t
        #     flat_cpu = t_tensor.detach().cpu().reshape(-1)
        #     zero_mask = (flat_cpu == 0)
        #     diffs = (self.timesteps.unsqueeze(-1) - flat_cpu.unsqueeze(0)).abs()
        #     idx = torch.argmin(diffs, dim=0)
        #     sigmas = self.sigmas[idx].to(device=t_tensor.device, dtype=self.sigmas.dtype)
        #     sigmas = sigmas.reshape(t_tensor.shape)
        #     if zero_mask.any():
        #         sigmas = sigmas.clone()
        #         sigmas.view(-1)[zero_mask] = 0
        #     return sigmas

        t_value = float(t)
        # if t_value == 0.0:
        #     return torch.zeros((), device=self.sigmas.device, dtype=self.sigmas.dtype)
        t_cpu = torch.tensor(t_value)
        idx = torch.argmin((self.train_timesteps - t_cpu).abs())
        return self.train_sigmas[idx]

    def step_from_to(self, model_output: torch.Tensor, timestep_from: torch.Tensor, timestep_to: torch.Tensor | None, sample: torch.Tensor) -> torch.Tensor:
        """
        Use an explicit (from, to) timestep pair to update one step:
            x_{to} = x_{from} + model_output * (sigma(to) - sigma(from))
        This method can be used to advance two modalities separately along their respective column sequences.
        """
        sigma_from = self.timestep_to_sigma(timestep_from)
        if timestep_to is None:
            # Final step: align the boundary behavior of the base implementation
            sigma_to = torch.tensor(1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0, device=sigma_from.device, dtype=sigma_from.dtype)
        else:
            sigma_to = self.timestep_to_sigma(timestep_to)
        # print(f"sigma_from: {sigma_from}, sigma_to: {sigma_to}")
        prev_sample = sample + model_output * (sigma_to - sigma_from)
        return prev_sample

    def _refresh_pair_cache(self) -> None:
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")

        def _apply_postprocess(pairs: torch.Tensor, source: str) -> torch.Tensor:
            if self._pair_postprocess_fn is None:
                return pairs
            if self._pair_postprocess_requires_source:
                modified = self._pair_postprocess_fn(pairs, source=source)
            else:
                modified = self._pair_postprocess_fn(pairs)
            if not isinstance(modified, torch.Tensor):
                raise TypeError("pair_postprocess return value must be a torch.Tensor")
            if modified.shape != pairs.shape:
                raise ValueError("pair_postprocess return tensor shape must be the same as the input")
            return modified

        base_pairs_timesteps = self._make_pairs_from_vector(self.timesteps)
        base_pairs_sigmas = self._make_pairs_from_vector(self.sigmas)

        self.pair_timesteps = _apply_postprocess(base_pairs_timesteps, "timesteps")
        self.pair_sigmas = _apply_postprocess(base_pairs_sigmas, "sigmas")
