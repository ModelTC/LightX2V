from __future__ import annotations

import os
from collections.abc import Collection
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import (
    BoundCapability,
    CheckpointCapability,
    ConsistencyCapability,
    DistillationCapability,
    FlowMatchingSFTCapability,
    LossResult,
    ParallelCapability,
    SFTStepContext,
    TrainableModelCapability,
)
from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.parallel import (
    apply_parallel,
    set_parallel_gradient_sync,
)
from lightx2v_train.utils.image_size_buckets import parse_image_size_buckets

from .latent_geometry import LatentGeometry


def _require_single_prompt(prompt):
    if isinstance(prompt, str):
        return True
    prompts = list(prompt)
    if len(prompts) != 1:
        raise ValueError(f"Training requires exactly one prompt per rank; physical batch sizes greater than 1 are not supported, got {len(prompts)} prompts.")
    return False


def _negative_prompt(conditioning, fallback, scalar=False):
    value = conditioning.get("negative_prompt")
    if value is None:
        prompts = []
    elif isinstance(value, str):
        prompts = [value]
    else:
        prompts = list(value)
    if not prompts:
        prompt = fallback
    elif len(prompts) == 1:
        prompt = prompts[0]
    else:
        raise ValueError(f"Training requires exactly one negative prompt per rank; got {len(prompts)} negative prompts.")
    prompt = prompt if isinstance(prompt, str) and prompt.strip() else fallback
    return prompt if scalar else [prompt]


def _require_singleton_tensor(value, name):
    if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != 1:
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(f"{name} must have leading dimension 1; physical batch sizes greater than 1 are not supported, got {shape}.")
    return value


class CommonTrainableCapability(BoundCapability, TrainableModelCapability):
    def configure(self, train_type: str, lora_config: Mapping[str, Any]) -> None:
        if train_type == "lora":
            rank = int(lora_config.get("rank", 16))
            self.model.add_lora(
                rank,
                int(lora_config.get("alpha", rank)),
                lora_config.get("target_modules"),
            )
            self.model.set_lora_trainable()
            return
        if train_type != "full":
            raise ValueError(f"Unsupported train type {train_type!r}; expected 'lora' or 'full'.")
        self.model.set_full_trainable()

    def restore(self, train_type: str) -> None:
        if train_type == "lora":
            self.model.set_lora_trainable()
        elif train_type == "full":
            self.model.set_full_trainable()
        else:
            raise ValueError(f"Unsupported train type {train_type!r}; expected 'lora' or 'full'.")

    def parameters(self):
        return self.model.trainable_parameters()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def set_eval(self) -> None:
        self.model.set_denoiser_eval()

    def log_structure(self) -> None:
        self.model.log_model_structure()


class CommonParallelCapability(BoundCapability, ParallelCapability):
    def apply(self, config) -> None:
        apply_parallel(self.model, config)

    def set_gradient_sync(self, enabled: bool) -> None:
        set_parallel_gradient_sync(self.model, enabled)

    def is_fsdp(self) -> bool:
        return self.model.is_fsdp2_wrapped()

    def state_module(self):
        return self.model.fsdp2_state_module()


class CommonCheckpointCapability(BoundCapability, CheckpointCapability):
    _CONSISTENCY_AUXILIARY_WEIGHTS_NAME = "consistency_auxiliary.safetensors"

    def _consistency_auxiliary_parameter_names(self) -> tuple[str, ...]:
        capabilities = self.model.ensure_capabilities()
        if not capabilities.supports(ConsistencyCapability):
            return ()
        return capabilities.require(ConsistencyCapability).auxiliary_parameter_names()

    def save_weights(self, save_dir, train_type) -> None:
        if train_type == "lora":
            self.model.save_lora_weights(
                save_dir,
                auxiliary_parameter_names=self._consistency_auxiliary_parameter_names(),
                auxiliary_weights_name=self._CONSISTENCY_AUXILIARY_WEIGHTS_NAME,
            )
        elif is_main_process():
            torch.save(
                self.model.denoiser_module().state_dict(),
                os.path.join(save_dir, "model_state.pt"),
            )

    def load_weights(self, save_dir, train_type) -> None:
        if train_type == "lora":
            self.model.load_lora_weights_for_resume(save_dir)
            self.model.load_auxiliary_weights(
                save_dir,
                self._consistency_auxiliary_parameter_names(),
                weights_name=self._CONSISTENCY_AUXILIARY_WEIGHTS_NAME,
            )
            return
        path = os.path.join(save_dir, "model_state.pt")
        if not os.path.exists(path):
            raise RuntimeError(f"model_state.pt not found in {save_dir}")
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        self.model.denoiser_module().load_state_dict(state_dict)

    def save_consolidated(self, output_path) -> None:
        self.model.save_consolidated_weights(output_path)

    def save_full_model(self, output_path) -> None:
        self.model.save_full_model(output_path)


class GenericFlowMatchingCapability(BoundCapability, FlowMatchingSFTCapability):
    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: SFTStepContext,
    ) -> LossResult:
        scheduler = context.noise_scheduler
        broadcast = context.broadcast

        with torch.no_grad():
            latent = _require_singleton_tensor(
                broadcast(self.model.encode_to_latent(batch)),
                "Flow-matching latent",
            )
            noise = broadcast(torch.randn_like(latent, dtype=context.running_dtype))
            latent_hw = latent.shape[-2:]
            timestep_or_sigma = broadcast(
                scheduler.sample_timestep_or_sigma(
                    latent_hw=latent_hw,
                )
            )
            noisy_latent = scheduler.add_noise(
                latent,
                noise,
                timestep_or_sigma,
            )
            condition = broadcast(self.model.encode_condition(batch))

        prediction = self.model.predict_denoiser_output(
            noisy_latent,
            timestep_or_sigma,
            condition,
        )
        target = scheduler.build_train_gt(latent, noise)
        loss = (prediction.float() - target.float()).square().mean()
        return LossResult(loss=loss)


class GenericConsistencyCapability(BoundCapability, ConsistencyCapability):
    def configure(self, features: Collection[str]) -> None:
        features = frozenset(features)
        if features:
            names = ", ".join(sorted(features))
            raise NotImplementedError(f"{type(self.model).__name__} does not support consistency features: {names}.")

    def restore_trainable_auxiliary(self) -> None:
        pass

    def auxiliary_parameter_names(self) -> tuple[str, ...]:
        return ()

    def encode_latent(self, batch):
        return _require_singleton_tensor(
            self.model.encode_to_latent(batch),
            "Consistency latent",
        )

    def encode_condition(self, batch):
        return self.model.encode_condition(batch)

    def sampling_latent_hw(self, batch, clean) -> tuple[int, int]:
        del batch
        return int(clean.shape[-2]), int(clean.shape[-1])

    def predict(self, request, path):
        prediction = self.model.predict_denoiser_output(
            request.sample,
            request.time,
            request.condition,
            **request.model_kwargs,
        )
        return path.convert_prediction(
            request.sample,
            prediction,
            request.time,
            source_type=self.model.denoiser_prediction_type(),
            target_type=request.prediction_type,
        )

    def predict_log_variance(self, time):
        del time
        raise NotImplementedError(f"{type(self.model).__name__} does not provide a consistency log-variance head.")

    def set_frozen(self, training: bool = False) -> None:
        denoiser = self.model.denoiser_module()
        denoiser.requires_grad_(False)
        denoiser.train(training)

    def denoiser(self):
        return self.model.denoiser_module()


class GenericDistillationCapability(BoundCapability, DistillationCapability):
    """Distribution-matching operations shared by image flow models."""

    def __init__(
        self,
        model,
        *,
        latent_geometry: LatentGeometry | None = None,
        guidance_in_denoiser_space: bool = False,
    ) -> None:
        super().__init__(model)
        self._latent_geometry = latent_geometry
        self._guidance_in_denoiser_space = bool(guidance_in_denoiser_space)

    @property
    def device(self):
        return self.model.device

    @property
    def default_negative_prompt(self):
        return None

    @property
    def default_lora_target_modules(self):
        return None

    def latent_shape(
        self,
        batch,
        shape_config,
        image_sizes,
        broadcast,
    ):
        conditioning = batch["conditioning"]
        prompt = conditioning.get("prompt", "")
        _require_single_prompt(prompt)
        meta = batch.get("meta", {})
        height = self._target_dimension(
            meta,
            "target_height",
        )
        width = self._target_dimension(
            meta,
            "target_width",
        )
        height = int(broadcast(height))
        width = int(broadcast(width))

        if image_sizes:
            configured_buckets = {bucket.spatial_size for bucket in parse_image_size_buckets(image_sizes)}
            if (height, width) not in configured_buckets:
                configured = ", ".join(f"{bucket_height}x{bucket_width}" for bucket_height, bucket_width in sorted(configured_buckets))
                raise ValueError(f"Image DMD sample size {height}x{width} is not in training.dmd.image_sizes: [{configured}].")

        if self._latent_geometry is None:
            raise NotImplementedError(f"{type(self).__name__} must override latent_shape() or be constructed with a latent_geometry adapter.")
        return self._latent_geometry.shape(self.model, height, width)

    @staticmethod
    def _target_dimension(meta, key):
        if key not in meta:
            raise KeyError(f"Image DMD prompt-only samples require meta.target_height and meta.target_width; missing {key}. target_image is not used.")
        value = meta[key]
        if torch.is_tensor(value):
            values = value.detach().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            values = [item.item() if torch.is_tensor(item) else item for item in value]
        else:
            values = [value]
        values = [int(item) for item in values]
        if len(values) != 1:
            raise ValueError(f"Image DMD requires exactly one {key} value per rank, got {values}.")
        if any(item <= 0 for item in values):
            raise ValueError(f"Image DMD {key} must be positive, got {values}.")
        return values[0]

    def encode_conditions(
        self,
        batch,
        negative_prompt,
        guidance_scale,
        broadcast,
    ):
        conditioning = batch["conditioning"]
        prompt = conditioning.get("prompt", "")
        scalar = _require_single_prompt(prompt)
        with torch.no_grad():
            positive = self.model.encode_prompt_condition(prompt)
            if guidance_scale > 1:
                negative = self.model.encode_prompt_condition(
                    _negative_prompt(
                        conditioning,
                        negative_prompt,
                        scalar=scalar,
                    )
                )
            else:
                negative = None
        return (
            broadcast(positive),
            broadcast(negative) if negative is not None else None,
        )

    def predict_velocity(self, latents, sigma, condition):
        _require_singleton_tensor(latents, "DMD latent")
        return self.model.predict_denoiser_output(latents, sigma, condition)

    def predict_guided_velocity(
        self,
        latents,
        sigma,
        condition,
        negative_condition,
        guidance_scale,
        cfg_norm,
    ):
        if negative_condition is None:
            return self.predict_velocity(latents, sigma, condition)

        if self._guidance_in_denoiser_space:
            denoiser_input = self.model.prepare_denoiser_input(latents)
            positive = self.model.denoise(
                denoiser_input,
                sigma,
                condition,
            )
            negative = self.model.denoise(
                denoiser_input,
                sigma,
                negative_condition,
            )
            prediction = self._cfg(
                positive,
                negative,
                guidance_scale,
                cfg_norm,
            )
            return self.model.postprocess_denoiser_output(
                prediction,
                denoiser_input,
            )
        return self._cfg(
            self.predict_velocity(latents, sigma, condition),
            self.predict_velocity(latents, sigma, negative_condition),
            guidance_scale,
            cfg_norm,
        )

    @staticmethod
    def _cfg(positive, negative, scale, norm):
        prediction = negative + scale * (positive - negative)
        if norm in (None, "none"):
            return prediction
        if norm == "layer_norm":
            positive_norm = torch.norm(positive, dim=-1, keepdim=True)
            guided_norm = torch.norm(prediction, dim=-1, keepdim=True)
            return prediction * (positive_norm / guided_norm.clamp_min(1e-12))
        if norm == "scalar":
            ratio = torch.norm(positive) / torch.norm(prediction).clamp_min(1e-12)
            return prediction * min(1.0, ratio.item())
        raise ValueError(f"Unsupported cfg_norm: {norm}")

    def initial_latents(self, latent_shape, dtype, broadcast):
        if int(latent_shape[0]) != 1:
            raise ValueError(f"DMD latent shape must start with 1, got {latent_shape}.")
        return broadcast(torch.randn(latent_shape, device=self.device, dtype=dtype))

    @staticmethod
    def latent_hw(latent_shape):
        return latent_shape[-2:]

    @staticmethod
    def random_noise_like(latents, dtype, broadcast):
        return broadcast(torch.randn_like(latents, dtype=dtype))

    @staticmethod
    def add_noise(scheduler, latents, noise, sigma):
        return scheduler.add_noise(latents, noise, sigma)

    @staticmethod
    def training_target(latents, noise):
        return noise - latents.float()

    @staticmethod
    def step(scheduler, velocity, step_index, sample):
        return scheduler.step_by_index(velocity, step_index, sample)

    @staticmethod
    def x0_from_velocity(sample, velocity, sigma):
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        expanded = sigma.reshape(
            sigma.shape[0],
            *([1] * (sample.ndim - 1)),
        )
        return sample + (torch.zeros_like(expanded) - expanded) * velocity

    @staticmethod
    def regression_loss(prediction, target):
        return F.mse_loss(
            prediction.float(),
            target.float(),
            reduction="mean",
        )

    @staticmethod
    def dmd_loss(latents, fake_x0, teacher_x0):
        with torch.no_grad():
            gradient = fake_x0 - teacher_x0
            dimensions = tuple(range(1, latents.ndim))
            normalizer = (
                (latents - teacher_x0)
                .abs()
                .mean(
                    dim=dimensions,
                    keepdim=True,
                )
            )
            gradient = torch.nan_to_num(gradient / normalizer)
        return 0.5 * F.mse_loss(
            latents.float(),
            (latents.float() - gradient.float()).detach(),
            reduction="mean",
        )

    @staticmethod
    def detach(value):
        return value.detach()

    @staticmethod
    def to_dtype(value, dtype):
        return value.to(dtype=dtype)

    def extract_real_latents(self, batch, dtype, broadcast):
        with torch.no_grad():
            latent = batch["inputs"].get("latents")
            if latent is None:
                latent = self.model.encode_to_latent(batch)
            latent = latent.to(device=self.device, dtype=dtype)
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            _require_singleton_tensor(latent, "DMD real latent")
        return broadcast(latent)

    def set_training(self, enabled: bool) -> None:
        self.model.denoiser_module().train(enabled)

    def denoiser(self):
        return self.model.denoiser_module()
