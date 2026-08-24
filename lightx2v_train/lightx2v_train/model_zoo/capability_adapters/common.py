from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lightx2v_train.data.training_cache import preserve_cache_dtype
from lightx2v_train.model_capabilities import (
    BoundCapability,
    CheckpointCapability,
    ConsistencyModelCapability,
    DistributionMatchingCapability,
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


def _move_cached_value(value, model, key=None):
    if torch.is_tensor(value):
        kwargs = {"device": model.device}
        if value.is_floating_point() and not preserve_cache_dtype(key):
            kwargs["dtype"] = model.running_dtype
        return value.to(**kwargs)
    if isinstance(value, Mapping):
        return {name: _move_cached_value(item, model, name) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_cached_value(item, model, key) for item in value)
    if isinstance(value, list):
        return [_move_cached_value(item, model, key) for item in value]
    return value


def _cached_latent(batch, model):
    latent = batch.get("inputs", {}).get("latents")
    if latent is not None:
        return _move_cached_value(latent, model)
    cache_path = batch.get("meta", {}).get("training_cache_path")
    if cache_path is not None:
        raise KeyError(f"Training cache {cache_path} has no inputs.latents entry.")
    return None


def _cached_condition(batch, model, role=None):
    conditioning = batch.get("conditioning", {})
    if "positive" not in conditioning:
        cache_path = batch.get("meta", {}).get("training_cache_path")
        if cache_path is not None:
            raise KeyError(f"Training cache {cache_path} has no conditioning.positive entry.")
        return None
    role = role or conditioning.get("active", "positive")
    if role not in conditioning:
        cache_path = batch.get("meta", {}).get("training_cache_path", "<unknown>")
        raise KeyError(f"Training cache {cache_path} has no conditioning.{role} entry.")
    condition = dict(conditioning.get("shared", {}))
    condition.update(conditioning[role])
    return _move_cached_value(condition, model)


def _cached_condition_pair(batch, model, *, require_negative):
    positive = _cached_condition(batch, model)
    if positive is None:
        return None
    negative = _cached_condition(batch, model, role="negative") if require_negative else None
    return positive, negative


def _prompt_or_default(value, default):
    return value if isinstance(value, str) else default


def _uses_prompt_dropout(model):
    return float(model.config.get("data", {}).get("train", {}).get("prompt_dropout_rate", 0.0)) > 0


def _consistency_requires_negative(model):
    config = model.config["training"].get("consistency", {})
    algorithm = str(config.get("algorithm", "cm")).lower()
    teacher = config.get("teacher", {})
    guidance_scale = teacher.get("guidance_scale")
    if algorithm in {"mean_flow", "meanflow"}:
        guidance = config.get("guidance", {})
        return any(
            value is not None
            for value in (
                guidance.get("condition_dropout_probability"),
                guidance.get("scale", guidance_scale),
                guidance.get("mixture_ratio"),
            )
        )
    if algorithm == "pcm":
        return guidance_scale is not None and float(guidance_scale) != 1.0
    return str(config.get("mode", "ct")).lower() == "cd" and guidance_scale is not None and float(guidance_scale) != 1.0


def _target_cache_inputs(model, batch, *, required):
    target = batch.get("inputs", {}).get("target_pixel_values")
    if target is None:
        if required:
            raise ValueError("Training-cache encoding requires inputs.target_pixel_values.")
        return {}
    return {"latents": model.encode_to_cache_latent(batch)}


def _encode_cache_conditions(model, batch, prompts, contextual_roles):
    conditions = model.encode_condition_roles(
        batch,
        prompts,
        contextual_roles=contextual_roles,
    )
    result = {}
    for role, condition in conditions.items():
        if not isinstance(condition, Mapping):
            raise TypeError(f"Encoded {role} condition must be a mapping, got {type(condition).__name__}.")
        result[role] = dict(condition)
    return result


def _cache_values_equal(left, right):
    if left is right:
        return True
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_cache_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_cache_values_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _factor_shared_conditions(conditions, shared_keys):
    shared = {}
    for key in shared_keys:
        if not all(key in condition for condition in conditions.values()):
            continue
        values = [condition[key] for condition in conditions.values()]
        if not all(_cache_values_equal(values[0], value) for value in values[1:]):
            raise ValueError(f"Condition field {key!r} is declared shared but differs between roles.")
        shared[key] = values[0]
        for condition in conditions.values():
            condition.pop(key)
    return shared


def _training_cache_data(model, batch, *, inputs, prompts, contextual_roles=(), conditioning_meta=None):
    conditions = _encode_cache_conditions(model, batch, prompts, contextual_roles)
    shared = _factor_shared_conditions(conditions, model.shared_condition_keys)
    conditioning = {
        "prompt": batch["conditioning"]["prompt"],
        **conditions,
        **(conditioning_meta or {}),
    }
    if "unconditional" in conditions:
        conditioning["unconditional_prompt"] = model.unconditional_prompt
    if shared:
        conditioning["shared"] = shared

    meta = dict(batch.get("meta", {}))
    target = batch.get("inputs", {}).get("target_pixel_values")
    if target is not None:
        meta["target_height"] = int(target.shape[-2])
        meta["target_width"] = int(target.shape[-1])
    return {
        "inputs": inputs,
        "conditioning": conditioning,
        "meta": meta,
    }


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
    _CONSISTENCY_MODEL_AUXILIARY_WEIGHTS_NAME = "consistency_auxiliary.safetensors"

    def _consistency_model_auxiliary_parameter_names(self) -> tuple[str, ...]:
        capabilities = self.model.ensure_capabilities()
        if not capabilities.supports(ConsistencyModelCapability):
            return ()
        return capabilities.require(ConsistencyModelCapability).auxiliary_parameter_names()

    def save_weights(self, save_dir, train_type) -> None:
        if train_type == "lora":
            self.model.save_lora_weights(
                save_dir,
                auxiliary_parameter_names=self._consistency_model_auxiliary_parameter_names(),
                auxiliary_weights_name=self._CONSISTENCY_MODEL_AUXILIARY_WEIGHTS_NAME,
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
                self._consistency_model_auxiliary_parameter_names(),
                weights_name=self._CONSISTENCY_MODEL_AUXILIARY_WEIGHTS_NAME,
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
    def encode_training_cache(self, batch):
        prompt = batch["conditioning"]["prompt"]
        prompts = {"positive": prompt}
        if _uses_prompt_dropout(self.model):
            prompts["unconditional"] = self.model.unconditional_prompt
        return _training_cache_data(
            self.model,
            batch,
            inputs=_target_cache_inputs(self.model, batch, required=True),
            prompts=prompts,
            contextual_roles=prompts,
        )

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: SFTStepContext,
    ) -> LossResult:
        scheduler = context.noise_scheduler
        broadcast = context.broadcast

        with torch.no_grad():
            latent = _cached_latent(batch, self.model)
            if latent is None:
                latent = self.model.encode_to_latent(batch)
            latent = _require_singleton_tensor(
                broadcast(latent),
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
            condition = _cached_condition(batch, self.model)
            if condition is None:
                condition = self.model.encode_condition(batch)
            condition = broadcast(condition)

        prediction = self.model.predict_denoiser_output(
            noisy_latent,
            timestep_or_sigma,
            condition,
        )
        target = scheduler.build_train_gt(latent, noise)
        loss = (prediction.float() - target.float()).square().mean()
        return LossResult(loss=loss)


class GenericConsistencyModelCapability(BoundCapability, ConsistencyModelCapability):
    def encode_training_cache(self, batch):
        prompt = batch["conditioning"]["prompt"]
        teacher = self.model.config["training"].get("consistency", {}).get("teacher", {})
        negative_prompt = _prompt_or_default(teacher.get("negative_prompt"), self.model.unconditional_prompt)
        prompts = {"positive": prompt}
        conditioning_meta = {}
        if _consistency_requires_negative(self.model):
            prompts["negative"] = negative_prompt
            conditioning_meta["negative_prompt"] = negative_prompt
        if _uses_prompt_dropout(self.model):
            prompts["unconditional"] = self.model.unconditional_prompt
        return _training_cache_data(
            self.model,
            batch,
            inputs=_target_cache_inputs(self.model, batch, required=True),
            prompts=prompts,
            contextual_roles=prompts,
            conditioning_meta=conditioning_meta,
        )

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
        latent = _cached_latent(batch, self.model)
        if latent is None:
            latent = self.model.encode_to_latent(batch)
        return _require_singleton_tensor(
            latent,
            "Consistency latent",
        )

    def encode_condition(self, batch):
        condition = _cached_condition(batch, self.model)
        return self.model.encode_condition(batch) if condition is None else condition

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


class GenericDistributionMatchingCapability(BoundCapability, DistributionMatchingCapability):
    """Distribution-matching operations shared by image flow models."""

    cache_uses_sample_context = False

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
        return self.model.unconditional_prompt

    @property
    def default_lora_target_modules(self):
        return None

    def encode_training_cache(self, batch):
        training = self.model.config["training"]
        teacher = training.get("teacher", {})
        dmd = training.get("dmd", {})
        guidance_scale = float(teacher.get("guidance_scale", dmd.get("guidance_scale", 3.0)))
        negative_prompt = _prompt_or_default(
            teacher.get(
                "negative_prompt",
                dmd.get("negative_prompt", self.default_negative_prompt),
            ),
            self.model.unconditional_prompt,
        )
        prompt = batch["conditioning"]["prompt"]
        prompts = {"positive": prompt}
        if _uses_prompt_dropout(self.model):
            prompts["unconditional"] = self.model.unconditional_prompt
        conditioning_meta = {}
        if guidance_scale > 1.0:
            prompts["negative"] = negative_prompt
            conditioning_meta["negative_prompt"] = negative_prompt
        return _training_cache_data(
            self.model,
            batch,
            inputs=_target_cache_inputs(self.model, batch, required=False),
            prompts=prompts,
            contextual_roles=prompts if self.cache_uses_sample_context else (),
            conditioning_meta=conditioning_meta,
        )

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
        cached = _cached_condition_pair(
            batch,
            self.model,
            require_negative=guidance_scale > 1,
        )
        if cached is not None:
            positive, negative = cached
            return broadcast(positive), broadcast(negative) if negative is not None else None
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
            denoiser_input = self.model.prepare_denoiser_input(latents, condition=condition)
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
            latent = _cached_latent(batch, self.model)
            if latent is None:
                latent = self.model.encode_to_latent(batch)
            latent = latent.to(device=self.device, dtype=dtype)
            if latent.ndim == 4 and latent.shape[0] != 1:
                latent = latent.unsqueeze(0)
            _require_singleton_tensor(latent, "DMD real latent")
        return broadcast(latent)

    def set_training(self, enabled: bool) -> None:
        self.model.denoiser_module().train(enabled)

    def denoiser(self):
        return self.model.denoiser_module()
