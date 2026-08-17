from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from .base import (
    CapabilityDenoiser,
    ConsistencyBatch,
    ConsistencyObjective,
    ConsistencyStepContext,
    DenoiserRequest,
    ObjectiveOutput,
    RectifiedFlowPath,
    expand_time,
    require_singleton_clean,
)
from .config import MeanFlowConfig
from .jvp import math_attention_for_forward_ad
from .objective_factory import CONSISTENCY_OBJECTIVE_REGISTER


def _where_condition(mask: Tensor, positive: Any, negative: Any):
    if torch.is_tensor(positive):
        if not torch.is_tensor(negative):
            raise TypeError(f"MeanFlow condition branches have incompatible types: Tensor and {type(negative).__name__}.")
        return torch.where(expand_time(mask, positive.ndim), positive, negative)
    if isinstance(positive, Mapping):
        if not isinstance(negative, Mapping) or positive.keys() != negative.keys():
            raise ValueError("MeanFlow condition mappings must have identical keys.")
        return {key: _where_condition(mask, value, negative[key]) for key, value in positive.items()}
    if isinstance(positive, (list, tuple)):
        if not isinstance(negative, type(positive)) or len(positive) != len(negative):
            raise ValueError("MeanFlow condition sequences must have identical types and lengths.")
        values = [_where_condition(mask, value, negative[index]) for index, value in enumerate(positive)]
        return type(positive)(values)
    if positive == negative:
        return positive
    raise ValueError(f"MeanFlow cannot mix non-tensor condition metadata with different values: {positive!r} and {negative!r}.")


@CONSISTENCY_OBJECTIVE_REGISTER("mean_flow")
@CONSISTENCY_OBJECTIVE_REGISTER("meanflow")
class MeanFlowObjective(ConsistencyObjective):
    """MeanFlow objective with endpoint conditioning and JVP/finite differences."""

    algorithm_name = "mean_flow"
    model_capabilities = frozenset({"endpoint_time"})

    def __init__(self, config, path: RectifiedFlowPath):
        self.config = MeanFlowConfig.from_mapping(config)
        self.path = path
        self.requires_teacher = self.config.mode == "cd"
        uses_training_guidance = any(
            value is not None
            for value in (
                self.config.condition_dropout_probability,
                self.config.guidance_scale,
                self.config.guidance_mixture_ratio,
            )
        )
        self.requires_negative_condition = uses_training_guidance
        self.negative_prompt = self.config.teacher.negative_prompt

    def sample_training_state(
        self,
        clean: Tensor,
        scheduler,
        context: ConsistencyStepContext,
    ) -> Mapping[str, Tensor]:
        require_singleton_clean(clean)
        first = scheduler.sample_timestep_or_sigma(latent_hw=context.latent_hw).to(clean.device).float()
        second = scheduler.sample_timestep_or_sigma(latent_hw=context.latent_hw).to(clean.device).float()
        t = torch.maximum(first, second)
        random_r = torch.minimum(first, second)
        use_random_r = torch.rand(1, device=clean.device) < self.config.random_endpoint_probability
        r = torch.where(use_random_r, random_r, t)
        return {
            "noise": torch.randn_like(clean),
            "t": t,
            "r": r,
            "random_endpoint_mask": use_random_r,
            "iteration": torch.tensor(context.iteration, device=clean.device, dtype=torch.int64),
        }

    def compute(
        self,
        batch: ConsistencyBatch,
        training_state: Mapping[str, Tensor],
        student: CapabilityDenoiser,
        teacher: Optional[CapabilityDenoiser] = None,
        references: Optional[Mapping[str, CapabilityDenoiser]] = None,
    ) -> ObjectiveOutput:
        del references
        clean = batch.clean
        noise = training_state["noise"]
        t = training_state["t"].float()
        r = training_state["r"].float()
        x_t = self.path.interpolate(clean, noise, t)
        condition, dxt_dt = self._target_velocity(batch, x_t, noise, t, student, teacher)

        velocity_jvp = self._jvp(x_t, t, r, dxt_dt, condition, student)
        velocity = student.predict(
            DenoiserRequest(
                sample=x_t,
                time=t,
                condition=condition,
                prediction_type="velocity",
                model_kwargs={"endpoint_time": r},
            )
        )
        loss, tangent, weight, warmup = self._loss(
            velocity,
            velocity_jvp,
            dxt_dt,
            t,
            r,
            int(training_state["iteration"].item()),
        )
        flow_matching_error = (velocity.float() - (noise.float() - clean.float())).square().flatten(1).mean(dim=1)
        return ObjectiveOutput(
            loss=loss.mean(),
            metrics={
                "mean_flow_loss": loss.detach().mean(),
                "mean_flow_velocity_mse": flow_matching_error.detach().mean(),
                "mean_flow_jvp_norm": torch.linalg.vector_norm(velocity_jvp.flatten(1), dim=1).detach().mean(),
                "mean_flow_tangent_norm": torch.linalg.vector_norm(tangent.flatten(1), dim=1).detach().mean(),
                "mean_flow_weight": weight.detach().mean(),
                "mean_flow_t": t.detach().mean(),
                "mean_flow_r": r.detach().mean(),
                "mean_flow_random_endpoint_fraction": training_state["random_endpoint_mask"].float().mean(),
                "mean_flow_warmup": warmup,
            },
        )

    @torch.no_grad()
    def _target_velocity(
        self,
        batch: ConsistencyBatch,
        x_t: Tensor,
        noise: Tensor,
        t: Tensor,
        student: CapabilityDenoiser,
        teacher: Optional[CapabilityDenoiser],
    ):
        if self.config.mode == "cd":
            if teacher is None:
                raise RuntimeError("MeanFlow consistency distillation requires a frozen teacher denoiser.")
            velocity = teacher.predict(DenoiserRequest(x_t, t, batch.condition, prediction_type="velocity"))
            if self.config.guidance_scale is not None:
                if batch.negative_condition is None:
                    raise RuntimeError("MeanFlow teacher CFG requires a negative condition.")
                negative = teacher.predict(DenoiserRequest(x_t, t, batch.negative_condition, prediction_type="velocity"))
                scale = self._time_limited_value(t, self.config.guidance_scale, outside=1.0)
                velocity = negative + expand_time(scale, velocity.ndim) * (velocity - negative)
            return batch.condition, velocity

        velocity = noise - batch.clean
        if self.config.guidance_scale is None and self.config.guidance_mixture_ratio is None:
            return batch.condition, velocity
        if batch.negative_condition is None:
            raise RuntimeError("MeanFlow training guidance requires a negative condition.")

        negative = student.predict(
            DenoiserRequest(
                x_t,
                t,
                batch.negative_condition,
                prediction_type="velocity",
                model_kwargs={"endpoint_time": t},
            )
        )
        scale_value = 1.0 if self.config.guidance_scale is None else self.config.guidance_scale
        scale = self._time_limited_value(t, scale_value, outside=1.0)
        if self.config.guidance_mixture_ratio is None:
            guided = negative + expand_time(scale, velocity.ndim) * (velocity - negative)
        else:
            conditional = student.predict(
                DenoiserRequest(
                    x_t,
                    t,
                    batch.condition,
                    prediction_type="velocity",
                    model_kwargs={"endpoint_time": t},
                )
            )
            mixture = self._time_limited_value(t, self.config.guidance_mixture_ratio, outside=0.0)
            guided = expand_time(scale, velocity.ndim) * velocity + expand_time(1.0 - scale - mixture, velocity.ndim) * negative + expand_time(mixture, velocity.ndim) * conditional

        dropout = self.config.condition_dropout_probability
        if dropout is None:
            return batch.condition, guided
        keep_condition = torch.rand(t.shape[0], device=t.device) >= dropout
        mixed_condition = _where_condition(keep_condition, batch.condition, batch.negative_condition)
        mixed_velocity = torch.where(expand_time(keep_condition, velocity.ndim), guided, velocity)
        return mixed_condition, mixed_velocity

    def _time_limited_value(self, t: Tensor, value: float, *, outside: float) -> Tensor:
        active = (t >= self.config.guidance_time_start) & (t <= self.config.guidance_time_end)
        return torch.where(active, torch.full_like(t, value), torch.full_like(t, outside))

    @torch.no_grad()
    def _jvp(
        self,
        x_t: Tensor,
        t: Tensor,
        r: Tensor,
        dxt_dt: Tensor,
        condition,
        student: CapabilityDenoiser,
    ) -> Tensor:
        def model_fn(sample, time, endpoint):
            return student.predict(
                DenoiserRequest(
                    sample,
                    time,
                    condition,
                    prediction_type="velocity",
                    model_kwargs={"endpoint_time": endpoint},
                )
            )

        if self.config.jvp.method == "exact":
            tangents = (dxt_dt.to(x_t.dtype), torch.ones_like(t), torch.zeros_like(r))
            rng_devices = [x_t.device] if x_t.device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices), math_attention_for_forward_ad(x_t.device.type):
                return torch.func.jvp(model_fn, (x_t, t, r), tangents)[1]

        work_t = t.double()
        work_r = r.double()
        epsilon = torch.full_like(work_t, self.config.jvp.epsilon)
        forward_valid = work_t + epsilon <= 1.0
        backward_valid = (work_t - epsilon >= 0.0) & (work_t - epsilon > work_r)
        central = forward_valid & backward_valid
        forward = forward_valid & ~backward_valid
        backward = ~forward_valid & backward_valid

        plus_t = work_t.clone()
        minus_t = work_t.clone()
        factor = torch.zeros_like(work_t)
        plus_t[central] += epsilon[central]
        minus_t[central] -= epsilon[central]
        factor[central] = 0.5 / epsilon[central]
        plus_t[forward] += epsilon[forward]
        factor[forward] = 1.0 / epsilon[forward]
        minus_t[backward] -= epsilon[backward]
        factor[backward] = 1.0 / epsilon[backward]

        plus_x = x_t.double() + expand_time(plus_t - work_t, x_t.ndim) * dxt_dt.double()
        minus_x = x_t.double() + expand_time(minus_t - work_t, x_t.ndim) * dxt_dt.double()
        rng_devices = [x_t.device] if x_t.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            plus = model_fn(plus_x.to(x_t.dtype), plus_t.to(t.dtype), work_r.to(r.dtype))
        with torch.random.fork_rng(devices=rng_devices):
            minus = model_fn(minus_x.to(x_t.dtype), minus_t.to(t.dtype), work_r.to(r.dtype))
        return (plus.double() - minus.double()) * expand_time(factor, plus.ndim)

    def _loss(
        self,
        velocity: Tensor,
        velocity_jvp: Tensor,
        dxt_dt: Tensor,
        t: Tensor,
        r: Tensor,
        iteration: int,
    ):
        velocity = velocity.double()
        velocity_jvp = velocity_jvp.double()
        dxt_dt = dxt_dt.double()
        delta = expand_time((t.double() - r.double()).clamp_min(0.0), velocity.ndim)
        if self.config.tangent_warmup_steps:
            warmup = min(1.0, iteration / self.config.tangent_warmup_steps)
        else:
            warmup = 1.0

        if self.config.loss_type == "l2":
            tangent = dxt_dt - warmup * delta * velocity_jvp
            squared_error = (velocity - tangent).square().flatten(1).sum(dim=1)
            weight = self._weight(squared_error)
            loss = squared_error * weight
        else:
            tangent = dxt_dt - velocity.detach() - warmup * delta * velocity_jvp
            if self.config.spatially_normalized_tangent:
                tangent = tangent / (tangent[0].numel() ** 0.5)
            tangent_norm = torch.linalg.vector_norm(tangent.flatten(1), dim=1)
            weight = self._weight(tangent_norm)
            target = (velocity + tangent * expand_time(weight, tangent.ndim)).detach()
            loss = (velocity - target).square().flatten(1).sum(dim=1)
        return loss, tangent, weight, warmup

    def _weight(self, value: Tensor) -> Tensor:
        method, *arguments = self.config.norm_method.split("_")
        if method == "poly" and len(arguments) == 1:
            return (value + self.config.norm_constant).pow(-float(arguments[0]))
        if method == "exp" and len(arguments) == 2:
            constant, scale = map(float, arguments)
            return constant * torch.exp(scale * value + self.config.norm_constant)
        raise ValueError("MeanFlow loss.norm_method must be 'poly_<power>' or 'exp_<constant>_<scale>'.")
