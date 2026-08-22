from __future__ import annotations

from typing import Mapping, Optional

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
from .cm import classifier_free_guidance
from .config import SCMConfig
from .jvp import math_attention_for_forward_ad
from .objective_factory import CONSISTENCY_OBJECTIVE_REGISTER


@CONSISTENCY_OBJECTIVE_REGISTER("scm")
class SCMObjective(ConsistencyObjective):
    """Continuous-time simplified Consistency Model (sCM) objective.

    The public model still predicts rectified-flow velocity. This
    objective performs FastGen's SNR-matched TrigFlow preconditioning at the
    model boundary, keeping the backbone and its checkpoints interoperable.
    """

    algorithm_name = "scm"
    model_capabilities = frozenset({"log_variance"})

    def __init__(self, config, path: RectifiedFlowPath):
        self.config = SCMConfig.from_mapping(config)
        self.path = path
        self.requires_teacher = self.config.mode == "cd"
        guidance_scale = self.config.teacher.guidance_scale
        self.requires_negative_condition = self.requires_teacher and guidance_scale is not None and guidance_scale != 1.0
        self.negative_prompt = self.config.teacher.negative_prompt

    def sample_training_state(
        self,
        clean: Tensor,
        scheduler,
        context: ConsistencyStepContext,
    ) -> Mapping[str, Tensor]:
        require_singleton_clean(clean)
        t = scheduler.sample_timestep_or_sigma(latent_hw=context.latent_hw).to(clean.device).float()
        return {
            "noise": torch.randn_like(clean) * self.config.sigma_data,
            "t": t,
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
        z = training_state["noise"]
        t = training_state["t"].float()
        sigma_data = self.config.sigma_data

        alpha = 1.0 - t
        sigma = t
        t_hat = torch.atan2(sigma, alpha * sigma_data)
        alpha_hat = torch.cos(t_hat)
        sigma_hat = torch.sin(t_hat)
        x_hat = expand_time(alpha_hat, clean.ndim).to(clean) * clean + expand_time(sigma_hat, clean.ndim).to(clean) * z

        if self.config.mode == "ct":
            dxt_dt = -expand_time(sigma_hat, clean.ndim).to(clean) * clean + expand_time(alpha_hat, clean.ndim).to(clean) * z
        else:
            if teacher is None:
                raise RuntimeError("sCM consistency distillation requires a frozen teacher denoiser.")
            with torch.no_grad():
                teacher_flow, _, _ = self._predict_trig_flow(teacher, x_hat, t_hat, batch.condition)
                dxt_dt = sigma_data * teacher_flow
                guidance_scale = self.config.teacher.guidance_scale
                if guidance_scale is not None and guidance_scale != 1.0:
                    if batch.negative_condition is None:
                        raise RuntimeError("sCM teacher CFG requires a negative condition.")
                    negative_flow, _, _ = self._predict_trig_flow(
                        teacher,
                        x_hat,
                        t_hat,
                        batch.negative_condition,
                    )
                    guided_flow = classifier_free_guidance(
                        teacher_flow,
                        negative_flow,
                        guidance_scale,
                        self.config.teacher.cfg_norm,
                    )
                    dxt_dt = sigma_data * guided_flow

        flow, original_t, x0_prediction = self._predict_trig_flow(student, x_hat, t_hat, batch.condition)
        log_variance = student.predict_log_variance(original_t).reshape(-1)
        flow_jvp = self._jvp(clean, z, x_hat, t_hat, dxt_dt, batch.condition, student)

        loss, unweighted, tangent, warmup = self._loss(
            flow,
            flow_jvp,
            x_hat,
            dxt_dt,
            log_variance,
            sigma,
            t_hat,
            int(training_state["iteration"].item()),
        )
        return ObjectiveOutput(
            loss=loss.mean(),
            metrics={
                "scm_unweighted": unweighted.detach().mean(),
                "scm_log_variance": log_variance.detach().mean(),
                "scm_t": t.detach().mean(),
                "scm_t_hat": t_hat.detach().mean(),
                "scm_jvp_norm": torch.linalg.vector_norm(flow_jvp.flatten(1), dim=1).detach().mean(),
                "scm_tangent_norm": torch.linalg.vector_norm(tangent.flatten(1), dim=1).detach().mean(),
                "scm_warmup": warmup,
                "scm_x0_norm": torch.linalg.vector_norm(x0_prediction.detach().flatten(1), dim=1).mean(),
            },
        )

    def _predict_trig_flow(self, denoiser: CapabilityDenoiser, x_hat: Tensor, t_hat: Tensor, condition):
        sigma_data = self.config.sigma_data
        tangent = sigma_data * torch.tan(t_hat.double())
        original_t = (tangent / (1.0 + tangent)).to(t_hat.dtype)
        coefficient = torch.sqrt((1.0 - original_t.double()).square() + (original_t.double() / sigma_data).square()).to(x_hat.dtype)
        original_sample = x_hat * expand_time(coefficient, x_hat.ndim)
        x0_prediction = denoiser.predict(
            DenoiserRequest(
                sample=original_sample,
                time=original_t,
                condition=condition,
                prediction_type="x0",
            )
        )
        denominator = expand_time(torch.sin(t_hat), x_hat.ndim).to(x_hat).clamp_min(1e-6)
        trig_velocity = (expand_time(torch.cos(t_hat), x_hat.ndim).to(x_hat) * x_hat - x0_prediction) / denominator
        return trig_velocity / sigma_data, original_t, x0_prediction

    @torch.no_grad()
    def _jvp(
        self,
        clean: Tensor,
        z: Tensor,
        x_hat: Tensor,
        t_hat: Tensor,
        dxt_dt: Tensor,
        condition,
        student: CapabilityDenoiser,
    ) -> Tensor:
        def model_fn(sample, time):
            return self._predict_trig_flow(student, sample, time, condition)[0]

        if self.config.jvp.method == "exact":
            v_t = torch.cos(t_hat) * torch.sin(t_hat)
            v_x = expand_time(v_t, dxt_dt.ndim).to(dxt_dt) * dxt_dt
            rng_devices = [x_hat.device] if x_hat.device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices), math_attention_for_forward_ad(x_hat.device.type):
                return torch.func.jvp(model_fn, (x_hat, t_hat), (v_x, v_t))[1]

        work_t = t_hat.double().clamp(1e-5, torch.pi / 2 - 1e-5)
        epsilon = (self.config.jvp.epsilon * work_t.abs()).clamp_min(1e-6)
        plus_t = (work_t + epsilon).clamp_max(torch.pi / 2 - 1e-5)
        minus_t = (work_t - epsilon).clamp_min(1e-5)
        plus_x = expand_time(torch.cos(plus_t), clean.ndim) * clean.double() + expand_time(torch.sin(plus_t), clean.ndim) * z.double()
        minus_x = expand_time(torch.cos(minus_t), clean.ndim) * clean.double() + expand_time(torch.sin(minus_t), clean.ndim) * z.double()
        rng_devices = [x_hat.device] if x_hat.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            plus = model_fn(plus_x.to(x_hat.dtype), plus_t.to(t_hat.dtype))
        with torch.random.fork_rng(devices=rng_devices):
            minus = model_fn(minus_x.to(x_hat.dtype), minus_t.to(t_hat.dtype))
        v_t = torch.cos(work_t) * torch.sin(work_t)
        factor = expand_time(v_t / (plus_t - minus_t), plus.ndim)
        return (plus.double() - minus.double()) * factor

    def _loss(
        self,
        flow: Tensor,
        flow_jvp: Tensor,
        x_hat: Tensor,
        dxt_dt: Tensor,
        log_variance: Tensor,
        sigma: Tensor,
        t_hat: Tensor,
        iteration: int,
    ):
        flow = flow.double()
        detached_flow = flow.detach()
        flow_jvp = flow_jvp.double()
        x_hat = x_hat.double()
        dxt_dt = dxt_dt.double()
        alpha_hat = torch.cos(t_hat.double())
        sigma_hat = torch.sin(t_hat.double())
        if self.config.tangent_warmup_steps:
            warmup = min(1.0, iteration / self.config.tangent_warmup_steps)
        else:
            warmup = 1.0

        g1 = -expand_time(alpha_hat.square(), flow.ndim) * (self.config.sigma_data * detached_flow - dxt_dt)
        g2 = -(expand_time(alpha_hat * sigma_hat, flow.ndim) * x_hat + self.config.sigma_data * flow_jvp)
        tangent = g1 + warmup * g2
        tangent_norm = torch.linalg.vector_norm(tangent.flatten(1), dim=1)
        if self.config.spatially_normalized_tangent:
            tangent_norm = tangent_norm / (tangent[0].numel() ** 0.5)
        tangent = tangent / expand_time(tangent_norm + self.config.tangent_warmup_constant, tangent.ndim)

        unweighted = (flow - detached_flow - tangent).square().flatten(1).mean(dim=1)
        prior_weight = sigma.double().clamp_min(self.config.min_denominator).reciprocal() if self.config.prior_weighting else torch.ones_like(sigma, dtype=torch.float64)
        dimension = x_hat[0].numel() if self.config.normalize_by_numel else 1.0
        log_variance = log_variance.double()
        loss = prior_weight * torch.exp(-log_variance) * unweighted / dimension + log_variance
        return loss, unweighted, tangent, warmup
