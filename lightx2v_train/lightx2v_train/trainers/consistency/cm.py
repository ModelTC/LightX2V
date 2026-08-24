from __future__ import annotations

from dataclasses import dataclass
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
from .config import CMConfig, CMLossConfig, CMTimePairConfig
from .objective_factory import CONSISTENCY_OBJECTIVE_REGISTER


@dataclass(frozen=True)
class CMTimePair:
    t: Tensor
    r: Tensor
    ratio: float
    stage: int


@dataclass(frozen=True)
class CMLossTerms:
    weighted: Tensor
    unweighted: Tensor
    prediction: Tensor
    target: Tensor


class CMTimePairSampler:
    """Build the adjacent time pair used by the CM curriculum.

    The default ``ect`` mapping and ratio curriculum match FastGen's CM
    implementation.  ``linear`` is also available for rectified-flow
    experiments without changing the objective itself.
    """

    def __init__(self, config: CMTimePairConfig):
        self.config = config

    def sample(self, t: Tensor, context: ConsistencyStepContext) -> CMTimePair:
        stage = int((context.iteration * context.global_batch_size) // (self.config.kimg_per_stage * 1000.0))
        ratio = min(1.0 - 1.0 / self.config.q ** (stage + 1), self.config.ratio_limit)

        # Pair construction remains in fp32 even when the model runs in bf16;
        # otherwise late curriculum stages can round t and r to the same time.
        t = t.float()
        if self.config.mapping in {"ect", "sigmoid"}:
            r = t - t * (1.0 - ratio) * (1.0 + 8.0 * torch.sigmoid(-t))
        else:
            r = ratio * t

        r = torch.clamp(r, min=self.config.min_r)
        # Keep a strictly positive interval for the loss weighting.  The
        # boundary r=0 remains valid and is handled exactly by the objective.
        largest_safe_r = torch.clamp(t - self.config.safety_epsilon, min=0.0)
        r = torch.minimum(r, largest_safe_r)
        return CMTimePair(t=t, r=r, ratio=ratio, stage=stage)


class CMLoss:
    """Vector-distance CM loss with configurable interval weighting."""

    _WEIGHT_ALIASES = {
        "default": "inverse_delta",
        "sqrt": "inverse_sqrt_delta",
        "one": "none",
    }

    def __init__(self, config: CMLossConfig):
        self.config = config
        self.dtype = torch.float64 if config.computation_dtype == "float64" else torch.float32

    def __call__(self, prediction: Tensor, target: Tensor, t: Tensor, r: Tensor):
        difference = prediction.to(self.dtype) - target.to(self.dtype)
        squared_distance = difference.flatten(1).square()
        if self.config.normalize_by_numel:
            squared_distance = squared_distance.mean(dim=1)
        else:
            squared_distance = squared_distance.sum(dim=1)

        if self.config.distance == "squared_l2":
            unweighted = squared_distance
        else:
            l2_distance = torch.sqrt(squared_distance)
            if self.config.distance == "pseudo_huber":
                constant = self.config.huber_constant
                unweighted = torch.sqrt(l2_distance.square() + constant**2) - constant
            else:
                unweighted = l2_distance

        delta = (t.to(self.dtype) - r.to(self.dtype)).clamp_min(self.config.min_denominator)
        weighting = self._WEIGHT_ALIASES.get(self.config.weighting, self.config.weighting)
        if weighting == "inverse_delta":
            weighted = unweighted / delta
        elif weighting == "inverse_sqrt_delta":
            weighted = unweighted / torch.sqrt(delta)
        elif weighting in {"c_out", "c_out_sq"}:
            # EDM c_out after matching the RF signal/noise ratio
            # sigma_edm=t/(1-t), written in a form that stays finite at t=1.
            time = t.to(self.dtype)
            sigma_data = self.config.sigma_data
            c_out = time * sigma_data / torch.sqrt(time.square() + sigma_data**2 * (1.0 - time).square()).clamp_min(self.config.min_denominator)
            denominator = c_out.square() if weighting == "c_out_sq" else c_out
            weighted = unweighted / denominator.clamp_min(self.config.min_denominator)
        elif weighting == "sigma_sq":
            weighted = unweighted / t.to(self.dtype).square().clamp_min(self.config.min_denominator)
        else:
            weighted = unweighted
        return weighted, unweighted


def classifier_free_guidance(
    conditional: Tensor,
    unconditional: Tensor,
    scale: float,
    norm: str,
) -> Tensor:
    guided = unconditional + scale * (conditional - unconditional)
    if norm == "none":
        return guided
    if norm == "layer_norm":
        conditional_norm = torch.linalg.vector_norm(conditional, dim=-1, keepdim=True)
        guided_norm = torch.linalg.vector_norm(guided, dim=-1, keepdim=True)
        return guided * (conditional_norm / guided_norm.clamp_min(1e-12))
    conditional_norm = torch.linalg.vector_norm(conditional)
    guided_norm = torch.linalg.vector_norm(guided)
    scale_correction = torch.clamp(conditional_norm / guided_norm.clamp_min(1e-12), max=1.0)
    return guided * scale_correction


@CONSISTENCY_OBJECTIVE_REGISTER("cm")
class CMObjective(ConsistencyObjective):
    """Consistency Training (CT) and Consistency Distillation (CD)."""

    algorithm_name = "cm"

    def __init__(self, config, path: RectifiedFlowPath):
        self.config = CMConfig.from_mapping(config)
        self.path = path
        self.time_pair_sampler = CMTimePairSampler(self.config.time_pair)
        self.loss_fn = CMLoss(self.config.loss)
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
        sampled_t = scheduler.sample_timestep_or_sigma(latent_hw=context.latent_hw).to(clean.device)
        pair = self.time_pair_sampler.sample(sampled_t, context)
        return {
            "noise": torch.randn_like(clean),
            "t": pair.t,
            "r": pair.r,
            "ratio": torch.tensor(pair.ratio, device=clean.device, dtype=torch.float32),
            "stage": torch.tensor(pair.stage, device=clean.device, dtype=torch.float32),
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
        terms = self.compute_loss_terms(batch, training_state, student, teacher)
        t = training_state["t"]
        r = training_state["r"]
        delta = t.float() - r.float()
        return ObjectiveOutput(
            loss=terms.weighted.mean(),
            metrics={
                "cm_unweighted": terms.unweighted.detach().mean(),
                "cm_t": t.detach().float().mean(),
                "cm_r": r.detach().float().mean(),
                "cm_delta": delta.detach().mean(),
                "cm_ratio": training_state["ratio"].detach(),
                "cm_stage": training_state["stage"].detach(),
            },
        )

    def compute_loss_terms(
        self,
        batch: ConsistencyBatch,
        training_state: Mapping[str, Tensor],
        student: CapabilityDenoiser,
        teacher: Optional[CapabilityDenoiser] = None,
    ) -> CMLossTerms:
        clean = batch.clean
        noise = training_state["noise"]
        t = training_state["t"]
        r = training_state["r"]
        noisy_t = self.path.interpolate(clean, noise, t)

        if self.config.mode == "ct":
            noisy_r = self.path.interpolate(clean, noise, r)
        else:
            if teacher is None:
                raise RuntimeError("CM consistency distillation requires a frozen teacher denoiser.")
            noisy_r = self._distill_to_r(
                noisy_t,
                t,
                r,
                batch.condition,
                batch.negative_condition,
                teacher,
            )

        # The first forward is forked so the target forward sees exactly the
        # same dropout RNG state.  Exiting fork_rng restores that state before
        # evaluating the stop-gradient target branch.
        rng_devices = [clean.device] if clean.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            prediction = student.predict(
                DenoiserRequest(
                    sample=noisy_t,
                    time=t,
                    condition=batch.condition,
                    prediction_type="x0",
                )
            )
        with torch.no_grad():
            target_candidate = student.predict(
                DenoiserRequest(
                    sample=noisy_r,
                    time=r,
                    condition=batch.condition,
                    prediction_type="x0",
                )
            )

        # D(x, 0)=x is the CM boundary condition.  For data-paired CT/CD,
        # FastGen anchors an r=0 target to the clean sample explicitly.
        positive_r = expand_time(r > 0, clean.ndim)
        target_candidate = torch.nan_to_num(target_candidate)
        target = torch.where(positive_r, target_candidate, clean)

        weighted, unweighted = self.loss_fn(prediction, target, t, r)
        return CMLossTerms(
            weighted=weighted,
            unweighted=unweighted,
            prediction=prediction,
            target=target,
        )

    @torch.no_grad()
    def _distill_to_r(
        self,
        noisy_t: Tensor,
        t: Tensor,
        r: Tensor,
        condition,
        negative_condition,
        teacher: CapabilityDenoiser,
    ) -> Tensor:
        conditional_velocity = teacher.predict(
            DenoiserRequest(
                sample=noisy_t,
                time=t,
                condition=condition,
                prediction_type="velocity",
            )
        )
        guidance_scale = self.config.teacher.guidance_scale
        if guidance_scale is None or guidance_scale == 1.0:
            velocity = conditional_velocity
        else:
            if negative_condition is None:
                raise RuntimeError("CM teacher CFG requires a negative condition.")
            unconditional_velocity = teacher.predict(
                DenoiserRequest(
                    sample=noisy_t,
                    time=t,
                    condition=negative_condition,
                    prediction_type="velocity",
                )
            )
            velocity = classifier_free_guidance(
                conditional_velocity,
                unconditional_velocity,
                guidance_scale,
                self.config.teacher.cfg_norm,
            )
        return self.path.euler_step(noisy_t, velocity, t, r)
