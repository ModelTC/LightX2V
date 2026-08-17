from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from .base import (
    CapabilityDenoiser,
    ConsistencyBatch,
    ConsistencyStepContext,
    DenoiserRequest,
    ObjectiveOutput,
    RectifiedFlowPath,
    ReferenceModelSpec,
    expand_time,
    require_singleton_clean,
)
from .cm import CMObjective
from .config import TCMConfig
from .objective_factory import CONSISTENCY_OBJECTIVE_REGISTER


def _distributed_mask_summary(mask: Tensor) -> tuple[bool, bool]:
    """Return global ``(all, any)`` without creating FSDP branch skew."""
    all_value = torch.tensor(bool(mask.all()), device=mask.device, dtype=torch.int32)
    any_value = torch.tensor(bool(mask.any()), device=mask.device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(all_value, op=dist.ReduceOp.MIN)
        dist.all_reduce(any_value, op=dist.ReduceOp.MAX)
    return bool(all_value.item()), bool(any_value.item())


class TwoStageDenoiser:
    """Route low times through frozen stage 1 and high times through stage 2."""

    def __init__(self, stage1: CapabilityDenoiser, stage2: CapabilityDenoiser, transition_time: float):
        self.stage1 = stage1
        self.stage2 = stage2
        self.transition_time = transition_time

    def predict(self, request: DenoiserRequest) -> Tensor:
        second_stage = request.time >= self.transition_time
        all_second, any_second = _distributed_mask_summary(second_stage)
        if all_second:
            return self.stage2.predict(request)

        rng_devices = [request.sample.device] if request.sample.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            stage1_prediction = self.stage1.predict(request)
        if not any_second:
            return stage1_prediction

        stage2_prediction = self.stage2.predict(request)
        mask = expand_time(second_stage, stage2_prediction.ndim)
        return torch.where(mask, stage2_prediction, stage1_prediction)


@CONSISTENCY_OBJECTIVE_REGISTER("tcm")
class TCMObjective(CMObjective):
    """Stage-2 Consistency Model training with a frozen stage-1 boundary."""

    algorithm_name = "tcm"

    def __init__(self, config, path: RectifiedFlowPath):
        super().__init__(config, path)
        self.tcm_config = TCMConfig.from_mapping(config)

    @property
    def reference_model_specs(self) -> tuple[ReferenceModelSpec, ...]:
        return (
            ReferenceModelSpec(
                role="stage1",
                checkpoint=self.tcm_config.stage1_checkpoint,
                # FastGen intentionally keeps stage 1 in train mode so its
                # dropout RNG can be matched to the stage-2 forward.
                training_mode=True,
            ),
        )

    @property
    def student_initialization_checkpoint(self) -> str:
        return self.tcm_config.stage1_checkpoint

    def sample_training_state(
        self,
        clean: Tensor,
        scheduler,
        context: ConsistencyStepContext,
    ) -> Mapping[str, Tensor]:
        require_singleton_clean(clean)
        t = scheduler.sample_timestep_or_sigma(latent_hw=context.latent_hw).to(clean.device).float()
        t = t.clamp_min(self.tcm_config.transition_time + self.config.time_pair.safety_epsilon)

        boundary_mask = torch.rand(1, device=clean.device) < self.tcm_config.boundary_probability
        t = torch.where(
            boundary_mask,
            torch.full_like(t, self.tcm_config.transition_time + self.config.time_pair.safety_epsilon),
            t,
        )

        pair = self.time_pair_sampler.sample(t, context)
        return {
            "noise": torch.randn_like(clean),
            "t": pair.t,
            "r": pair.r,
            "boundary_mask": boundary_mask,
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
        if references is None or "stage1" not in references:
            raise RuntimeError("TCM requires the frozen 'stage1' consistency model.")

        two_stage = TwoStageDenoiser(
            references["stage1"],
            student,
            self.tcm_config.transition_time,
        )
        terms = self.compute_loss_terms(batch, training_state, two_stage, teacher)
        boundary_mask = training_state["boundary_mask"].bool()
        zero = terms.weighted.sum() * 0.0
        regular_loss = terms.weighted[~boundary_mask].mean() if (~boundary_mask).any() else zero
        boundary_loss = terms.weighted[boundary_mask].mean() if boundary_mask.any() else zero
        regular_unweighted = terms.unweighted[~boundary_mask].mean() if (~boundary_mask).any() else zero.detach()
        loss = regular_loss + self.tcm_config.boundary_weight * boundary_loss

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "tcm_regular": regular_loss.detach(),
                "tcm_boundary": boundary_loss.detach(),
                "tcm_unweighted": regular_unweighted.detach(),
                "tcm_boundary_fraction": boundary_mask.float().mean(),
                "tcm_t": training_state["t"].detach().mean(),
                "tcm_r": training_state["r"].detach().mean(),
                "tcm_ratio": training_state["ratio"].detach(),
                "tcm_stage": training_state["stage"].detach(),
            },
        )
