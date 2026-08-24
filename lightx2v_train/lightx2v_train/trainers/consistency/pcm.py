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
    require_singleton_clean,
)
from .cm import classifier_free_guidance
from .config import PCMConfig, PCMLossConfig, PCMSolverConfig
from .objective_factory import CONSISTENCY_OBJECTIVE_REGISTER


@dataclass(frozen=True)
class PCMTimeState:
    solver_index: Tensor
    phase_index: Tensor
    phase_start_index: Tensor
    t: Tensor
    r: Tensor
    s: Tensor


class PCMTimeGrid:
    """Discrete teacher-solver grid and equal-width PCM phase partition.

    Grid indices run from low to high noise, matching the reference SD3
    ``EulerSolver``. For an index ``i``, ``r`` is the adjacent lower-noise
    solver time and ``s`` is the lower boundary shared by the whole phase.
    """

    def __init__(self, config: PCMSolverConfig):
        self.config = config

    def sample(self, scheduler, *, latent_hw, device) -> PCMTimeState:
        current, previous = self.build_grid(scheduler, latent_hw=latent_hw, device=device)
        solver_index = torch.randint(0, self.config.num_solver_steps, (1,), device=device)
        phase_starts = self.phase_start_indices(device)
        phase_index = torch.bucketize(solver_index, phase_starts, right=True) - 1
        phase_start_index = phase_starts[phase_index]
        return PCMTimeState(
            solver_index=solver_index,
            phase_index=phase_index,
            phase_start_index=phase_start_index,
            t=current[solver_index],
            r=previous[solver_index],
            s=previous[phase_start_index],
        )

    def build_grid(self, scheduler, *, latent_hw, device) -> tuple[Tensor, Tensor]:
        steps = self.config.num_solver_steps
        max_time = float(scheduler.max_t)
        current = torch.arange(1, steps + 1, device=device, dtype=torch.float32) * (max_time / steps)
        raw_boundary = scheduler.min_t if self.config.boundary_time is None else self.config.boundary_time
        boundary = torch.tensor([raw_boundary], device=device, dtype=torch.float32)

        if scheduler.do_time_shift:
            current = scheduler.time_shift(current, latent_hw=latent_hw)
            if raw_boundary > 0.0:
                boundary = scheduler.time_shift(boundary, latent_hw=latent_hw)
        if not boundary.item() < current[0].item():
            raise ValueError(f"PCM boundary time must be smaller than the first solver time; got boundary={boundary.item():.6f}, first={current[0].item():.6f}.")
        previous = torch.cat([boundary, current[:-1]])
        return current, previous

    def phase_start_indices(self, device) -> Tensor:
        phases = torch.arange(self.config.num_phases, device=device, dtype=torch.int64)
        return torch.div(
            phases * self.config.num_solver_steps,
            self.config.num_phases,
            rounding_mode="floor",
        )


class PCMLoss:
    """Elementwise pseudo-Huber/L2 loss used by the released PCM scripts."""

    def __init__(self, config: PCMLossConfig):
        self.config = config
        self.dtype = torch.float64 if config.computation_dtype == "float64" else torch.float32

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        difference = prediction.to(self.dtype) - target.to(self.dtype)
        if self.config.distance == "l2":
            elementwise = difference.square()
        else:
            constant = self.config.huber_constant
            elementwise = torch.sqrt(difference.square() + constant**2) - constant
        return elementwise.flatten(1).mean(dim=1)


def pcm_inference_sigmas(
    num_inference_steps: int,
    num_solver_steps: int,
    *,
    scheduler=None,
    latent_hw=None,
) -> list[float]:
    """Return the descending PCM phase schedule used by inference.

    PCM is trained against phase boundaries after the scheduler's time shift.
    Custom inference sigmas otherwise bypass scheduler shifting, so apply the
    same transform here when a scheduler is provided.
    """
    if not 1 <= num_inference_steps <= num_solver_steps:
        raise ValueError("PCM inference steps must be in [1, num_solver_steps].")
    indices = torch.div(
        torch.arange(num_inference_steps, dtype=torch.int64) * num_solver_steps,
        num_inference_steps,
        rounding_mode="floor",
    )
    sigmas = (num_solver_steps - indices).float() / num_solver_steps
    if scheduler is not None and scheduler.do_time_shift:
        sigmas = scheduler.time_shift(sigmas, latent_hw=latent_hw)
    return sigmas.tolist()


@CONSISTENCY_OBJECTIVE_REGISTER("pcm")
class PCMObjective(ConsistencyObjective):
    """Phased Consistency Model distillation for rectified-flow backbones."""

    algorithm_name = "pcm"
    requires_teacher = True

    def __init__(self, config, path: RectifiedFlowPath):
        self.config = PCMConfig.from_mapping(config)
        self.path = path
        self.time_grid = PCMTimeGrid(self.config.solver)
        self.loss_fn = PCMLoss(self.config.loss)
        guidance_scale = self.config.teacher.guidance_scale
        self.requires_negative_condition = guidance_scale is not None and guidance_scale != 1.0
        self.negative_prompt = self.config.teacher.negative_prompt

    def sample_training_state(
        self,
        clean: Tensor,
        scheduler,
        context: ConsistencyStepContext,
    ) -> Mapping[str, Tensor]:
        require_singleton_clean(clean)
        time = self.time_grid.sample(
            scheduler,
            latent_hw=context.latent_hw,
            device=clean.device,
        )
        return {
            "noise": torch.randn_like(clean),
            "solver_index": time.solver_index,
            "phase_index": time.phase_index,
            "phase_start_index": time.phase_start_index,
            "t": time.t,
            "r": time.r,
            "s": time.s,
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
        if teacher is None:
            raise RuntimeError("PCM requires a frozen diffusion teacher denoiser.")

        clean = batch.clean
        t = training_state["t"]
        r = training_state["r"]
        s = training_state["s"]
        noisy_t = self.path.interpolate(clean, training_state["noise"], t)

        # Match dropout masks between the trainable prediction and the
        # stop-gradient student target while leaving global RNG unchanged.
        rng_devices = [clean.device] if clean.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            prediction_velocity = student.predict(DenoiserRequest(noisy_t, t, batch.condition, prediction_type="velocity"))
        prediction = self.path.euler_step(noisy_t, prediction_velocity, t, s)

        with torch.no_grad():
            teacher_velocity = self._teacher_velocity(
                noisy_t,
                t,
                batch.condition,
                batch.negative_condition,
                teacher,
            )
            noisy_r = self.path.euler_step(noisy_t, teacher_velocity, t, r)
            target_velocity = student.predict(DenoiserRequest(noisy_r, r, batch.condition, prediction_type="velocity"))
            target = self.path.euler_step(noisy_r, target_velocity, r, s)

        per_sample_loss = self.loss_fn(prediction, target)
        return ObjectiveOutput(
            loss=per_sample_loss.mean(),
            metrics={
                "pcm_unweighted": per_sample_loss.detach().mean(),
                "pcm_t": t.detach().float().mean(),
                "pcm_r": r.detach().float().mean(),
                "pcm_s": s.detach().float().mean(),
                "pcm_teacher_step": (t.float() - r.float()).detach().mean(),
                "pcm_phase_span": (t.float() - s.float()).detach().mean(),
                "pcm_phase": training_state["phase_index"].detach().float().mean(),
                "pcm_solver_index": training_state["solver_index"].detach().float().mean(),
            },
        )

    @torch.no_grad()
    def _teacher_velocity(
        self,
        sample: Tensor,
        time: Tensor,
        condition,
        negative_condition,
        teacher: CapabilityDenoiser,
    ) -> Tensor:
        conditional = teacher.predict(DenoiserRequest(sample, time, condition, prediction_type="velocity"))
        scale = self.config.teacher.guidance_scale
        if scale is None or scale == 1.0:
            return conditional
        if negative_condition is None:
            raise RuntimeError("PCM teacher CFG requires a negative condition.")
        unconditional = teacher.predict(DenoiserRequest(sample, time, negative_condition, prediction_type="velocity"))
        return classifier_free_guidance(
            conditional,
            unconditional,
            scale,
            self.config.teacher.cfg_norm,
        )
