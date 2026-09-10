import copy
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
)

from ..dmd.config import DmdScheduleConfig


def _phase_step_index(match_timestep, num_train_timesteps, num_steps, config_name):
    if num_steps <= 0:
        raise ValueError(f"{config_name}.num_inference_steps must be positive.")
    if not 0 < match_timestep < num_train_timesteps:
        raise ValueError("training.dmd.phased.match_timestep must lie strictly between 0 and scheduler.num_train_timesteps.")
    index, remainder = divmod((num_train_timesteps - match_timestep) * num_steps, num_train_timesteps)
    if remainder:
        raise ValueError(f"training.dmd.phased.match_timestep={match_timestep} is not a boundary of the uniform {num_steps}-step schedule from {config_name}.num_inference_steps.")
    return index


def parse_role_lora_config(
    role_config,
    role,
    default_target_modules,
):
    train_type = role_config["train_type"]
    if train_type not in {"full", "lora"}:
        raise ValueError(f"training.{role}.train_type must be 'full' or 'lora'.")
    if train_type == "full":
        return None
    lora_config = copy.deepcopy(role_config["lora"])
    lora_config["rank"] = int(lora_config["rank"])
    lora_config["alpha"] = int(lora_config["alpha"])
    if "target_modules" not in lora_config:
        lora_config["target_modules"] = list(default_target_modules)
    return lora_config


@dataclass(frozen=True)
class PhasedDmdConfig(DmdScheduleConfig):
    """Parsed configuration unique to dual-region phased DMD."""

    phased: Dict[str, Any]
    match_timestep: int
    match_step_index: int
    infer_boundary_step_index: int
    score_timestep_margin: int
    eps: float
    dmd_norm_clip_min: float
    guidance_distill: float
    student_2: Dict[str, Any]
    fake_2: Dict[str, Any]
    enable_fake_low_high: bool
    student_2_train_type: str
    fake_2_train_type: str
    fake_low_high_train_type: str
    student_2_lora: Optional[Dict[str, Any]]
    fake_2_lora: Optional[Dict[str, Any]]
    fake_low_high_lora: Optional[Dict[str, Any]]
    student_2_optimizer: Dict[str, Any]
    fake_2_optimizer: Dict[str, Any]
    fake_low_high_optimizer: Dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        config,
        *,
        base_config,
        dmd_config,
        infer_config,
        max_train_iters,
        default_lora_target_modules,
    ):
        phased = dmd_config.get("phased", {})
        if not isinstance(phased, dict):
            raise ValueError("training.dmd.phased must be a mapping.")
        match_timestep = int(phased.get("match_timestep", 500))
        match_step_index = _phase_step_index(
            match_timestep,
            base_config.num_train_timestep,
            base_config.num_inference_steps,
            "training.dmd",
        )
        inference_step_count = int(infer_config.get("num_inference_steps", base_config.num_inference_steps))
        configured_boundary = _phase_step_index(
            match_timestep,
            base_config.num_train_timestep,
            inference_step_count,
            "inference",
        )
        infer_boundary = int(
            infer_config.get(
                "boundary_step_index",
                configured_boundary,
            )
        )
        if infer_boundary != configured_boundary:
            raise ValueError(f"inference.boundary_step_index must be {configured_boundary} for match_timestep={match_timestep} and inference.num_inference_steps={inference_step_count}.")
        if max_train_iters < 2:
            raise ValueError("phased_dmd requires training.max_train_iters >= 2 so both High and Low regions are trained.")

        margin = int(phased.get("score_timestep_margin", 20))
        if not (0 < margin < base_config.num_train_timestep - match_timestep):
            raise ValueError("training.dmd.phased.score_timestep_margin must be positive and leave score timesteps above the phase boundary.")
        eps = float(phased.get("eps", 1.0e-8))
        norm_clip_min = float(dmd_config.get("norm_clip_min", 1.0e-4))
        if eps <= 0 or norm_clip_min <= 0:
            raise ValueError("training.dmd.phased.eps and training.dmd.norm_clip_min must be positive.")

        training = config["training"]
        student_2 = training["student_2"]
        fake_2 = training["fake_2"]
        student_2_lora = parse_role_lora_config(
            student_2,
            "student_2",
            default_lora_target_modules,
        )
        fake_2_lora = parse_role_lora_config(
            fake_2,
            "fake_2",
            default_lora_target_modules,
        )
        fake_2_optimizer = copy.deepcopy(fake_2["optimizer"])
        return cls(
            **base_config.__dict__,
            phased=phased,
            match_timestep=match_timestep,
            match_step_index=match_step_index,
            infer_boundary_step_index=infer_boundary,
            score_timestep_margin=margin,
            eps=eps,
            dmd_norm_clip_min=norm_clip_min,
            guidance_distill=float(phased.get("guidance_distill", 6.0)),
            student_2=student_2,
            fake_2=fake_2,
            enable_fake_low_high=bool(phased.get("enable_fake_low_high", True)),
            student_2_train_type=student_2["train_type"],
            fake_2_train_type=fake_2["train_type"],
            fake_low_high_train_type=fake_2["train_type"],
            student_2_lora=student_2_lora,
            fake_2_lora=fake_2_lora,
            fake_low_high_lora=copy.deepcopy(fake_2_lora),
            student_2_optimizer=copy.deepcopy(student_2["optimizer"]),
            fake_2_optimizer=fake_2_optimizer,
            fake_low_high_optimizer=copy.deepcopy(fake_2_optimizer),
        )
