import copy
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
)

from ..dmd.config import VideoDmdConfig


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
class PhasedDmdConfig(VideoDmdConfig):
    """Parsed configuration unique to dual-region phased DMD."""

    phased: Dict[str, Any]
    match_timestep: int
    match_step_index: int
    infer_boundary_step_index: int
    score_timestep_margin: int
    score_timestep_min: int
    score_timestep_max: int
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
        if dmd_config.get("cdm", {}).get("enabled", False):
            raise ValueError("video_phased_dmd does not support training.dmd.cdm.")
        phased = dmd_config.get("phased", {})
        if not isinstance(phased, dict):
            raise ValueError("training.dmd.phased must be a mapping.")
        match_timestep = int(phased.get("match_timestep", 500))
        if match_timestep not in base_config.denoising_step_list:
            raise ValueError("training.dmd.phased.match_timestep must appear in training.dmd.denoising_step_list.")
        match_step_index = base_config.denoising_step_list.index(match_timestep)
        if not (0 < match_step_index < len(base_config.denoising_step_list)):
            raise ValueError("training.dmd.phased.match_timestep must split training.dmd.denoising_step_list into non-empty High and Low regions.")

        inference_steps = infer_config.get("denoising_step_list")
        if not isinstance(inference_steps, list):
            raise ValueError("video_phased_dmd requires inference.denoising_step_list.")
        inference_step_count = int(
            infer_config.get(
                "num_inference_steps",
                len(inference_steps),
            )
        )
        if inference_step_count != len(inference_steps):
            raise ValueError("inference.num_inference_steps must equal the length of inference.denoising_step_list.")
        if match_timestep not in inference_steps:
            raise ValueError("training.dmd.phased.match_timestep must appear in inference.denoising_step_list.")
        configured_boundary = inference_steps.index(match_timestep)
        infer_boundary = int(
            infer_config.get(
                "boundary_step_index",
                configured_boundary,
            )
        )
        if infer_boundary != configured_boundary:
            raise ValueError("inference.boundary_step_index must equal the position of training.dmd.phased.match_timestep in inference.denoising_step_list.")
        if "training_target" in phased:
            raise ValueError("training.dmd.phased.training_target is no longer supported. Phased DMD always trains High and Low regions together.")
        if max_train_iters < 2:
            raise ValueError("video_phased_dmd requires training.max_train_iters >= 2 so both High and Low regions are trained.")

        margin = int(phased.get("score_timestep_margin", 20))
        score_min = int(phased["score_timestep_min"])
        score_max = int(phased["score_timestep_max"])
        if not (1 <= score_min < score_max <= base_config.num_train_timestep):
            raise ValueError(f"training.dmd.phased score timestep bounds must satisfy 1 <= min < max <= {base_config.num_train_timestep}.")
        if score_min >= match_timestep or score_max < match_timestep + margin:
            raise ValueError("training.dmd.phased score timestep bounds must leave non-empty Low and High ranges around match_timestep.")
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
            score_timestep_min=score_min,
            score_timestep_max=score_max,
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
