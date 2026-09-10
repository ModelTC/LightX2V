import copy
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
)

import torch

from lightx2v_train.utils.generation_shapes import parse_generation_shapes
from lightx2v_train.utils.utils import get_running_dtype


def _lora_config(
    role_config: Dict[str, Any],
    role: str,
) -> Optional[Dict[str, Any]]:
    train_type = role_config["train_type"]
    if train_type != "lora":
        return None
    config = copy.deepcopy(role_config["lora"])
    config["rank"] = int(config["rank"])
    config["alpha"] = int(config["alpha"])
    return config


@dataclass(frozen=True)
class DmdConfig:
    """Parsed configuration shared by every DMD trainer."""

    student: Dict[str, Any]
    fake: Dict[str, Any]
    teacher: Dict[str, Any]
    dmd: Dict[str, Any]
    student_train_type: str
    fake_train_type: str
    student_lora: Optional[Dict[str, Any]]
    fake_lora: Optional[Dict[str, Any]]
    num_inference_steps: int
    fake_update_ratio: int
    guidance_scale: float
    negative_prompt: Optional[str]
    cfg_norm: str
    generation_shapes: list | None
    random_schedule_enabled: bool
    random_schedule_num_steps_min: int
    random_schedule_num_steps_max: int
    random_schedule_sigma_min: float
    random_schedule_sigma_max: float
    random_schedule_sampling_method: str
    latent_dtype: torch.dtype | None

    @classmethod
    def from_mapping(
        cls,
        config,
        *,
        default_negative_prompt=None,
    ):
        training = config["training"]
        if "train_type" in training:
            raise ValueError("DMD trainers use training.student.train_type and training.fake.train_type; remove training.train_type.")
        if "lora" in training:
            raise ValueError("DMD trainers do not read training.lora. Use training.student.lora and training.fake.lora.")

        student = training["student"]
        fake = training["fake"]
        teacher = training.get("teacher", {})
        dmd = training["dmd"]
        num_inference_steps = int(dmd.get("num_inference_steps", 4))
        if num_inference_steps <= 0:
            raise ValueError("training.dmd.num_inference_steps must be positive.")
        configured_negative_prompt = teacher.get("negative_prompt")
        negative_prompt = default_negative_prompt if default_negative_prompt is not None else configured_negative_prompt

        random_schedule = dmd.get("random_schedule", {})
        latent_dtype = dmd.get("latent_dtype")
        if latent_dtype is not None:
            latent_dtype = get_running_dtype(str(latent_dtype).lower())
        generation_shapes = dmd.get("generation_shapes")
        if generation_shapes is not None:
            parse_generation_shapes(generation_shapes)
        return cls(
            student=student,
            fake=fake,
            teacher=teacher,
            dmd=dmd,
            student_train_type=student["train_type"],
            fake_train_type=fake["train_type"],
            student_lora=_lora_config(student, "student"),
            fake_lora=_lora_config(fake, "fake"),
            num_inference_steps=num_inference_steps,
            fake_update_ratio=max(
                1,
                int(dmd.get("fake_update_ratio", 1)),
            ),
            guidance_scale=float(teacher.get("guidance_scale", 3.0)),
            negative_prompt=negative_prompt,
            cfg_norm=teacher.get("cfg_norm", "layer_norm"),
            generation_shapes=generation_shapes,
            random_schedule_enabled=bool(random_schedule.get("enabled", False)),
            random_schedule_num_steps_min=int(random_schedule.get("num_steps_min", 1)),
            random_schedule_num_steps_max=int(
                random_schedule.get(
                    "num_steps_max",
                    num_inference_steps,
                )
            ),
            random_schedule_sigma_min=float(random_schedule.get("sigma_min", 0.02)),
            random_schedule_sigma_max=float(random_schedule.get("sigma_max", 0.98)),
            random_schedule_sampling_method=random_schedule.get(
                "sampling_method",
                "stratified",
            ),
            latent_dtype=latent_dtype,
        )


@dataclass(frozen=True)
class DmdScheduleConfig:
    """Rollout schedule and optional student checkpoint configuration."""

    num_train_timestep: int
    num_inference_steps: int
    ts_schedule: bool
    ts_schedule_max: bool
    student_checkpoint_path: Optional[str]
    student_checkpoint_strict: bool

    @classmethod
    def from_mapping(
        cls,
        config,
        *,
        dmd_config,
        student_config,
        num_inference_steps,
    ):
        scheduler_config = config["scheduler"]
        num_train_timestep = int(scheduler_config.get("num_train_timesteps", 1000))
        return cls(
            num_train_timestep=num_train_timestep,
            num_inference_steps=num_inference_steps,
            ts_schedule=bool(dmd_config.get("ts_schedule", False)),
            ts_schedule_max=bool(dmd_config.get("ts_schedule_max", False)),
            student_checkpoint_path=student_config.get("checkpoint_path"),
            student_checkpoint_strict=bool(student_config.get("checkpoint_strict", True)),
        )
