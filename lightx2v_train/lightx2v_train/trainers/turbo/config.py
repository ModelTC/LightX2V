"""Configuration and explicit update accounting for H3 Turbo distillation."""

import math
from dataclasses import asdict, dataclass, fields
from numbers import Real


@dataclass(frozen=True)
class TurboConfig:
    num_inference_steps: int = 8
    dcm_steps: int = 2000
    scm_steps: int = 500
    joint_steps: int = 4500
    fake_update_ratio: int = 4
    video_shift: float = 12.0
    audio_shift: float = 3.0
    dcm_total_steps: int = 48
    dcm_skip_steps: int = 1
    loss_scale: float = 100.0
    loss_scale_dmd: float = 1.0
    scm_normalizer_epsilon: float = 0.1
    dmd_normalizer_floor: float = 1e-5
    tangent_warmup_steps: int = 1000
    generator_time_mean: float = -0.8
    generator_time_std: float = 1.6
    score_time_mean: float = 0.0
    score_time_std: float = 1.6
    video_weight: float = 1.0
    audio_weight: float = 1.0

    @property
    def total_steps(self):
        """Student optimizer updates; fake updates are additional."""
        return self.dcm_steps + self.scm_steps + self.joint_steps

    def stage_at(self, student_updates):
        if not 0 <= student_updates < self.total_steps:
            raise ValueError("Student cursor is outside the Turbo curriculum")
        if student_updates < self.dcm_steps:
            return "dcm"
        if student_updates < self.dcm_steps + self.scm_steps:
            return "scm"
        return "joint"

    def fake_updates_before(self, student_updates):
        if not 0 <= student_updates <= self.total_steps:
            raise ValueError("Invalid Turbo student cursor")
        return max(0, student_updates - self.dcm_steps - self.scm_steps) * self.fake_update_ratio

    def consumed_microbatches(self, student_updates, accumulation=1):
        return (student_updates + self.fake_updates_before(student_updates)) * accumulation

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, config):
        training = config["training"]
        if training.get("method") != "turbo":
            raise ValueError("Turbo requires training.method=turbo")
        options = training.get("turbo", {})
        if not isinstance(options, dict):
            raise ValueError("training.turbo must be a mapping")
        unknown = set(options) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training.turbo options: {sorted(unknown)}")
        result = cls(**options)
        integer_fields = (
            "num_inference_steps",
            "dcm_steps",
            "scm_steps",
            "joint_steps",
            "fake_update_ratio",
            "dcm_total_steps",
            "dcm_skip_steps",
            "tangent_warmup_steps",
        )
        for name in integer_fields:
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"training.turbo.{name} must be an integer")
        for name in ("dcm_steps", "scm_steps", "joint_steps", "tangent_warmup_steps"):
            if getattr(result, name) < 0:
                raise ValueError(f"training.turbo.{name} must be non-negative")
        if result.total_steps < 1 or training.get("max_train_iters") != result.total_steps:
            raise ValueError("max_train_iters must equal dcm_steps + scm_steps + joint_steps (STUDENT updates)")
        if min(result.num_inference_steps, result.fake_update_ratio, result.dcm_total_steps) < 1:
            raise ValueError("Turbo inference steps, fake ratio and dCM grid size must be positive")
        if not 1 <= result.dcm_skip_steps <= result.dcm_total_steps:
            raise ValueError("Turbo requires 1 <= dcm_skip_steps <= dcm_total_steps")
        for name in (
            "video_shift",
            "audio_shift",
            "loss_scale",
            "loss_scale_dmd",
            "scm_normalizer_epsilon",
            "dmd_normalizer_floor",
            "generator_time_std",
            "score_time_std",
            "video_weight",
            "audio_weight",
        ):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"training.turbo.{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"training.turbo.{name} must be finite and positive")
        for name in ("generator_time_mean", "score_time_mean"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"training.turbo.{name} must be finite")
        if config["model"].get("name") not in {"minimax_h3_t2av", "minimax_h3_ref2av"}:
            raise ValueError("Turbo currently supports minimax_h3_t2av and minimax_h3_ref2av")
        if "dmd" in training:
            raise ValueError("Turbo reads training.turbo, not training.dmd; remove the old PDMD options")
        if training.get("teacher", {}).get("guidance_scale", 1.0) != 1.0:
            raise ValueError("H3 Turbo uses the direct conditional teacher: guidance_scale must be 1")
        if config.get("scheduler", {}).get("time_shift_settings", {}).get("do_time_shift", False):
            raise ValueError("Turbo owns modality shifts; scheduler.do_time_shift must be false")
        sp = config.get("distributed", {}).get("sequence_parallel", {})
        if sp.get("enabled", False) and int(sp.get("size", 1)) != 1:
            raise ValueError("H3 Turbo exact JVP currently requires sequence_parallel.size=1")
        matching = config["model"].get("capabilities", {}).get("distribution_matching", {})
        for name in ("official_pdmd", "projected_dmd"):
            if matching.get(name, False):
                raise ValueError(f"Turbo uses ordinary DMD; remove {name}")
        for name, value in (("video_flow_shift", result.video_shift), ("audio_flow_shift", result.audio_shift)):
            if name in matching and float(matching[name]) != value:
                raise ValueError(f"Turbo {name} differs between training and model metadata")
        if training.get("student", {}).get("checkpoint_path"):
            raise ValueError("Start Turbo from the base model or resume a complete Turbo checkpoint")
        if config.get("resume", {}).get("allow_distribution_matching_transition", False):
            raise ValueError("Turbo resume requires an unchanged recipe; distribution-matching transitions are not supported")
        accumulation = training.get("gradient_accumulation_iters", 1)
        if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
            raise ValueError("Turbo gradient_accumulation_iters must be a positive integer")
        limit = training.get("save_total_limit")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("Turbo save_total_limit must be a positive integer or null")
        return result
