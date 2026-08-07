from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


def _mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


@dataclass(frozen=True)
class CMTimePairConfig:
    mapping: str = "ect"
    q: float = 2.0
    ratio_limit: float = 0.999
    kimg_per_stage: float = 12500.0
    min_r: float = 0.0
    safety_epsilon: float = 1e-6

    @classmethod
    def from_mapping(cls, value):
        value = _mapping(value, "training.consistency.time_pair")
        result = cls(
            mapping=str(value.get("mapping", "ect")).lower(),
            q=float(value.get("q", 2.0)),
            ratio_limit=float(value.get("ratio_limit", 0.999)),
            kimg_per_stage=float(value.get("kimg_per_stage", 12500.0)),
            min_r=float(value.get("min_r", 0.0)),
            safety_epsilon=float(value.get("safety_epsilon", 1e-6)),
        )
        if result.mapping not in {"ect", "sigmoid", "linear"}:
            raise ValueError("training.consistency.time_pair.mapping must be 'ect', 'sigmoid', or 'linear'.")
        if result.q <= 1.0:
            raise ValueError("training.consistency.time_pair.q must be greater than 1.")
        if not 0.0 <= result.ratio_limit < 1.0:
            raise ValueError("training.consistency.time_pair.ratio_limit must be in [0, 1).")
        if result.kimg_per_stage <= 0.0:
            raise ValueError("training.consistency.time_pair.kimg_per_stage must be positive.")
        if result.min_r < 0.0:
            raise ValueError("training.consistency.time_pair.min_r must be non-negative.")
        if result.safety_epsilon <= 0.0:
            raise ValueError("training.consistency.time_pair.safety_epsilon must be positive.")
        return result


@dataclass(frozen=True)
class CMLossConfig:
    distance: str = "pseudo_huber"
    huber_constant: float = 1e-8
    weighting: str = "inverse_delta"
    normalize_by_numel: bool = False
    computation_dtype: str = "float32"
    min_denominator: float = 1e-12
    sigma_data: float = 0.5

    @classmethod
    def from_mapping(cls, value):
        value = _mapping(value, "training.consistency.loss")
        result = cls(
            distance=str(value.get("distance", "pseudo_huber")).lower(),
            huber_constant=float(value.get("huber_constant", 1e-8)),
            weighting=str(value.get("weighting", "inverse_delta")).lower(),
            normalize_by_numel=bool(value.get("normalize_by_numel", False)),
            computation_dtype=str(value.get("computation_dtype", "float32")).lower(),
            min_denominator=float(value.get("min_denominator", 1e-12)),
            sigma_data=float(value.get("sigma_data", 0.5)),
        )
        if result.distance not in {"pseudo_huber", "l2", "squared_l2"}:
            raise ValueError("training.consistency.loss.distance must be 'pseudo_huber', 'l2', or 'squared_l2'.")
        if result.huber_constant < 0.0:
            raise ValueError("training.consistency.loss.huber_constant must be non-negative.")
        if result.weighting not in {
            "inverse_delta",
            "inverse_sqrt_delta",
            "none",
            "default",
            "sqrt",
            "one",
            "c_out",
            "c_out_sq",
            "sigma_sq",
        }:
            raise ValueError("Unsupported training.consistency.loss.weighting.")
        if result.computation_dtype not in {"float32", "float64"}:
            raise ValueError("training.consistency.loss.computation_dtype must be 'float32' or 'float64'.")
        if result.min_denominator <= 0.0:
            raise ValueError("training.consistency.loss.min_denominator must be positive.")
        if result.sigma_data <= 0.0:
            raise ValueError("training.consistency.loss.sigma_data must be positive.")
        return result


@dataclass(frozen=True)
class CMTeacherConfig:
    guidance_scale: Optional[float] = None
    negative_prompt: str = " "
    cfg_norm: str = "none"

    @classmethod
    def from_mapping(cls, value):
        value = _mapping(value, "training.consistency.teacher")
        raw_scale = value.get("guidance_scale")
        result = cls(
            guidance_scale=None if raw_scale is None else float(raw_scale),
            negative_prompt=str(value.get("negative_prompt", " ")),
            cfg_norm=str(value.get("cfg_norm", "none")).lower(),
        )
        if result.guidance_scale is not None and result.guidance_scale < 0.0:
            raise ValueError("training.consistency.teacher.guidance_scale must be non-negative.")
        if result.cfg_norm not in {"none", "layer_norm", "scalar"}:
            raise ValueError("training.consistency.teacher.cfg_norm must be 'none', 'layer_norm', or 'scalar'.")
        return result


@dataclass(frozen=True)
class CMConfig:
    mode: str
    time_pair: CMTimePairConfig
    loss: CMLossConfig
    teacher: CMTeacherConfig

    @classmethod
    def from_mapping(cls, config):
        consistency = _mapping(config["training"].get("consistency"), "training.consistency")
        mode = str(consistency.get("mode", "ct")).lower()
        if mode not in {"ct", "cd"}:
            raise ValueError("training.consistency.mode must be 'ct' or 'cd'.")
        return cls(
            mode=mode,
            time_pair=CMTimePairConfig.from_mapping(consistency.get("time_pair")),
            loss=CMLossConfig.from_mapping(consistency.get("loss")),
            teacher=CMTeacherConfig.from_mapping(consistency.get("teacher")),
        )


@dataclass(frozen=True)
class JVPConfig:
    method: str = "finite_difference"
    epsilon: float = 1e-3

    @classmethod
    def from_mapping(cls, value, *, default_epsilon):
        value = _mapping(value, "training.consistency.jvp")
        method = str(value.get("method", "finite_difference")).lower()
        aliases = {"fd": "finite_difference", "finite-difference": "finite_difference"}
        result = cls(
            method=aliases.get(method, method),
            epsilon=float(value.get("epsilon", default_epsilon)),
        )
        if result.method not in {"finite_difference", "exact"}:
            raise ValueError("training.consistency.jvp.method must be 'finite_difference' or 'exact'.")
        if result.epsilon <= 0.0:
            raise ValueError("training.consistency.jvp.epsilon must be positive.")
        return result


@dataclass(frozen=True)
class SCMConfig:
    mode: str
    sigma_data: float
    tangent_warmup_steps: int
    tangent_warmup_constant: float
    prior_weighting: bool
    spatially_normalized_tangent: bool
    normalize_by_numel: bool
    min_denominator: float
    jvp: JVPConfig
    teacher: CMTeacherConfig

    @classmethod
    def from_mapping(cls, config):
        consistency = _mapping(config["training"].get("consistency"), "training.consistency")
        loss = _mapping(consistency.get("loss"), "training.consistency.loss")
        mode = str(consistency.get("mode", "ct")).lower()
        if mode not in {"ct", "cd"}:
            raise ValueError("training.consistency.mode must be 'ct' or 'cd'.")
        result = cls(
            mode=mode,
            sigma_data=float(consistency.get("sigma_data", 0.5)),
            tangent_warmup_steps=int(loss.get("tangent_warmup_steps", 10000)),
            tangent_warmup_constant=float(loss.get("tangent_warmup_constant", 0.1)),
            prior_weighting=bool(loss.get("prior_weighting", True)),
            spatially_normalized_tangent=bool(loss.get("spatially_normalized_tangent", True)),
            normalize_by_numel=bool(loss.get("normalize_by_numel", True)),
            min_denominator=float(loss.get("min_denominator", 1e-12)),
            jvp=JVPConfig.from_mapping(consistency.get("jvp"), default_epsilon=1e-3),
            teacher=CMTeacherConfig.from_mapping(consistency.get("teacher")),
        )
        if result.sigma_data <= 0.0:
            raise ValueError("training.consistency.sigma_data must be positive.")
        if result.tangent_warmup_steps < 0:
            raise ValueError("tangent_warmup_steps must be non-negative.")
        if result.tangent_warmup_constant <= 0.0 or result.min_denominator <= 0.0:
            raise ValueError("sCM normalization constants must be positive.")
        return result


@dataclass(frozen=True)
class TCMConfig:
    cm: CMConfig
    transition_time: float
    boundary_probability: float
    boundary_weight: float
    stage1_checkpoint: str

    @classmethod
    def from_mapping(cls, config):
        consistency = _mapping(config["training"].get("consistency"), "training.consistency")
        stage1_checkpoint = str(consistency.get("stage1_checkpoint", "")).strip()
        result = cls(
            cm=CMConfig.from_mapping(config),
            transition_time=float(consistency.get("transition_time", 0.5)),
            boundary_probability=float(consistency.get("boundary_probability", 0.25)),
            boundary_weight=float(consistency.get("boundary_weight", 0.1)),
            stage1_checkpoint=stage1_checkpoint,
        )
        if not 0.0 < result.transition_time <= 1.0:
            raise ValueError("training.consistency.transition_time must be in (0, 1].")
        if not 0.0 <= result.boundary_probability <= 1.0:
            raise ValueError("training.consistency.boundary_probability must be in [0, 1].")
        if result.boundary_weight < 0.0:
            raise ValueError("training.consistency.boundary_weight must be non-negative.")
        if not result.stage1_checkpoint:
            raise ValueError("TCM requires training.consistency.stage1_checkpoint.")
        return result


@dataclass(frozen=True)
class PCMSolverConfig:
    num_solver_steps: int
    num_phases: int
    boundary_time: Optional[float]

    @classmethod
    def from_mapping(cls, value):
        value = _mapping(value, "training.consistency.solver")
        raw_boundary = value.get("boundary_time")
        result = cls(
            num_solver_steps=int(value.get("num_solver_steps", 50)),
            num_phases=int(value.get("num_phases", 4)),
            boundary_time=None if raw_boundary is None else float(raw_boundary),
        )
        if result.num_solver_steps <= 0:
            raise ValueError("PCM solver.num_solver_steps must be positive.")
        if not 1 <= result.num_phases <= result.num_solver_steps:
            raise ValueError("PCM solver.num_phases must be in [1, num_solver_steps].")
        if result.boundary_time is not None and not 0.0 <= result.boundary_time < 1.0:
            raise ValueError("PCM solver.boundary_time must be in [0, 1).")
        return result


@dataclass(frozen=True)
class PCMLossConfig:
    distance: str
    huber_constant: float
    computation_dtype: str

    @classmethod
    def from_mapping(cls, value):
        value = _mapping(value, "training.consistency.loss")
        distance = str(value.get("distance", "pseudo_huber")).lower()
        aliases = {"huber": "pseudo_huber", "mse": "l2", "squared_l2": "l2"}
        result = cls(
            distance=aliases.get(distance, distance),
            huber_constant=float(value.get("huber_constant", 1e-3)),
            computation_dtype=str(value.get("computation_dtype", "float32")).lower(),
        )
        if result.distance not in {"pseudo_huber", "l2"}:
            raise ValueError("PCM loss.distance must be 'pseudo_huber' or 'l2'.")
        if result.huber_constant < 0.0:
            raise ValueError("PCM loss.huber_constant must be non-negative.")
        if result.computation_dtype not in {"float32", "float64"}:
            raise ValueError("PCM loss.computation_dtype must be 'float32' or 'float64'.")
        return result


@dataclass(frozen=True)
class PCMConfig:
    mode: str
    solver: PCMSolverConfig
    loss: PCMLossConfig
    teacher: CMTeacherConfig

    @classmethod
    def from_mapping(cls, config):
        consistency = _mapping(config["training"].get("consistency"), "training.consistency")
        mode = str(consistency.get("mode", "cd")).lower()
        if mode != "cd":
            raise ValueError("The published PCM algorithm is distillation-only; set training.consistency.mode to 'cd'.")
        return cls(
            mode=mode,
            solver=PCMSolverConfig.from_mapping(consistency.get("solver")),
            loss=PCMLossConfig.from_mapping(consistency.get("loss")),
            teacher=CMTeacherConfig.from_mapping(consistency.get("teacher")),
        )


@dataclass(frozen=True)
class MeanFlowConfig:
    mode: str
    random_endpoint_probability: float
    loss_type: str
    norm_method: str
    norm_constant: float
    tangent_warmup_steps: int
    spatially_normalized_tangent: bool
    min_denominator: float
    condition_dropout_probability: Optional[float]
    guidance_scale: Optional[float]
    guidance_mixture_ratio: Optional[float]
    guidance_time_start: float
    guidance_time_end: float
    jvp: JVPConfig
    teacher: CMTeacherConfig

    @classmethod
    def from_mapping(cls, config):
        consistency = _mapping(config["training"].get("consistency"), "training.consistency")
        loss = _mapping(consistency.get("loss"), "training.consistency.loss")
        sampling = _mapping(consistency.get("sampling"), "training.consistency.sampling")
        guidance = _mapping(consistency.get("guidance"), "training.consistency.guidance")
        teacher = _mapping(consistency.get("teacher"), "training.consistency.teacher")
        mode = str(consistency.get("mode", "ct")).lower()
        raw_dropout = guidance.get("condition_dropout_probability")
        raw_scale = guidance.get("scale", teacher.get("guidance_scale"))
        raw_mixture = guidance.get("mixture_ratio")
        result = cls(
            mode=mode,
            random_endpoint_probability=float(sampling.get("random_endpoint_probability", 0.0)),
            loss_type=str(loss.get("type", "opt_grad")).lower(),
            norm_method=str(loss.get("norm_method", "poly_1.0")).lower(),
            norm_constant=float(loss.get("norm_constant", 0.1)),
            tangent_warmup_steps=int(loss.get("tangent_warmup_steps", 0)),
            spatially_normalized_tangent=bool(loss.get("spatially_normalized_tangent", False)),
            min_denominator=float(loss.get("min_denominator", 1e-12)),
            condition_dropout_probability=None if raw_dropout is None else float(raw_dropout),
            guidance_scale=None if raw_scale is None else float(raw_scale),
            guidance_mixture_ratio=None if raw_mixture is None else float(raw_mixture),
            guidance_time_start=float(guidance.get("time_start", 0.0)),
            guidance_time_end=float(guidance.get("time_end", 1.0)),
            jvp=JVPConfig.from_mapping(consistency.get("jvp"), default_epsilon=1e-4),
            teacher=CMTeacherConfig.from_mapping(consistency.get("teacher")),
        )
        if result.mode not in {"ct", "cd"}:
            raise ValueError("training.consistency.mode must be 'ct' or 'cd'.")
        if not 0.0 <= result.random_endpoint_probability <= 1.0:
            raise ValueError("random_endpoint_probability must be in [0, 1].")
        if result.loss_type not in {"l2", "opt_grad"}:
            raise ValueError("MeanFlow loss.type must be 'l2' or 'opt_grad'.")
        if result.tangent_warmup_steps < 0 or result.norm_constant <= 0.0 or result.min_denominator <= 0.0:
            raise ValueError("MeanFlow normalization values are invalid.")
        if result.condition_dropout_probability is not None and not 0.0 <= result.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1].")
        if result.guidance_scale is not None and result.guidance_scale < 0.0:
            raise ValueError("MeanFlow guidance scale must be non-negative.")
        if result.guidance_time_start > result.guidance_time_end:
            raise ValueError("guidance.time_start cannot be greater than guidance.time_end.")
        return result
