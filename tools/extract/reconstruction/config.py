"""Configuration for the independent reconstruction entry point."""

import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path

from ..lora_extractor import ExtractConfig


@dataclass
class ReconstructionConfig(ExtractConfig):
    rank: int = 64
    alpha: float = 8.0
    dtype: str = "fp32"
    key_prefix: str = "transformer"
    algorithm: str = "qdiff"
    calibration: str = ""
    calibration_manifest: str | None = None
    calibration_counts: dict[str, int] | None = None
    calibration_inference_config: str = "tools/extract/configs/qwen_image_21_calibration.json"
    model_config: str | None = None
    work_dir: str = "outputs/reconstruction"
    initialization: str = "svd"
    initial_lora: str | None = None
    svd_output: str | None = None
    compute_dtype: str = "bf16"
    iterations: int = 1000
    temporal_iterations: int = 1000
    learning_rate: float = 5e-5
    batch_size: int | None = None
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0
    qdrop_input_mix: bool = False
    qdrop_student_prob: float = 0.5
    seed: int = 42
    checkpointing: bool = True
    log_every: int = 25
    step_indices: list[int] | None = None
    keep_cache: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.key_prefix == "auto":
            self.key_prefix = "transformer"
        if self.key_prefix != "transformer":
            raise ValueError("Qwen-Image 2.1 reconstruction uses key_prefix=transformer")
        if self.algorithm not in ("qdiff", "tfmq"):
            raise ValueError("algorithm must be qdiff or tfmq")
        if self.initialization not in ("svd", "random", "lora"):
            raise ValueError("initialization must be svd, random or lora")
        if (self.initialization == "lora") != bool(self.initial_lora):
            raise ValueError("initial_lora is required only with initialization=lora")
        if not self.calibration:
            raise ValueError("calibration must point to a native trace .pt file or a directory of traces")
        if self.calibration_counts is not None:
            counts = self.calibration_counts
            if not isinstance(counts, dict) or set(counts) != {"t2i", "i2i_single", "i2i_multi"}:
                raise ValueError("calibration_counts must specify t2i, i2i_single and i2i_multi request counts")
            if any(type(count) is not int or count < 0 for count in counts.values()) or sum(counts.values()) == 0:
                raise ValueError("calibration_counts must contain non-negative integers with a positive total")
        if self.batch_size is not None:
            if type(self.batch_size) is not int or self.batch_size < 1:
                raise ValueError("batch_size must be a positive integer (global batch across all ranks)")
            if self.gradient_accumulation != 1:
                raise ValueError("Set batch_size or gradient_accumulation; batch_size determines accumulation automatically")
        if self.compute_dtype not in ("fp32", "float32", "bf16", "bfloat16"):
            raise ValueError("compute_dtype must be fp32 or bf16 (FP32 LoRA master parameters)")
        if type(self.qdrop_input_mix) is not bool:
            raise ValueError("qdrop_input_mix must be a boolean")
        if type(self.qdrop_student_prob) not in (int, float) or not math.isfinite(self.qdrop_student_prob) or not 0 <= self.qdrop_student_prob <= 1:
            raise ValueError("qdrop_student_prob must be a finite probability in [0, 1]")
        for name in ("iterations", "temporal_iterations", "gradient_accumulation", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.step_indices is not None and (not self.step_indices or any(type(i) is not int or i < 0 for i in self.step_indices) or len(set(self.step_indices)) != len(self.step_indices)):
            raise ValueError("step_indices must contain unique non-negative integers")
        for name in ("calibration", "calibration_manifest", "calibration_inference_config", "model_config", "work_dir", "initial_lora", "svd_output"):
            value = getattr(self, name)
            if value:
                setattr(self, name, os.path.expandvars(os.path.expanduser(value)))
        self.svd_output = self.svd_output or str(Path(self.output).with_suffix(".svd.safetensors"))
        if not self.svd_output.endswith(".safetensors") or Path(self.svd_output).resolve() == Path(self.output).resolve():
            raise ValueError("svd_output must be a separate .safetensors file")


def load_config(path, overrides=None):
    path = Path(path)
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text())
    else:
        import yaml

        values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise TypeError("Config must contain a mapping")
    unknown = set(values) - {field.name for field in fields(ReconstructionConfig)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return ReconstructionConfig(**values)
