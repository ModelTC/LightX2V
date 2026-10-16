"""Qwen-Image preprocessing and replay-transport configuration."""

from dataclasses import dataclass

from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


@dataclass
class QwenImage21PipelineConfig(DiffusionPipelineConfig):
    """Official preprocessing settings and bounded replay transport shapes."""

    output_resolution: int = 1024
    max_reference_tokens: int = 8192
    max_reference_images: int = 1
    replay_prompt_length: int = 4096
    use_kv_cache: bool = False
    init_same_noise: bool = True
