"""Consistency-model capabilities for Wan-family video backbones."""

from lightx2v_train.model_zoo.capability_adapters.common import (
    _consistency_requires_negative,
    _prompt_or_default,
)
from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    ProjectedTimeEmbeddingAdapter,
    SinusoidalTimeEmbeddingAdapter,
    TimeConditionedConsistencyModelCapability,
)
from lightx2v_train.model_zoo.wan.training_cache import encode_wan_video_cache


class WanConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Bind generic consistency extensions to Wan's sinusoidal time MLP."""

    def __init__(self, model) -> None:
        super().__init__(
            model,
            SinusoidalTimeEmbeddingAdapter(
                embedding_module_path="time_embedding",
                embedding_dimension_path="dim",
                frequency_dimension_path="freq_dim",
                time_scale=float(model.num_train_timesteps),
            ),
        )

    def encode_training_cache(self, batch):
        teacher = self.model.config["training"].get("consistency", {}).get("teacher", {})
        extra_prompts = {}
        conditioning_meta = {}
        if _consistency_requires_negative(self.model):
            negative_prompt = _prompt_or_default(
                teacher.get("negative_prompt"),
                self.model.unconditional_prompt,
            )
            extra_prompts["negative"] = negative_prompt
            conditioning_meta["negative_prompt"] = negative_prompt
        return encode_wan_video_cache(
            self.model,
            batch,
            extra_prompts=extra_prompts,
            conditioning_meta=conditioning_meta,
        )


class LingBotConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Bind generic consistency extensions to LingBot's time embedding."""

    def __init__(self, model) -> None:
        super().__init__(
            model,
            ProjectedTimeEmbeddingAdapter(
                hook_module_path="time_embedder",
                projection_module_path="time_proj",
                embedding_module_path="time_embedder",
                embedding_dimension_path="config.hidden_size",
                time_scale=1000.0,
            ),
        )
