"""Consistency-model capabilities for Wan-family video backbones."""

from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    ProjectedTimeEmbeddingAdapter,
    SinusoidalTimeEmbeddingAdapter,
    TimeConditionedConsistencyModelCapability,
)


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
