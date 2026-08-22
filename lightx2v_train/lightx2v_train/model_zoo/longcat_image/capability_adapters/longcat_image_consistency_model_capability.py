"""Consistency-model capability for LongCat-Image."""

from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    ProjectedTimeEmbeddingAdapter,
    TimeConditionedConsistencyModelCapability,
)


class LongCatImageConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Bind generic consistency extensions to LongCat's time embedding."""

    def __init__(self, model) -> None:
        super().__init__(
            model,
            ProjectedTimeEmbeddingAdapter(
                hook_module_path="time_embed",
                projection_module_path="time_embed.time_proj",
                embedding_module_path="time_embed.timestep_embedder",
                embedding_dimension_path="inner_dim",
                time_scale=1000.0,
            ),
        )
