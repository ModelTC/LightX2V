"""Consistency-model capability for Flux2 backbones."""

from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    ProjectedTimeEmbeddingAdapter,
    TimeConditionedConsistencyModelCapability,
)


class Flux2ConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Bind generic consistency extensions to Flux2 timestep conditioning."""

    def __init__(self, model) -> None:
        super().__init__(
            model,
            ProjectedTimeEmbeddingAdapter(
                hook_module_path="time_guidance_embed",
                projection_module_path="time_guidance_embed.time_proj",
                embedding_module_path="time_guidance_embed.timestep_embedder",
                embedding_dimension_path="inner_dim",
                time_scale=1000.0,
            ),
        )
