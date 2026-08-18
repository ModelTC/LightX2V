"""Consistency-model capability for Qwen-Image backbones."""

from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    ProjectedTimeEmbeddingAdapter,
    TimeConditionedConsistencyModelCapability,
)


class QwenImageConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Bind the generic consistency extensions to Qwen's time embedding."""

    def __init__(self, model) -> None:
        super().__init__(
            model,
            ProjectedTimeEmbeddingAdapter(
                hook_module_path="time_text_embed",
                projection_module_path="time_text_embed.time_proj",
                embedding_module_path="time_text_embed.timestep_embedder",
                embedding_dimension_path="inner_dim",
            ),
            # Keep existing Qwen consistency checkpoints byte-for-byte compatible.
            endpoint_module_name="r_timestep_embedder",
            log_variance_module_name="logvar_linear",
        )
