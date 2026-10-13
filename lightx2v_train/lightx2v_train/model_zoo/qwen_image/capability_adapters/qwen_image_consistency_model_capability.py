"""Consistency-model capability for Qwen-Image backbones."""

import torch

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
            endpoint_module_name="r_timestep_embedder",
            log_variance_module_name="logvar_linear",
        )


class QwenImage21TimeEmbeddingAdapter(ProjectedTimeEmbeddingAdapter):
    """Condition target rows while preserving the extra t=0 prefix row."""

    def encode(self, denoiser, embedder, time):
        # The 2.1 projection applies its own 1000x scaling and expects a vector.
        return super().encode(denoiser, embedder, time.reshape(-1))

    def add_endpoint_embedding(self, denoiser, endpoint_embedder, endpoint_time, hook_module, output):
        if not denoiser.config.causal_condition:
            return super().add_endpoint_embedding(denoiser, endpoint_embedder, endpoint_time, hook_module, output)
        embedding = self.encode(denoiser, endpoint_embedder, endpoint_time)
        batch_size = output.shape[0] - 1
        if embedding.shape[0] not in (1, batch_size):
            raise ValueError(f"Expected one endpoint time or {batch_size} endpoint times, got {embedding.shape[0]}.")
        target_rows = output[:-1] + embedding.to(device=output.device, dtype=output.dtype)
        # Keep the text/reference prefix at t = 0 for every endpoint.
        return torch.cat((target_rows, output[-1:]), dim=0)


class QwenImage21ConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Bind generic consistency features to the 2.1 timestep embedding."""

    def __init__(self, model) -> None:
        super().__init__(
            model,
            QwenImage21TimeEmbeddingAdapter(
                hook_module_path="time_text_embed",
                projection_module_path="time_text_embed.time_proj",
                embedding_module_path="time_text_embed.timestep_embedder",
                embedding_dimension_path="inner_dim",
            ),
            endpoint_module_name="r_timestep_embedder",
            log_variance_module_name="logvar_linear",
        )
