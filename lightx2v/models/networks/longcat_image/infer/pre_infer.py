import torch.nn.functional as F

from lightx2v.utils.envs import *

from .module_io import LongCatImagePreInferModuleOutput


class LongCatImagePreInfer:
    """Pre-processing inference for LongCat Image Transformer."""

    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)
        self.head_dim = config["attention_head_dim"]
        self.rope = None
        self.clear_rope_cache()

    def clear_rope_cache(self):
        self._rope_cache = {True: None, False: None}

    def set_rope(self, rope):
        self.rope = rope
        self.clear_rope_cache()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.clear_rope_cache()

    def prepare_rope_cache(self, image_rotary_emb):
        if self.rope is None:
            raise RuntimeError("LongCatImagePreInfer RoPE is not initialized.")

        image_rotary_emb = self.rope.prepare_freqs(image_rotary_emb, rotary_dim=self.head_dim)
        image_rotary_positions = self.rope.prepare_positions(image_rotary_emb)
        return image_rotary_emb, image_rotary_positions

    def get_rope_cache(self, image_rotary_emb):
        branch = bool(self.scheduler.infer_condition)
        sources = tuple(image_rotary_emb[:2]) if isinstance(image_rotary_emb, tuple) else (image_rotary_emb,)
        cached = self._rope_cache[branch]
        if cached is not None:
            cached_sources, cached_value = cached
            same_sources = len(cached_sources) == len(sources) and all(
                cached_source is source for cached_source, source in zip(cached_sources, sources)
            )
            if same_sources:
                return cached_value

        prepared = self.prepare_rope_cache(image_rotary_emb)
        self._rope_cache[branch] = (sources, prepared)
        return prepared

    def infer(self, weights, hidden_states, encoder_hidden_states):
        """
        Run pre-processing: embed inputs and compute timestep embedding.

        Args:
            weights: Pre-processing weights (x_embedder, context_embedder, time_embed)
            hidden_states: Latent image tensor [B, L, C] (currently only B=1 supported)
            encoder_hidden_states: Text embeddings [B, L, D] (currently only B=1 supported)

        Returns:
            LongCatImagePreInferModuleOutput with processed tensors

        Note:
            Current implementation only supports batch_size=1. The squeeze(0) operations
            assume single batch input. Batch inference support would require refactoring
            the entire inference pipeline.
        """
        # Validate batch size (currently only batch_size=1 is supported)
        if hidden_states.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {hidden_states.shape[0]}")
        if encoder_hidden_states.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {encoder_hidden_states.shape[0]}")

        # Embed image latents: x_embedder (squeeze batch dim since B=1)
        hidden_states = weights.x_embedder.apply(hidden_states.squeeze(0))

        # Embed text context: context_embedder
        encoder_hidden_states = weights.context_embedder.apply(encoder_hidden_states.squeeze(0))

        # Timestep embedding
        # time_proj is sinusoidal (computed in scheduler), then pass through MLP
        timesteps_proj = self.scheduler.timesteps_proj  # [B, 256]

        # timestep_embedder: linear_1 -> silu -> linear_2
        temb = weights.timestep_embedder_linear_1.apply(timesteps_proj)
        temb = F.silu(temb)
        temb = weights.timestep_embedder_linear_2.apply(temb)

        # Get rotary embeddings from scheduler
        if self.scheduler.infer_condition:
            image_rotary_emb = self.scheduler.image_rotary_emb
        else:
            image_rotary_emb = self.scheduler.negative_image_rotary_emb
        image_rotary_emb, image_rotary_positions = self.get_rope_cache(image_rotary_emb)

        # For I2I task: get input image latents and output sequence length
        input_image_latents = None
        output_seq_len = None
        if hasattr(self.scheduler, "input_image_latents") and self.scheduler.input_image_latents is not None:
            # Embed input image latents
            input_image_latents = weights.x_embedder.apply(self.scheduler.input_image_latents.squeeze(0))
            output_seq_len = self.scheduler.output_seq_len

        return LongCatImagePreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            image_rotary_positions=image_rotary_positions,
            input_image_latents=input_image_latents,
            output_seq_len=output_seq_len,
        )
