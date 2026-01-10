import torch
import torch.nn.functional as F

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

from .module_io import LongCatImagePreInferModuleOutput


class LongCatImagePreInfer:
    """Pre-processing inference for LongCat Image Transformer."""

    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, encoder_hidden_states):
        """
        Run pre-processing: embed inputs and compute timestep embedding.

        Args:
            weights: Pre-processing weights (x_embedder, context_embedder, time_embed)
            hidden_states: Latent image tensor [B, L, C]
            encoder_hidden_states: Text embeddings [B, L, D]

        Returns:
            LongCatImagePreInferModuleOutput with processed tensors
        """
        # Embed image latents: x_embedder
        hidden_states = F.linear(
            hidden_states.squeeze(0),
            weights.x_embedder_weight,
            weights.x_embedder_bias
        )

        # Embed text context: context_embedder
        encoder_hidden_states = F.linear(
            encoder_hidden_states.squeeze(0),
            weights.context_embedder_weight,
            weights.context_embedder_bias
        )

        # Timestep embedding
        # time_proj is sinusoidal (computed in scheduler), then pass through MLP
        timesteps_proj = self.scheduler.timesteps_proj  # [B, 256]

        # timestep_embedder: linear_1 -> silu -> linear_2
        temb = F.linear(
            timesteps_proj,
            weights.timestep_embedder_linear_1_weight,
            weights.timestep_embedder_linear_1_bias
        )
        temb = F.silu(temb)
        temb = F.linear(
            temb,
            weights.timestep_embedder_linear_2_weight,
            weights.timestep_embedder_linear_2_bias
        )

        # Get rotary embeddings from scheduler
        if self.scheduler.infer_condition:
            image_rotary_emb = self.scheduler.image_rotary_emb
        else:
            image_rotary_emb = self.scheduler.negative_image_rotary_emb

        # For I2I task: get input image latents and output sequence length
        input_image_latents = None
        output_seq_len = None
        if hasattr(self.scheduler, 'input_image_latents') and self.scheduler.input_image_latents is not None:
            # Embed input image latents
            input_image_latents = F.linear(
                self.scheduler.input_image_latents.squeeze(0),
                weights.x_embedder_weight,
                weights.x_embedder_bias
            )
            output_seq_len = self.scheduler.output_seq_len

        return LongCatImagePreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            input_image_latents=input_image_latents,
            output_seq_len=output_seq_len
        )
