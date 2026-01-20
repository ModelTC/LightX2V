import torch

from lightx2v.models.networks.ltx2.infer.module_io import LTX2PreInferModuleOutput, TransformerArgs
from lightx2v.models.networks.ltx2.infer.utils import *
from lightx2v.utils.envs import *


class LTX2PreInfer:
    """
    Pre-inference module for LTX2 transformer.

    Handles all input preprocessing before transformer blocks.
    """

    def __init__(self, config):
        self.config = config
        # Video config
        self.num_attention_heads = self.config["num_attention_heads"]
        self.inner_dim = self.config["num_attention_heads"] * config["attention_head_dim"]
        self.positional_embedding_max_pos = [20, 2048, 2048]

        # Audio config
        self.audio_num_attention_heads = self.config.get("audio_num_attention_heads", 32)
        self.audio_attention_head_dim = self.config.get("audio_attention_head_dim", 64)
        self.audio_inner_dim = self.audio_num_attention_heads * self.audio_attention_head_dim
        self.audio_cross_attention_dim = self.config["audio_cross_attention_dim"]
        self.audio_positional_embedding_max_pos = [config["audio_pos_embed_max_pos"]]

        # Common config
        self.timestep_scale_multiplier = self.config["timestep_scale_multiplier"]
        self.av_ca_timestep_scale_multiplier = self.config["cross_attn_timestep_scale_multiplier"]
        self.double_precision_rope = self.config.get("double_precision_rope", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=10000.0,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=LTXRopeType.SPLIT,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def __infer_video(self, weights, inputs, av_ca_factor):
        """Process video modality data."""
        # Get video modality data
        v_latent = self.scheduler.video_latent_state.latent
        v_positions = self.scheduler.video_latent_state.positions
        v_timesteps = self.scheduler.video_timesteps_from_mask()
        if self.scheduler.infer_condition:
            v_context = inputs["text_encoder_output"]["v_context_p"]
        else:
            v_context = inputs["text_encoder_output"]["v_context_n"]

        # 1. Patchify projection
        video_x = weights.patchify_proj.apply(v_latent)

        # 2. Timestep embeddings (adaln)
        v_timestep = v_timesteps * self.timestep_scale_multiplier
        v_timesteps_proj = get_timestep_embedding(v_timestep.flatten()).to(GET_DTYPE())

        v_emb_linear_1_out = weights.adaln_single_emb_timestep_embedder_linear_1.apply(v_timesteps_proj)
        v_emb_linear_1_out = torch.nn.functional.silu(v_emb_linear_1_out)
        v_embedded_timestep = weights.adaln_single_emb_timestep_embedder_linear_2.apply(v_emb_linear_1_out)
        v_adaln_timestep = weights.adaln_single_linear.apply(torch.nn.functional.silu(v_embedded_timestep))

        # 3. Caption projection
        v_context = weights.caption_projection_linear_1.apply(v_context.squeeze(0))
        v_context = torch.nn.functional.gelu(v_context, approximate="tanh")
        v_context = weights.caption_projection_linear_2.apply(v_context)

        # 4. Positional embeddings
        v_pe = self._prepare_positional_embeddings(
            positions=v_positions.unsqueeze(0),  # No unsqueeze, pass directly
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.num_attention_heads,
            x_dtype=v_latent.dtype,
        )

        # 5. Cross-attention positional embeddings
        v_cross_pe = self._prepare_positional_embeddings(
            positions=v_positions.unsqueeze(0)[:, 0:1, :],  # No unsqueeze, directly slice
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[20],
            use_middle_indices_grid=True,
            num_attention_heads=self.num_attention_heads,
            x_dtype=v_latent.dtype,
        )

        # 6. Cross-attention timestep embeddings
        # Video cross scale-shift AdaLN
        v_cross_proj = get_timestep_embedding(v_timestep.flatten()).to(GET_DTYPE())
        v_cross_emb_1 = weights.av_ca_video_scale_shift_adaln_single_emb_linear_1.apply(v_cross_proj)
        v_cross_emb_1 = torch.nn.functional.silu(v_cross_emb_1)
        v_cross_emb_2 = weights.av_ca_video_scale_shift_adaln_single_emb_linear_2.apply(v_cross_emb_1)
        v_cross_scale_shift_timestep = weights.av_ca_video_scale_shift_adaln_single_linear.apply(torch.nn.functional.silu(v_cross_emb_2))

        # Video cross gate AdaLN
        v_gate_timestep_flat = v_timestep.flatten() * av_ca_factor
        v_gate_proj = get_timestep_embedding(v_gate_timestep_flat).to(GET_DTYPE())
        v_gate_emb_1 = weights.av_ca_a2v_gate_adaln_single_emb_linear_1.apply(v_gate_proj)
        v_gate_emb_1 = torch.nn.functional.silu(v_gate_emb_1)
        v_gate_emb_2 = weights.av_ca_a2v_gate_adaln_single_emb_linear_2.apply(v_gate_emb_1)
        v_cross_gate_timestep = weights.av_ca_a2v_gate_adaln_single_linear.apply(torch.nn.functional.silu(v_gate_emb_2))

        # Return TransformerArgs structure
        return TransformerArgs(
            x=video_x,
            context=v_context,
            context_mask=None,
            timesteps=v_adaln_timestep,
            embedded_timestep=v_embedded_timestep,
            positional_embeddings=v_pe,
            cross_positional_embeddings=v_cross_pe,
            cross_scale_shift_timestep=v_cross_scale_shift_timestep,
            cross_gate_timestep=v_cross_gate_timestep,
        )

    def __infer_audio(self, weights, inputs, av_ca_factor):
        """Process audio modality data."""
        # Get audio modality data
        a_latent = self.scheduler.audio_latent_state.latent
        a_positions = self.scheduler.audio_latent_state.positions
        a_timesteps = self.scheduler.audio_timesteps_from_mask()
        if self.scheduler.infer_condition:
            a_context = inputs["text_encoder_output"]["a_context_p"]
        else:
            a_context = inputs["text_encoder_output"]["a_context_n"]

        # 1. Audio patchify projection
        audio_x = weights.audio_patchify_proj.apply(a_latent)

        # 2. Audio timestep embeddings (adaln)
        a_timestep = a_timesteps * self.timestep_scale_multiplier
        a_timesteps_proj = get_timestep_embedding(a_timestep.flatten()).to(GET_DTYPE())

        a_emb_linear_1_out = weights.audio_adaln_single_emb_timestep_embedder_linear_1.apply(a_timesteps_proj)
        a_emb_linear_1_out = torch.nn.functional.silu(a_emb_linear_1_out)
        a_embedded_timestep = weights.audio_adaln_single_emb_timestep_embedder_linear_2.apply(a_emb_linear_1_out)
        a_adaln_timestep = weights.audio_adaln_single_linear.apply(torch.nn.functional.silu(a_embedded_timestep))

        # 3. Audio caption projection
        a_context = weights.audio_caption_projection_linear_1.apply(a_context.squeeze(0))
        a_context = torch.nn.functional.gelu(a_context, approximate="tanh")
        a_context = weights.audio_caption_projection_linear_2.apply(a_context)

        # 4. Audio positional embeddings
        # Note: audio positions already have batch dim [B, 1, T, 2], unlike video [3, num_patches, 2]
        a_pe = self._prepare_positional_embeddings(
            positions=a_positions,  # Already has batch dim
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.audio_num_attention_heads,
            x_dtype=a_latent.dtype,
        )

        # 5. Audio cross-attention positional embeddings (for video cross attention)
        a_cross_pe = self._prepare_positional_embeddings(
            positions=a_positions[:, 0:1, :],  # Already has batch dim
            inner_dim=self.audio_cross_attention_dim,  # Use audio_cross_attention_dim
            max_pos=[20],
            use_middle_indices_grid=True,
            num_attention_heads=self.audio_num_attention_heads,
            x_dtype=a_latent.dtype,
        )

        # 6. Audio cross-attention timestep embeddings
        # Audio cross scale-shift AdaLN
        a_cross_proj = get_timestep_embedding(a_timestep.flatten()).to(GET_DTYPE())
        a_cross_emb_1 = weights.av_ca_audio_scale_shift_adaln_single_emb_linear_1.apply(a_cross_proj)
        a_cross_emb_1 = torch.nn.functional.silu(a_cross_emb_1)
        a_cross_emb_2 = weights.av_ca_audio_scale_shift_adaln_single_emb_linear_2.apply(a_cross_emb_1)
        a_cross_scale_shift_timestep = weights.av_ca_audio_scale_shift_adaln_single_linear.apply(torch.nn.functional.silu(a_cross_emb_2))

        # Audio cross gate AdaLN
        a_gate_timestep_flat = a_timestep.flatten() * av_ca_factor
        a_gate_proj = get_timestep_embedding(a_gate_timestep_flat).to(GET_DTYPE())
        a_gate_emb_1 = weights.av_ca_v2a_gate_adaln_single_emb_linear_1.apply(a_gate_proj)
        a_gate_emb_1 = torch.nn.functional.silu(a_gate_emb_1)
        a_gate_emb_2 = weights.av_ca_v2a_gate_adaln_single_emb_linear_2.apply(a_gate_emb_1)
        a_cross_gate_timestep = weights.av_ca_v2a_gate_adaln_single_linear.apply(torch.nn.functional.silu(a_gate_emb_2))
        # Return TransformerArgs structure
        return TransformerArgs(
            x=audio_x,
            context=a_context,
            context_mask=None,
            timesteps=a_adaln_timestep,
            embedded_timestep=a_embedded_timestep,
            positional_embeddings=a_pe,
            cross_positional_embeddings=a_cross_pe,
            cross_scale_shift_timestep=a_cross_scale_shift_timestep,
            cross_gate_timestep=a_cross_gate_timestep,
        )

    @torch.no_grad()
    def infer(self, weights, inputs):
        """Main inference entry point."""
        # Calculate AV cross-attention factor (used by both video and audio)
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

        # Process video and audio modalities
        video_args = self.__infer_video(weights, inputs, av_ca_factor)
        audio_args = self.__infer_audio(weights, inputs, av_ca_factor)
        return LTX2PreInferModuleOutput(
            video_args=video_args,
            audio_args=audio_args,
        )
