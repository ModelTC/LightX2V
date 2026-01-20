"""
Transformer inference module for LTX2 transformer model.

This module handles transformer block inference including:
- Self-attention
- Cross-attention
- Feed-forward network
- Audio-video cross-attention (if applicable)
"""

import torch

from lightx2v.models.networks.ltx2.infer.module_io import LTX2PreInferModuleOutput
from lightx2v.models.networks.ltx2.infer.triton_ops import fuse_scale_shift_kernel, fused_rmsnorm_modulate
from lightx2v.models.networks.ltx2.infer.utils import LTXRopeType, apply_rotary_emb, modulate_torch_naive, modulate_with_rmsnorm_torch_naive, rmsnorm_torch_naive
from lightx2v.models.networks.wan.infer.triton_ops import norm_infer


class LTX2TransformerInfer:
    """
    Transformer inference module for LTX2 transformer.

    Handles all transformer block inference.
    """

    def __init__(self, config):
        """
        Initialize transformer inference module.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.rope_type = LTXRopeType(config["rope_type"])
        self.blocks_num = config.get("num_layers", 48)
        self.v_num_heads = config.get("num_attention_heads", 32)
        self.v_head_dim = config.get("attention_head_dim", 128)
        self.a_num_heads = config.get("audio_num_attention_heads", 32)
        self.a_head_dim = config.get("audio_attention_head_dim", 64)
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        if config.get("norm_modulate_backend", "triton") == "triton":
            self.norm_infer_func = norm_infer
            self.modulate_func = fuse_scale_shift_kernel
            self.modulate_with_rmsnorm_func = fused_rmsnorm_modulate
        else:
            self.norm_infer_func = rmsnorm_torch_naive
            self.modulate_func = modulate_torch_naive
            self.modulate_with_rmsnorm_func = modulate_with_rmsnorm_torch_naive

    def set_scheduler(self, scheduler):
        """Set the scheduler for inference."""
        self.scheduler = scheduler

    def _infer_attn(
        self,
        attn_phase,
        x: torch.Tensor,
        context=None,
        pe=None,
        k_pe=None,
        is_audio=False,
    ) -> torch.Tensor:
        q = attn_phase.to_q.apply(x)
        context = x if context is None else context
        k = attn_phase.to_k.apply(context)
        v = attn_phase.to_v.apply(context)
        q = attn_phase.q_norm.apply(q)
        k = attn_phase.k_norm.apply(k)
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type).squeeze()
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type).squeeze()
        out = attn_phase.attn_func.apply(
            q=q.view(
                -1,
                self.v_num_heads if not is_audio else self.a_num_heads,
                self.v_head_dim if not is_audio else self.a_head_dim,
            ),
            k=k.view(
                -1,
                self.v_num_heads if not is_audio else self.a_num_heads,
                self.v_head_dim if not is_audio else self.a_head_dim,
            ),
            v=v.view(
                -1,
                self.v_num_heads if not is_audio else self.a_num_heads,
                self.v_head_dim if not is_audio else self.a_head_dim,
            ),
            max_seqlen_q=q.size(0),
        )
        return attn_phase.to_out.apply(out)

    def _infer_ffn(self, ffn_phase, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network."""
        x = ffn_phase.net_0_proj.apply(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return ffn_phase.net_2.apply(x)

    @torch.no_grad()
    def infer_block(self, block, vx, ax, pre_infer_out: LTX2PreInferModuleOutput):
        """
        Perform inference for a single transformer block.

        Args:
            block: Single LTX2TransformerBlock instance
            vx: Video hidden states
            ax: Audio hidden states
            pre_infer_out: LTX2PreInferModuleOutput from pre-inference

        Returns:
            Tuple of (vx, ax) after processing this block
        """
        # Video self-attention and cross-attention
        vshift_msa, vscale_msa, vgate_msa = self._get_ada_values(
            block.scale_shift_table.tensor,
            pre_infer_out.video_args.timesteps,
            slice(0, 3),
        )

        norm_vx = self.modulate_with_rmsnorm_func(vx, vscale_msa, vshift_msa, weight=None, bias=None, eps=1e-6)
        vx = (
            vx
            + self._infer_attn(
                attn_phase=block.compute_phases[0],
                x=norm_vx,
                pe=pre_infer_out.video_args.positional_embeddings,
                is_audio=False,
            )
            * vgate_msa
        )

        vx = vx + self._infer_attn(
            attn_phase=block.compute_phases[1],
            x=self.norm_infer_func(vx, weight=None, bias=None, eps=1e-6),
            context=pre_infer_out.video_args.context,
            is_audio=False,
        )

        del vshift_msa, vscale_msa, vgate_msa

        # Audio self-attention and cross-attention
        ashift_msa, ascale_msa, agate_msa = self._get_ada_values(
            block.audio_scale_shift_table.tensor,
            pre_infer_out.audio_args.timesteps,
            slice(0, 3),
        )

        norm_ax = self.modulate_with_rmsnorm_func(ax, ascale_msa, ashift_msa, weight=None, bias=None, eps=1e-6)
        ax = (
            ax
            + self._infer_attn(
                attn_phase=block.compute_phases[2],
                x=norm_ax,
                pe=pre_infer_out.audio_args.positional_embeddings,
                is_audio=True,
            )
            * agate_msa
        )

        ax = ax + self._infer_attn(
            attn_phase=block.compute_phases[3],
            x=self.norm_infer_func(ax, weight=None, bias=None, eps=1e-6),
            context=pre_infer_out.audio_args.context,
            is_audio=True,
        )

        del ashift_msa, ascale_msa, agate_msa

        # Audio-video cross-attention
        vx_norm3 = self.norm_infer_func(vx, weight=None, bias=None, eps=1e-6)
        ax_norm3 = self.norm_infer_func(ax, weight=None, bias=None, eps=1e-6)

        # Get audio scale-shift values
        (
            scale_ca_audio_hidden_states_a2v,
            shift_ca_audio_hidden_states_a2v,
            scale_ca_audio_hidden_states_v2a,
            shift_ca_audio_hidden_states_v2a,
            gate_out_v2a,
        ) = self._get_av_ca_ada_values(
            block.scale_shift_table_a2v_ca_audio.tensor,
            pre_infer_out.audio_args.cross_scale_shift_timestep,
            pre_infer_out.audio_args.cross_gate_timestep,
        )

        # Get video scale-shift values
        (
            scale_ca_video_hidden_states_a2v,
            shift_ca_video_hidden_states_a2v,
            scale_ca_video_hidden_states_v2a,
            shift_ca_video_hidden_states_v2a,
            gate_out_a2v,
        ) = self._get_av_ca_ada_values(
            block.scale_shift_table_a2v_ca_video.tensor,
            pre_infer_out.video_args.cross_scale_shift_timestep,
            pre_infer_out.video_args.cross_gate_timestep,
        )

        # Audio-to-video cross-attention
        # vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
        # ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v

        vx_scaled = self.modulate_func(vx_norm3, scale_ca_video_hidden_states_a2v, shift_ca_video_hidden_states_a2v)
        ax_scaled = self.modulate_func(ax_norm3, scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v)

        vx = (
            vx
            + self._infer_attn(
                attn_phase=block.compute_phases[4],
                x=vx_scaled,
                context=ax_scaled,
                pe=pre_infer_out.video_args.cross_positional_embeddings,
                k_pe=pre_infer_out.audio_args.cross_positional_embeddings,
                is_audio=True,
            )
            * gate_out_a2v
        )

        # Video-to-audio cross-attention
        # ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
        # vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
        ax_scaled = self.modulate_func(ax_norm3, scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a)
        vx_scaled = self.modulate_func(vx_norm3, scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a)

        ax = (
            ax
            + self._infer_attn(
                attn_phase=block.compute_phases[5],
                x=ax_scaled,
                context=vx_scaled,
                pe=pre_infer_out.audio_args.cross_positional_embeddings,
                k_pe=pre_infer_out.video_args.cross_positional_embeddings,
                is_audio=True,
            )
            * gate_out_v2a
        )

        del gate_out_a2v, gate_out_v2a
        del (
            scale_ca_video_hidden_states_a2v,
            shift_ca_video_hidden_states_a2v,
            scale_ca_audio_hidden_states_a2v,
            shift_ca_audio_hidden_states_a2v,
            scale_ca_video_hidden_states_v2a,
            shift_ca_video_hidden_states_v2a,
            scale_ca_audio_hidden_states_v2a,
            shift_ca_audio_hidden_states_v2a,
        )

        # Video feed-forward
        vshift_mlp, vscale_mlp, vgate_mlp = self._get_ada_values(
            block.scale_shift_table.tensor,
            pre_infer_out.video_args.timesteps,
            slice(3, None),
        )
        vx_scaled = self.modulate_with_rmsnorm_func(vx, vscale_mlp, vshift_mlp, weight=None, bias=None, eps=1e-6)
        vx = vx + self._infer_ffn(block.compute_phases[6], vx_scaled) * vgate_mlp
        del vshift_mlp, vscale_mlp, vgate_mlp

        # Audio feed-forward
        ashift_mlp, ascale_mlp, agate_mlp = self._get_ada_values(
            block.audio_scale_shift_table.tensor,
            pre_infer_out.audio_args.timesteps,
            slice(3, None),
        )
        ax_scaled = self.modulate_with_rmsnorm_func(ax, ascale_mlp, ashift_mlp, weight=None, bias=None, eps=1e-6)
        ax = ax + self._infer_ffn(block.compute_phases[7], ax_scaled) * agate_mlp
        del ashift_mlp, ascale_mlp, agate_mlp

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return vx, ax

    def infer(self, weights, pre_infer_out: LTX2PreInferModuleOutput):
        """
        Perform transformer blocks inference.

        Args:
            weights: LTX2TransformerWeights instance
            pre_infer_out: LTX2PreInferModuleOutput from pre-inference

        Returns:
            Tuple of (video_x, audio_x, video_timestep, audio_timestep) after transformer blocks
        """
        vx = pre_infer_out.video_args.x
        ax = pre_infer_out.audio_args.x

        # Process all transformer blocks
        for block_idx in range(self.blocks_num):
            block = weights.blocks[block_idx]
            vx, ax = self.infer_block(block, vx, ax, pre_infer_out)

        return vx, ax, pre_infer_out.video_args.embedded_timestep, pre_infer_out.audio_args.embedded_timestep

    def _get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        """Get adaptive values from scale-shift table (no batch dimension)."""
        # timestep shape: [seq_len, total_dim] where total_dim = num_ada_params * hidden_dim
        # scale_shift_table shape: [num_ada_params, hidden_dim]
        num_ada_params = scale_shift_table.shape[0]

        # Reshape timestep to [seq_len, num_ada_params, hidden_dim]
        timestep_reshaped = timestep.reshape(timestep.shape[0], num_ada_params, -1)

        ada_values = (scale_shift_table[indices].unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype) + timestep_reshaped[:, indices, :]).unbind(dim=1)
        return ada_values

    def _get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get AV cross-attention Ada values (no batch dimension)."""
        # scale_shift_timestep shape: [seq_len, total_dim]
        # gate_timestep shape: [seq_len, total_dim]
        # scale_shift_table shape: [5, hidden_dim] (4 for scale/shift + 1 for gate)

        # Process scale-shift values (4 parameters)
        num_ss_params = num_scale_shift_values  # Should be 4
        ss_table = scale_shift_table[:num_ss_params, :]  # [4, hidden_dim]
        ss_reshaped = scale_shift_timestep.reshape(scale_shift_timestep.shape[0], num_ss_params, -1)  # [seq_len, 4, hidden_dim]
        scale_shift_ada_values = (ss_table.unsqueeze(0).to(device=scale_shift_timestep.device, dtype=scale_shift_timestep.dtype) + ss_reshaped).unbind(
            dim=1
        )  # Returns 4 tensors of shape [seq_len, hidden_dim]

        # Process gate values (1 parameter)
        gate_table = scale_shift_table[num_ss_params:, :]  # [1, hidden_dim]
        gate_reshaped = gate_timestep.reshape(gate_timestep.shape[0], gate_table.shape[0], -1)  # [seq_len, 1, hidden_dim]
        gate_values = (gate_table.unsqueeze(0).to(device=gate_timestep.device, dtype=gate_timestep.dtype) + gate_reshaped).unbind(dim=1)  # Returns 1 tensor of shape [seq_len, hidden_dim]

        return (*scale_shift_ada_values, *gate_values)
