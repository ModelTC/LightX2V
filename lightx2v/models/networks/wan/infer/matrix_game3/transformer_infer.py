"""Transformer inference for Matrix-Game-3.0.

Implements the MG3.0 WanAttentionBlock forward pass in LightX2V's
decomposed weight/infer architecture. The block execution order is:

    self_attn → cam_injection → cross_attn → action_model → ffn

This closely follows the official MG3 `WanAttentionBlock.forward()`.
"""

import math

import torch
from einops import rearrange

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_func

        FLASH_ATTN_3_AVAILABLE = False
    except ImportError:
        FLASH_ATTN_3_AVAILABLE = False

from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def rope_apply_with_indices(x, grid_sizes, freqs, indices):
    """Apply RoPE using explicit frame indices (for memory-aware attention).

    Rather than assuming sequential frame positions 0..F-1, this uses the
    provided ``indices`` list so that memory frames and prediction frames
    can have non-contiguous positional encodings.

    Args:
        x: Shape [B, S, num_heads, head_dim]
        grid_sizes: Shape [B, 3]  (F, H, W)
        freqs: Pre-computed RoPE frequencies
        indices: List of frame indices to use for positional encoding
    """
    n = x.shape[2]  # num_heads
    f, h, w = grid_sizes[0].tolist()

    if freqs.dim() == 2:
        # Standard freqs: [max_seq, head_dim//2]
        c = freqs.shape[1]
        c_t = c - 2 * (c // 3)
        c_h = c // 3
        c_w = c // 3
        freq_parts = freqs.split([c_t, c_h, c_w], dim=1)

        freq_t = freq_parts[0][indices]
        cos_sin = torch.cat(
            [
                freq_t.view(f, 1, 1, -1).expand(f, h, w, -1),
                freq_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
    elif freqs.dim() == 3:
        # Head-specific freqs: [num_heads, max_seq, head_dim_per_head//2]
        c = freqs.shape[2]
        c_t = c - 2 * (c // 3)
        c_h = c // 3
        c_w = c // 3
        freq_parts = freqs.split([c_t, c_h, c_w], dim=2)

        freq_t = freq_parts[0][:, indices, :]  # [n, f, c_t]
        cos_sin = torch.cat(
            [
                freq_t.permute(1, 0, 2).unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w, -1),
                freq_parts[1][:, :h, :].permute(1, 0, 2).unsqueeze(0).unsqueeze(3).expand(f, -1, -1, w, -1),
                freq_parts[2][:, :w, :].permute(1, 0, 2).unsqueeze(0).unsqueeze(2).expand(f, -1, h, -1, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, n, -1)
    else:
        raise ValueError(f"Unexpected freqs shape: {freqs.shape}")

    cos_sin = cos_sin.to(x.device)
    # Apply RoPE
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_complex * cos_sin).flatten(-2)
    return out.type_as(x)


class WanMtxg3TransformerInfer(WanTransformerInfer):
    """Transformer inference backend for Matrix-Game-3.0.

    Extends the base ``WanTransformerInfer`` to handle:
    - Memory-aware self-attention with indexed RoPE
    - Per-block camera plucker injection (scale/shift)
    - ActionModule forward pass for keyboard/mouse conditioning
    """

    def __init__(self, config):
        super().__init__(config)
        self.action_config = config.get("action_config", {})
        self.action_blocks = set(self.action_config.get("blocks", []))

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.cos_sin = pre_infer_out.cos_sin
        self.freqs = pre_infer_out.freqs
        self.reset_infer_states()
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def infer_main_blocks(self, blocks, pre_infer_out):
        x = pre_infer_out.x
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out)
        return x

    def infer_block(self, block, x, pre_infer_out):
        """Execute one MG3.0 transformer block.

        Phase order:
        0: self_attn
        1: cam_injection
        2: cross_attn
        3: action_model (only on action blocks)
        4 (or 3): ffn
        """
        has_action = self.block_idx in self.action_blocks

        # --- Modulation (6-way) ---
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )

        # --- Phase 0: Self-Attention (with memory-aware RoPE) ---
        y_out = self._infer_self_attn_mg3(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
            pre_infer_out,
        )

        # Gate and residual
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x = x + y_out * gate_msa.squeeze()

        # --- Phase 1: Camera Plucker Injection ---
        if pre_infer_out.plucker_emb is not None:
            x = self._infer_cam_injection(block.compute_phases[1], x, pre_infer_out.plucker_emb)

        # --- Phase 2: Cross-Attention ---
        cross_phase = block.compute_phases[2]
        norm3_out = cross_phase.norm3.apply(x)
        n, d = self.num_heads, self.head_dim
        q = cross_phase.cross_attn_norm_q.apply(cross_phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = cross_phase.cross_attn_norm_k.apply(cross_phase.cross_attn_k.apply(pre_infer_out.context)).view(-1, n, d)
        v = cross_phase.cross_attn_v.apply(pre_infer_out.context).view(-1, n, d)

        if self.cross_attn_cu_seqlens_q is None:
            self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32).to(q.device)
        if self.cross_attn_cu_seqlens_kv is None:
            self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]], dtype=torch.int32).to(k.device)

        attn_out = cross_phase.cross_attn_1.apply(
            q=q, k=k, v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )
        attn_out = cross_phase.cross_attn_o.apply(attn_out)
        x = x + attn_out

        # --- Phase 3: ActionModule (optional) ---
        if has_action:
            action_phase_idx = 3
            action_phase = block.compute_phases[action_phase_idx]
            x = self._infer_action_module(
                action_phase, x, pre_infer_out
            )

        # --- Phase 4 (or 3): FFN ---
        ffn_phase_idx = 4 if has_action else 3
        ffn_phase = block.compute_phases[ffn_phase_idx]
        norm2_out = ffn_phase.norm2.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.sensitive_layer_dtype)
        norm2_out = norm2_out * (1 + c_scale_msa.squeeze()) + c_shift_msa.squeeze()
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.infer_dtype)

        y = ffn_phase.ffn_0.apply(norm2_out)
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = ffn_phase.ffn_2.apply(y)

        # FFN gate + residual
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y.to(self.sensitive_layer_dtype) * c_gate_msa.squeeze()
        else:
            x = x + y * c_gate_msa.squeeze()

        return x

    def _infer_self_attn_mg3(self, phase, x, shift_msa, scale_msa, pre_infer_out):
        """Self-attention with memory-aware indexed RoPE."""
        cos_sin = self.cos_sin

        norm1_out = phase.norm1.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        norm1_out = norm1_out * (1 + scale_msa.squeeze()) + shift_msa.squeeze()
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        # Memory-aware RoPE
        memory_length = getattr(pre_infer_out, "memory_length", 0)
        memory_latent_idx = getattr(pre_infer_out, "memory_latent_idx", None)
        predict_latent_idx = getattr(pre_infer_out, "predict_latent_idx", None)
        grid_sizes = pre_infer_out.grid_sizes

        if memory_length > 0:
            hw = grid_sizes.tuple[1] * grid_sizes.tuple[2]
            # Split into memory and prediction parts
            q_memory = q[:memory_length * hw].unsqueeze(0)
            k_memory = k[:memory_length * hw].unsqueeze(0)
            q_pred = q[memory_length * hw:].unsqueeze(0)
            k_pred = k[memory_length * hw:].unsqueeze(0)

            # Build grid_sizes tensors
            f_total = grid_sizes.tuple[0]
            h, w = grid_sizes.tuple[1], grid_sizes.tuple[2]
            grid_sizes_mem = torch.tensor([[memory_length, h, w]], dtype=torch.long, device=q.device)
            grid_sizes_pred = torch.tensor([[f_total - memory_length, h, w]], dtype=torch.long, device=q.device)

            # RoPE with explicit indices
            mem_indices = memory_latent_idx if memory_latent_idx is not None else list(range(memory_length))
            q_memory = rope_apply_with_indices(q_memory, grid_sizes_mem, self.freqs, mem_indices)
            k_memory = rope_apply_with_indices(k_memory, grid_sizes_mem, self.freqs, mem_indices)

            if predict_latent_idx is not None:
                if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                    pred_indices = list(range(predict_latent_idx[0], predict_latent_idx[1]))
                else:
                    pred_indices = predict_latent_idx
            else:
                pred_indices = list(range(grid_sizes_pred[0, 0].item()))

            q_pred = rope_apply_with_indices(q_pred, grid_sizes_pred, self.freqs, pred_indices)
            k_pred = rope_apply_with_indices(k_pred, grid_sizes_pred, self.freqs, pred_indices)

            q = torch.cat([q_memory.squeeze(0), q_pred.squeeze(0)], dim=0)
            k = torch.cat([k_memory.squeeze(0), k_pred.squeeze(0)], dim=0)
        else:
            # No memory — MG3 official behavior still uses indexed RoPE.
            q_unsq = q.unsqueeze(0)
            k_unsq = k.unsqueeze(0)
            grid_sizes_t = torch.tensor(
                [[grid_sizes.tuple[0], grid_sizes.tuple[1], grid_sizes.tuple[2]]],
                dtype=torch.long, device=q.device,
            )
            if predict_latent_idx is not None:
                if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
                    pred_indices = list(range(predict_latent_idx[0], predict_latent_idx[1]))
                else:
                    pred_indices = predict_latent_idx
            else:
                pred_indices = list(range(grid_sizes.tuple[0]))
            q = rope_apply_with_indices(q_unsq, grid_sizes_t, self.freqs, pred_indices).squeeze(0)
            k = rope_apply_with_indices(k_unsq, grid_sizes_t, self.freqs, pred_indices).squeeze(0)

        img_qkv_len = q.shape[0]
        if self.self_attn_cu_seqlens_qkv is None:
            self.self_attn_cu_seqlens_qkv = torch.tensor([0, img_qkv_len], dtype=torch.int32).to(q.device)

        attn_out = phase.self_attn_1.apply(
            q=q, k=k, v=v,
            cu_seqlens_q=self.self_attn_cu_seqlens_qkv,
            cu_seqlens_kv=self.self_attn_cu_seqlens_qkv,
            max_seqlen_q=img_qkv_len,
            max_seqlen_kv=img_qkv_len,
        )

        y = phase.self_attn_o.apply(attn_out)
        return y

    def _infer_cam_injection(self, cam_phase, x, plucker_emb):
        """Apply per-block camera plucker injection via scale/shift modulation.

        From official MG3:
            c2ws_hidden = cam_injector_layer2(silu(cam_injector_layer1(plucker_emb)))
            c2ws_hidden = c2ws_hidden + plucker_emb
            cam_scale = cam_scale_layer(c2ws_hidden)
            cam_shift = cam_shift_layer(c2ws_hidden)
            x = (1 + cam_scale) * x + cam_shift
        """
        hidden = cam_phase.cam_injector_layer1.apply(plucker_emb)
        hidden = torch.nn.functional.silu(hidden)
        hidden = cam_phase.cam_injector_layer2.apply(hidden)
        hidden = hidden + plucker_emb

        cam_scale = cam_phase.cam_scale_layer.apply(hidden)
        cam_shift = cam_phase.cam_shift_layer.apply(hidden)
        x = (1.0 + cam_scale) * x + cam_shift
        return x

    def _infer_action_module(self, phase, x, pre_infer_out):
        """ActionModule forward: keyboard + mouse conditioning via cross-attention.

        This implements the official MG3 ActionModule logic in the LightX2V
        weight/infer separation style. The module:
        1. Processes mouse condition through mouse_mlp
        2. Applies temporal self-attention with QKV (t_qkv)
        3. Projects back via proj_mouse
        4. Processes keyboard condition through keyboard_embed
        5. Applies keyboard cross-attention
        6. Projects back via proj_keyboard
        """
        grid_sizes = pre_infer_out.grid_sizes
        f, h, w = grid_sizes.tuple
        S = h * w

        mouse_cond = pre_infer_out.mouse_cond
        keyboard_cond = pre_infer_out.keyboard_cond

        x_in = x.unsqueeze(0)  # [1, FHW, C]

        # --- Mouse conditioning ---
        if mouse_cond is not None:
            hidden_states = rearrange(x_in, "B (T S) C -> (B S) T C", T=f, S=S)

            # Mouse MLP
            mouse_input = torch.cat([hidden_states, mouse_cond.expand(S, -1, -1) if mouse_cond.shape[0] == 1 else mouse_cond], dim=-1)
            mouse_out = phase.mouse_mlp_0.apply(mouse_input.reshape(-1, mouse_input.shape[-1]))
            mouse_out = torch.nn.functional.gelu(mouse_out, approximate="tanh")
            mouse_out = phase.mouse_mlp_2.apply(mouse_out)
            mouse_out = phase.mouse_mlp_3.apply(mouse_out)
            mouse_out = mouse_out.reshape(S, f, -1)

            # Mouse temporal self-attention with QKV
            mouse_qkv = phase.t_qkv.apply(mouse_out.reshape(-1, mouse_out.shape[-1]))
            mouse_qkv = mouse_qkv.reshape(S, f, 3, self.num_heads, self.head_dim)
            q_m, k_m, v_m = mouse_qkv.permute(2, 0, 1, 3, 4).unbind(0)

            # QK norm (RMSNorm)
            q_m = phase.img_attn_q_norm.apply(q_m.reshape(-1, self.head_dim)).reshape(S, f, self.num_heads, self.head_dim)
            k_m = phase.img_attn_k_norm.apply(k_m.reshape(-1, self.head_dim)).reshape(S, f, self.num_heads, self.head_dim)

            # Flash attention
            if FLASH_ATTN_3_AVAILABLE:
                mouse_attn = flash_attn_interface.flash_attn_func(q_m, k_m, v_m)
            else:
                mouse_attn = flash_attn_func(q_m, k_m, v_m)

            mouse_attn = rearrange(mouse_attn, "(B S) T h d -> B (T S) (h d)", B=1, S=S)
            mouse_proj = phase.proj_mouse.apply(mouse_attn.squeeze(0)).unsqueeze(0)
            x_in = x_in + mouse_proj

        # --- Keyboard conditioning ---
        if keyboard_cond is not None:
            # Keyboard embed
            kb_emb = phase.keyboard_embed_0.apply(keyboard_cond.reshape(-1, keyboard_cond.shape[-1]))
            kb_emb = torch.nn.functional.silu(kb_emb)
            kb_emb = phase.keyboard_embed_2.apply(kb_emb)
            kb_emb = kb_emb.reshape(keyboard_cond.shape[0], keyboard_cond.shape[1], -1)

            # Keyboard cross-attention: query from hidden states, key/value from keyboard
            mouse_q = phase.mouse_attn_q.apply(x_in.squeeze(0)).unsqueeze(0)
            keyboard_kv = phase.keyboard_attn_kv.apply(kb_emb.reshape(-1, kb_emb.shape[-1]))
            keyboard_kv = keyboard_kv.reshape(1, -1, keyboard_kv.shape[-1])

            HD = mouse_q.shape[-1]
            D = HD // self.num_heads
            q_k = mouse_q.view(1, -1, self.num_heads, D)
            kv_split = keyboard_kv.view(1, -1, 2, self.num_heads, D)
            k_k, v_k = kv_split.permute(2, 0, 1, 3, 4).unbind(0)

            # QK norm
            q_k_flat = q_k.reshape(-1, D)
            k_k_flat = k_k.reshape(-1, D)
            q_k = phase.key_attn_q_norm.apply(q_k_flat).reshape(1, -1, self.num_heads, D)
            k_k = phase.key_attn_k_norm.apply(k_k_flat).reshape(1, -1, self.num_heads, D)

            # Flash attention
            if FLASH_ATTN_3_AVAILABLE:
                kb_attn = flash_attn_interface.flash_attn_func(q_k, k_k, v_k)
            else:
                kb_attn = flash_attn_func(q_k, k_k, v_k)

            kb_attn = rearrange(kb_attn, "B L H D -> B L (H D)")
            kb_proj = phase.proj_keyboard.apply(kb_attn.squeeze(0)).unsqueeze(0)
            x_in = x_in + kb_proj

        return x_in.squeeze(0)

    @property
    def freqs(self):
        """Access the pre-infer's freqs for RoPE with indices."""
        return self._freqs

    @freqs.setter
    def freqs(self, value):
        self._freqs = value

    def infer_non_blocks(self, weights, x, e):
        """Head processing — same as base but handles per-token time embeddings."""
        if e.dim() == 2:
            modulation = weights.head_modulation.tensor
            e_parts = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            modulation = weights.head_modulation.tensor.unsqueeze(2)
            e_parts = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e_parts = [ei.squeeze(1) for ei in e_parts]
        else:
            modulation = weights.head_modulation.tensor
            e_parts = (modulation + e.unsqueeze(1)).chunk(2, dim=1)

        x = weights.norm.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype)
        x = x * (1 + e_parts[1].squeeze()) + e_parts[0].squeeze()
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.infer_dtype)
        x = weights.head.apply(x)
        return x

    def set_freqs(self, freqs):
        """Set RoPE frequencies from pre_infer."""
        self._freqs = freqs
