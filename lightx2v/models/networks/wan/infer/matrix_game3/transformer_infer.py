"""Transformer inference for Matrix-Game-3.0.

Implements the MG3.0 WanAttentionBlock forward pass in LightX2V's
decomposed weight/infer architecture. The block execution order is:

    self_attn → cam_injection → cross_attn → action_model → ffn

This closely follows the official MG3 `WanAttentionBlock.forward()`.
"""

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

from lightx2v.models.networks.wan.infer.matrix_game2.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
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
                freq_t.permute(1, 0, 2).view(f, 1, 1, n, -1).expand(f, h, w, n, -1),
                freq_parts[1][:, :h, :].permute(1, 0, 2).view(1, h, 1, n, -1).expand(f, h, w, n, -1),
                freq_parts[2][:, :w, :].permute(1, 0, 2).view(1, 1, w, n, -1).expand(f, h, w, n, -1),
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
        # Official Matrix-Game-3 blocks are always instantiated with
        # `use_memory=True`, which slightly changes the cross-attention residual path.
        self.use_memory = True
        self.action_config = config.get("action_config", {})
        self.action_blocks = set(self.action_config.get("blocks", []))
        self.vae_time_compression_ratio = int(self.action_config.get("vae_time_compression_ratio", 4))
        self.windows_size = int(self.action_config.get("windows_size", 3))
        self.action_patch_size = list(self.action_config.get("patch_size", [1, 2, 2]))
        self.action_rope_theta = float(self.action_config.get("rope_theta", 256))
        self.enable_mouse = bool(self.action_config.get("enable_mouse", True))
        self.enable_keyboard = bool(self.action_config.get("enable_keyboard", True))
        self.action_heads_num = int(self.action_config.get("heads_num", 16))
        self.mouse_hidden_dim = int(self.action_config.get("mouse_hidden_dim", 1024))
        self.keyboard_hidden_dim = int(self.action_config.get("keyboard_hidden_dim", 1024))
        self.mouse_qk_dim_list = list(self.action_config.get("mouse_qk_dim_list", [8, 28, 28]))
        self.rope_dim_list = list(self.action_config.get("rope_dim_list", [8, 28, 28]))

    def _get_action_rotary_pos_embed(self, video_length, head_dim, rope_dim_list=None):
        target_ndim = 3
        latents_size = [video_length, self.action_patch_size[1], self.action_patch_size[2]]

        if isinstance(self.action_patch_size, int):
            rope_sizes = [s // self.action_patch_size for s in latents_size]
            patch_t = self.action_patch_size
        else:
            rope_sizes = [s // self.action_patch_size[idx] for idx, s in enumerate(latents_size)]
            patch_t = self.action_patch_size[0]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal the action attention head dim"

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.action_rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        usable = video_length * rope_sizes[1] * rope_sizes[2] // patch_t
        return freqs_cos[-usable:], freqs_sin[-usable:]

    def _run_flash_attention(self, q, k, v, causal=False):
        if FLASH_ATTN_3_AVAILABLE:
            try:
                return flash_attn_interface.flash_attn_func(q, k, v, causal=causal)
            except TypeError:
                return flash_attn_interface.flash_attn_func(q, k, v)
        if "flash_attn_func" in globals():
            try:
                return flash_attn_func(q, k, v, causal=causal)
            except TypeError:
                return flash_attn_func(q, k, v)

        q_pt = q.transpose(1, 2)
        k_pt = k.transpose(1, 2)
        v_pt = v.transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=causal).transpose(1, 2).contiguous()

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
        # Match the official MG3 block semantics:
        # when `use_memory=True`, norm3 is applied in-place on the residual stream
        # before cross-attention, so the action/ffn branches see the normalized x.
        if pre_infer_out.mouse_cond is not None or self.use_memory:
            x = cross_phase.norm3.apply(x)
            norm3_out = x
        else:
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
            q=q,
            k=k,
            v=v,
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
            x = self._infer_action_module(action_phase, x, pre_infer_out)

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
            q_memory = q[: memory_length * hw].unsqueeze(0)
            k_memory = k[: memory_length * hw].unsqueeze(0)
            q_pred = q[memory_length * hw :].unsqueeze(0)
            k_pred = k[memory_length * hw :].unsqueeze(0)

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
                dtype=torch.long,
                device=q.device,
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
            q=q,
            k=k,
            v=v,
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
        """ActionModule forward aligned with the official MG3 implementation."""
        tt, th, tw = pre_infer_out.grid_sizes.tuple
        spatial_tokens = th * tw
        pad_t = self.vae_time_compression_ratio * self.windows_size

        mouse_cond = pre_infer_out.mouse_cond
        keyboard_cond = pre_infer_out.keyboard_cond
        mouse_cond_memory = pre_infer_out.mouse_cond_memory
        keyboard_cond_memory = pre_infer_out.keyboard_cond_memory

        x_in = x.unsqueeze(0)  # [1, T*S, C]
        hidden_states = x_in
        memory_length = 0

        if self.enable_mouse and mouse_cond is not None:
            batch_size, num_frames, mouse_dim = mouse_cond.shape
            assert (((num_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0) or (num_frames % self.vae_time_compression_ratio == 0)
            if ((num_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0:
                num_feats = int((num_frames - 1) / self.vae_time_compression_ratio) + 1
                mouse_cond = torch.cat([mouse_cond[:, 0:1, :].repeat(1, pad_t, 1), mouse_cond], dim=1)
            else:
                num_feats = num_frames // self.vae_time_compression_ratio
                mouse_cond = torch.cat(
                    [mouse_cond[:, 0:1, :].repeat(1, pad_t - self.vae_time_compression_ratio, 1), mouse_cond],
                    dim=1,
                )

            mouse_groups = [
                mouse_cond[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t,
                    :,
                ]
                for i in range(num_feats)
            ]
            mouse_groups = torch.stack(mouse_groups, dim=1)
            if mouse_cond_memory is not None:
                memory_length = mouse_cond_memory.shape[1]
                mouse_memory = mouse_cond_memory.unsqueeze(2).repeat(1, 1, pad_t, 1)
                mouse_groups = torch.cat([mouse_memory, mouse_groups], dim=1)

            hidden_states_mouse = rearrange(x_in, "B (T S) C -> (B S) T C", T=tt, S=spatial_tokens)
            mouse_groups = mouse_groups.unsqueeze(-1).repeat(1, 1, 1, 1, spatial_tokens)
            mouse_groups = rearrange(mouse_groups, "b t window d s -> (b s) t (window d)")
            if mouse_groups.shape[1] != tt:
                raise ValueError(f"matrix-game-3 mouse condition window mismatch: expected latent T={tt}, got {mouse_groups.shape[1]}")

            mouse_input = torch.cat([hidden_states_mouse, mouse_groups], dim=-1)
            mouse_hidden = phase.mouse_mlp_0.apply(mouse_input.reshape(-1, mouse_input.shape[-1]))
            mouse_hidden = torch.nn.functional.gelu(mouse_hidden, approximate="tanh")
            mouse_hidden = phase.mouse_mlp_2.apply(mouse_hidden)
            mouse_hidden = phase.mouse_mlp_3.apply(mouse_hidden)
            mouse_hidden = mouse_hidden.reshape(batch_size * spatial_tokens, tt, -1)

            mouse_head_dim = self.mouse_hidden_dim // self.action_heads_num
            mouse_qkv = phase.t_qkv.apply(mouse_hidden.reshape(-1, mouse_hidden.shape[-1]))
            mouse_qkv = mouse_qkv.reshape(batch_size * spatial_tokens, tt, 3, self.action_heads_num, mouse_head_dim)
            q_m, k_m, v_m = mouse_qkv.permute(2, 0, 1, 3, 4).unbind(0)

            q_m = phase.img_attn_q_norm.apply(q_m.reshape(-1, mouse_head_dim)).reshape(batch_size * spatial_tokens, tt, self.action_heads_num, mouse_head_dim)
            k_m = phase.img_attn_k_norm.apply(k_m.reshape(-1, mouse_head_dim)).reshape(batch_size * spatial_tokens, tt, self.action_heads_num, mouse_head_dim)

            if memory_length > 0:
                freqs_memory = self._get_action_rotary_pos_embed(memory_length, mouse_head_dim, self.mouse_qk_dim_list)
                q_mem, k_mem = apply_rotary_emb(q_m[:, :memory_length], k_m[:, :memory_length], freqs_memory, head_first=False)
                q_m[:, :memory_length] = q_mem
                k_m[:, :memory_length] = k_mem

                pred_length = tt - memory_length
                if pred_length > 0:
                    freqs_pred = self._get_action_rotary_pos_embed(pred_length, mouse_head_dim, self.mouse_qk_dim_list)
                    q_pred, k_pred = apply_rotary_emb(q_m[:, memory_length:], k_m[:, memory_length:], freqs_pred, head_first=False)
                    q_m[:, memory_length:] = q_pred
                    k_m[:, memory_length:] = k_pred
            else:
                freqs = self._get_action_rotary_pos_embed(tt, mouse_head_dim, self.mouse_qk_dim_list)
                q_m, k_m = apply_rotary_emb(q_m, k_m, freqs, head_first=False)

            mouse_attn = self._run_flash_attention(q_m, k_m, v_m, causal=False)
            mouse_attn = rearrange(mouse_attn, "(b s) t h d -> b (t s) (h d)", b=batch_size, s=spatial_tokens)
            mouse_proj = phase.proj_mouse.apply(mouse_attn.reshape(-1, mouse_attn.shape[-1])).reshape(batch_size, tt * spatial_tokens, -1)
            hidden_states = x_in + mouse_proj
        else:
            hidden_states = x_in

        if self.enable_keyboard and keyboard_cond is not None:
            batch_size, num_frames, _ = keyboard_cond.shape
            assert (((num_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0) or (num_frames % self.vae_time_compression_ratio == 0)
            if ((num_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0:
                num_feats = int((num_frames - 1) / self.vae_time_compression_ratio) + 1
                keyboard_cond = torch.cat([keyboard_cond[:, 0:1, :].repeat(1, pad_t, 1), keyboard_cond], dim=1)
            else:
                num_feats = num_frames // self.vae_time_compression_ratio
                keyboard_cond = torch.cat(
                    [keyboard_cond[:, 0:1, :].repeat(1, pad_t - self.vae_time_compression_ratio, 1), keyboard_cond],
                    dim=1,
                )

            keyboard_hidden = phase.keyboard_embed_0.apply(keyboard_cond.reshape(-1, keyboard_cond.shape[-1]))
            keyboard_hidden = torch.nn.functional.silu(keyboard_hidden)
            keyboard_hidden = phase.keyboard_embed_2.apply(keyboard_hidden)
            keyboard_hidden = keyboard_hidden.reshape(batch_size, keyboard_cond.shape[1], -1)

            keyboard_groups = [
                keyboard_hidden[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t,
                    :,
                ]
                for i in range(num_feats)
            ]
            keyboard_groups = torch.stack(keyboard_groups, dim=1)
            if keyboard_cond_memory is not None:
                memory_length = keyboard_cond_memory.shape[1]
                keyboard_memory = phase.keyboard_embed_0.apply(keyboard_cond_memory.reshape(-1, keyboard_cond_memory.shape[-1]))
                keyboard_memory = torch.nn.functional.silu(keyboard_memory)
                keyboard_memory = phase.keyboard_embed_2.apply(keyboard_memory)
                keyboard_memory = keyboard_memory.reshape(batch_size, memory_length, -1)
                keyboard_memory = keyboard_memory.unsqueeze(2).repeat(1, 1, pad_t, 1)
                keyboard_groups = torch.cat([keyboard_memory, keyboard_groups], dim=1)

            if keyboard_groups.shape[1] != tt:
                raise ValueError(f"matrix-game-3 keyboard condition window mismatch: expected latent T={tt}, got {keyboard_groups.shape[1]}")

            keyboard_groups = keyboard_groups.reshape(batch_size, keyboard_groups.shape[1], -1)
            mouse_q = phase.mouse_attn_q.apply(hidden_states.reshape(-1, hidden_states.shape[-1])).reshape(batch_size, tt * spatial_tokens, -1)
            keyboard_kv = phase.keyboard_attn_kv.apply(keyboard_groups.reshape(-1, keyboard_groups.shape[-1]))
            keyboard_kv = keyboard_kv.reshape(batch_size, keyboard_groups.shape[1], -1)

            keyboard_head_dim = self.keyboard_hidden_dim // self.action_heads_num
            q_k = mouse_q.view(batch_size, -1, self.action_heads_num, keyboard_head_dim)
            kv = keyboard_kv.view(batch_size, -1, 2, self.action_heads_num, keyboard_head_dim)
            k_k, v_k = kv.permute(2, 0, 1, 3, 4).unbind(0)

            q_k = phase.key_attn_q_norm.apply(q_k.reshape(-1, keyboard_head_dim)).reshape(batch_size, -1, self.action_heads_num, keyboard_head_dim)
            k_k = phase.key_attn_k_norm.apply(k_k.reshape(-1, keyboard_head_dim)).reshape(batch_size, -1, self.action_heads_num, keyboard_head_dim)

            q_k = rearrange(q_k, "b (t s) h d -> (b s) t h d", s=spatial_tokens)
            if memory_length > 0:
                freqs_memory = self._get_action_rotary_pos_embed(memory_length, keyboard_head_dim, self.mouse_qk_dim_list)
                q_mem, k_mem = apply_rotary_emb(q_k[:, :memory_length], k_k[:, :memory_length], freqs_memory, head_first=False)
                q_k[:, :memory_length] = q_mem
                k_k[:, :memory_length] = k_mem

                pred_length = tt - memory_length
                if pred_length > 0:
                    freqs_pred = self._get_action_rotary_pos_embed(pred_length, keyboard_head_dim, self.mouse_qk_dim_list)
                    q_pred, k_pred = apply_rotary_emb(q_k[:, memory_length:], k_k[:, memory_length:], freqs_pred, head_first=False)
                    q_k[:, memory_length:] = q_pred
                    k_k[:, memory_length:] = k_pred
            else:
                freqs = self._get_action_rotary_pos_embed(tt, keyboard_head_dim, self.rope_dim_list)
                q_k, k_k = apply_rotary_emb(q_k, k_k, freqs, head_first=False)

            k_k = k_k.repeat(spatial_tokens, 1, 1, 1)
            v_k = v_k.repeat(spatial_tokens, 1, 1, 1)
            kb_attn = self._run_flash_attention(q_k, k_k, v_k, causal=False)
            kb_attn = rearrange(kb_attn, "(b s) t h d -> b (t s) (h d)", b=batch_size, s=spatial_tokens)
            kb_proj = phase.proj_keyboard.apply(kb_attn.reshape(-1, kb_attn.shape[-1])).reshape(batch_size, tt * spatial_tokens, -1)
            hidden_states = hidden_states + kb_proj

        return hidden_states.squeeze(0)

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
