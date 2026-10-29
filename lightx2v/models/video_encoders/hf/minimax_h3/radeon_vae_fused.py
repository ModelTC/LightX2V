"""MiniMax-H3 video VAE ViT decoder blocks with fused aiter gfx1201 HIP ops (FP16 infer / FP32 residual)."""

from __future__ import annotations

import types

import torch


def _eps(norm) -> float:
    return float(norm.eps) if norm.eps is not None else torch.finfo(torch.float32).eps


def _fused_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    from aiter.ops.gfx1201 import h3_ops

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, num_frames * height * width, num_channels)
    hidden_states = self.proj_in(hidden_states)
    num_patches = hidden_states.shape[1]
    hidden_states = hidden_states.to(self.sensitive_layer_dtype)

    register_tokens = self.register_tokens.expand(batch_size, -1, -1)
    cls_token = torch.zeros_like(hidden_states[:, :1, :])
    hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1).contiguous()

    grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0 for size in (num_frames, height, width)]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
    suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
    cos, sin = self.rope(torch.cat([position_ids, suffix_ids], dim=1))

    blocks = self.transformer_blocks
    rows = batch_size * hidden_states.shape[1]
    dim = hidden_states.shape[-1]
    # rotary tables are identical across heads and batch here (position ids are expanded)
    cos16 = cos[0, :, 0].to(torch.float16).contiguous().repeat(batch_size, 1)
    sin16 = sin[0, :, 0].to(torch.float16).contiguous().repeat(batch_size, 1)
    h = hidden_states
    normed = torch.empty(h.shape, device=h.device, dtype=torch.float16)
    h3_ops.vae_residual_rms(h, None, None, blocks[0].norm1.weight, normed, _eps(blocks[0].norm1))
    for index, block in enumerate(blocks):
        attn = block.attn
        query = attn.to_q(normed)
        key = attn.to_k(normed)
        value = attn.to_v(normed)
        h3_ops.vae_qk_norm_rope(query.view(rows, -1), key.view(rows, -1), cos16, sin16, attn.heads, _eps(attn.norm_q))
        query = query.unflatten(2, (attn.heads, attn.dim_head))
        key = key.unflatten(2, (attn.heads, attn.dim_head))
        value = value.unflatten(2, (attn.heads, attn.dim_head))
        out = attn.calculate.apply(query, key, value, max_seqlen_q=query.shape[1], max_seqlen_kv=key.shape[1])
        out = attn.to_out[0](out.view(batch_size, query.shape[1], attn.inner_dim))
        normed = torch.empty(h.shape, device=h.device, dtype=torch.float16)
        h3_ops.vae_residual_rms(h, out.contiguous(), block.scale1, block.norm2.weight, normed, _eps(block.norm2))
        proj = block.ff.net[0].proj(normed)
        ff = block.ff.net[2](h3_ops.vae_swiglu(proj.view(rows, -1)).view(batch_size, -1, proj.shape[-1] // 2))
        if index + 1 < len(blocks):
            nxt = blocks[index + 1].norm1
            normed = torch.empty(h.shape, device=h.device, dtype=torch.float16)
            h3_ops.vae_residual_rms(h, ff.contiguous(), block.scale2, nxt.weight, normed, _eps(nxt))
        else:
            h3_ops.vae_residual_rms(h, ff.contiguous(), block.scale2, None, None, 0.0)

    hidden_states = self.norm_out(h).to(self.infer_dtype)
    hidden_states = self.proj_out(hidden_states)[:, :num_patches, :]
    patch_size, patch_size_t = self.patch_size, self.patch_size_t
    hidden_states = hidden_states.view(batch_size, num_frames, height, width, self.out_channels, patch_size_t, patch_size, patch_size)
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(batch_size, self.out_channels, num_frames * patch_size_t, height * patch_size, width * patch_size)


def supported(decoder) -> bool:
    block = decoder.transformer_blocks[0]
    return (
        decoder.infer_dtype == torch.float16
        and decoder.sensitive_layer_dtype == torch.float32
        and not decoder.use_compile
        and block.attn.to_qkv is None
        and isinstance(block.attn.to_q, torch.nn.Linear)
        and type(block.attn.to_q) is torch.nn.Linear
        and block.attn.dim_head == 64
        and block.norm1.weight is not None
        and block.norm1.weight.dtype == torch.float32
    )


def install(vae) -> bool:
    decoder = vae.decoder
    if not supported(decoder):
        return False
    decoder._native_forward = decoder.forward
    decoder.forward = types.MethodType(_fused_forward, decoder)
    return True
