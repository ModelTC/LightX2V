# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import lightx2v.common.ops.sparse as sp
from lightx2v.common.ops.sparse.basic import SparseTensor, VarLenTensor
from lightx2v.models.networks.trellis2.utils.position import RotaryPositionEmbedder


class Trellis2TransformerInfer:
    """Dense and sparse flow computation with explicit weight arguments."""

    @staticmethod
    def _timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def infer_timestep(self, weights, t):
        t_freq = self._timestep_embedding(t, weights.frequency_embedding_size)
        t_emb = weights.mlp(t_freq)
        return t_emb

    def infer_dense_flow(self, weights, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], weights.in_channels, *[weights.resolution] * 3], (
            f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], weights.in_channels, *[weights.resolution] * 3]}"
        )
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = weights.input_layer(h)
        if weights.pe_mode == "ape":
            h = h + weights.pos_emb[None]
        t_emb = self.infer_timestep(weights.t_embedder, t)
        if weights.share_mod:
            t_emb = weights.adaLN_modulation(t_emb)
        t_emb = t_emb.to(dtype=weights.dtype)
        h = h.to(dtype=weights.dtype)
        cond = cond.to(dtype=weights.dtype)
        for block in weights.blocks:
            h = self.infer_dense_block(block, h, t_emb, cond, weights.rope_phases)
        h = h.to(dtype=x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = weights.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[weights.resolution] * 3).contiguous()
        return h

    def infer_sparse_flow(
        self, weights, x: sp.SparseTensor, t: torch.Tensor, cond: Union[torch.Tensor, List[torch.Tensor]], concat_cond: Optional[sp.SparseTensor] = None, **kwargs
    ) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        h = weights.input_layer(x)
        h = h.to(dtype=weights.dtype)
        t_emb = self.infer_timestep(weights.t_embedder, t)
        if weights.share_mod:
            t_emb = weights.adaLN_modulation(t_emb)
        t_emb = t_emb.to(dtype=weights.dtype)
        cond = cond.to(dtype=weights.dtype)
        if weights.pe_mode == "ape":
            pe = weights.pos_embedder(h.coords[:, 1:])
            h = h + pe.to(dtype=weights.dtype)
        for block in weights.blocks:
            h = self.infer_sparse_block(block, h, t_emb, cond)
        h = h.to(dtype=x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = weights.out_layer(h)
        return h

    def infer_qk_norm(self, weights, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * weights.gamma * weights.scale).to(x.dtype)

    def infer_attention(self, weights, x: torch.Tensor, context: Optional[torch.Tensor] = None, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        (B, L, C) = x.shape
        if weights._type == "self":
            qkv = weights.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, weights.num_heads, -1)
            if weights.qk_rms_norm or weights.use_rope:
                (q, k, v) = qkv.unbind(dim=2)
                if weights.qk_rms_norm:
                    q = self.infer_qk_norm(weights.q_rms_norm, q)
                    k = self.infer_qk_norm(weights.k_rms_norm, k)
                if weights.use_rope:
                    assert phases is not None, "Phases must be provided for RoPE"
                    q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                    k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
                h = weights.attention(q, k, v)
            else:
                h = weights.attention(qkv)
        else:
            Lkv = context.shape[1]
            q = weights.to_q(x)
            kv = weights.to_kv(context)
            q = q.reshape(B, L, weights.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, weights.num_heads, -1)
            if weights.qk_rms_norm:
                q = self.infer_qk_norm(weights.q_rms_norm, q)
                (k, v) = kv.unbind(dim=2)
                k = self.infer_qk_norm(weights.k_rms_norm, k)
                h = weights.attention(q, k, v)
            else:
                h = weights.attention(q, kv)
        h = h.reshape(B, L, -1)
        h = weights.to_out(h)
        return h

    def infer_feed_forward(self, weights, x: torch.Tensor) -> torch.Tensor:
        return weights.mlp(x)

    def infer_dense_block(self, weights, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights.share_mod:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (weights.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = weights.adaLN_modulation(mod).chunk(6, dim=1)
        h = weights.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.infer_attention(weights.self_attn, h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = weights.norm2(x)
        h = self.infer_attention(weights.cross_attn, h, context)
        x = x + h
        h = weights.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.infer_feed_forward(weights.mlp, h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def infer_sparse_block(self, weights, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if weights.share_mod:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (weights.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = weights.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(weights.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = weights.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(weights.norm2(x.feats))
        h = weights.cross_attn(h, context)
        x = x + h
        h = x.replace(weights.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.infer_feed_forward(weights.mlp, h)
        h = h * gate_mlp
        x = x + h
        return x
