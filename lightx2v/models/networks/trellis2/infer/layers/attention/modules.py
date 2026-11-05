# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Optional

import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.module_infer import ModuleInfer
from lightx2v.models.networks.trellis2.infer.layers.attention.rope import RotaryPositionEmbedderInfer


class MultiHeadRMSNormInfer(ModuleInfer):
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class MultiHeadAttentionInfer(ModuleInfer):
    def infer(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        (B, L, C) = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.qk_rms_norm or self.use_rope:
                (q, k, v) = qkv.unbind(dim=2)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    assert phases is not None, "Phases must be provided for RoPE"
                    q = RotaryPositionEmbedderInfer.apply_rotary_embedding(q, phases)
                    k = RotaryPositionEmbedderInfer.apply_rotary_embedding(k, phases)
                h = self.attention(q, k, v)
            else:
                h = self.attention(qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                (k, v) = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = self.attention(q, k, v)
            else:
                h = self.attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h
