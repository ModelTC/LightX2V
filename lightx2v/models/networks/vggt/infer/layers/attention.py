# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class AttentionInfer(ModuleInfer):
    def infer(self, x, pos=None):
        (batch, tokens, channels) = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        (q, k) = (self.q_norm(q), self.k_norm(k))
        if self.rope is not None:
            (q, k) = (self.rope(q, pos), self.rope(k, pos))
        out = self.attention(q, k, v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)
