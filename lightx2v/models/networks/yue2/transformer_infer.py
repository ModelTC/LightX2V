"""Native AR prefill/decode and bidirectional NAR transformer execution."""

import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer


def rotary(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    cos, sin = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class YuE2TransformerInfer(BaseTransformerInfer):
    def __init__(self, config, phase):
        self.phase = phase
        self.cpu_offload = config.get("cpu_offload", False)
        # Keep AR eager: compiler rounding can change sampled tokens.
        self.init_compile(dict(config) | {"use_compile": config.get("use_compile", False) and phase == "nar"})
        self.blocks_executed = 0

    def get_compile_block_key(self, block_idx, block):
        # Offload reuses each slot's graph across logical layers.
        return id(block) if self.cpu_offload else block_idx

    def infer(self, block_idx, block, *args):
        self.blocks_executed += 1
        return self.run_block(block_idx, block, *args)

    def infer_block(self, block, x, cos, sin, cache_k=None, cache_v=None, position=None):
        h = block.input_norm.apply(x)
        q = block.q_norm.apply(block.q.apply(h).view(-1, block.q_heads, block.head_dim))
        k = block.k_norm.apply(block.k.apply(h).view(-1, block.kv_heads, block.head_dim))
        v = block.v.apply(h).view(-1, block.kv_heads, block.head_dim)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if self.phase == "decode":
            cache_k.index_copy_(0, position, k)
            cache_v.index_copy_(0, position, v)
            keys, values = cache_k, cache_v
        elif self.phase == "nar":
            keys, values = torch.cat((cache_k, k)), torch.cat((cache_v, v))
        else:
            keys, values = k, v
        h = block.attention.apply(q, keys, values, causal=self.phase == "prefill")
        x = x + block.o.apply(h)
        h = block.mlp_norm.apply(x)
        x = x + block.down.apply(F.silu(block.gate.apply(h)) * block.up.apply(h))
        return (x, k, v) if self.phase == "prefill" else x
