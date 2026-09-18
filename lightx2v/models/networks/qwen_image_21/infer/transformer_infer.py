import torch
import torch.nn.functional as F


def apply_rope(x, frequencies):
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * frequencies[:, None]).flatten(-2).to(x.dtype)


class QwenImage21TransformerInfer:
    def __init__(self, config):
        self.config = config
        self.heads = config["num_attention_heads"]

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer_block(self, block, x, state, index, cache, modulation):
        scale1, gate1, scale2, gate2 = modulation
        h = block.norm1.apply(x) * scale1
        q = block.norm_q.apply(block.q.apply(h).reshape(-1, self.heads, self.config["attention_head_dim"]))
        k = block.norm_k.apply(block.k.apply(h).reshape_as(q))
        v = block.v.apply(h).reshape_as(q)
        q, k = apply_rope(q, state.rotary), apply_rope(k, state.rotary)
        prefix = state.layout.prefix_len
        if state.cached:
            k = torch.cat((cache.k_cache(index), k))
            v = torch.cat((cache.v_cache(index), v))
        else:
            cache.store_kv(k[:prefix], v[:prefix], index)
        if state.cached:
            attention = block.attention.apply(q, k, v)
        else:
            attention = x.new_zeros(x.shape)
            for begin, end, is_text in state.layout.segments + [(prefix, len(q), False)]:
                mask = None
                if is_text:
                    mask = torch.arange(end, device=x.device)[None] <= torch.arange(begin, end, device=x.device)[:, None]
                op = block.prefix_attention if mask is not None else block.attention
                attention[begin:end] = op.apply(q[begin:end], k[:end], v[:end], attn_mask=mask)
        x = x + gate1 * block.out.apply(attention)
        h = block.norm2.apply(x) * scale2
        x = x + gate2 * block.down.apply(F.silu(block.gate.apply(h)) * block.up.apply(h))
        return x.clamp(-65504, 65504) if x.dtype == torch.float16 else x

    def infer(self, weights, state, cache):
        x = state.hidden_states
        # Modulation is shared by every block in this step.
        scale1, gate1, scale2, gate2 = state.modulation.chunk(4, -1)
        modulation = tuple(value[state.rows] for value in (1 + scale1, gate1.tanh(), 1 + scale2, gate2.tanh()))
        for index, block in enumerate(weights.blocks):
            x = self.infer_block(block, x, state, index, cache, modulation)
        return x
