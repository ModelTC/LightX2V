"""Dense partitions for the native Animate2 packed reference-attention mask."""

import math

import torch

from lightx2v.common.ops.attn.flex_attn import animate2_token_layout
from lightx2v.common.ops.attn.template import AttnWeightTemplate
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER


def partition_attention(q, k, v, backend, *, origin_len, origin_area, log_scale=0.0):
    from lightx2v.common.ops.attn.ring_attn import _merge_attention_blocks

    area, valid_q, valid_ref, total_q, total_ref = animate2_token_layout(origin_len, origin_area)
    if math.isnan(log_scale) or log_scale == math.inf:
        raise ValueError("Expected finite log_scale or -inf")
    if q.ndim != 3 or k.shape != v.shape or k.ndim != 3 or q.shape[1:] != k.shape[1:]:
        raise ValueError("Expected compatible SHD Q/K/V")
    if q.shape[0] != total_q or k.shape[0] != total_q + total_ref:
        raise ValueError("Packed padding differs from native Flex layout")
    heads, dim = q.shape[1:]

    def dense(query, key, value):
        output, lse = backend.apply_with_lse(query, key, value)
        count = math.prod(query.shape[:-2])
        if output.shape not in ((count, heads * dim), query.shape) or lse.shape != (count, heads):
            raise ValueError("Dense backend output/LSE ABI differs from native contract")
        return output.reshape(count, heads, dim), lse.float()

    # A constant bias on frame 1 shifts its LSE and leaves its normalized output unchanged.
    main_k = torch.cat((k[:area], k[2 * area : valid_q]), dim=0)
    main_v = torch.cat((v[:area], v[2 * area : valid_q]), dim=0)
    main, lse = dense(q[:valid_q], main_k, main_v)
    del main_k, main_v
    if log_scale != -math.inf:
        biased, biased_lse = dense(q[:valid_q], k[area : 2 * area], v[area : 2 * area])
        main, lse = _merge_attention_blocks(main, lse, biased, biased_lse + log_scale)
        del biased, biased_lse

    # Query frame t sees only reference frame t-1 in addition to the generation keys.
    reference, reference_lse = dense(
        q[area:valid_q].reshape(-1, area, heads, dim),
        k[total_q : total_q + valid_ref].reshape(-1, area, heads, dim),
        v[total_q : total_q + valid_ref].reshape(-1, area, heads, dim),
    )
    tail, _ = _merge_attention_blocks(main[area:], lse[area:], reference, reference_lse)
    output = torch.zeros_like(q)
    output[:area] = main[:area]
    output[area:valid_q] = tail
    return output


class PartitionedFlexAttnWeight(AttnWeightTemplate):
    """Use native dense attention/LSE while preserving Flex packing metadata."""

    def __init__(self, backend_type):
        if backend_type not in {"flash_attn2", "flash_attn3", "sage_attn2"}:
            raise ValueError(f"Unsupported Animate2 partition backend: {backend_type}")
        super().__init__(backend_type)
        self.dense_attention = ATTN_WEIGHT_REGISTER[backend_type]()

    def compute_stats(self, arguments, output):
        return {"status": "composite", "note": "Count attention FLOPs in the dense apply_with_lse subcalls."}

    def apply(self, q, k, v, *, origin_len, origin_area, log_scale=0.0, **kwargs):
        return partition_attention(q, k, v, self.dense_attention, origin_len=origin_len, origin_area=origin_area, log_scale=log_scale)
