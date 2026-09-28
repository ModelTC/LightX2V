import os

import torch
from loguru import logger


_REPORTED_ROUTES = set()
_REPORTED_INIT = False


def _report_route(query, key, value, checks, block_idx):
    reasons = tuple(name for name, passed in checks.items() if not passed)
    signature = (block_idx, str(query.device), str(query.dtype), tuple(query.shape), tuple(key.shape), tuple(value.shape), query.stride(), key.stride(), value.stride(), reasons)
    if signature not in _REPORTED_ROUTES:
        _REPORTED_ROUTES.add(signature)
        logger.info("gfx1201_sage_attn route={} rank={} block={} qkv_shapes={} dtype={} qkv_strides={} reasons={}",
                    "fallback_aiter" if reasons else "custom", os.environ.get("RANK", "?"),
                    block_idx, (tuple(query.shape), tuple(key.shape), tuple(value.shape)), query.dtype,
                    (query.stride(), key.stride(), value.stride()), reasons)

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER


@PLATFORM_ATTN_WEIGHT_REGISTER("gfx1201_sage_attn")
class RadeonCoreswAttnWeight(AttnWeightTemplate):
    def __init__(self):
        global _REPORTED_INIT
        self.config = {}
        if not _REPORTED_INIT:
            _REPORTED_INIT = True
            logger.info("gfx1201_sage_attn initialized rank={} platform={} adapter={}",
                        os.environ.get("RANK", "?"), os.environ.get("PLATFORM", "auto"), __file__)

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None,
              max_seqlen_q=None, max_seqlen_kv=None, drop_rate=0,
              attn_mask=None, causal=False, scheduler=None, block_idx=None, **kwargs):
        if drop_rate or causal or attn_mask is not None or kwargs:
            raise ValueError("gfx1201_sage_attn supports only unmasked noncausal inference without extra options")
        if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError("Expected SHD or BSHD Q/K/V")
        if any(tensor.requires_grad for tensor in (q, k, v)):
            raise ValueError("gfx1201_sage_attn is inference-only")
        if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
            raise ValueError("Q/K/V must share device and dtype")
        if q.device.type != "cuda" or torch.version.hip is None:
            raise RuntimeError("gfx1201_sage_attn requires ROCm")
        if q.ndim == 3:
            query, key, value = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        else:
            query, key, value = q, k, v
        batch, sequence, heads, dimension = query.shape
        if key.shape[0] != batch or value.shape != key.shape:
            raise ValueError("Expected equal batch counts and equal K/V shapes")
        key_sequence = key.shape[1]
        query_lengths = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sequence
        key_lengths = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * key_sequence
        cu_query = query_lengths if cu_seqlens_q is None else cu_seqlens_q.to(q.device)
        cu_key = key_lengths if cu_seqlens_kv is None else cu_seqlens_kv.to(q.device)
        maximum_query = sequence if max_seqlen_q is None else max_seqlen_q
        maximum_key = key_sequence if max_seqlen_kv is None else max_seqlen_kv
        architecture = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        checks = {
            "RADEON_CORESW_ATTENTION": os.environ.get("RADEON_CORESW_ATTENTION", "1") != "0",
            "architecture": architecture == "gfx1201",
            "dtype_bf16": query.dtype == torch.bfloat16,
            "equal_qkv_shape": query.shape == key.shape == value.shape,
            "head_dim_128": dimension == 128,
            "positive_shape": batch > 0 and heads > 0 and sequence > 0,
            "index_range": batch * ((sequence + 31) // 32) * 32 * heads * dimension <= 2**31 - 1,
            "contiguous_qkv": all(tensor.is_contiguous() for tensor in (query, key, value)),
            "max_sequence": maximum_query == sequence and maximum_key == key_sequence,
        }
        supported = all(checks.values())
        if supported:
            checks["regular_cu_seqlens"] = torch.equal(cu_query, query_lengths) and torch.equal(cu_key, key_lengths)
            supported = checks["regular_cu_seqlens"]
        _report_route(query, key, value, checks, block_idx)
        if supported:
            from aiter.ops.gfx1201.sage_attention import gfx1201_sage_attention

            output = gfx1201_sage_attention(query, key, value)
        else:
            from aiter import flash_attn_varlen_func

            output = flash_attn_varlen_func(
                query.reshape(-1, heads, dimension),
                key.reshape(-1, key.shape[2], key.shape[3]),
                value.reshape(-1, value.shape[2], value.shape[3]),
                cu_query, cu_key, maximum_query, maximum_key,
            )
        return output.reshape(batch * sequence, heads * dimension)
