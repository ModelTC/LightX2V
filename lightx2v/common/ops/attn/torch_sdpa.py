from contextlib import nullcontext

import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


def _use_h3_mps_query_chunks(q, k, v, chunk_size, scope, attn_mask, causal, drop_rate):
    return (
        scope == "minimax_h3_dit"
        and isinstance(chunk_size, int)
        and chunk_size > 0
        and q.device.type == k.device.type == v.device.type == "mps"
        and q.ndim == 4
        and q.shape == k.shape == v.shape
        and q.shape[0] == 1
        and q.shape[1] == 56
        and q.shape[-1] == 128
        and q.shape[2] > 0
        and attn_mask is None
        and not causal
        and drop_rate == 0
    )


def _query_chunked_sdpa(q, k, v, chunk_size):
    # Each query still attends to every key/value. Only the query workspace is
    # bounded; there is no context truncation or change to the softmax domain.
    return torch.cat(
        [F.scaled_dot_product_attention(q[:, :, start : start + chunk_size, :], k, v, attn_mask=None, dropout_p=0.0, is_causal=False) for start in range(0, q.shape[2], chunk_size)],
        dim=2,
    )


@ATTN_WEIGHT_REGISTER("torch_sdpa")
class TorchSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        # Hunyuan3D upstream Attention uses SDPA flash kernel (see hy3dshape hunyuandit.py).
        # Matching this context is required for bit-identical attention vs the reference.
        sdpa_ctx = nullcontext()
        if kwargs.get("model_cls") == "hunyuan3d":
            sdpa_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True,
            )
        with sdpa_ctx:
            chunk_size = kwargs.get("mps_sdpa_query_chunk_size", 0)
            if _use_h3_mps_query_chunks(q, k, v, chunk_size, kwargs.get("attention_scope"), attn_mask, causal, drop_rate):
                x = _query_chunked_sdpa(q, k, v, chunk_size)
            else:
                # q/k/v are (B, H, S, D) here, so head count is dim 1. GQA models such as
                # neopp (32 q heads, 8 kv heads) need SDPA to broadcast the kv groups.
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=drop_rate,
                    is_causal=causal,
                    enable_gqa=q.shape[1] != k.shape[1],
                )
        x = x.transpose(1, 2)
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out.squeeze(0)
