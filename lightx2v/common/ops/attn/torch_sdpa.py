from contextlib import nullcontext

import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("torch_sdpa")
class TorchSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    @staticmethod
    def _cu_bounds(cu_seqlens, sequence_length, name):
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens_{name} must be one-dimensional, got shape {tuple(cu_seqlens.shape)}")
        bounds = tuple(int(value) for value in cu_seqlens.tolist())
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != sequence_length:
            raise ValueError(f"cu_seqlens_{name} must start at 0 and end at {sequence_length}, got {bounds}")
        if any(start > stop for start, stop in zip(bounds[:-1], bounds[1:])):
            raise ValueError(f"cu_seqlens_{name} must be nondecreasing, got {bounds}")
        return bounds

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
        softmax_scale = kwargs.get("softmax_scale")
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        def run_sdpa(query, key, value, mask):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
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
                # query/key/value are (B, H, S, D) here, so head count is dim 1.
                # GQA models such as neopp (32 q heads, 8 kv heads) need SDPA to
                # broadcast the kv groups.
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=drop_rate,
                    is_causal=causal,
                    scale=softmax_scale,
                    enable_gqa=query.shape[1] != key.shape[1],
                )
            return output.transpose(1, 2)

        packed_varlen = cu_seqlens_q is not None or cu_seqlens_kv is not None
        if not packed_varlen:
            if q.ndim == 3:
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            x = run_sdpa(q, k, v, attn_mask)
            b, s, a, d = x.shape
            return x.reshape(b, s, a * d).squeeze(0)

        if cu_seqlens_q is None or cu_seqlens_kv is None:
            raise ValueError("cu_seqlens_q and cu_seqlens_kv must either both be set or both be None")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Packed varlen Torch SDPA expects unbatched q/k/v tensors shaped [tokens, heads, dim]")
        q_bounds = self._cu_bounds(cu_seqlens_q, q.shape[0], "q")
        kv_bounds = self._cu_bounds(cu_seqlens_kv, k.shape[0], "kv")
        if len(q_bounds) != len(kv_bounds):
            raise ValueError(f"Packed q and kv must contain the same number of sequences, got {q_bounds} and {kv_bounds}")
        if v.shape[0] != k.shape[0]:
            raise ValueError(f"Packed k and v sequence lengths must match, got {k.shape[0]} and {v.shape[0]}")

        output = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
        for q_start, q_stop, kv_start, kv_stop in zip(q_bounds[:-1], q_bounds[1:], kv_bounds[:-1], kv_bounds[1:]):
            if q_start == q_stop:
                continue
            segment_mask = None if attn_mask is None else attn_mask[..., q_start:q_stop, kv_start:kv_stop]
            segment = run_sdpa(
                q[q_start:q_stop].unsqueeze(0),
                k[kv_start:kv_stop].unsqueeze(0),
                v[kv_start:kv_stop].unsqueeze(0),
                segment_mask,
            )
            output[q_start:q_stop].copy_(segment[0])
        return output.flatten(1)
