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
    def _cu_bounds(cu_seqlens, seq_len, name):
        bounds = cu_seqlens.tolist()
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != seq_len or any(start > stop for start, stop in zip(bounds, bounds[1:])):
            raise ValueError(f"Invalid cu_seqlens_{name}")
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
            # Match Hunyuan3D's flash-only upstream dispatch.
            sdpa_ctx = nullcontext()
            if kwargs.get("model_cls") == "hunyuan3d":
                sdpa_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True,
                )
            with sdpa_ctx:
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=drop_rate,
                    is_causal=causal,
                    scale=softmax_scale,
                    enable_gqa=query.shape[1] != key.shape[1],
                ).transpose(1, 2)

        if (cu_seqlens_q is None) != (cu_seqlens_kv is None):
            raise ValueError("cu_seqlens_q and cu_seqlens_kv must either both be set or both be None")

        use_packed = q.ndim == 3 and cu_seqlens_q is not None
        if use_packed:
            if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
                raise ValueError("cu_seqlens_q and cu_seqlens_kv must describe the same number of sequences")
            if cu_seqlens_q.numel() < 2:
                raise ValueError("cu_seqlens must contain at least two boundaries")
            use_packed = cu_seqlens_q.numel() > 2

        # A 4D tensor already has an explicit batch dimension. A two-entry
        # cu_seqlens tensor describes one sequence, so it uses the same dense path.
        if not use_packed:
            if q.ndim == 3:
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            return run_sdpa(q, k, v, attn_mask).flatten(2).squeeze(0)

        if any(x.ndim != 3 for x in (q, k, v)):
            raise ValueError("Packed Torch SDPA expects 3D q/k/v")

        q_bounds = self._cu_bounds(cu_seqlens_q, q.shape[0], "q")
        kv_bounds = self._cu_bounds(cu_seqlens_kv, k.shape[0], "kv")
        if len(q_bounds) != len(kv_bounds):
            raise ValueError("Packed q and kv must contain the same number of sequences")
        if v.shape[0] != k.shape[0]:
            raise ValueError("Packed k and v sequence lengths must match")

        # PyTorch SDPA has no varlen API, so process each cu_seqlens segment independently.
        output = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
        for q_start, q_stop, kv_start, kv_stop in zip(q_bounds, q_bounds[1:], kv_bounds, kv_bounds[1:]):
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
