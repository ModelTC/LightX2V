"""CUTLASS-SYCL CUTE Flash Attention for Intel XPU."""

import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.ops.attn.template import AttnWeightTemplate

try:
    from omni_xpu_kernel import cute
except ImportError as exc:
    cute = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _cute_sdp(q, k, v):
    if cute is None:
        raise RuntimeError(
            "intel_xpu_cute_attn requires omni-xpu-kernel"
        ) from _IMPORT_ERROR
    return cute.sdp(q, k, v)


@ATTN_WEIGHT_REGISTER("intel_xpu_cute_attn")
class IntelXpuCuteAttnWeight(AttnWeightTemplate):
    """CUTE self-attention adapter for LightX2V's varlen attention API."""

    def __init__(self):
        self.config = {}
        self._logged_backend = False

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        **kwargs,
    ):
        if q.ndim == 4:
            batch_size = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        else:
            batch_size = 1

        if not self._logged_backend:
            h3_route = (
                q.dtype == torch.bfloat16
                and q.shape[0] >= 38208
                and q.shape[1:] == (7, 128)
            )
            logger.info(
                "intel_xpu_cute_attn: backend={}, shape={}, dtype={}",
                "cute_h3_bf16.sdp" if h3_route else "cute_fmha.sdp",
                (1, *q.shape),
                q.dtype,
            )
            self._logged_backend = True

        if cu_seqlens_q is None or batch_size == 1:
            output = _cute_sdp(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            return output.squeeze(0).reshape(q.shape[0], -1)

        outputs = []
        for index in range(cu_seqlens_q.shape[0] - 1):
            q_start = cu_seqlens_q[index].item()
            q_end = cu_seqlens_q[index + 1].item()
            kv_start = cu_seqlens_kv[index].item()
            kv_end = cu_seqlens_kv[index + 1].item()
            output = _cute_sdp(
                q[q_start:q_end].unsqueeze(0),
                k[kv_start:kv_end].unsqueeze(0),
                v[kv_start:kv_end].unsqueeze(0),
            )
            outputs.append(output.squeeze(0).reshape(q_end - q_start, -1))
        return torch.cat(outputs, dim=0)
