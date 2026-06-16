import torch

from lightx2v_platform.ops.rope.rope_template import GET_DTYPE, RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


def GET_SENSITIVE_DTYPE():
    import os
    DTYPE_MAP = {
        "BF16": torch.bfloat16,
        "FP16": torch.float16,
        "FP32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
    }
    flag = os.getenv("SENSITIVE_LAYER_DTYPE", "None")
    if flag == "None":
        return GET_DTYPE()
    return DTYPE_MAP[flag]


@PLATFORM_ROPE_REGISTER("npu_rope")
class NpuRope(RopeTemplate):
    def __init__(self):
        super().__init__()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def _apply_rope_fp32(self, xq, xk, cos_sin_cache):
        n = xq.size(1)
        seq_len = cos_sin_cache.size(0)
        xq_fp32 = torch.view_as_complex(
            xq[:seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        )
        xk_fp32 = torch.view_as_complex(
            xk[:seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        )
        xq_rot = torch.view_as_real(xq_fp32 * cos_sin_cache).flatten(2)
        xk_rot = torch.view_as_real(xk_fp32 * cos_sin_cache).flatten(2)
        if xq.size(0) > seq_len:
            xq_rot = torch.cat([xq_rot, xq[seq_len:]], dim=0)
            xk_rot = torch.cat([xk_rot, xk[seq_len:]], dim=0)
        return xq_rot.to(self.infer_dtype), xk_rot.to(self.infer_dtype)

    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor):
        s, n, d = xq.shape
        seq_len = cos_sin_cache.size(0)
        cos = cos_sin_cache.real
        sin = cos_sin_cache.imag
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        xq_part = xq[:seq_len]
        xk_part = xk[:seq_len]

        if torch_npu is not None and hasattr(torch_npu, "npu_rotary_mul"):
            if self.sensitive_layer_dtype != self.infer_dtype:
                xq_part = xq_part.float()
                xk_part = xk_part.float()
                cos = cos.float() if cos.dtype != torch.float32 else cos
                sin = sin.float() if sin.dtype != torch.float32 else sin
            if not xq_part.is_contiguous():
                xq_part = xq_part.contiguous()
            if not xk_part.is_contiguous():
                xk_part = xk_part.contiguous()
            xq_rotated = torch_npu.npu_rotary_mul(xq_part, cos, sin, "interleave")
            xk_rotated = torch_npu.npu_rotary_mul(xk_part, cos, sin, "interleave")
            if s > seq_len:
                xq = torch.cat([xq_rotated.to(self.infer_dtype), xq[seq_len:]], dim=0)
                xk = torch.cat([xk_rotated.to(self.infer_dtype), xk[seq_len:]], dim=0)
            else:
                xq = xq_rotated.to(self.infer_dtype)
                xk = xk_rotated.to(self.infer_dtype)
            return xq, xk

        return self._apply_rope_fp32(xq, xk, cos_sin_cache)
