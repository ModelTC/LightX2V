from functools import lru_cache

import torch

from lightx2v_platform.ops.rope.rope_template import GET_DTYPE, RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


@lru_cache(maxsize=None)
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
    if flag not in DTYPE_MAP:
        raise ValueError(f"Unsupported SENSITIVE_LAYER_DTYPE: {flag}. Expected one of {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[flag]


@PLATFORM_ROPE_REGISTER("npu_rope")
class NpuRope(RopeTemplate):
    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "interleaved":
            raise ValueError("NpuRope only supports interleaved layout.")
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    @staticmethod
    def _normalize_cos_sin(freqs, rotary_dim, positions=None):
        if torch.is_tensor(freqs) and torch.is_complex(freqs):
            cos = freqs.real.repeat_interleave(2, dim=-1)
            sin = freqs.imag.repeat_interleave(2, dim=-1)
        elif torch.is_tensor(freqs):
            if freqs.shape[-1] != rotary_dim:
                raise ValueError(f"A real RoPE cache must concatenate half-width cosine and sine values and have width {rotary_dim}, got {freqs.shape[-1]}.")
            cos, sin = freqs.chunk(2, dim=-1)
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        elif isinstance(freqs, tuple) and len(freqs) >= 2:
            cos, sin = freqs[:2]
            if cos.shape != sin.shape:
                raise ValueError(f"RoPE cosine/sine shapes must match, got {cos.shape} and {sin.shape}.")
            if cos.shape[-1] == rotary_dim // 2:
                cos = cos.repeat_interleave(2, dim=-1)
                sin = sin.repeat_interleave(2, dim=-1)
            elif cos.shape[-1] != rotary_dim:
                raise ValueError(f"RoPE frequency width must be {rotary_dim // 2} or {rotary_dim}, got {cos.shape[-1]}.")
        else:
            raise TypeError(f"NpuRope expects a complex tensor, a concatenated real cosine/sine cache, or a (cos, sin) tuple, got {type(freqs)!r}.")

        # Keep the prepared cache compact. The fused NPU call below adds the
        # singleton batch/head axes required by its SBND/S11D contract.
        if cos.ndim == 3 and cos.shape[-2] == 1:
            cos = cos.squeeze(-2)
            sin = sin.squeeze(-2)
        if cos.ndim != 2:
            raise ValueError(f"NpuRope frequency tensors must have shape [L, D] or [L, 1, D], got {cos.shape}.")
        if positions is not None:
            positions = positions.reshape(-1).to(device=cos.device, dtype=torch.long)
            cos = cos.index_select(0, positions)
            sin = sin.index_select(0, positions)
        return cos, sin

    def prepare_freqs(self, freqs, rotary_dim=None):
        if isinstance(freqs, tuple) or (torch.is_tensor(freqs) and not torch.is_complex(freqs)):
            if rotary_dim is None:
                raise ValueError("rotary_dim is required for real RoPE frequencies.")
            cos, sin = self._normalize_cos_sin(freqs, rotary_dim)
            return cos.contiguous(), sin.contiguous()
        return freqs

    def _apply_rope_fp32(self, xq, xk, cos, sin):
        seq_len = cos.size(0)

        def rotate_interleaved(x):
            pairs = x.reshape(*x.shape[:-1], -1, 2)
            first, second = pairs.unbind(dim=-1)
            return torch.stack((-second, first), dim=-1).flatten(-2)

        xq_part = xq[:seq_len].to(torch.float32)
        xk_part = xk[:seq_len].to(torch.float32)
        cos = cos.to(torch.float32).unsqueeze(-2)
        sin = sin.to(torch.float32).unsqueeze(-2)
        xq_rot = xq_part * cos + rotate_interleaved(xq_part) * sin
        xk_rot = xk_part * cos + rotate_interleaved(xk_part) * sin
        if xq.size(0) > seq_len:
            xq_rot = torch.cat([xq_rot, xq[seq_len:].to(torch.float32)], dim=0)
            xk_rot = torch.cat([xk_rot, xk[seq_len:].to(torch.float32)], dim=0)
        return xq_rot.to(self.infer_dtype), xk_rot.to(self.infer_dtype)

    def apply(self, xq: torch.Tensor, xk: torch.Tensor, freqs, positions=None, **kwargs):
        if xq.ndim != 3 or xk.ndim != 3:
            raise ValueError(f"NpuRope expects [L, H, D] tensors, got q={xq.shape}, k={xk.shape}.")
        s, _, d = xq.shape
        cos, sin = self._normalize_cos_sin(freqs, d, positions=positions)
        seq_len = cos.size(0)
        if seq_len > s:
            raise ValueError(f"RoPE sequence length {seq_len} exceeds query length {s}.")
        xq_part = xq[:seq_len]
        xk_part = xk[:seq_len]

        if torch_npu is not None and hasattr(torch_npu, "npu_rotary_mul"):
            if self.sensitive_layer_dtype != self.infer_dtype:
                xq_part = xq_part.to(self.sensitive_layer_dtype)
                xk_part = xk_part.to(self.sensitive_layer_dtype)
                cos = cos.to(self.sensitive_layer_dtype)
                sin = sin.to(self.sensitive_layer_dtype)
            if not xq_part.is_contiguous():
                xq_part = xq_part.contiguous()
            if not xk_part.is_contiguous():
                xk_part = xk_part.contiguous()
            if not cos.is_contiguous():
                cos = cos.contiguous()
            if not sin.is_contiguous():
                sin = sin.contiguous()
            # aclnnRotaryPositionEmbedding requires q/k in SBND layout and
            # cos/sin in S11D layout.
            xq_sbnd = xq_part.unsqueeze(1)
            xk_sbnd = xk_part.unsqueeze(1)
            cos_s11d = cos.unsqueeze(1).unsqueeze(1)
            sin_s11d = sin.unsqueeze(1).unsqueeze(1)
            xq_rotated = torch_npu.npu_rotary_mul(xq_sbnd, cos_s11d, sin_s11d, "interleave").squeeze(1)
            xk_rotated = torch_npu.npu_rotary_mul(xk_sbnd, cos_s11d, sin_s11d, "interleave").squeeze(1)
            if s > seq_len:
                xq = torch.cat([xq_rotated.to(self.infer_dtype), xq[seq_len:]], dim=0)
                xk = torch.cat([xk_rotated.to(self.infer_dtype), xk[seq_len:]], dim=0)
            else:
                xq = xq_rotated.to(self.infer_dtype)
                xk = xk_rotated.to(self.infer_dtype)
            return xq, xk

        return self._apply_rope_fp32(xq, xk, cos, sin)
