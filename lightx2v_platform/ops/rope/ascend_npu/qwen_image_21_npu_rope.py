import torch

from lightx2v_platform.ops.rope.ascend_npu.npu_rope import NpuRope
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _is_npu_tensor(tensor):
    return tensor.device.type == "npu"


@PLATFORM_ROPE_REGISTER("qwen_image_21_npu_rope")
class QwenImage21NpuRope(NpuRope):
    """Qwen-Image-2.1 interleaved RoPE backed by ``npu_rotary_mul``.

    Qwen-Image-2.1 computes RoPE in FP32 and rounds the result back to the
    projection dtype.  The expanded cosine/sine tables are prepared once per
    request and sliced for condition prefill and target denoising.
    """

    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if compute_dtype != torch.float32:
            raise ValueError("QwenImage21NpuRope requires FP32 RoPE computation.")

    def prepare_freqs(self, freqs, rotary_dim=None):
        if rotary_dim is None:
            raise ValueError("rotary_dim is required for Qwen-Image-2.1 RoPE frequencies.")
        cos, sin = self._normalize_cos_sin(freqs, rotary_dim)
        return cos.float().contiguous(), sin.float().contiguous()

    @staticmethod
    def _rotate_interleaved(x):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _fallback(self, x, cos, sin):
        input_dtype = x.dtype
        x = x.float()
        cos = cos.float().unsqueeze(-2)
        sin = sin.float().unsqueeze(-2)
        return (x * cos + self._rotate_interleaved(x) * sin).to(input_dtype)

    def apply(self, xq: torch.Tensor, xk: torch.Tensor, freqs, positions=None, **kwargs):
        if xq.ndim != 3 or xk.ndim != 3:
            raise ValueError(f"QwenImage21NpuRope expects [L, H, D] tensors, got q={xq.shape}, k={xk.shape}.")
        if xq.shape != xk.shape:
            raise ValueError(f"QwenImage21NpuRope expects matching Q/K shapes, got q={xq.shape}, k={xk.shape}.")

        seq_len, _, head_dim = xq.shape
        cos, sin = self._normalize_cos_sin(freqs, head_dim, positions=positions)
        if cos.shape[0] != seq_len:
            raise ValueError(f"QwenImage21NpuRope expects one frequency row per token, got tokens={seq_len}, freqs={cos.shape[0]}.")

        if (
            torch_npu is None
            or not hasattr(torch_npu, "npu_rotary_mul")
            or not (_is_npu_tensor(xq) and _is_npu_tensor(xk))
        ):
            return self._fallback(xq, cos, sin), self._fallback(xk, cos, sin)

        # The Ascend operator consumes SBND activations and S11D tables.  Keep
        # both inputs in FP32 to match the original Qwen-Image-2.1 arithmetic.
        cos = cos.float().contiguous().unsqueeze(1).unsqueeze(1)
        sin = sin.float().contiguous().unsqueeze(1).unsqueeze(1)

        def apply_single(x):
            input_dtype = x.dtype
            x = x.float().contiguous().unsqueeze(1)
            output = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")
            return output.squeeze(1).to(input_dtype)

        return apply_single(xq), apply_single(xk)
