import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER


@PLATFORM_ROPE_REGISTER("minimax_h3_xpu_rope")
class MiniMaxH3XpuRope(RopeTemplate):
    """MiniMax-H3 partial split-half RoPE backed by lightx2v_kernel_xpu."""

    def __init__(self, layout="split_half", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "split_half":
            raise ValueError("MiniMaxH3XpuRope only supports split_half layout")

    @staticmethod
    def _can_use_xpu_kernel(x, cos, sin, rotary_dim):
        if not (x.device.type == "xpu" and x.dtype == torch.bfloat16 and x.ndim == 3 and x.shape[-1] == 128 and rotary_dim == 96 and cos.dtype == torch.float32 and sin.dtype == torch.float32):
            return False
        try:
            import sycl_kernels

            return sycl_kernels.has_minimax_h3_rope()
        except (AttributeError, ImportError, OSError, RuntimeError):
            return False

    def _torch_apply_single(self, x, cos, sin, rotary_dim):
        if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
            raise ValueError(f"rotary_dim must be positive, even, and <= head_size; got rotary_dim={rotary_dim}, head_size={x.shape[-1]}")
        if cos.shape != sin.shape or cos.shape[-1] != rotary_dim:
            raise ValueError(f"cos and sin must have matching width {rotary_dim}, got {tuple(cos.shape)} and {tuple(sin.shape)}")

        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        first, second = x_rot.to(self.compute_dtype).chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        while cos.ndim < x_rot.ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        output = x_rot.to(self.compute_dtype) * cos.to(self.compute_dtype)
        output.add_(rotated * sin.to(self.compute_dtype))
        output = output.to(x.dtype)
        return torch.cat((output, x_pass), dim=-1) if x_pass.shape[-1] else output

    def apply_single(self, x, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else int(rotary_dim)
        if self._can_use_xpu_kernel(x, cos, sin, rotary_dim):
            import sycl_kernels

            return sycl_kernels.minimax_h3_rope_cached(x.contiguous(), cos.contiguous(), sin.contiguous())
        return self._torch_apply_single(x, cos, sin, rotary_dim)

    def apply(self, query, key, freqs, rotary_dim=None, **kwargs):
        return (
            self.apply_single(query, freqs, rotary_dim=rotary_dim, **kwargs),
            self.apply_single(key, freqs, rotary_dim=rotary_dim, **kwargs),
        )
