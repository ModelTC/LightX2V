import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


@PLATFORM_ROPE_REGISTER("mlu_rope")
class MluRope(RopeTemplate):
    """MLU fused RoPE with an optional FP32 arithmetic contract.

    MiniMax-H3 stores a full-width split-half cosine/sine cache and rotates
    only the leading cache-width channels. ``torch_mlu_ops.apply_rotary``
    implements that layout directly. When ``compute_dtype`` differs from the
    input dtype, the private converted tensor is used as the in-place work
    buffer, which avoids the clone/cat intermediates of the generic path.
    """

    def __init__(self, layout="split_half", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if tmo is None or not hasattr(tmo, "apply_rotary"):
            raise RuntimeError("mlu_rope requires torch_mlu_ops with apply_rotary support.")

    def _normalize_freqs(self, freqs, rotary_dim, device):
        if not isinstance(freqs, tuple) or len(freqs) != 2:
            raise TypeError("mlu_rope expects a (cos, sin) tuple.")
        cos, sin = freqs
        if not torch.is_tensor(cos) or not torch.is_tensor(sin) or cos.shape != sin.shape:
            raise ValueError("mlu_rope cosine and sine caches must be matching tensors.")
        if cos.ndim != 2:
            raise ValueError(f"mlu_rope expects [sequence, rotary_dim] caches, got {tuple(cos.shape)}.")
        rotary_dim = cos.shape[-1] if rotary_dim is None else int(rotary_dim)
        if cos.shape[-1] != rotary_dim:
            raise ValueError(f"mlu_rope cache width {cos.shape[-1]} does not match rotary_dim {rotary_dim}.")
        return (
            cos.to(device=device, dtype=self.compute_dtype).contiguous(),
            sin.to(device=device, dtype=self.compute_dtype).contiguous(),
        )

    def apply_single(self, x, freqs, rotary_dim=None, **kwargs):
        if x.ndim != 3:
            raise ValueError(f"mlu_rope expects [sequence, heads, head_dim], got {tuple(x.shape)}.")
        cos, sin = self._normalize_freqs(freqs, rotary_dim, x.device)
        if cos.shape[0] != x.shape[0]:
            raise ValueError(f"mlu_rope cache sequence {cos.shape[0]} does not match input {x.shape[0]}.")
        if cos.shape[-1] > x.shape[-1] or cos.shape[-1] % 2:
            raise ValueError("mlu_rope rotary dimension must be even and no larger than head_dim.")

        # The padded [1, S, H, D] form avoids constructing cu_seqlens for every
        # transformer block. The cache rows are already aligned to token rows,
        # so position_ids=None correctly selects positions 0..S-1.
        if x.dtype == self.compute_dtype:
            output = torch.empty_like(x)
            work = x
        else:
            output = x.to(self.compute_dtype)
            work = output
        tmo.apply_rotary(
            work.unsqueeze(0),
            sin,
            cos,
            position_ids=None,
            cu_seqlens=None,
            interleaved=self.layout == "interleaved",
            discrete=False,
            dynamic_ntk=False,
            max_seqlen=x.shape[0],
            output=output.unsqueeze(0),
        )
        return output if output.dtype == x.dtype else output.to(x.dtype)

    def apply(self, q, k, freqs, rotary_dim=None, **kwargs):
        return (
            self.apply_single(q, freqs, rotary_dim=rotary_dim, **kwargs),
            self.apply_single(k, freqs, rotary_dim=rotary_dim, **kwargs),
        )
