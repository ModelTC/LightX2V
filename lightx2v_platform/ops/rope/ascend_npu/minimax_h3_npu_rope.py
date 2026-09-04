"""Fused partial rotate-half RoPE for MiniMax-H3 on Ascend.

MiniMax-H3 rotates only the leading ``rotary_dim`` (96 of 128) channels of each
head with a rotate-half (split-half) pairing; the remaining channels pass
through. The rotary part is fused into a single NPU kernel through the
MindIE-SD ``rotary_position_embedding`` op (``npu_rotary_mul`` mode ``half``),
and the pass-through channels are re-catenated outside.

MindIE-SD is imported only when this implementation is selected. When it is
not installed, the implementation delegates to ``TorchRealRope``
(``split_half``), the original decomposed flow.
"""

from functools import lru_cache

import torch
from loguru import logger

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER


@lru_cache(maxsize=1)
def _load_mindiesd_rope():
    try:
        from mindiesd.layers import rotary_position_embedding

        return rotary_position_embedding
    except ImportError:
        logger.warning("MindIE-SD is unavailable; MiniMax-H3 will use the torch RoPE fallback.")
        return None
    except Exception as exc:
        raise RuntimeError("MindIE-SD could not be loaded for the MiniMax-H3 Ascend RoPE backend.") from exc


@PLATFORM_ROPE_REGISTER("minimax_h3_npu_rope")
class MiniMaxH3NpuRope(RopeTemplate):
    """Partial split-half RoPE fused via the MindIE-SD rotary_position_embedding op."""

    def __init__(self, layout="split_half", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "split_half":
            raise ValueError("MiniMaxH3NpuRope only supports split_half layout")
        self._mindiesd_rope = _load_mindiesd_rope()
        self._fallback_rope = None

    def _fallback(self):
        if self._fallback_rope is None:
            from lightx2v.common.ops.rope import TorchRealRope

            self._fallback_rope = TorchRealRope(layout=self.layout, compute_dtype=self.compute_dtype)
        return self._fallback_rope

    @staticmethod
    def _validate_inputs(x, freqs, rotary_dim):
        if not torch.is_tensor(x) or x.ndim != 3:
            shape = getattr(x, "shape", None)
            raise ValueError(f"MiniMaxH3NpuRope expects x with shape [L, H, D], got {shape}.")
        if not isinstance(freqs, tuple) or len(freqs) != 2:
            raise TypeError("MiniMaxH3NpuRope expects freqs to be a (cos, sin) tuple.")
        cos, sin = freqs
        if not torch.is_tensor(cos) or not torch.is_tensor(sin):
            raise TypeError("MiniMaxH3NpuRope expects cos and sin to be tensors.")
        if cos.shape != sin.shape:
            raise ValueError(f"RoPE cosine/sine shapes must match, got {cos.shape} and {sin.shape}.")
        if cos.ndim != 2:
            raise ValueError(f"MiniMax-H3 RoPE frequencies must have shape [L, rotary_dim], got {cos.shape}.")
        if cos.shape[0] != x.shape[0]:
            raise ValueError(f"RoPE sequence length must match the input: x={x.shape[0]}, freqs={cos.shape[0]}.")
        if cos.device != x.device or sin.device != x.device:
            raise ValueError(f"RoPE inputs must share a device, got x={x.device}, cos={cos.device}, sin={sin.device}.")

        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(f"rotary_dim must be a positive even number, got {rotary_dim}.")
        if rotary_dim > x.shape[-1]:
            raise ValueError(f"rotary_dim={rotary_dim} exceeds the input head dimension {x.shape[-1]}.")
        if rotary_dim != cos.shape[-1]:
            raise ValueError(f"rotary_dim={rotary_dim} must match the frequency width {cos.shape[-1]}.")
        return cos, sin, rotary_dim

    def apply(self, xq, xk, freqs, rotary_dim=None, **kwargs):
        if self._mindiesd_rope is None:
            self._validate_inputs(xq, freqs, rotary_dim)
            self._validate_inputs(xk, freqs, rotary_dim)
            return self._fallback().apply(xq, xk, freqs, rotary_dim=rotary_dim, **kwargs)
        return (
            self._apply_single(xq, freqs, rotary_dim),
            self._apply_single(xk, freqs, rotary_dim),
        )

    def apply_single(self, x, freqs, rotary_dim=None, **kwargs):
        if self._mindiesd_rope is None:
            self._validate_inputs(x, freqs, rotary_dim)
            return self._fallback().apply_single(x, freqs, rotary_dim=rotary_dim, **kwargs)
        return self._apply_single(x, freqs, rotary_dim)

    def _apply_single(self, x, freqs, rotary_dim):
        # x: [L, H, D]; rotate the leading rotary_dim channels, pass the rest.
        cos, sin, rotary_dim = self._validate_inputs(x, freqs, rotary_dim)
        x_rot = x[..., :rotary_dim].contiguous()
        x_pass = x[..., rotary_dim:]
        # The fused MindIE kernel requires x/cos/sin to share a dtype. The
        # torch fallback continues to honor RopeTemplate.compute_dtype.
        cos = cos.to(x.dtype).contiguous()
        sin = sin.to(x.dtype).contiguous()
        # mindiesd rotary_position_embedding takes x in [B,N,S,D]/[B,S,N,D]/
        # [S,B,N,D] and 4-D cos/sin ([S,1,1,D] S11D is the SBND pairing);
        # its 2-D [S,D] path assumes [B,S,N,D], so pass S11D explicitly.
        rotated = self._mindiesd_rope(
            x_rot.unsqueeze(1),  # [L, 1, H, D] SBND
            cos.unsqueeze(1).unsqueeze(1),  # [L, 1, 1, D] S11D
            sin.unsqueeze(1).unsqueeze(1),
            rotated_mode="rotated_half",
            head_first=False,
            fused=True,
        ).squeeze(1)
        rotated = rotated.to(x.dtype)
        if x_pass.shape[-1]:
            return torch.cat((rotated, x_pass), dim=-1)
        return rotated
