"""Fused partial rotate-half RoPE for MiniMax-H3 on Ascend.

MiniMax-H3 rotates only the leading ``rotary_dim`` (96 of 128) channels of each
head with a rotate-half (split-half) pairing; the remaining channels pass
through. The rotary part is fused into a single NPU kernel through the
MindIE-SD ``rotary_position_embedding`` op (``npu_rotary_mul`` mode ``half``),
and the pass-through channels are re-catenated outside.

Input/output guards for the fused op live in MindIE-SD
(``mindiesd.layers.rope.check_input_params``). Fallback: when mindiesd is not
installed, delegate to ``TorchRealRope`` (split_half), the original decomposed
flow.
"""

import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER

try:
    from mindiesd.layers import rotary_position_embedding as _mindiesd_rope
except ImportError:
    _mindiesd_rope = None


@PLATFORM_ROPE_REGISTER("minimax_h3_npu_rope")
class MiniMaxH3NpuRope(RopeTemplate):
    """Partial split-half RoPE fused via the MindIE-SD rotary_position_embedding op."""

    def __init__(self, layout="split_half", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "split_half":
            raise ValueError("MiniMaxH3NpuRope only supports split_half layout")

    def _fallback(self):
        from lightx2v.common.ops.rope import TorchRealRope

        return TorchRealRope(layout=self.layout, compute_dtype=self.compute_dtype)

    def apply(self, xq, xk, freqs, rotary_dim=None, **kwargs):
        if _mindiesd_rope is None:
            return self._fallback().apply(xq, xk, freqs, rotary_dim=rotary_dim, **kwargs)
        return (
            self._apply_single(xq, freqs, rotary_dim),
            self._apply_single(xk, freqs, rotary_dim),
        )

    def apply_single(self, x, freqs, rotary_dim=None, **kwargs):
        if _mindiesd_rope is None:
            return self._fallback().apply_single(x, freqs, rotary_dim=rotary_dim, **kwargs)
        return self._apply_single(x, freqs, rotary_dim)

    def _apply_single(self, x, freqs, rotary_dim):
        # x: [L, H, D]; rotate the leading rotary_dim channels, pass the rest.
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        x_rot = x[..., :rotary_dim].contiguous()
        x_pass = x[..., rotary_dim:]
        cos = cos.to(x.dtype).contiguous()
        sin = sin.to(x.dtype).contiguous()
        # mindiesd rotary_position_embedding takes x in [B,N,S,D]/[B,S,N,D]/
        # [S,B,N,D] and 4-D cos/sin ([S,1,1,D] S11D is the SBND pairing);
        # its 2-D [S,D] path assumes [B,S,N,D], so pass S11D explicitly.
        rotated = _mindiesd_rope(
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
