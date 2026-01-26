import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER


@PLATFORM_ROPE_REGISTER("gcu_wan_rope")
class GcuWanRope(RopeTemplate):
    def __init__(self):
        super().__init__()

    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor):
        """
        Apply WAN RoPE using PyTorch operations, optimized for GCU.

        This implementation converts tensors to float32 for computation
        to avoid dtype mismatch issues on GCU hardware.

        Args:
            xq: Query tensor
            xk: Key tensor
            cos_sin_cache: Cosine and sine cache for rotary embedding

        Returns:
            Tuple of (xq, xk) with rotary embedding applied
        """
        n = xq.size(1)
        seq_len = cos_sin_cache.size(0)

        xq_seq = xq[:seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        xk_seq = xk[:seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        # Apply rotary embedding using real arithmetic
        cos = cos_sin_cache.real
        sin = cos_sin_cache.imag
        xq_real = xq_seq[..., 0]
        xq_imag = xq_seq[..., 1]
        xk_real = xk_seq[..., 0]
        xk_imag = xk_seq[..., 1]
        xq_rotated = torch.stack([xq_real * cos - xq_imag * sin, xq_real * sin + xq_imag * cos], dim=-1).flatten(2)
        xk_rotated = torch.stack([xk_real * cos - xk_imag * sin, xk_real * sin + xk_imag * cos], dim=-1).flatten(2)

        # Concatenate rotated part with remaining part if any
        if xq.size(0) > seq_len:
            xq = torch.cat([xq_rotated, xq[seq_len:]], dim=0)
            xk = torch.cat([xk_rotated, xk[seq_len:]], dim=0)
        else:
            xq = xq_rotated
            xk = xk_rotated

        return xq.to(self.infer_dtype), xk.to(self.infer_dtype)
