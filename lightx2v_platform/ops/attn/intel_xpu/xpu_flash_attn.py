"""
Intel XPU Flash Attention operator for LightX2V.

Uses sycl_kernels.sdp — a hand-written ESIMD/SYCL flash-attention
kernel for Intel Arc / Meteor Lake / Panther Lake iGPUs.

Layout convention (WAN varlen format):
  q / k / v : [S, num_heads, head_dim]   (S = total tokens across batch)
  cu_seqlens : [B+1]  int32  cumulative sequence lengths
  output     : [S, num_heads * head_dim]
"""

import warnings

import torch
import torch.nn.functional as F

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER


try:
    import sycl_kernels as _sycl_mod
    _sdp_fn = _sycl_mod.sdp
except ImportError:
    warnings.warn(
        "\n"
        "[intel_xpu_flash_attn] sycl_kernels not found — falling back to torch SDPA.\n"
        "  For best performance on Intel Arc GPU, build and install the ESIMD kernel:\n"
        "    cd lightx2v_kernel_xpu\n"
        "    conda activate lightx2v_kernel\n"
        "    call build_la.bat\n"
        "    pip install dist\\sycl_kernels-0.0.1-cp311-abi3-win_amd64.whl "
        "--force-reinstall --no-deps\n",
        stacklevel=2,
    )
    _sycl_mod = None
    _sdp_fn = None


def _sdp(q4d, k4d, v4d):
    """
    Unified SDP dispatch.  q4d/k4d/v4d are [1, L, H, D] fp16 or bf16 on XPU.

    - sycl_kernels available  → hand-written ESIMD Flash Attention
    - fallback                → torch scaled_dot_product_attention
                                (requires layout permute: [B,L,H,D] ↔ [B,H,L,D])
    """
    if _sdp_fn is not None:
        return _sdp_fn(q4d, k4d, v4d)

    # torch SDPA expects [B, H, L, D]
    q_t = q4d.permute(0, 2, 1, 3).contiguous()
    k_t = k4d.permute(0, 2, 1, 3).contiguous()
    v_t = v4d.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.permute(0, 2, 1, 3)  # → [1, L_q, H, D]




@ATTN_WEIGHT_REGISTER("intel_xpu_flash_attn")
class IntelXpuFlashAttnWeight(AttnWeightTemplate):
    """
    Flash Attention for Intel XPU.

    Registered as "intel_xpu_flash_attn".  Select it in a config JSON via:
        "self_attn_1_type":  "intel_xpu_flash_attn"
        "cross_attn_1_type": "intel_xpu_flash_attn"
        "cross_attn_2_type": "intel_xpu_flash_attn"

    Behaviour:
      - Single sequence (batch_size == 1, or cu_seqlens is None):
          One kernel call, no loop overhead.
      - Multi-sequence varlen (cu_seqlens provided, batch_size > 1):
          Splits by cu_seqlens, runs one kernel call per sequence, cats results.
          Correct for cross-attention where Q and KV can have different lengths.

    Kernel selection (automatic):
      - sycl_kernels installed → ESIMD Flash Attention (PTL-H, doubleGRF)
      - sycl_kernels not found → torch.nn.functional.scaled_dot_product_attention
    """

    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):

        # ── normalise 4-D input [B, S, H, D] → [B*S, H, D] ──────────────────
        if q.ndim == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        else:
            bs = 1

        # q/k/v are now [S, H, D]
        total_q = q.shape[0]

        if cu_seqlens_q is None or bs == 1:
            # ── fast single-sequence path ─────────────────────────────────────
            # kernel expects [1, L, H, D]; returns [1, L_q, H, D]
            x = _sdp(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

            # [1, L_q, H, D] → [L_q, H*D]
            return x.squeeze(0).reshape(total_q, -1)

        # ── varlen path: one SDPA call per sequence in the batch ──────────────
        batch_size = cu_seqlens_q.shape[0] - 1
        outputs = []

        for i in range(batch_size):
            qs = cu_seqlens_q[i].item()
            qe = cu_seqlens_q[i + 1].item()
            ks = cu_seqlens_kv[i].item()
            ke = cu_seqlens_kv[i + 1].item()

            x_i = _sdp(
                q[qs:qe].unsqueeze(0),
                k[ks:ke].unsqueeze(0),
                v[ks:ke].unsqueeze(0),
            )

            # [1, L_q, H, D] → [Sq, H*D]
            outputs.append(x_i.squeeze(0).reshape(qe - qs, -1))

        return torch.cat(outputs, dim=0)  # [total_S, H*D]
