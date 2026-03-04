"""
Intel XPU Flash Attention operator for LightX2V.

Uses sycl_kernels.sdp_bf16io — a hand-written ESIMD/SYCL flash-attention
kernel for Intel Arc / Meteor Lake / Panther Lake iGPUs.

Layout convention (WAN varlen format):
  q / k / v : [S, num_heads, head_dim]   (S = total tokens across batch)
  cu_seqlens : [B+1]  int32  cumulative sequence lengths
  output     : [S, num_heads * head_dim]
"""

import torch

import sycl_kernels

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER


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
            # kernel expects [1, L, H, D]; returns [1, L_q, H, D] bf16
            x = sycl_kernels.sdp(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

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

            x_i = sycl_kernels.sdp(
                    q[qs:qe].unsqueeze(0),
                    k[ks:ke].unsqueeze(0),
                    v[ks:ke].unsqueeze(0),
                )

            # [1, L_q, H, D] → [Sq, H*D]
            outputs.append(x_i.squeeze(0).reshape(qe - qs, -1))


        return torch.cat(outputs, dim=0)  # [total_S, H*D]
