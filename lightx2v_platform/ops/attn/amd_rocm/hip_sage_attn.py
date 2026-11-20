"""SageAttention HIP kernels for AMD gfx1201 (RDNA4), with an INT4/INT8 per-layer split for MiniMax-H3.

Select with ``attn_type=hip_sage_amd_rocm``. Every call uses the INT8 Q.K^T kernel by default. When the
model enables ``hip_sage_setting.mixed_precision`` (only MiniMax-H3 is validated), DiT blocks before
``int8_first_block`` (default 45 of 50) use the faster INT4 Q.K^T kernel; the last blocks keep INT8 because
their Q/K outliers (|q|, |k| up to ~10x the other blocks) break INT4 (per-block output cosine 0.78-0.85).
The kernels are built on first use with hipcc (see ``hip_sage/jit.py``); only gfx1201 is supported.
"""

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

from . import hip_sage

MIXED_PRECISION_MODELS = ("minimax_h3",)
DEFAULT_INT8_FIRST_BLOCK = 45


@PLATFORM_ATTN_WEIGHT_REGISTER("hip_sage_amd_rocm")
class AmdHipSageAttnWeight(AttnWeightTemplate):
    """Dense non-causal SageAttention (smoothed INT4/INT8 QK, FP8 PV) for gfx1201."""

    def __init__(self):
        self.config = {}
        if not hip_sage.is_supported_device():
            raise RuntimeError(f"hip_sage_amd_rocm requires an AMD {hip_sage.ARCH} GPU with a ROCm build of PyTorch.")
        self.int8_first_block = None

    def set_config(self, config=None):
        """``config``: ``{"model": str, "mixed_precision": bool, "int8_first_block": int}``."""
        super().set_config(config)
        if not self.config.get("mixed_precision", False):
            self.int8_first_block = None
            return
        model = self.config.get("model")
        if model not in MIXED_PRECISION_MODELS:
            raise ValueError(f"hip_sage_amd_rocm mixed INT4/INT8 precision is only validated for {MIXED_PRECISION_MODELS}, got model={model!r}.")
        self.int8_first_block = int(self.config.get("int8_first_block", DEFAULT_INT8_FIRST_BLOCK))

    def bits_for(self, block_idx):
        if self.int8_first_block is not None and block_idx is not None and block_idx < self.int8_first_block:
            return 4
        return 8

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        block_idx=None,
        softmax_scale=None,
        **kwargs,
    ):
        if kwargs.get("causal", False):
            raise NotImplementedError("hip_sage_amd_rocm implements non-causal attention only.")
        for cu in (cu_seqlens_q, cu_seqlens_kv):
            if cu is not None and cu.numel() > 2:
                raise NotImplementedError("hip_sage_amd_rocm supports a single sequence (no varlen batches).")
        if q.ndim == 4:
            if q.shape[0] != 1:
                raise NotImplementedError("hip_sage_amd_rocm supports batch size 1.")
            q, k, v = q[0], k[0], v[0]
        if not hip_sage.supports(q, k, v):
            raise ValueError(f"hip_sage_amd_rocm expects bf16 [N, H<=64, 128] Q/K/V of equal shape, got q={tuple(q.shape)} {q.dtype}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
        out = hip_sage.attention(q, k, v, softmax_scale, self.bits_for(block_idx))
        return out.reshape(q.shape[0], -1)
