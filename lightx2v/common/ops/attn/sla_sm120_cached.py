"""SM120 sparse SageAttention with SLA routing and a persistent V layout."""

import torch

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .sage_sm90_fused import make_transposed_v
from .template import AttnWeightTemplate


@torch.library.custom_op("lightx2v::sla_sage2_sm120_cached", mutates_args=(), device_types="cuda")
def sla_sage2_sm120_cached(q: torch.Tensor, k: torch.Tensor, transposed_v: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    import spas_sage_attn._fused as fused
    import spas_sage_attn._qattn as qattn

    from .utils.sla_util import get_block_map
    from .utils.sparge_util import block_map_incremental_lut_triton, quantize_qk_blocks

    if q.ndim != 3 or k.ndim != 3 or q.shape[1:] != k.shape[1:] or q.shape[-1] not in (64, 128):
        raise ValueError("Cached SLA requires [tokens, heads, dim] Q/K with matching heads and head_dim 64 or 128")
    if not q.is_cuda or k.device != q.device or transposed_v.device != q.device or q.dtype != k.dtype or q.dtype != transposed_v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Cached SLA requires FP16 or BF16 tensors on the same CUDA device")
    expected_padded = (k.shape[0] + 127) // 128 * 128
    if transposed_v.shape != (1, q.shape[-1], q.shape[1], expected_padded) or not transposed_v.is_contiguous():
        raise ValueError("Cached SLA requires contiguous [1, dim, heads, padded_key_tokens] V storage")

    qh = q.unsqueeze(0).transpose(1, 2).contiguous()
    kh = k.unsqueeze(0).transpose(1, 2).contiguous()
    center = kh.mean(dim=-2, keepdim=True)
    mask, _, _ = get_block_map(qh, kh, topk_ratio, 128, 64, center=center)
    lut, valid = block_map_incremental_lut_triton(mask)
    qi, qs = quantize_qk_blocks(qh, None, 128)
    ki, ks = quantize_qk_blocks(kh, center, 64)
    # The native NHD API supports non-contiguous head/row strides. These views
    # retain the quantizer's exact bytes and avoid another Q/K layout copy.
    qi, ki = qi.transpose(1, 2), ki.transpose(1, 2)
    vf = torch.empty_like(transposed_v, dtype=torch.float8_e4m3fn)
    vs = q.new_empty((1, q.shape[1], q.shape[-1]), dtype=torch.float32)
    fused.scale_fuse_quant_cuda(transposed_v, vf, vs, k.shape[0], 2.25, 0)
    output = torch.empty_like(q.unsqueeze(0))
    threshold = q.new_full((q.shape[1],), 1e6, dtype=torch.float32)
    qattn.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(qi, ki, vf, output, lut, valid, threshold, qs, ks, vs, 0, False, 1, q.shape[-1] ** -0.5, 0)
    return output.reshape(q.shape[0], -1)


@sla_sage2_sm120_cached.register_fake
def _sla_sage2_sm120_cached_fake(q, k, transposed_v, topk_ratio):
    return q.new_empty((q.shape[0], q.shape[1] * q.shape[2]))


@ATTN_WEIGHT_REGISTER("sla_sage2_sm120_cached")
class SlaSage2Sm120CachedWeight(AttnWeightTemplate):
    sparsity_ratio = 0.8

    def __init__(self, config=None):
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("sla_sage2_sm120_cached requires an SM120 GPU")
        self.set_config(config or {})

    def set_config(self, config):
        self.config = dict(config)
        ratio = float(self.config.get("sparsity_ratio", type(self).sparsity_ratio))
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"Cached SLA sparsity_ratio must be in [0, 1), got {ratio}")
        self.topk_ratio = 1 - ratio

    def apply(self, q, k, v, **kwargs):
        workspace = make_transposed_v(v.contiguous().unsqueeze(0), k.shape[0])
        return self.apply_pretransposed(q, k, workspace)

    def apply_pretransposed(self, q, k, workspace):
        return sla_sage2_sm120_cached(q, k, workspace, self.topk_ratio)
