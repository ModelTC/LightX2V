"""SM120 SageAttention accepting the causal cache's pretransposed V layout."""

import torch

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@torch.library.custom_op("lightx2v::sage_sm120_transposed_v", mutates_args=(), device_types="cuda")
def sage_sm120_transposed_v(q: torch.Tensor, k: torch.Tensor, transposed_v: torch.Tensor) -> torch.Tensor:
    from sageattention import _fused, _qattn_sm89
    from sageattention.quant import per_warp_int8

    dim = q.shape[-1]
    if dim not in (64, 128):
        raise ValueError("sage_attn2_sm120_cached requires head_dim 64 or 128")
    qi, qs, ki, ks = per_warp_int8(q, k, k.mean(dim=1, keepdim=True), tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
    vf = torch.empty_like(transposed_v, dtype=torch.float8_e4m3fn)
    vs = q.new_empty(q.shape[0], k.shape[2], dim, dtype=torch.float32)
    _fused.scale_fuse_quant_cuda(transposed_v, vf, vs, k.shape[1], 2.25, 0)
    output = torch.empty_like(q)
    _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(qi, ki, vf, output, qs, ks, vs, 0, 0, 2, dim**-0.5, 0)
    return output


@sage_sm120_transposed_v.register_fake
def sage_sm120_transposed_v_fake(q, k, transposed_v):
    return torch.empty_like(q)


@ATTN_WEIGHT_REGISTER("sage_attn2_sm120_cached")
class SageAttn2Sm120CachedWeight(AttnWeightTemplate):
    def __init__(self):
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("sage_attn2_sm120_cached requires an SM120 GPU")
        self.config = {}

    def apply(self, q, k, v, **kwargs):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        from sageattention import _fused

        batch, length, heads, dim = v.shape
        workspace = v.new_empty(batch, dim, heads, (length + 63) // 64 * 64)
        _fused.transpose_pad_permute_cuda(v, workspace, 0)
        return sage_sm120_transposed_v(q, k, workspace).flatten(0, 1).flatten(1)

    def apply_pretransposed(self, q, k, workspace):
        return sage_sm120_transposed_v(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), workspace).flatten(0, 1).flatten(1)
