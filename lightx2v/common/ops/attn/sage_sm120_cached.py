"""SM120 SageAttention with cached V layouts and static quantized K/V."""

import torch

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .sage_sm90_fused import make_transposed_v
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


def prepare_sage_sm120_kv(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize static [tokens, heads, dim] K/V exactly as native SM120 Sage."""
    from sageattention import _fused
    from sageattention.quant import per_channel_fp8

    if k.ndim != 3 or k.shape != v.shape or k.shape[-1] not in (64, 128):
        raise ValueError("SM120 prepared K/V require matching [tokens, heads, dim] tensors with head_dim 64 or 128")
    if not k.is_cuda or k.device != v.device or k.dtype != v.dtype or k.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("SM120 prepared K/V require FP16 or BF16 tensors on the same CUDA device")
    k, v = k.contiguous().unsqueeze(0), v.contiguous().unsqueeze(0)
    torch.cuda.set_device(k.device)
    ki = torch.empty_like(k, dtype=torch.int8)
    ks = k.new_empty((1, k.shape[2], (k.shape[1] + 63) // 64), dtype=torch.float32)
    # Native per_warp_int8 uses this fused per-block path for K, with its BF16 mean.
    km = k.mean(dim=1, keepdim=True).squeeze(1)
    _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, ki, ks, 64, 0)
    vf, vs, _ = per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25, smooth_v=False)
    return ki, ks, vf, vs


@torch.library.custom_op("lightx2v::sage_sm120_cached_kv", mutates_args=(), device_types="cuda")
def sage_sm120_cached_kv(q: torch.Tensor, ki: torch.Tensor, ks: torch.Tensor, vf: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    """Attend to prepared static K/V, quantizing only the current query."""
    from sageattention import _fused, _qattn_sm89

    q = q.contiguous().unsqueeze(0)
    dim = q.shape[-1]
    if q.ndim != 4 or dim not in (64, 128):
        raise ValueError("SM120 cached-K/V attention requires [tokens, heads, dim] Q with head_dim 64 or 128")
    qi = torch.empty_like(q, dtype=torch.int8)
    qs = q.new_empty((1, q.shape[2], ((q.shape[1] + 127) // 128) * 4), dtype=torch.float32)
    _fused.quant_per_warp_int8_cuda(q, qi, qs, 128, 32, 0)
    output = torch.empty_like(q)
    _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(qi, ki, vf, output, qs, ks, vs, 0, 0, 2, dim**-0.5, 0)
    return output.flatten(0, 1).flatten(1)


@sage_sm120_cached_kv.register_fake
def sage_sm120_cached_kv_fake(q, ki, ks, vf, vs):
    return q.new_empty((q.shape[0], q.shape[1] * q.shape[2]))


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
        return sage_sm120_transposed_v(q, k, make_transposed_v(v, k.shape[1])).flatten(0, 1).flatten(1)

    def apply_pretransposed(self, q, k, workspace):
        return sage_sm120_transposed_v(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), workspace).flatten(0, 1).flatten(1)

    def prepare_kv(self, k, v):
        return prepare_sage_sm120_kv(k, v)

    def apply_prepared(self, q, prepared):
        return sage_sm120_cached_kv(q, *prepared)
