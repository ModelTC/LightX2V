import torch

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


def update_transposed_v(v, workspace, start):
    from .kernels.sage_sm90_fused_quant import update_transposed_v_kernel

    batch, length, heads, dim = v.shape
    if length:
        grid = ((length + 31) // 32, heads, batch * ((dim + 31) // 32))
        update_transposed_v_kernel[grid](v, workspace, length, heads, dim, workspace.shape[-1], start, *v.stride())


def make_transposed_v(v, total_tokens):
    batch, _, heads, dim = v.shape
    workspace = v.new_zeros(batch, dim, heads, (total_tokens + 127) // 128 * 128)
    update_transposed_v(v, workspace, 0)
    return workspace


@torch.library.custom_op("lightx2v::sage_sm90_transposed_v", mutates_args=(), device_types="cuda")
def sage_sm90_transposed_v(q: torch.Tensor, k: torch.Tensor, transposed_v: torch.Tensor) -> torch.Tensor:
    from sageattention import _fused, _qattn_sm90
    from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel

    from .kernels.sage_sm90_fused_quant import center_quant_key

    batch, queries, q_heads, dim = q.shape
    if dim not in (64, 128):
        raise ValueError("sage_attn2_sm90_fused requires head_dim 64 or 128")
    keys, k_heads = k.shape[1:3]
    q_int8, k_int8 = torch.empty_like(q, dtype=torch.int8), torch.empty_like(k, dtype=torch.int8)
    q_scale = q.new_empty(batch, q_heads, (queries + 63) // 64 * 32, dtype=torch.float32)
    k_scale = k.new_empty(batch, k_heads, (keys + 127) // 128 * 4, dtype=torch.float32)
    mean = k.mean(dim=1, keepdim=True)
    quant_query_per_thread_int8_kernel[((queries + 63) // 64 * 32, q_heads, batch)](
        q, q_int8, q_scale, queries, q.stride(0), q.stride(2), q.stride(1), q_int8.stride(0), q_int8.stride(2), q_int8.stride(1), q_scale.stride(0), q_scale.stride(1), C=dim, BLK=16
    )
    center_quant_key[((keys + 127) // 128 * 4, k_heads, batch)](k, mean, k_int8, k_scale, keys, k_heads, dim)
    v_fp8 = torch.empty_like(transposed_v, dtype=torch.float8_e4m3fn)
    v_scale = q.new_empty(batch, k_heads, dim, dtype=torch.float32)
    _fused.scale_fuse_quant_cuda(transposed_v, v_fp8, v_scale, transposed_v.shape[-1], 448.0, 0)
    output = torch.empty_like(q)
    _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, output, q_scale, k_scale, v_scale, 0, 0, 3, dim**-0.5, 0)
    return output


@sage_sm90_transposed_v.register_fake
def sage_sm90_transposed_v_fake(q, k, transposed_v):
    return torch.empty_like(q)


@ATTN_WEIGHT_REGISTER("sage_attn2_sm90_fused")
class SageAttn2Sm90FusedWeight(AttnWeightTemplate):
    def __init__(self):
        if torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("sage_attn2_sm90_fused requires an SM90 GPU")
        self.config = {}

    def apply(self, q, k, v, **kwargs):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        workspace = make_transposed_v(v, k.shape[1])
        return sage_sm90_transposed_v(q, k, workspace).flatten(0, 1).flatten(1)

    def apply_pretransposed(self, q, k, workspace):
        output = sage_sm90_transposed_v(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), workspace)
        return output.flatten(0, 1).flatten(1)
