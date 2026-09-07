import math
import threading

import torch

_COMPILED = {}
_COMPILE_LOCK = threading.Lock()


def _mark_shd_dynamic(tensor):
    stride_order = (0, 1, 2)
    return tensor.mark_layout_dynamic(leading_dim=2).mark_compact_shape_dynamic(mode=0, stride_order=stride_order).mark_compact_shape_dynamic(mode=1, stride_order=stride_order)


def _mark_1d_dynamic(tensor):
    return tensor.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))


def _mark_sparse_count_dynamic(tensor):
    stride_order = (0, 1, 2)
    return (
        tensor.mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=stride_order)
        .mark_compact_shape_dynamic(mode=1, stride_order=stride_order)
        .mark_compact_shape_dynamic(mode=2, stride_order=stride_order)
    )


def _mark_sparse_indices_dynamic(tensor):
    stride_order = (0, 1, 2, 3)
    return (
        tensor.mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(mode=0, stride_order=stride_order)
        .mark_compact_shape_dynamic(mode=1, stride_order=stride_order)
        .mark_compact_shape_dynamic(mode=2, stride_order=stride_order)
    )


def _to_cute(tensor, element_type):
    from cutlass.cute.runtime import from_dlpack

    cute_tensor = from_dlpack(tensor, assumed_align=16)
    cute_tensor.element_type = element_type
    return cute_tensor


def _cutlass_dtype(dtype):
    import cutlass

    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    raise TypeError(f"unsupported CuTeDSL sparse FMHA dtype: {dtype}")


def _validate_inputs(q, k, v, cu_seqlens, max_seqlen, block_count, block_indices):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("CuTeDSL sparse FMHA expects [total_tokens, heads, dim] Q/K/V")
    if k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("CuTeDSL sparse FMHA requires matching token counts and K/V shapes")
    if q.shape[2] != k.shape[2] or not 0 < q.shape[2] <= 128:
        raise ValueError("CuTeDSL sparse FMHA requires matching head dimensions in [1, 128]")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("the number of Q heads must be divisible by the number of KV heads")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn) or any(t.dtype != q.dtype for t in (k, v)):
        raise TypeError("CuTeDSL sparse FMHA requires matching FP16, BF16, or FP8 Q/K/V")
    if not all(t.is_cuda and t.is_contiguous() for t in (q, k, v)):
        raise ValueError("CuTeDSL sparse FMHA requires contiguous CUDA Q/K/V")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be a one-dimensional int32 tensor")
    if max_seqlen <= 0:
        raise ValueError("max_seqlen must be positive")
    if block_count.dtype != torch.int32 or block_indices.dtype != torch.int32:
        raise TypeError("sparse metadata must use int32")
    if block_count.ndim != 3 or block_indices.ndim != 4:
        raise ValueError("sparse block count/index metadata must have ranks 3 and 4")
    if block_indices.shape[:3] != block_count.shape:
        raise ValueError("sparse block count/index prefixes must match")
    batch_size = cu_seqlens.shape[0] - 1
    q_blocks = (max_seqlen + 255) // 256
    if block_count.shape != (batch_size, q.shape[1], q_blocks):
        raise ValueError(f"block_count must have shape [{batch_size}, {q.shape[1]}, {q_blocks}]")
    if block_indices.shape[-1] <= 0:
        raise ValueError("every sparse metadata row must retain at least one KV block")


def _make_args(q, k, v, out, cu_seqlens, max_seqlen, block_count, block_indices):
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass.cute.typing import Float32, Int32

    scale = 1.0 / math.sqrt(q.shape[-1])
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    input_element_type = _cutlass_dtype(q.dtype)
    output_element_type = _cutlass_dtype(out.dtype)
    return (
        _mark_shd_dynamic(_to_cute(q, input_element_type)),
        _mark_shd_dynamic(_to_cute(k, input_element_type)),
        _mark_shd_dynamic(_to_cute(v, input_element_type)),
        _mark_shd_dynamic(_to_cute(out, output_element_type)),
        _mark_1d_dynamic(_to_cute(cu_seqlens, cutlass.Int32)),
        Int32(max_seqlen),
        Float32(scale * math.log2(math.e)),
        Float32(scale),
        Float32(1.0),
        _mark_sparse_count_dynamic(_to_cute(block_count, cutlass.Int32)),
        _mark_sparse_indices_dynamic(_to_cute(block_indices, cutlass.Int32)),
        stream,
    )


@torch.compiler.disable
def cutedsl_sparse_fmha(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    block_count,
    block_indices,
    output_dtype=None,
):
    """Run the standalone Blackwell 256x128 block-sparse ViT FMHA."""
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    block_count = block_count.to(device=q.device, dtype=torch.int32).contiguous()
    block_indices = block_indices.to(device=q.device, dtype=torch.int32).contiguous()
    _validate_inputs(q, k, v, cu_seqlens, max_seqlen, block_count, block_indices)

    from cutlass import cute
    from cutlass.cute.typing import Float32

    from .kernels import cutedsl_fmha_helpers as fmha_utils
    from .kernels.cutedsl_sparse_fmha_kernel import (
        BlackwellFusedMultiHeadAttentionForward,
    )

    if output_dtype is None:
        output_dtype = q.dtype
    out = torch.empty(q.shape, device=q.device, dtype=output_dtype)
    args = _make_args(q, k, v, out, cu_seqlens, max_seqlen, block_count, block_indices)
    capacity = block_indices.shape[-1]
    key = (q.device.index, q.dtype, out.dtype, q.shape[-1], capacity)

    compiled = _COMPILED.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                mma_k = ((q.shape[-1] + 15) // 16) * 16
                actual_head_dim = q.shape[-1] if mma_k != q.shape[-1] else None
                kernel = BlackwellFusedMultiHeadAttentionForward(
                    Float32,
                    Float32,
                    (128, 128, mma_k),
                    True,
                    fmha_utils.MaskEnum.RESIDUAL_MASK,
                    is_causal=False,
                    use_sliding_window=False,
                    actual_head_dim=actual_head_dim,
                )
                compiled = cute.compile(kernel.__call_vit_sparse__, *args)
                _COMPILED[key] = compiled

    compiled(*args)
    return out
