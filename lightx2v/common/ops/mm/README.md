# Shared projections and GELU/linear fusion

`optimized_linear.py` provides loaded-weight operators that other models can use
without EdenLive configuration fields. Backend implementations are grouped by
operation: `projection_group.py` contains FP8-SGL and INT8-ConvRot shared/packed
projections; `fused_gelu_linear.py` contains both GELU/quantization fusions.

```python
from lightx2v.common.ops.mm.optimized_linear import make_gelu_linear, make_projection_group

qkv = make_projection_group((q_linear, k_linear, v_linear), packed=True)
gelu_linear = make_gelu_linear(ffn_output_linear, enabled=can_fuse)

q, k, v = qkv.apply(x) if qkv is not None else tuple(linear.apply(x) for linear in (q_linear, k_linear, v_linear))
output = gelu_linear.apply(hidden)
```

Create these operators after loading resident weights, before compilation and
CUDA Graph capture. Rebuild them after weight reloads. The GELU fusion matches
the compiled GELU(tanh) path with `emulate_precision_casts=True`; callers using
different precision semantics should pass `enabled=False`. Model offload,
parallelism and graph eligibility remain the caller's responsibility.

## Supported kernels

| Backend | Shared / packed projections | GELU + activation quantization |
| --- | --- | --- |
| `fp8-sgl` | Exact `MMWeightWfp8channelAfp8channeldynamicSgl`; resident CUDA E4M3 weights in transposed `[K, N]` storage; FP32 per-output scales; contiguous 2D BF16/FP16 input and matching output dtype. GPU and GEMM shape support comes from the installed SGL kernel. | BF16 2D input, `1 <= K <= 16384`; FP8-SGL output GEMM. Tested full-model hidden width: 14336. GELU rounds to BF16 before the original per-token FP8 scale/quantization arithmetic. |
| `int8-convrot` | Exact `MMWeightWint8ConvRot`; SM120; group 256; `M >= 3`; `256 <= K <= 4096`, divisible by 256; contiguous INT8 `[N, K]` weights with `N` divisible by 16; FP32 per-output scales; BF16/FP16 input/output. Shares the native comfy-kitchen rotation and quantizer, with its original CUTLASS/cuBLAS INT32 GEMM and epilogue. | SM120, BF16 input/output, `M >= 3`, **K=14336**, group 256, output width divisible by 16; scalar or per-output FP32 scales. Retains the GELU, rotated activation and scale rounding boundaries before native comfy-kitchen GEMM. Other widths/groups/architectures use the original linear. |

The ConvRot fusion CUDA source is
[`lightx2v_kernel/csrc/quantization/convrot_gelu_quant.cu`](../../../../lightx2v_kernel/csrc/quantization/convrot_gelu_quant.cu).
Its launcher compiles it lazily into PyTorch's extension cache during
`make_gelu_linear`, before graph capture. Missing CUDA source/compiler or an
older comfy-kitchen backend disables this fusion. Installed main-package wheels
without the kernel source retain the original ConvRot implementation.

## Fallback and precision

Unsupported backends, mixed backend groups, incompatible dtype/layout/shape,
adapters, lazy loading and offload buffers keep the original module operations.
An unsupported projection group returns `None`; an unsupported GELU wrapper
executes `module.apply(F.gelu(hidden, approximate="tanh"))`. Runtime guards also
reject changed module tensor identities or newly enabled adapters/offload state.
The supported ConvRot path respects comfy-kitchen's selected CUDA backend;
forcing another backend keeps its original math.

Packing concatenates the existing quantized weights/scales/biases without
requantizing them. Original modules reference slices of that packed storage,
avoiding persistent duplicate weights and keeping their fallback usable. Packed
outputs are strided split views; downstream normalization and attention kernels
must support that row stride or explicitly materialize contiguous tensors.

These optimizations preserve the computation of the same quantized backend.
They do not remove the quantization error already introduced by FP8 or INT8.
