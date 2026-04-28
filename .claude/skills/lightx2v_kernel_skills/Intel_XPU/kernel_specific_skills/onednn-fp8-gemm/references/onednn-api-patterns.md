# oneDNN API Patterns for Quantized GEMM

## Engine and Stream Setup

### From raw SYCL queue
```cpp
sycl::queue q(sycl::gpu_selector_v);
dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);
```

### From PyTorch XPU stream
```cpp
auto device = input.device();
sycl::queue& q = c10::xpu::getCurrentXPUStream(device.index()).queue();
dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);
```

## FP8 Per-N Scale Pattern

### Critical: set_scales_mask vs set_scales

```cpp
// CORRECT: JIT kernel
attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);

// WRONG: forces ocl:ref:any fallback
attr.set_scales(DNNL_ARG_WEIGHTS, 2, {}, dt::f32);
```

The `set_scales_mask` API uses an implicit f32 dtype and allows the JIT compiler to select
the optimal kernel. The `set_scales` API with explicit dtype constrains kernel selection
and results in the slow reference fallback for FP8.

### Bias Fusion

Bias can be fused into the matmul with no additional attribute configuration:
```cpp
// Just add bias to the execution args
args.insert({DNNL_ARG_BIAS, dnnl::memory(bias_md, eng, ptr_bias)});
```

Bias dtype should match the activation dtype (FP16 bias for FP16 activations).

## INT4 Block-wise Scale Pattern

### Scale mask decomposition

For weights B with logical shape `[K, N]`:
- Mask bit 0 (value 1): per-K grouping
- Mask bit 1 (value 2): per-N grouping
- Mask 3 = both bits: block-wise (groups along K, individual N)

The `{block_size, 1}` group parameter means:
- Group `block_size` consecutive K values together
- Each N column has its own scale within each group

### Zero-point

Scalar zero-point (mask=0, no group):
```cpp
attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dt::u8);
```

## Memory Allocation

All buffers must be SYCL device memory:
```cpp
auto* ptr = sycl::malloc_device<T>(count, queue);
queue.memcpy(ptr, host_data, count * sizeof(T));
queue.wait();
```

For INT4 (U4), the buffer size is `N * K / 2` bytes (2 values packed per byte).

## Execution Args Keys

| Key | Meaning |
|-----|---------|
| `DNNL_ARG_SRC` | Input activations A |
| `DNNL_ARG_WEIGHTS` | Quantized weights B |
| `DNNL_ARG_DST` | Output C |
| `DNNL_ARG_BIAS` | Optional bias vector |
| `DNNL_ARG_ATTR_SCALES \| DNNL_ARG_WEIGHTS` | Weight dequant scales |
| `DNNL_ARG_ATTR_ZERO_POINTS \| DNNL_ARG_WEIGHTS` | Weight zero points (INT4) |
