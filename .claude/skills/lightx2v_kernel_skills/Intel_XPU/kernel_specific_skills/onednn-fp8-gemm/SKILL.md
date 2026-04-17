---
name: onednn-fp8-gemm
description: >
  Use this skill when implementing, optimizing, or debugging quantized GEMM kernels using
  oneDNN on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) or newer Intel XPU. Xe2 is the GPU architecture; LNL and BMG are product names. Covers FP16/BF16 x FP8_E4M3 with
  per-N scale, FP16 x FP8 with block-wise scale along K, FP16 x INT4 (U4) with block-wise
  scale + zero-point, 2D block quantization emulation via repeat_interleave, bias fusion,
  and the critical API differences between set_scales_mask (JIT) vs set_scales (ref fallback).
  Use whenever the user mentions oneDNN FP8 GEMM, quantized matmul, W8A16, W4A16,
  per-N scale, block-wise FP8, block-wise INT4, 2D block quantization, or dnnl matmul
  primitive on Intel GPU.
---

# oneDNN Quantized GEMM on Xe2 (Lunar Lake/LNL, Battlemage/BMG)

Specialized knowledge for implementing FP16/BF16 x FP8 and FP16 x INT4 quantized GEMM
using oneDNN's matmul primitive on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) discrete GPU.

---

## Quick-Reference Rules (must follow every time)

### Supported Configurations on BMG (oneDNN 2025.2)

| I/O dtype | Weights | Scale | oneDNN Impl | Performance |
|-----------|---------|-------|-------------|-------------|
| **FP16** | FP8_E4M3 | per-N | `jit:gemm:any` | **~130 TFLOPS (96% of 135T peak)** |
| **BF16** | FP8_E4M3 | per-N | `jit:gemm:any` | **~96 TFLOPS (71% of peak)** |
| **FP16** | FP8_E4M3 | block-K per-N | `jit:gemm:any` | **~88-110 TFLOPS** |
| BF16 | FP8_E4M3 | block-K per-N | `jit:gemm:any` | **BROKEN** (wrong results on v3.7) |
| **FP16** | U4 (INT4) | block-wise | `jit:gemm:any` | **~130 TFLOPS** |
| FP32 | FP8_E4M3 | per-N | `ocl:ref:any` | Very slow (reference) |

**KEY DIFFERENCE FROM PTL**: On BMG with oneDNN 2025.2, **both FP16xFP8 and BF16xFP8 have
optimized JIT kernels**. On PTL, only FP16xFP8 had JIT; BF16xFP8 fell back to `ocl:ref:any`.

### Layout Convention
- **A**: `[M, K]` row-major (`format_tag::ab`) -- FP16 or BF16 activations
- **B**: logical `[K, N]`, physical `[N, K]` (`format_tag::ba`) -- FP8_E4M3 or U4 weights
- **scale**: `[N]` FP32 for FP8; `[n_groups, N]` FP16 for INT4
- **C**: `[M, N]` row-major (`format_tag::ab`) -- same dtype as A

### CRITICAL: set_scales_mask vs set_scales

| API | Use Case | JIT? | Notes |
|-----|----------|------|-------|
| `set_scales_mask(DNNL_ARG_WEIGHTS, 2)` | FP8 per-N | **YES** | Implicit f32 dtype, required for JIT |
| `set_scales(DNNL_ARG_WEIGHTS, 2, {}, dt::f32)` | FP8 per-N | **NO** | Forces `ocl:ref:any` fallback! |
| `set_scales(DNNL_ARG_WEIGHTS, 3, {blk,1}, dt::f16)` | INT4 block-wise | **YES** | Explicit dtype required for groups |

**Always check implementation string after creating primitive_desc:**
```cpp
std::string impl = pd.impl_info_str();
if (impl.find("ref") != std::string::npos) {
    printf("WARNING: Fallback to slow reference implementation!\n");
}
```

---

## Pattern 1: FP16/BF16 x FP8_E4M3 with Per-N Scale

```
C[M,N] = A[M,K] x dequant(B[K,N])  where dequant(B) = B_fp8 * scale[N]
```

### Memory Descriptors
```cpp
using dt = dnnl::memory::data_type;
using ft = dnnl::memory::format_tag;

dnnl::memory::desc a_md({M, K}, dt::f16, ft::ab);     // or dt::bf16
dnnl::memory::desc b_md({K, N}, dt::f8_e4m3, ft::ba); // logical [K,N], physical [N,K]
dnnl::memory::desc scale_md({N}, dt::f32, ft::a);
dnnl::memory::desc c_md({M, N}, dt::f16, ft::ab);     // same dtype as A
```

### Primitive Attributes
```cpp
dnnl::primitive_attr attr;
attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);  // per-N, MUST use set_scales_mask
attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
```

### Execute
```cpp
dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
dnnl::matmul prim(pd);

std::unordered_map<int, dnnl::memory> args = {
    {DNNL_ARG_SRC,                            dnnl::memory(a_md, eng, ptr_A)},
    {DNNL_ARG_WEIGHTS,                        dnnl::memory(b_md, eng, ptr_B)},
    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_md, eng, ptr_scales)},
    {DNNL_ARG_DST,                            dnnl::memory(c_md, eng, ptr_C)},
};

// Optional bias: [N] in same dtype as A
if (has_bias) {
    dnnl::memory::desc bias_md({N}, dt::f16, ft::a);
    args.insert({DNNL_ARG_BIAS, dnnl::memory(bias_md, eng, ptr_bias)});
}

prim.execute(s, args);
```

### FP8 E4M3 Format
- 1 sign + 4 exponent + 3 mantissa, bias=7, no inf
- Max value: +/-448, min normal: 2^(-6)
- Per-N absmax quantization: `scale[n] = max(|B[n,:]|) / 448`

---

## Pattern 2: FP16 x INT4 (U4) with Block-wise Scale + Zero-point

```
C[M,N] = A[M,K] x dequant(B[K,N])
where dequant(B) = (B_u4 - zero_point) * scale[group, N]
```

### Memory Descriptors
```cpp
int block_size = 64;  // quantization group size along K
int n_groups = K / block_size;

dnnl::memory::desc a_md({M, K}, dt::f16, ft::ab);
dnnl::memory::desc w_md({K, N}, dt::u4, ft::ba);  // 4-bit unsigned, transposed
dnnl::memory::desc scale_md({n_groups, N}, dt::f16, ft::ba);  // block-wise
dnnl::memory::desc zp_md({1}, dt::u8, ft::a);  // scalar zero point
dnnl::memory::desc c_md({M, N}, dt::f16, ft::ab);
```

### Primitive Attributes
```cpp
dnnl::primitive_attr attr;
// Block-wise scales: mask = (1<<0)|(1<<1) = 3, group = {block_size, 1}
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1) + (1 << 0), {block_size, 1}, dt::f16);
attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dt::u8);
attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
```

### Execute
```cpp
std::unordered_map<int, dnnl::memory> args = {
    {DNNL_ARG_SRC,                                dnnl::memory(a_md, eng, ptr_A)},
    {DNNL_ARG_WEIGHTS,                            dnnl::memory(w_md, eng, ptr_W)},
    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,     dnnl::memory(scale_md, eng, ptr_scales)},
    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(zp_md, eng, ptr_zp)},
    {DNNL_ARG_DST,                                dnnl::memory(c_md, eng, ptr_C)},
};
prim.execute(s, args);
```

### INT4 Weight Packing (PyTorch side)
```python
# Pack two U4 values per byte: low nibble first
packed[...] = qweight[:, 0::2]           # even indices -> low nibble
packed[...] |= qweight[:, 1::2] << 4     # odd indices -> high nibble
```

### INT4 Per-group Absmax Quantization
```python
grouped = weight.reshape(-1, group_size)
max_vals = grouped.abs().max(-1).values
scales = max_vals / 7.0                   # map to [1, 15] with zp=8
qweight = clamp(round(grouped / scales.unsqueeze(-1) + 8), 0, 15).to(uint8)
```

---

## Pattern 3: FP16 x FP8_E4M3 with Block-wise Scale (1D along K)

```
C[M,N] = A[M,K] x dequant(B[K,N])
where dequant(B[gk*bk+k, n]) = B_fp8[gk*bk+k, n] * scale[gk, n]
```

Block-wise along K with per-N granularity within each block. Same pattern as
INT4 (Pattern 2) but with FP8 weights and FP32 scales.

### Memory Descriptors
```cpp
int block_k = 128;  // block size along K
int n_groups_k = K / block_k;

dnnl::memory::desc a_md({M, K}, dt::f16, ft::ab);
dnnl::memory::desc b_md({K, N}, dt::f8_e4m3, ft::ba);  // logical [K,N], physical [N,K]
dnnl::memory::desc scale_md({n_groups_k, N}, dt::f32, ft::ab);  // block-wise FP32
dnnl::memory::desc c_md({M, N}, dt::f16, ft::ab);
```

### Primitive Attributes
```cpp
dnnl::primitive_attr attr;
// mask=3 = (1<<0)|(1<<1): per both K and N dims of logical B[K,N]
// groups={block_k, 1}: block along K, per-N within each block
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1) + (1 << 0),
                {block_k, 1}, dnnl::memory::data_type::f32);
attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
```

### Emulating 2D Block Quantization (block_k x block_n)

True 2D block scales `[K/bk, N/bn]` are NOT supported on oneDNN v3.7 (2025.2) --
primitive creation fails. Requires oneDNN v3.9+.

**Workaround**: Quantize with 2D blocks, then expand compact scales to 1D format:
```python
# Quantize weight [N, K] with 2D blocks (128x128)
# scales_compact: [K//128, N//128]  -- one scale per block

# Expand to 1D format: repeat each N-group scale 128 times
# scales_1d: [K//128, N]  -- what oneDNN v3.7 accepts
scales_1d = scales_compact.repeat_interleave(128, dim=1).contiguous()

# Call oneDNN with expanded scales
out = sycl_kernels.onednn_w8a16_fp8_block(x, qweight, scales_1d, block_k=128)
```

Scale memory overhead: compact `[32, 32]` = 4 KB → expanded `[32, 4096]` = 512 KB
(still small vs weight tensor ~16 MB).

### Known Issues (oneDNN v3.7 / 2025.2)

- **FP16 block-wise: WORKS** -- `jit:gemm:any`, rel_rms = 0.0004
- **BF16 block-wise: BROKEN** -- selects `jit:gemm:any` but produces garbage output (rel_rms ~164). Use FP16 only for block-wise FP8.
- **2D native `{128, 128}` groups**: primitive creation fails, needs oneDNN v3.9+

---

## Scale Mask Cheat Sheet

| Quantization | API | Mask | Group | Dtype |
|-------------|-----|------|-------|-------|
| FP8 per-N | `set_scales_mask()` | 2 | -- | implicit f32 |
| FP8 block-K | `set_scales()` | 3 | `{block_k,1}` | `dt::f32` |
| INT4 block-wise | `set_scales()` | 3 | `{block_size,1}` | `dt::f16` |

- **Mask=0**: per-tensor (single scale for entire weight)
- **Mask=1**: per-K (per input channel)
- **Mask=2**: per-N (per output channel) -- most common for weight quantization
- **Mask=3**: per both dims = block-wise (groups along K, per-N)

**Note**: FP8 per-N MUST use `set_scales_mask`, NOT `set_scales`. FP8 block-wise
uses `set_scales` with `dt::f32`. INT4 block-wise uses `set_scales` with `dt::f16`.

---

## Required Headers and Initialization

```cpp
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

sycl::queue q(sycl::gpu_selector_v);
dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);
```

In PyTorch context: `auto q = c10::xpu::getCurrentXPUStream(device.index()).queue();`

---

## Build Commands

### Standalone C++ (no PyTorch)
```batch
set TORCH_INC=<conda_env>\Lib\site-packages\torch\include
set DNNL_LIB=<conda_env>\Library\lib
icpx test.cpp -fsycl -o test.exe -I"%TORCH_INC%" -L"%DNNL_LIB%" -ldnnl -std=c++20
```

Where `<conda_env>` provides `oneapi/dnnl/dnnl.hpp` via torch's include and `dnnl.lib` via Library/lib.

### Within PyTorch extension (CMake)
```cmake
find_package(IntelSYCL REQUIRED)
find_package(Torch REQUIRED)
target_link_libraries(${MODULE} torch torch_python dnnl)
```

---

## Performance Results (BMG, oneDNN 2025.2)

### Pure C++ ULT (no PyTorch overhead)

| Config | Time | TFLOPS | Impl |
|--------|------|--------|------|
| FP16xFP8 attn [512,4096,4096] | 0.134 ms | 128.6 | jit:gemm:any |
| FP16xFP8 FFN up [512,10240,4096] | 0.384 ms | 111.9 | jit:gemm:any |
| FP16xFP8 FFN down [512,4096,10240] | 0.329 ms | 130.5 | jit:gemm:any |
| BF16xFP8 attn [512,4096,4096] | 0.179 ms | 96.0 | jit:gemm:any |
| BF16xFP8 FFN up [512,10240,4096] | 0.509 ms | 84.3 | jit:gemm:any |
| FP16xU4 GEMV [1,4096,2048] | 0.009 ms | 1.8 | jit:gemm:any |
| FP16xU4 attn [512,4096,4096] | 0.132 ms | 130.2 | jit:gemm:any |

### PyTorch Wrapper (lightx2v_kernel_xpu)

| Config | FP16xFP8 | BF16xFP8 | Speedup vs native |
|--------|----------|----------|-------------------|
| attn [512,4096,4096] | 0.15ms / 112T | 0.20ms / 85T | 4.3-4.7x |
| FFN up [512,10240,4096] | 0.40ms / 107T | 0.55ms / 78T | 3.4-3.9x |
| FFN down [512,4096,10240] | 0.36ms / 119T | 0.48ms / 89T | 3.8-4.2x |

BMG peak: **135 TFLOPS FP16 XMX**. FP16xFP8 reaches ~96% in pure C++, ~83% through PyTorch.

---

## Correctness Testing

- Reference: FP32 CPU with per-N dequantization (FP8) or per-group dequantization (INT4)
- Expected rel_rms: **< 0.01** for FP8, **< 0.05** for INT4 (higher due to 4-bit quantization)
- Check for NaN before computing error
- All tests verified on BMG: rel_rms 0.0002-0.0034 for FP8, 0.0003 for INT4

---

## Sample Code

| Asset | Description |
|-------|-------------|
| `assets/test_onednn.cpp` | Complete pure C++ ULT: FP16xFP8, BF16xFP8, FP16xU4, bias -- all 4 patterns |
| `assets/fp16xfp8_per_n_scale.cpp` | Detailed FP16 x FP8 standalone example with profiling |

---

## Reference Files

| File | Contents |
|------|----------|
| `references/onednn-api-patterns.md` | Full API patterns for FP8/INT4, scale masks, execution args, bias fusion |
| `references/performance-results.md` | Detailed BMG benchmark results and roofline analysis |
| `references/troubleshooting.md` | Common errors: ref fallback, BF16 issues, build problems |
