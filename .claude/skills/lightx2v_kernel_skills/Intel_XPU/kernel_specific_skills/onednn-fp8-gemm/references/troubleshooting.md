# oneDNN Quantized GEMM Troubleshooting

## Reference Fallback Detection

### Symptom: Extremely slow performance (10-100x expected)

**Check**: Print the implementation string after creating the primitive descriptor:
```cpp
auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
printf("impl: %s\n", pd.impl_info_str().c_str());
```

- `jit:gemm:any` = optimized JIT kernel (good)
- `ocl:ref:any` = reference fallback (bad)

### Common causes of ref fallback

1. **Using `set_scales()` instead of `set_scales_mask()` for FP8 per-N**:
   ```cpp
   // BAD: forces ref fallback
   attr.set_scales(DNNL_ARG_WEIGHTS, 2, {}, dt::f32);

   // GOOD: enables JIT
   attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
   ```

2. **Missing `set_fpmath_mode`**: While not always strictly required, enabling it ensures
   the JIT compiler has maximum flexibility:
   ```cpp
   attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
   ```

3. **FP32 output dtype**: JIT kernels only support f16/bf16 output for FP8 inputs.
   Using f32 output forces ref fallback.

4. **Unsupported scale configuration**: Per-tensor (mask=0) or per-K (mask=1) may not
   have JIT support for all dtype combinations.

## Build Errors

### "cannot find -ldnnl"
- Install `onednn_devel`: `pip install onednn_devel==2025.2.0`
- Check library path: `<conda_env>/Library/lib/dnnl.lib`
- Add `-L"<conda_env>/Library/lib"` to compile command

### "dnnl.hpp not found"
- Headers come from torch's include dir: `<conda_env>/Lib/site-packages/torch/include`
- Add `-I"<torch_include>"` to compile command
- Alternative: install `onednn_devel` which puts headers in `<conda_env>/Library/include`

### ur_api.h conflicts
- Conda's Python env ships its own `ur_api.h` that conflicts with oneAPI's version
- Solution: create a shim header that includes the compiler's version (see kernel_xpu_integration skill)

## Runtime Errors

### "No SYCL GPU device found"
- Check Intel GPU driver: `sycl-ls` should show the GPU
- Ensure oneAPI runtime is loaded: run `setvars.bat` first
- Check `SYCL_DEVICE_FILTER=gpu` if set

### Segfault in matmul execute
- Verify all device pointers are valid (allocated with `sycl::malloc_device`)
- Ensure host-to-device copies completed (`queue.wait()` before execute)
- Check buffer sizes match memory descriptors (especially for INT4: N*K/2 bytes)

### BF16 on older oneDNN
- oneDNN versions before 2025.2 may not have JIT for BF16xFP8 on BMG
- Workaround: use ESIMD dequant to BF16 + BF16xBF16 matmul pipeline
- Or upgrade to oneDNN 2025.2+

## Memory Layout Gotchas

### Weight storage: [N,K] physical, [K,N] logical
The `format_tag::ba` tells oneDNN that the physical layout is transposed from the logical shape.
- Logical shape in descriptor: `{K, N}`
- Physical storage: `[N, K]` (each row = one output channel)
- This is the standard weight layout for per-N quantization

### INT4 packing
- Two U4 values per byte: even index = low nibble, odd index = high nibble
- Buffer size: `N * K / 2` bytes
- K must be even (or padded to even)

### Scale memory for INT4
- Shape: `[n_groups, N]` where `n_groups = K / block_size`
- Layout: `format_tag::ba` (physical `[N, n_groups]`, logical `[n_groups, N]`)
- Dtype: `dt::f16` (not f32!)
