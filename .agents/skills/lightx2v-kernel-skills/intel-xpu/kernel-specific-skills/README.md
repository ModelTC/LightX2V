# Kernel Specific Skills

Reference index for the skills below; this file is not an invocable skill entry.

Available kernel references in this checkout. The bundled samples and historical benchmarks do not establish runtime support in LightX2V; Python exports are defined in `lightx2v_kernel/intel_xpu/csrc/entry.cpp` and `lightx2v_kernel/intel_xpu/python/sycl_kernels/__init__.py`, relative to the repository root.

## Sub-Skills

### SDP / Flash Attention (Xe2)

| Skill | Description |
|-------|-------------|
| [xe2-sdp-kernels](xe2-sdp-kernels/SKILL.md) | Flash Attention SDP kernels (prefill and decode) on Xe2: GQA, DPAS tiling; unified test assets have missing sparse-header dependencies |
| [xe2-sdp-bf16](xe2-sdp-bf16/SKILL.md) | BF16 SDP kernel variants on Xe2: ALU analysis, barrier interleave optimization, V-conversion strategies |
| [xe2-sdp-hd256](xe2-sdp-hd256/SKILL.md) | Head-dim 256 SDP kernel optimizations on Xe2: pipelined QK, MAD fusion, S-scatter techniques |

### GEMM (Matrix Multiply)

| Skill | Description |
|-------|-------------|
| [xe2-esimd-gemm](xe2-esimd-gemm/SKILL.md) | FP16 ESIMD GEMM kernels on Xe2: DPAS tiling, gather-based layouts, prefetch strategies |
| [onednn-fp8-gemm](onednn-fp8-gemm/SKILL.md) | OneDNN FP8 GEMM on Xe2: quantization patterns, API usage, performance tuning |

### GEMV (Matrix-Vector Multiply)

| Skill | Description |
|-------|-------------|
| [xe2-esimd-gemv](xe2-esimd-gemv/SKILL.md) | W4A16 and W8A16 quantized GEMV on Xe2: SLM reduction, dequant-dot fusion |
