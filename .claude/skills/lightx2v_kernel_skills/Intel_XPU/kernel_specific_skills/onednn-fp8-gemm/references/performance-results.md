# oneDNN Quantized GEMM Performance on BMG

## Hardware
- **GPU**: Intel Battlemage (BMG, Xe2 discrete)
- **Peak**: 135 TFLOPS FP16 XMX
- **Practical BW**: ~520 GB/s
- **oneDNN version**: 2025.2.0

## Pure C++ ULT Results (no PyTorch overhead)

### Test 1: FP16 x FP8_E4M3 (per-N scale)

| Shape [M,N,K] | Label | Time (ms) | TFLOPS | % Peak | Impl |
|----------------|-------|-----------|--------|--------|------|
| [512, 4096, 4096] | attn | 0.134 | 128.56 | 95.2% | jit:gemm:any |
| [512, 10240, 4096] | FFN up | 0.384 | 111.94 | 82.9% | jit:gemm:any |
| [512, 4096, 10240] | FFN down | 0.329 | 130.51 | 96.7% | jit:gemm:any |

Correctness: rel_rms = 0.0002 (all shapes)

### Test 2: BF16 x FP8_E4M3 (per-N scale)

| Shape [M,N,K] | Label | Time (ms) | TFLOPS | % Peak | Impl |
|----------------|-------|-----------|--------|--------|------|
| [512, 4096, 4096] | attn | 0.179 | 95.96 | 71.1% | jit:gemm:any |
| [512, 10240, 4096] | FFN up | 0.509 | 84.30 | 62.4% | jit:gemm:any |

Correctness: rel_rms = 0.0034

### Test 3: FP16 x INT4 (U4) (block-wise scale + zero-point, blk=64)

| Shape [M,N,K] | Label | Time (ms) | TFLOPS | % Peak | Impl |
|----------------|-------|-----------|--------|--------|------|
| [1, 4096, 2048] | GEMV | 0.009 | 1.80 | 1.3% | jit:gemm:any |
| [512, 4096, 4096] | attn | 0.132 | 130.24 | 96.5% | jit:gemm:any |

Correctness: rel_rms = 0.0003

### Test 4: FP16 x FP8 + bias
- [512, 4096, 4096]: rel_rms = 0.0128, PASS

## PyTorch Wrapper Results (lightx2v_kernel_xpu)

### FP16 x FP8

| Shape [M,N,K] | oneDNN FP8 | Native dequant+GEMM | Speedup |
|----------------|-----------|---------------------|---------|
| [512, 4096, 4096] | 0.15ms / 112T | 0.68ms / 25T | 4.7x |
| [512, 10240, 4096] | 0.40ms / 107T | 1.74ms / 25T | 4.3x |
| [512, 4096, 10240] | 0.36ms / 119T | 1.64ms / 26T | 4.6x |

### BF16 x FP8

| Shape [M,N,K] | oneDNN FP8 | Native dequant+GEMM | Speedup |
|----------------|-----------|---------------------|---------|
| [512, 4096, 4096] | 0.20ms / 85T | 0.68ms / 25T | 3.4x |
| [512, 10240, 4096] | 0.55ms / 78T | 1.82ms / 24T | 3.3x |

## Key Observations

1. **FP16xFP8 is faster than FP16xFP16** due to reduced weight memory traffic (1 byte vs 2 bytes per element)
2. **BF16xFP8 gets JIT on BMG** (oneDNN 2025.2) -- this was ref-only on PTL
3. **INT4 matches FP8 throughput** for compute-bound shapes (512x4096x4096)
4. **GEMV is memory-bound**: only 1.8 TFLOPS for M=1 (memory access dominates)
5. **PyTorch overhead**: ~15-20% reduction vs pure C++ (queue management, tensor validation)

## Roofline Analysis

For [512, 4096, 4096] FP16xFP8:
- FLOPs: 2 * 512 * 4096 * 4096 = 17.18 GFLOPs
- Bytes: A(512*4096*2) + B(4096*4096*1) + C(512*4096*2) = ~25 MB
- Arithmetic intensity: 17.18G / 25M = ~688 FLOPs/byte
- Clearly compute-bound (ridge point at ~260 FLOPs/byte for BMG)
