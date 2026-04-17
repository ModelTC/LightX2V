# Kernel Specific Skills

Implementation skills for specific ESIMD kernel types on Intel Xe2 and Xe3 GPUs.

## Sub-Skills

### SDP / Flash Attention (Xe2)

| Skill | Description |
|-------|-------------|
| [xe2-sdp-kernels](./xe2-sdp-kernels/SKILL.md) | Flash Attention SDP kernels (prefill and decode) on Xe2: GQA, paged KV-cache, DPAS tiling |
| [xe2-sdp-bf16](./xe2-sdp-bf16/SKILL.md) | BF16 SDP kernel variants on Xe2: ALU analysis, barrier interleave optimization, V-conversion strategies |
| [xe2-sdp-hd256](./xe2-sdp-hd256/SKILL.md) | Head-dim 256 SDP kernel optimizations on Xe2: pipelined QK, MAD fusion, S-scatter techniques |
| [xe2-sdp-sparse-mask](./xe2-sdp-sparse-mask/SKILL.md) | SDP with sparse attention masks on Xe2: mask pipeline, sparse kernel patterns |

### GEMM (Matrix Multiply)

| Skill | Description |
|-------|-------------|
| [xe2-esimd-gemm](./xe2-esimd-gemm/SKILL.md) | FP16 ESIMD GEMM kernels on Xe2: DPAS tiling, gather-based layouts, prefetch strategies |
| [onednn-fp8-gemm](./onednn-fp8-gemm/SKILL.md) | OneDNN FP8 GEMM on Xe2: quantization patterns, API usage, performance tuning |
| [xe3-onednn-fp16-gemm](./xe3-onednn-fp16-gemm/SKILL.md) | OneDNN FP16 GEMM on Xe3 (Panther Lake): setup and optimization |
| [xe3-onednn-fp8-gemm](./xe3-onednn-fp8-gemm/SKILL.md) | FP16/BF16 x FP8 quantized GEMM using OneDNN on Xe3: dequant pipelines, per-N scaling |

### GEMV (Matrix-Vector Multiply)

| Skill | Description |
|-------|-------------|
| [intel-esimd-gemv-fp16](./intel-esimd-gemv-fp16/skill.md) | High-performance FP16 GEMV kernels on Intel GPUs using ESIMD |
| [intel-esimd-gemv-w4a16](./intel-esimd-gemv-w4a16/skill.md) | W4A16 quantized GEMV kernels using ESIMD: 4-bit weight dequantization |
| [xe2-esimd-gemv](./xe2-esimd-gemv/SKILL.md) | W4A16 and W8A16 quantized GEMV on Xe2: SLM reduction, dequant-dot fusion |
| [gemv-gguf-quant](./gemv-gguf-quant/skill.md) | GGUF quantized GEMV (Q4_K/Q5_K/Q6_K/Q8_0) on Xe2: nibble unpacking, pre-shuffled bit buffers, >95% DRAM roofline |

### Attention Sub-Operations

| Skill | Description |
|-------|-------------|
| [intel-esimd-qk-gemm-gqa](./intel-esimd-qk-gemm-gqa/skill.md) | QK GEMM decode kernel with Grouped-Query Attention (GQA) |
| [intel-esimd-qk-gemm-softmax](./intel-esimd-qk-gemm-softmax/skill.md) | Fused QK GEMM + softmax kernel |
| [intel-esimd-qkv-gemm](./intel-esimd-qkv-gemm/skill.md) | QKV GEMM kernel for attention computation |
| [intel-esimd-kv-gemm](./intel-esimd-kv-gemm/skill.md) | KV GEMM kernel for score-value multiplication |
| [intel-esimd-pooling](./intel-esimd-pooling/skill.md) | QK max pooling kernel for sparse attention |

### TopK Kernels

| Skill | Description |
|-------|-------------|
| [intel-esimd-topk](./intel-esimd-topk/skill.md) | Top-K selection kernel using ESIMD |
| [intel-esimd-topk-runtime](./intel-esimd-topk-runtime/skill.md) | Runtime-configurable Top-K with dynamic K |
| [intel-esimd-topk-softmax](./intel-esimd-topk-softmax/skill.md) | Fused softmax + Top-K + normalize kernel |

### Linear Attention

| Skill | Description |
|-------|-------------|
| [xe2-linear-attention](./xe2-linear-attention/SKILL.md) | GLA (Gated Linear Attention) and GDN (Gated Delta Network) recurrent/decode kernels on Xe2 |

### MoE (Mixture of Experts)

| Skill | Description |
|-------|-------------|
| [moe-kernels](./moe-kernels/SKILL.md) | MoE kernel suite: routing (sigmoid top-K), decode GEMV, prefill GEMM, and vLLM integration |

### Sparse Attention (InfLLMv2)

| Skill | Description |
|-------|-------------|
| [infllmv2-sparse-attention](./infllmv2-sparse-attention/SKILL.md) | InfLLMv2 sparse attention suite: pattern generation, sparse SDP (prefill/decode), vLLM integration |

### Xe3 Kernels

| Skill | Description |
|-------|-------------|
| [xe3-esimd-kernels](./xe3-esimd-kernels/SKILL.md) | SYCL ESIMD kernels on Xe3 (Panther Lake): GLA decode/recurrent kernels |
