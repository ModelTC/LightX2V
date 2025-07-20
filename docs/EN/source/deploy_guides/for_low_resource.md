# Lightx2v Low-Resource Deployment Guide

## 📋 Overview

This guide is specifically designed for hardware resource-constrained environments, particularly configurations with **8GB VRAM + 16/32GB RAM**, providing detailed instructions on how to successfully run Lightx2v 14B models for 480p and 720p video generation.

Lightx2v is a powerful video generation model, but it requires careful optimization to run smoothly in resource-constrained environments. This guide provides a complete solution from hardware selection to software configuration, ensuring you can achieve the best video generation experience under limited hardware conditions.

## 🎯 Target Hardware Configuration

### Recommended Hardware Specifications

**GPU Requirements**:
- **VRAM**: 8GB (RTX 3060/3070/4060/4060Ti, etc.)
- **Architecture**: NVIDIA graphics cards with CUDA support

**System Memory**:
- **Minimum**: 16GB DDR4
- **Recommended**: 32GB DDR4/DDR5
- **Memory Speed**: 3200MHz or higher recommended

**Storage Requirements**:
- **Type**: NVMe SSD strongly recommended
- **Capacity**: At least 50GB available space
- **Speed**: Read speed of 3000MB/s or higher recommended

**CPU Requirements**:
- **Cores**: 8 cores or more recommended
- **Frequency**: 3.0GHz or higher recommended
- **Architecture**: Support for AVX2 instruction set

## ⚙️ Core Optimization Strategies

### 1. Environment Optimization

Before running Lightx2v, it's recommended to set the following environment variables to optimize performance:

```bash
# CUDA memory allocation optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable CUDA Graph mode to improve inference performance
export ENABLE_GRAPH_MODE=true

# Use BF16 precision for inference to reduce VRAM usage (default FP32 precision)
export DTYPE=BF16
```

**Optimization Details**:
- `expandable_segments:True`: Allows dynamic expansion of CUDA memory segments, reducing memory fragmentation
- `ENABLE_GRAPH_MODE=true`: Enables CUDA Graph to reduce kernel launch overhead
- `DTYPE=BF16`: Uses BF16 precision to reduce VRAM usage while maintaining quality

### 2. Quantization Strategy

Quantization is a key optimization technique in low-resource environments, reducing memory usage by lowering model precision.

#### Quantization Scheme Comparison

**FP8 Quantization** (Recommended for RTX 40 series):
```python
# Suitable for GPUs supporting FP8, providing better precision
dit_quant_scheme = "fp8"      # DIT model quantization
t5_quant_scheme = "fp8"       # T5 text encoder quantization
clip_quant_scheme = "fp8"     # CLIP visual encoder quantization
```

**INT8 Quantization** (Universal solution):
```python
# Suitable for all GPUs, minimal memory usage
dit_quant_scheme = "int8"     # 8-bit integer quantization
t5_quant_scheme = "int8"      # Text encoder quantization
clip_quant_scheme = "int8"    # Visual encoder quantization
```

### 3. Efficient Operator Selection Guide

Choosing the right operators can significantly improve inference speed and reduce memory usage.

#### Attention Operator Selection

**Recommended Priority**:
1. **[Sage Attention](https://github.com/thu-ml/SageAttention)** (Highest priority)

2. **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** (Universal solution)

#### Matrix Multiplication Operator Selection

**ADA Architecture GPUs** (RTX 40 series):

Recommended priority:
1. **[q8-kernel](https://github.com/KONAKONA666/q8_kernels)** (Highest performance, ADA architecture only)
2. **[sglang-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** (Balanced solution)
3. **[vllm-kernel](https://github.com/vllm-project/vllm)** (Universal solution)

**Other Architecture GPUs**:
1. **[sglang-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** (Recommended)
2. **[vllm-kernel](https://github.com/vllm-project/vllm)** (Alternative)

### 4. Parameter Offloading Strategy

Parameter offloading technology allows models to dynamically schedule parameters between CPU and disk, breaking through VRAM limitations.

#### Three-Level Offloading Architecture

```python
# Disk-CPU-GPU three-level offloading configuration
cpu_offload=True             # Enable CPU offloading
t5_cpu_offload=True          # Enable T5 encoder CPU offloading
offload_granularity=phase    # DIT model fine-grained offloading
t5_offload_granularity=block # T5 encoder fine-grained offloading
lazy_load = True             # Enable lazy loading mechanism
num_disk_workers = 2         # Disk I/O worker threads
```

#### Offloading Strategy Details

**Lazy Loading Mechanism**:
- Model parameters are loaded from disk to CPU on demand
- Reduces runtime memory usage
- Supports large models running with limited memory

**Disk Storage Optimization**:
- Use high-speed SSD to store model parameters
- Store model files grouped by blocks
- Refer to conversion script [documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme.md), specify `--save_by_block` parameter during conversion

### 5. VRAM Optimization Techniques

VRAM optimization strategies for 720p video generation.

#### CUDA Memory Management

```python
# CUDA memory cleanup configuration
clean_cuda_cache = True        # Timely cleanup of GPU cache
rotary_chunk = True            # Rotary position encoding chunked computation
rotary_chunk_size = 100        # Chunk size, adjustable based on VRAM
```

#### Chunked Computation Strategy

**Rotary Position Encoding Chunking**:
- Process long sequences in small chunks
- Reduce peak VRAM usage
- Maintain computational precision

### 6. VAE Optimization

VAE (Variational Autoencoder) is a key component in video generation, and optimizing VAE can significantly improve performance.

#### VAE Chunked Inference

```python
# VAE optimization configuration
use_tiling_vae = True          # Enable VAE chunked inference
```

#### [Lightweight VAE](https://github.com/madebyollin/taehv/blob/main/taew2_1.pth)

```python
# VAE optimization configuration
use_tiny_vae = True            # Use lightweight VAE
```

**VAE Optimization Effects**:
- Standard VAE: Baseline performance, 100% quality retention
- Standard VAE chunked: Reduces VRAM usage, increases inference time, 100% quality retention
- Lightweight VAE: Extremely low VRAM usage, video quality loss

### 7. Model Selection Strategy

Choosing the right model version is crucial for low-resource environments.

#### Recommended Model Comparison

**Distilled Models** (Strongly recommended):
- ✅ **[Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)**

- ✅ **[Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v)**

#### Performance Optimization Suggestions

When using the above distilled models, you can further optimize performance:
- Disable CFG: `"enable_cfg": false`
- Reduce inference steps: `infer_step: 4`
- Reference configuration files: [config](https://github.com/ModelTC/LightX2V/tree/main/configs/distill)

## 🚀 Complete Configuration Examples

### Pre-configured Templates

- **[14B Model 480p Video Generation Configuration](https://github.com/ModelTC/lightx2v/tree/main/configs/offload/disk/wan_i2v_phase_lazy_load_480p.json)**

- **[14B Model 720p Video Generation Configuration](https://github.com/ModelTC/lightx2v/tree/main/configs/offload/disk/wan_i2v_phase_lazy_load_720p.json)**

- **[1.3B Model 720p Video Generation Configuration](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/block/wan_t2v_1_3b.json)**
  - The inference bottleneck for 1.3B models is the T5 encoder, so the configuration file specifically optimizes for T5

**[Launch Script](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_lazy_load.sh)**

## 📚 Reference Resources

- [Parameter Offloading Mechanism Documentation](../method_tutorials/offload.md) - In-depth understanding of offloading technology principles
- [Quantization Technology Guide](../method_tutorials/quantization.md) - Detailed explanation of quantization technology
- [Gradio Deployment Guide](deploy_gradio.md) - Detailed Gradio deployment instructions

## ⚠️ Important Notes

1. **Hardware Requirements**: Ensure your hardware meets minimum configuration requirements
2. **Driver Version**: Recommend using the latest NVIDIA drivers (535+)
3. **CUDA Version**: Ensure CUDA version is compatible with PyTorch (recommend CUDA 11.8+)
4. **Storage Space**: Reserve sufficient disk space for model caching (at least 50GB)
5. **Network Environment**: Stable network connection required for initial model download
6. **Environment Variables**: Be sure to set the recommended environment variables to optimize performance

**Technical Support**: If you encounter issues, please submit an Issue to the project repository.
