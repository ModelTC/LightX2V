# Lightx2v 低资源部署指南

## 📋 概述

本指南专门针对硬件资源受限的环境，特别是**8GB显存 + 16/32GB内存**的配置，详细说明如何成功运行Lightx2v 14B模型进行480p和720p视频生成。

Lightx2v是一个强大的视频生成模型，但在资源受限的环境下需要精心优化才能流畅运行。本指南将为您提供从硬件选择到软件配置的完整解决方案，确保您能够在有限的硬件条件下获得最佳的视频生成体验。

## 🎯 目标硬件配置详解

### 推荐硬件规格

**GPU要求**:
- **显存**: 8GB (RTX 3060/3070/4060/4060Ti 等)
- **架构**: 支持CUDA的NVIDIA显卡

**系统内存**:
- **最低要求**: 16GB DDR4
- **推荐配置**: 32GB DDR4/DDR5
- **内存速度**: 建议3200MHz及以上

**存储要求**:
- **类型**: 强烈推荐NVMe SSD
- **容量**: 至少50GB可用空间
- **速度**: 读取速度建议3000MB/s以上

**CPU要求**:
- **核心数**: 建议8核心及以上
- **频率**: 建议3.0GHz及以上
- **架构**: 支持AVX2指令集

## ⚙️ 核心优化策略详解

### 1. 环境优化

在运行Lightx2v之前，建议设置以下环境变量以优化性能：

```bash
# CUDA内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启用CUDA Graph模式，提升推理性能
export ENABLE_GRAPH_MODE=true

# 使用BF16精度推理，减少显存占用（默认FP32精度）
export DTYPE=BF16
```

**优化说明**:
- `expandable_segments:True`: 允许CUDA内存段动态扩展，减少内存碎片
- `ENABLE_GRAPH_MODE=true`: 启用CUDA Graph，减少内核启动开销
- `DTYPE=BF16`: 使用BF16精度，在保持质量的同时减少显存占用

### 2. 量化策略

量化是低资源环境下的关键优化技术，通过降低模型精度来减少内存占用。

#### 量化方案对比

**FP8量化** (推荐用于RTX 40系列):
```python
# 适用于支持FP8的GPU，提供更好的精度
dit_quant_scheme = "fp8"      # DIT模型量化
t5_quant_scheme = "fp8"       # T5文本编码器量化
clip_quant_scheme = "fp8"     # CLIP视觉编码器量化
```

**INT8量化** (通用方案):
```python
# 适用于所有GPU，内存占用最小
dit_quant_scheme = "int8"     # 8位整数量化
t5_quant_scheme = "int8"      # 文本编码器量化
clip_quant_scheme = "int8"    # 视觉编码器量化
```
### 3. 高效算子选择指南

选择合适的算子可以显著提升推理速度和减少内存占用。

#### 注意力算子选择

**推荐优先级**:
1. **[Sage Attention](https://github.com/thu-ml/SageAttention)** (最高优先级)

2. **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** (通用方案)


#### 矩阵乘算子选择

**ADA架构显卡** (RTX 40系列):

推荐优先级:
1. **[q8-kernel](https://github.com/KONAKONA666/q8_kernels)** (最高性能，仅支持ADA架构)
2. **[sglang-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** (平衡方案)
3. **[vllm-kernel](https://github.com/vllm-project/vllm)** (通用方案)

**其他架构显卡**:
1. **[sglang-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** (推荐)
2. **[vllm-kernel](https://github.com/vllm-project/vllm)** (备选)

### 4. 参数卸载策略详解

参数卸载技术允许模型在CPU和磁盘之间动态调度参数，突破显存限制。

#### 三级卸载架构

```python
# 磁盘-CPU-GPU三级卸载配置
cpu_offload=True             # 启用CPU卸载
t5_cpu_offload=True          # 启用T5编码器CPU卸载
offload_granularity=phase    # DIT模型细粒度卸载
t5_offload_granularity=block # T5编码器细粒度卸载
lazy_load = True             # 启用延迟加载机制
num_disk_workers = 2         # 磁盘I/O工作线程数
```

#### 卸载策略详解

**延迟加载机制**:
- 模型参数按需从磁盘加载到CPU
- 减少运行时内存占用
- 支持大模型在有限内存下运行

**磁盘存储优化**:
- 使用高速SSD存储模型参数
- 按照block分组存储模型文件
- 参考转换脚本[文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)，转换时指定`--save_by_block`参数

### 5. 显存优化技术详解

针对720p视频生成的显存优化策略。

#### CUDA内存管理

```python
# CUDA内存清理配置
clean_cuda_cache = True        # 及时清理GPU缓存
rotary_chunk = True            # 旋转位置编码分块计算
rotary_chunk_size = 100        # 分块大小，可根据显存调整
```

#### 分块计算策略

**旋转位置编码分块**:
- 将长序列分成小块处理
- 减少峰值显存占用
- 保持计算精度

### 6. VAE优化详解

VAE (变分自编码器) 是视频生成的关键组件，优化VAE可以显著提升性能。

#### VAE分块推理

```python
# VAE优化配置
use_tiling_vae = True          # 启用VAE分块推理
```

#### [轻量级VAE](https://github.com/madebyollin/taehv/blob/main/taew2_1.pth)

```python
# VAE优化配置
use_tiny_vae = True            # 使用轻量级VAE
```

**VAE优化效果**:
- 标准VAE: 基准性能，100%质量保持
- 标准VAE分块: 降低显存，增加推理时间，100%质量保持
- 轻量VAE: 极低显存，视频质量有损


### 7. 模型选择策略

选择合适的模型版本对低资源环境至关重要。

#### 推荐模型对比

**蒸馏模型** (强烈推荐):
- ✅ **[Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)**

- ✅ **[Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v)**


#### 性能优化建议

使用上述蒸馏模型时，可以进一步优化性能：
- 关闭CFG: `"enable_cfg": false`
- 减少推理步数: `infer_step: 4`
- 参考配置文件: [config](https://github.com/ModelTC/LightX2V/tree/main/configs/distill)

## 🚀 完整配置示例

### 预配置模板

- **[14B模型480p视频生成配置](https://github.com/ModelTC/lightx2v/tree/main/configs/offload/disk/wan_i2v_phase_lazy_load_480p.json)**

- **[14B模型720p视频生成配置](https://github.com/ModelTC/lightx2v/tree/main/configs/offload/disk/wan_i2v_phase_lazy_load_720p.json)**

- **[1.3B模型720p视频生成配置](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/block/wan_t2v_1_3b.json)**
  - 1.3B模型推理瓶颈是T5 encoder，配置文件专门针对T5进行优化

**[启动脚本](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_lazy_load.sh)**


## 📚 参考资源

- [参数卸载机制文档](../method_tutorials/offload.md) - 深入了解卸载技术原理
- [量化技术指南](../method_tutorials/quantization.md) - 量化技术详细说明
- [Gradio部署指南](deploy_gradio.md) - Gradio部署详细说明

## ⚠️ 重要注意事项

1. **硬件要求**: 确保您的硬件满足最低配置要求
2. **驱动版本**: 建议使用最新的NVIDIA驱动 (535+)
3. **CUDA版本**: 确保CUDA版本与PyTorch兼容 (建议CUDA 11.8+)
4. **存储空间**: 预留足够的磁盘空间用于模型缓存 (至少50GB)
5. **网络环境**: 首次下载模型需要稳定的网络连接
6. **环境变量**: 务必设置推荐的环境变量以优化性能


**技术支持**: 如遇到问题，请提交Issue到项目仓库。
