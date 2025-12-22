# Wan2.1 模型转换指南

本指南介绍如何使用 model_converter 工具对 Wan DiT 模型进行格式转换和量化。

## 模型概述

### 支持的模型

| 模型名称 | 分辨率 | 参数量 | 模型类型 |
|---------|--------|--------|---------|
| Wan2.1-I2V-14B-480P | 480×832 | 14B | wan_dit |
| Wan2.1-I2V-14B-720P | 720×1280 | 14B | wan_dit |

### 支持的量化方案

| 量化方案 | 精度 | 压缩比 | 推荐场景 |
|---------|------|--------|---------|
| int8 | INT8 | ~50% | NVIDIA GPU / hygon_dcu |
| fp8 | FP8 E4M3 | ~50% | NVIDIA GPU (H100+) |
| mxfp4 | MxFP4 | ~75% | 极致压缩 |
| nvfp4 | NVIDIA FP4 | ~75% | NVIDIA GPU |

---

## 快速开始

### 方式 1: 使用命令行（推荐）

```bash
python -m tools.model_converter \
    --config tools/model_converter/configs/templates/wan_dit_int8_dcu.yaml
```

### 方式 2: 使用原始 converter.py

```bash
python tools/convert/converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output_int8 \
    --output_name wan2.1_i2v_480p_int8 \
    --output_ext .safetensors \
    --model_type wan_dit \
    --linear_type int8 \
    --quantized \
    --single_file
```

---

## 配置选项

### 核心配置项

#### 1. 模型类型配置

```yaml
source:
  type: wan_dit              # 模型类型（必填）
  path: /path/to/model       # 模型路径（必填）
  format: safetensors        # 文件格式
```

对应命令行参数：
```bash
--model_type wan_dit
--source /path/to/model
```

#### 2. 量化配置

```yaml
quantization:
  method: int8               # 量化方法: int8, fp8, mxfp4, nvfp4
  options:
    target_modules:          # 要量化的模块
      - self_attn            # 自注意力层
      - cross_attn           # 交叉注意力层
      - ffn                  # 前馈网络
    per_channel: true        # 使用 per-channel 量化
    symmetric: true          # 使用对称量化
```

对应命令行参数：
```bash
--linear_type int8
--quantized
# target_modules 由 --model_type wan_dit 自动设置
```

**注意**: 
- `target_modules` 在原始 converter.py 中通过 `model_type_keys_map` 自动配置
- Wan DiT 默认量化模块：`["self_attn", "cross_attn", "ffn"]`
- **不量化**: `norm` 层、`embedding` 层、`head` 层

#### 3. 输出配置

```yaml
output:
  path: /path/to/output      # 输出目录
  name: wan_dit_int8         # 输出文件名
  copy_metadata: true        # 复制配置文件
```

```yaml
target:
  format: lightx2v           # 输出格式
  precision: int8            # 精度标识
  layout: by_block           # 保存方式: by_block, single_file, chunked
```

对应命令行参数：
```bash
--output /path/to/output
--output_name wan_dit_int8
--single_file              # 或 --save_by_block
--copy_no_weight_files     # 复制元数据
```

**保存方式对比**:
- `single_file`: 单个文件，加载快，但需要大内存
- `by_block`: 按 block 分块，适合大模型
- `chunked`: 按固定大小分块，灵活性高

#### 4. 性能配置

```yaml
performance:
  parallel: true             # 并行转换
  device: cuda:0             # 设备: cuda:0, cpu
  num_workers: 4             # 并行线程数
```

对应命令行参数：
```bash
--parallel                 # 启用并行
--device cuda:0
```

---

## 使用示例

### 示例 1: INT8 量化

```bash
python tools/convert/converter.py \
    --source /data/models/Wan2.1-I2V-14B-480P/ \
    --output /data/models/Wan2.1-I2V-14B-480P_int8 \
    --output_name wan2.1_i2v_480p_int8 \
    --output_ext .safetensors \
    --model_type wan_dit \
    --linear_type int8 \
    --quantized \
    --single_file
```

**量化效果**:
- 原始大小: ~28 GB (FP16)
- 量化后: ~14 GB (INT8)
- 压缩比: ~50%

### 示例 2: FP8 量化（NVIDIA H100）

```bash
python tools/convert/converter.py \
    --source /data/models/Wan2.1-I2V-14B-720P/ \
    --output /data/models/Wan2.1-I2V-14B-720P_fp8 \
    --output_name wan2.1_i2v_720p_fp8 \
    --output_ext .safetensors \
    --model_type wan_dit \
    --linear_type fp8 \
    --quantized \
    --save_by_block
```

### 示例 3: 批量转换（480P + 720P）

```bash
#!/bin/bash
# 批量转换脚本

MODELS=(
    "Wan2.1-I2V-14B-480P"
    "Wan2.1-I2V-14B-720P"
)

for model in "${MODELS[@]}"; do
    echo "Converting $model..."
    python tools/convert/converter.py \
        --source /data/models/$model/ \
        --output /data/models/${model}_int8 \
        --output_name ${model,,}_int8 \
        --model_type wan_dit \
        --linear_type int8 \
        --quantized \
        --single_file
done
```

### 示例 4: 使用 YAML 配置

**创建配置文件** `wan_480p_int8.yaml`:
```yaml
source:
  type: wan_dit
  path: /data/models/Wan2.1-I2V-14B-480P

quantization:
  method: int8
  options:
    target_modules:
      - self_attn
      - cross_attn
      - ffn

output:
  path: /data/models/Wan2.1-I2V-14B-480P_int8
  name: wan2.1_i2v_480p_int8
```

**运行转换**:
```bash
python -m tools.model_converter --config wan_480p_int8.yaml
```