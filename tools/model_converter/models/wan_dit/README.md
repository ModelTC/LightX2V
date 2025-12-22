# Wan DiT Model Converter

Wan DiT系列模型的转换器，支持I2V和T2V模型。

## 支持的模型

- **Wan2.1-I2V-14B-480P** - 图像到视频生成
- **Wan2.1-T2V-14B** - 文本到视频生成
- **Wan-Animate-DiT** - 带适配器的动画生成模型

## 模型变体

### 标准 Wan DiT

```yaml
source:
  type: wan_dit
  path: /path/to/Wan2.1-I2V-14B-480P
```

特点：
- 标准的self-attention和cross-attention
- FFN模块
- 自动忽略音频适配器权重

### Wan Animate DiT

```yaml
source:
  type: wan_dit
  variant: wan_animate_dit  # 可选，会自动检测
  path: /path/to/Wan-Animate-DiT
```

特点：
- 包含face adapter
- 额外的adapter keys需要量化

## 格式转换

### LightX2V → Diffusers

```yaml
source:
  type: wan_dit
  path: /path/to/lightx2v_model

target:
  format: diffusers  # 转换为Diffusers格式
  layout: chunked    # 分块保存
```

键映射示例：
- `blocks.0.self_attn.q` → `blocks.0.attn1.to_q`
- `blocks.0.cross_attn.k` → `blocks.0.attn2.to_k`
- `head.head` → `proj_out`

### Diffusers → LightX2V

```yaml
source:
  type: wan_dit
  path: /path/to/diffusers_model
  format: diffusers

target:
  format: lightx2v
  layout: by_block  # 按block保存
```

## 量化配置

### 默认量化模块

```yaml
quantization:
  options:
    target_modules:
      - self_attn  # 自注意力
      - cross_attn # 交叉注意力
      - ffn        # 前馈网络
```

### 忽略的模块

自动忽略：
- `ca.*` - 音频适配器
- `audio.*` - 音频相关权重

### INT8 量化示例

```yaml
source:
  type: wan_dit
  path: /path/to/model

target:
  format: lightx2v
  layout: by_block

quantization:
  method: int8
  backend: dcu
```

预期效果：
- 原始大小：~27GB (FP32)
- INT8大小：~7GB
- 压缩率：~74%

### FP8 量化示例

```yaml
quantization:
  method: fp8
  backend: nvidia
```

预期效果：
- 原始大小：~27GB (FP32)
- FP8大小：~7GB
- 性能提升：2-3x (NVIDIA H100)

## LoRA支持

### 合并单个LoRA

```yaml
source:
  type: wan_dit
  path: /path/to/base_model

lora:
  enabled: true
  paths:
    - /path/to/style_lora.safetensors
  strengths: [1.0]
  alphas: [8.0]
```

### 合并多个LoRA

```yaml
lora:
  enabled: true
  paths:
    - /path/to/lora1.safetensors
    - /path/to/lora2.safetensors
  strengths: [1.0, 0.8]  # 第二个LoRA强度降低
  alphas: [8.0, 8.0]
```

### LoRA格式

自动支持以下格式：
- Standard: `key.lora_up.weight`, `key.lora_down.weight`
- Diffusers: `key_lora.up.weight`, `key_lora.down.weight`
- Diffusers V2: `key.lora_B.weight`, `key.lora_A.weight`

## ComfyUI导出

### FP8 ComfyUI格式

```yaml
source:
  type: wan_dit
  path: /path/to/model

target:
  format: comfyui
  precision: fp8
  layout: single_file  # ComfyUI要求单文件

quantization:
  method: fp8
  backend: nvidia

output:
  path: /path/to/comfyui/models
  name: wan_dit_fp8
```

特殊处理：
- 添加 `scaled_fp8` 标记
- 使用 `.scale_weight` 后缀存储scales
- 单文件格式

## 性能优化

### 大模型转换

```yaml
target:
  layout: by_block  # 按block保存，减少内存

performance:
  parallel: true     # 启用并行加速
  device: cuda:0     # GPU加速量化
  num_workers: 4
```

### DCU优化

```yaml
quantization:
  backend: dcu

performance:
  device: cuda:0  # DCU设备
```

环境变量：
```bash
export HIP_VISIBLE_DEVICES=0
export PLATFORM=hygon_dcu
export PYTHONPATH=/path/to/LightX2V:$PYTHONPATH
```

## 完整示例

### 示例1: DCU上INT8转换

```bash
# 1. 创建配置
cat > convert_config.yaml << EOF
source:
  type: wan_dit
  path: /data/models/Wan2.1-I2V-14B-480P

target:
  format: lightx2v
  layout: by_block

quantization:
  method: int8
  backend: dcu

output:
  path: /data/output/wan_int8
  name: wan_dit_int8_dcu

performance:
  device: cuda:0
EOF

# 2. 运行转换
export HIP_VISIBLE_DEVICES=0
python -m model_converter convert --config convert_config.yaml
```

### 示例2: LoRA + FP8 + ComfyUI

```bash
cat > lora_comfyui.yaml << EOF
source:
  type: wan_dit
  path: /data/models/base_model

lora:
  enabled: true
  paths:
    - /data/loras/style.safetensors
  strengths: [1.0]
  alphas: [8.0]

target:
  format: comfyui
  layout: single_file

quantization:
  method: fp8
  backend: nvidia

output:
  path: /data/comfyui/models
  name: wan_style_fp8
EOF

python -m model_converter convert --config lora_comfyui.yaml
```

### 示例3: 快速命令行模式

```bash
# 不使用配置文件，直接命令行
python -m model_converter convert \
  --model wan_dit \
  --source /data/models/Wan2.1-I2V-14B-480P \
  --target-format lightx2v \
  --quantization int8 \
  --backend dcu \
  --output /data/output/wan_int8 \
  --layout by_block
```

## 故障排除

### 问题：键不匹配

**症状**：LoRA应用失败，提示找不到匹配的键

**解决**：
```yaml
# 确保LoRA和模型格式匹配
# 如果模型是Diffusers格式，LoRA也应该是
source:
  format: diffusers  # 明确指定格式
```

### 问题：内存不足

**症状**：OOM错误

**解决**：
```yaml
target:
  layout: by_block  # 使用分块保存

performance:
  parallel: false   # 禁用并行
  device: cpu       # 使用CPU量化
```

### 问题：量化后精度下降

**症状**：量化后模型效果明显下降

**解决**：
1. 尝试FP8而不是INT8：
```yaml
quantization:
  method: fp8  # 通常比INT8精度更好
```

2. 调整量化范围：
```yaml
quantization:
  options:
    target_modules:
      - ffn  # 只量化FFN，保留attention精度
```

## 技术细节

### 模型结构

```
Wan DiT
├── condition_embedder
│   ├── text_embedder (text_embedding)
│   ├── time_embedder (time_embedding)
│   ├── time_proj (time_projection)
│   └── image_embedder (img_emb)
├── blocks[0..N]
│   ├── norm1
│   ├── attn1 (self_attn)
│   ├── norm2
│   ├── attn2 (cross_attn)
│   ├── norm3
│   ├── ffn
│   └── scale_shift_table (modulation)
└── proj_out (head)
```

### 键映射规则数量

- 总计：33条映射规则
- Attention：16条
- FFN：2条
- Embeddings：9条
- Norms：4条
- Other：2条

## 相关资源

- [配置模板](../../configs/templates/)
- [主文档](../../README.md)
- [迁移指南](../../MIGRATION.md)

