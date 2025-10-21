# 模型转换工具

这是一个功能强大的模型权重转换工具，支持格式转换、量化、LoRA融合等多种功能。

## 主要特性

- **格式转换**: 支持 PyTorch (.pth) 和 SafeTensors (.safetensors) 格式互转
- **模型量化**: 支持 INT8 和 FP8 量化，显著减小模型体积
- **架构转换**: 支持 LightX2V 和 Diffusers 架构互转
- **LoRA 融合**: 支持多种 LoRA 格式的加载和融合
- **多模型支持**: 支持 Wan DiT、Qwen Image DiT、T5、CLIP 等
- **灵活保存**: 支持单文件、按块、分块等多种保存方式
- **并行处理**: 大模型转换支持并行加速

## 支持的模型类型

- `wan_dit`: Wan DiT 系列模型（默认）
- `wan_animate_dit`: Wan Animate DiT 模型
- `qwen_image_dit`: Qwen Image DiT 模型
- `wan_t5`: Wan T5 文本编码器
- `wan_clip`: Wan CLIP 视觉编码器

## 核心参数说明

### 基础参数

- `-s, --source`: 输入路径（文件或目录）
- `-o, --output`: 输出目录路径
- `-o_e, --output_ext`: 输出格式，可选 `.pth` 或 `.safetensors`（默认）
- `-o_n, --output_name`: 输出文件名（默认: `converted`）
- `-t, --model_type`: 模型类型（默认: `wan_dit`）

### 架构转换参数

- `-d, --direction`: 转换方向
  - `None`: 不进行架构转换（默认）
  - `forward`: LightX2V → Diffusers
  - `backward`: Diffusers → LightX2V

### 量化参数

- `--quantized`: 启用量化
- `--bits`: 量化位宽，当前仅支持 8 位
- `--linear_dtype`: 线性层量化类型
  - `torch.int8`: INT8 量化
  - `torch.float8_e4m3fn`: FP8 量化
- `--non_linear_dtype`: 非线性层数据类型
  - `torch.bfloat16`: BF16
  - `torch.float16`: FP16
  - `torch.float32`: FP32（默认）
- `--device`: 量化使用的设备，可选 `cpu`（默认）或 `cuda`
- `--comfyui_mode`: ComfyUI 兼容模式
- `--full_quantized`: 全量化模式（ComfyUI 模式下有效）

### LoRA 参数

- `--lora_path`: LoRA 文件路径，支持多个（用空格分隔）
- `--lora_strength`: LoRA 强度系数，支持多个（默认: 1.0）
- `--alpha`: LoRA alpha 参数，支持多个
- `--lora_key_convert`: LoRA 键转换模式
  - `auto`: 自动检测（默认）
  - `same`: 使用原始键名
  - `convert`: 应用与模型相同的转换

### 保存参数

- `--single_file`: 保存为单个文件（注意: 大模型会消耗大量内存）
- `-b, --save_by_block`: 按块保存（推荐用于 backward 转换）
- `-c, --chunk-size`: 分块大小（默认: 100，0 表示不分块）
- `--copy_no_weight_files`: 复制源目录中的非权重文件

### 性能参数

- `--parallel`: 启用并行处理（默认: True）
- `--no-parallel`: 禁用并行处理

## 支持的 LoRA 格式

工具自动检测并支持以下 LoRA 格式:

1. **Standard**: `{key}.lora_up.weight` 和 `{key}.lora_down.weight`
2. **Diffusers**: `{key}_lora.up.weight` 和 `{key}_lora.down.weight`
3. **Diffusers V2**: `{key}.lora_B.weight` 和 `{key}.lora_A.weight`
4. **Diffusers V3**: `{key}.lora.up.weight` 和 `{key}.lora.down.weight`
5. **Mochi**: `{key}.lora_B` 和 `{key}.lora_A`（无 .weight 后缀）
6. **Transformers**: `{key}.lora_linear_layer.up.weight` 和 `{key}.lora_linear_layer.down.weight`
7. **Qwen**: `{key}.lora_B.default.weight` 和 `{key}.lora_A.default.weight`

此外还支持差值（diff）格式:
- `.diff`: 权重差值
- `.diff_b`: bias 差值
- `.diff_m`: modulation 差值

## 使用示例

### 1. 模型量化

#### 1.1 Wan DiT 量化为 INT8

**多个 safetensors，按 dit block 存储**
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan_int8 \
    --linear_dtype torch.int8 \
    --model_type wan_dit \
    --quantized \
    --save_by_block
```

**单个 safetensor 文件**
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_int8_lightx2v \
    --linear_dtype torch.int8 \
    --model_type wan_dit \
    --quantized \
    --single_file
```

#### 1.2 Wan DiT 量化为 FP8

**多个 safetensors，按 dit block 存储**
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan_fp8 \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --save_by_block
```

**单个 safetensor 文件**
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file
```

**ComfyUI 的 scaled_fp8 格式**
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_comfyui \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file \
    --comfyui_mode
```

**ComfyUI 的全 FP8 格式**
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_comfyui \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file \
    --comfyui_mode \
    --full_quantized
```

> **提示**: 对于其他 DIT 模型，切换 `--model_type` 参数即可

#### 1.3 T5 编码器量化

**INT8 量化**
```bash
python converter.py \
    --source /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_t5_umt5-xxl-enc-int8 \
    --linear_dtype torch.int8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_t5 \
    --quantized
```

**FP8 量化**
```bash
python converter.py \
    --source /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_t5_umt5-xxl-enc-fp8 \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_t5 \
    --quantized
```

#### 1.4 CLIP 编码器量化

**INT8 量化**
```bash
python converter.py \
    --source /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_clip_open-clip-xlm-roberta-large-vit-huge-14-int8 \
    --linear_dtype torch.int8 \
    --non_linear_dtype torch.float16 \
    --model_type wan_clip \
    --quantized
```

**FP8 量化**
```bash
python converter.py \
    --source /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8 \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.float16 \
    --model_type wan_clip \
    --quantized
```

### 2. LoRA 融合

#### 2.1 融合单个 LoRA

```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --single_file
```

#### 2.2 融合多个 LoRA

```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --model_type wan_dit \
    --lora_path /path/to/lora1.safetensors /path/to/lora2.safetensors \
    --lora_strength 1.0 0.8 \
    --single_file
```

#### 2.3 LoRA 融合后量化

**LoRA 融合 → FP8 量化**
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_quantized \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --quantized \
    --linear_dtype torch.float8_e4m3fn \
    --single_file
```

**LoRA 融合 → ComfyUI scaled_fp8**
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_quantized \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --quantized \
    --linear_dtype torch.float8_e4m3fn \
    --single_file \
    --comfyui_mode
```

**LoRA 融合 → ComfyUI 全 FP8**
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_quantized \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --quantized \
    --linear_dtype torch.float8_e4m3fn \
    --single_file \
    --comfyui_mode \
    --full_quantized
```

#### 2.4 LoRA 键转换模式

**自动检测模式（推荐）**
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert auto \
    --single_file
```

**使用原始键名（LoRA 已经是目标格式）**
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --direction forward \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert same \
    --single_file
```

**应用转换（LoRA 使用源格式）**
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --direction forward \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert convert \
    --single_file
```

### 3. 架构格式转换

#### 3.1 LightX2V → Diffusers

```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P \
    --output /path/to/Wan2.1-I2V-14B-480P-Diffusers \
    --output_ext .safetensors \
    --model_type wan_dit \
    --direction forward \
    --chunk-size 100
```

#### 3.2 Diffusers → LightX2V

```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P-Diffusers \
    --output /path/to/Wan2.1-I2V-14B-480P \
    --output_ext .safetensors \
    --model_type wan_dit \
    --direction backward \
    --save_by_block
```

### 4. 格式转换

#### 4.1 .pth → .safetensors

```bash
python converter.py \
    --source /path/to/model.pth \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name model
```

#### 4.2 多个 .safetensors → 单文件

```bash
python converter.py \
    --source /path/to/model_directory/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --single_file
```
