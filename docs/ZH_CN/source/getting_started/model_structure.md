# 模型格式与加载指南

## 📖 概述

LightX2V 是一个灵活的视频生成推理框架，支持多种模型来源和格式，为用户提供丰富的选择：

- ✅ **Wan 官方模型**：直接兼容 Wan2.1 和 Wan2.2 官方发布的完整模型
- ✅ **单文件模型**：支持 LightX2V 发布的单文件格式模型（包含量化版本）
- ✅ **LoRA 模型**：支持加载 LightX2V 发布的蒸馏 LoRA

本文档将详细介绍各种模型格式的使用方法、配置参数和最佳实践。

---

## 🗂️ 格式一：Wan 官方模型

### 模型仓库
- [Wan2.1 Collection](https://huggingface.co/collections/Wan-AI/wan21-68ac4ba85372ae5a8e282a1b)
- [Wan2.2 Collection](https://huggingface.co/collections/Wan-AI/wan22-68ac4ae80a8b477e79636fc8)

### 模型特点
- **官方保证**：Wan-AI 官方发布的完整模型，质量最高
- **完整组件**：包含所有必需的组件（DIT、T5、CLIP、VAE）
- **原始精度**：使用 BF16/FP32 精度，无量化损失
- **兼容性强**：与 Wan 官方工具链完全兼容

### Wan2.1 官方模型

#### 目录结构

以 [Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) 为例：

```
Wan2.1-I2V-14B-720P/
├── diffusion_pytorch_model-00001-of-00007.safetensors   # DIT 模型分片 1
├── diffusion_pytorch_model-00002-of-00007.safetensors   # DIT 模型分片 2
├── diffusion_pytorch_model-00003-of-00007.safetensors   # DIT 模型分片 3
├── diffusion_pytorch_model-00004-of-00007.safetensors   # DIT 模型分片 4
├── diffusion_pytorch_model-00005-of-00007.safetensors   # DIT 模型分片 5
├── diffusion_pytorch_model-00006-of-00007.safetensors   # DIT 模型分片 6
├── diffusion_pytorch_model-00007-of-00007.safetensors   # DIT 模型分片 7
├── diffusion_pytorch_model.safetensors.index.json       # 分片索引文件
├── models_t5_umt5-xxl-enc-bf16.pth                      # T5 文本编码器
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP 编码器
├── Wan2.1_VAE.pth                                       # VAE 编解码器
├── config.json                                          # 模型配置
├── xlm-roberta-large/                                   # CLIP tokenizer
├── google/                                              # T5 tokenizer
├── assets/
└── examples/
```

#### 使用方法

```bash
# 下载模型
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir ./models/Wan2.1-I2V-14B-720P

# 配置启动脚本
model_path=./models/Wan2.1-I2V-14B-720P
lightx2v_path=/path/to/LightX2V

# 运行推理
cd LightX2V/scripts
bash wan/run_wan_i2v.sh
```

### Wan2.2 官方模型

#### 目录结构

以 [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) 为例：

```
Wan2.2-I2V-A14B/
├── high_noise_model/                                    # 高噪声模型目录
│   ├── diffusion_pytorch_model-00001-of-00009.safetensors
│   ├── diffusion_pytorch_model-00002-of-00009.safetensors
│   ├── ...
│   ├── diffusion_pytorch_model-00009-of-00009.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json
├── low_noise_model/                                     # 低噪声模型目录
│   ├── diffusion_pytorch_model-00001-of-00009.safetensors
│   ├── diffusion_pytorch_model-00002-of-00009.safetensors
│   ├── ...
│   ├── diffusion_pytorch_model-00009-of-00009.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json
├── models_t5_umt5-xxl-enc-bf16.pth                      # T5 文本编码器
├── Wan2.1_VAE.pth                                       # VAE 编解码器
├── configuration.json                                   # 模型配置
├── google/                                              # T5 tokenizer
├── assets/                                              # 示例资源（可选）
└── examples/                                            # 示例文件（可选）
```

#### 使用方法

```bash
# 下载模型
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    --local-dir ./models/Wan2.2-I2V-A14B

# 配置启动脚本
model_path=./models/Wan2.2-I2V-A14B
lightx2v_path=/path/to/LightX2V

# 运行推理
cd LightX2V/scripts
bash wan22/run_wan22_moe_i2v.sh
```

### 可用模型列表

#### Wan2.1 官方模型列表

| 模型名称 | 下载链接 |
|---------|----------|
| Wan2.1-I2V-14B-720P | [链接](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| Wan2.1-I2V-14B-480P | [链接](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1-T2V-14B | [链接](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1-T2V-1.3B | [链接](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-FLF2V-14B-720P | [链接](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P) |
| Wan2.1-VACE-14B | [链接](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) |
| Wan2.1-VACE-1.3B | [链接](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) |

#### Wan2.2 官方模型列表

| 模型名称 | 下载链接 |
|---------|----------|
| Wan2.2-I2V-A14B | [链接](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| Wan2.2-T2V-A14B | [链接](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Wan2.2-TI2V-5B | [链接](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Wan2.2-Animate-14B | [链接](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) |

### 使用提示

> 💡 **量化模型使用**：如需使用量化模型，可参考[模型转换脚本](https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme_zh.md)进行转换，或直接使用下方格式二中的预转换量化模型
>
> 💡 **显存优化**：对于 RTX 4090 24GB 或更小显存的设备，建议结合量化技术和 CPU 卸载功能：
> - 量化配置：参考[量化技术文档](../method_tutorials/quantization.md)
> - CPU 卸载：参考[参数卸载文档](../method_tutorials/offload.md)
> - Wan2.1 配置：参考 [offload 配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/offload)
> - Wan2.2 配置：参考 [wan22 配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/wan22) 中以 `4090` 结尾的配置

---

## 🗂️ 格式二：LightX2V 单文件模型（推荐）

### 模型仓库
- [Wan2.1-LightX2V](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan2.2-LightX2V](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

### 模型特点
- **单文件管理**：单个 safetensors 文件，易于管理和部署
- **多精度支持**：提供原始精度、FP8、INT8 等多种精度版本
- **蒸馏加速**：支持 4-step 快速推理
- **工具兼容**：兼容 ComfyUI 等其他工具

**示例**：
- `wan2.1_i2v_720p_lightx2v_4step.safetensors` - 720P 图生视频原始精度
- `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors` - 720P 图生视频 FP8 量化
- `wan2.1_i2v_480p_int8_lightx2v_4step.safetensors` - 480P 图生视频 INT8 量化
- ...

### Wan2.1 单文件模型

#### 场景 A：下载单个模型文件

**步骤 1：选择并下载模型**

```bash
# 创建模型目录
mkdir -p ./models/wan2.1_i2v_720p

# 下载 720P 图生视频 FP8 量化模型
huggingface-cli download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p \
    --include "wan2.1_i2v_720p_lightx2v_4step.safetensors"
```

**步骤 2：手动组织其他模块**

目录结构如下
```
wan2.1_i2v_720p/
├── wan2.1_i2v_720p_lightx2v_4step.safetensors                    # 原始精度
└── t5/clip/vae/config.json/xlm-roberta-large/google等其他组件       # 需要手动组织
```

**步骤 3：配置启动脚本**

```bash
# 在启动脚本中设置（指向包含模型文件的目录）
model_path=./models/wan2.1_i2v_720p
lightx2v_path=/path/to/LightX2V

# 运行脚本
cd LightX2V/scripts
bash wan/run_wan_i2v_distill_4step_cfg.sh
```

> 💡 **提示**：当目录下只有一个模型文件时，LightX2V 会自动加载该文件。

#### 场景 B：下载多个模型文件

当您下载了多个不同精度的模型到同一目录时，需要在配置文件中明确指定使用哪个模型。

**步骤 1：下载多个模型**

```bash
# 创建模型目录
mkdir -p ./models/wan2.1_i2v_720p_multi

# 下载原始精度模型
huggingface-cli download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p_multi \
    --include "wan2.1_i2v_720p_lightx2v_4step.safetensors"

# 下载 FP8 量化模型
huggingface-cli download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p_multi \
    --include "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors"

# 下载 INT8 量化模型
huggingface-cli download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p_multi \
    --include "wan2.1_i2v_720p_int8_lightx2v_4step.safetensors"
```

**步骤 2：手动组织其他模块**

目录结构如下：

```
wan2.1_i2v_720p_multi/
├── wan2.1_i2v_720p_lightx2v_4step.safetensors                    # 原始精度
├── wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors   # FP8 量化
└── wan2.1_i2v_720p_int8_lightx2v_4step.safetensors              # INT8 量化
└── t5/clip/vae/config.json/xlm-roberta-large/google等其他组件       # 需要手动组织
```

**步骤 3：在配置文件中指定模型**

编辑配置文件（如 `configs/distill/wan_i2v_distill_4step_cfg.json`）：

```json
{
    // 使用原始精度模型
    "dit_original_ckpt": "./models/wan2.1_i2v_720p_multi/wan2.1_i2v_720p_lightx2v_4step.safetensors",

    // 或使用 FP8 量化模型
    // "dit_quantized_ckpt": "./models/wan2.1_i2v_720p_multi/wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    // "dit_quantized": true,
    // "dit_quant_scheme": "fp8-vllm",

    // 或使用 INT8 量化模型
    // "dit_quantized_ckpt": "./models/wan2.1_i2v_720p_multi/wan2.1_i2v_720p_int8_lightx2v_4step.safetensors",
    // "dit_quantized": true,
    // "dit_quant_scheme": "int8-vllm",

    // 其他配置...
}
```
### 使用提示

> 💡 **配置参数说明**：
> - **dit_original_ckpt**：用于指定原始精度模型（BF16/FP32/FP16）的路径
> - **dit_quantized_ckpt**：用于指定量化模型（FP8/INT8）的路径，需配合 `dit_quantized` 和 `dit_quant_scheme` 参数使用

**步骤 4：启动推理**

```bash
cd LightX2V/scripts
bash wan/run_wan_i2v_distill_4step_cfg.sh
```

### Wan2.2 单文件模型

#### 目录结构要求

使用 Wan2.2 单文件模型时，需要手动创建特定的目录结构：

```
wan2.2_models/
├── high_noise_model/                                    # 高噪声模型目录（必须）
│   └── wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors  # 高噪声模型文件
└── low_noise_model/                                     # 低噪声模型目录（必须）
│   └── wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors  # 低噪声模型文件
└── t5/vae/config.json/xlm-roberta-large/google等其他组件       # 需要手动组织
```

#### 场景 A：每个目录下只有一个模型文件

```bash
# 创建必需的子目录
mkdir -p ./models/wan2.2_models/high_noise_model
mkdir -p ./models/wan2.2_models/low_noise_model

# 下载高噪声模型到对应目录
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models/high_noise_model \
    --include "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"

# 下载低噪声模型到对应目录
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models/low_noise_model \
    --include "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"

# 配置启动脚本（指向父目录）
model_path=./models/wan2.2_models
lightx2v_path=/path/to/LightX2V

# 运行脚本
cd LightX2V/scripts
bash wan22/run_wan22_moe_i2v_distill.sh
```

> 💡 **提示**：当每个子目录下只有一个模型文件时，LightX2V 会自动加载。

#### 场景 B：每个目录下有多个模型文件

当您在 `high_noise_model/` 和 `low_noise_model/` 目录下分别放置了多个不同精度的模型时，需要在配置文件中明确指定。

```bash
# 创建目录
mkdir -p ./models/wan2.2_models_multi/high_noise_model
mkdir -p ./models/wan2.2_models_multi/low_noise_model

# 下载高噪声模型的多个版本
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models_multi/high_noise_model \
    --include "wan2.2_i2v_A14b_high_noise_*.safetensors"

# 下载低噪声模型的多个版本
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models_multi/low_noise_model \
    --include "wan2.2_i2v_A14b_low_noise_*.safetensors"
```

**目录结构**：

```
wan2.2_models_multi/
├── high_noise_model/
│   ├── wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors        # 原始精度
│   ├── wan2.2_i2v_A14b_high_noise_fp8_e4m3_lightx2v_4step.safetensors    # FP8 量化
│   └── wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors   # INT8 量化
└── low_noise_model/
│    ├── wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors         # 原始精度
│    ├── wan2.2_i2v_A14b_low_noise_fp8_e4m3_lightx2v_4step.safetensors     # FP8 量化
│    └── wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors    # INT8 量化
└── t5/vae/config.json/xlm-roberta-large/google等其他组件       # 需要手动组织
```

**配置文件设置**：

```json
{
    // 使用原始精度模型
    "high_noise_original_ckpt": "./models/wan2.2_models_multi/high_noise_model/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors",
    "low_noise_original_ckpt": "./models/wan2.2_models_multi/low_noise_model/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors",

    // 或使用 FP8 量化模型
    // "high_noise_quantized_ckpt": "./models/wan2.2_models_multi/high_noise_model/wan2.2_i2v_A14b_high_noise_fp8_e4m3_lightx2v_4step.safetensors",
    // "low_noise_quantized_ckpt": "./models/wan2.2_models_multi/low_noise_model/wan2.2_i2v_A14b_low_noise_fp8_e4m3_lightx2v_4step.safetensors",
    // "dit_quantized": true,
    // "dit_quant_scheme": "fp8-vllm"

    // 或使用 INT8 量化模型
    // "high_noise_quantized_ckpt": "./models/wan2.2_models_multi/high_noise_model/wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors",
    // "low_noise_quantized_ckpt": "./models/wan2.2_models_multi/low_noise_model/wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors",
    // "dit_quantized": true,
    // "dit_quant_scheme": "int8-vllm"
}
```

### 使用提示

> 💡 **配置参数说明**：
> - **high_noise_original_ckpt** / **low_noise_original_ckpt**：用于指定原始精度模型（BF16/FP32/FP16）的路径
> - **high_noise_quantized_ckpt** / **low_noise_quantized_ckpt**：用于指定量化模型（FP8/INT8）的路径，需配合 `dit_quantized` 和 `dit_quant_scheme` 参数使用


### 可用模型列表

#### Wan2.1 单文件模型列表

**图生视频模型（I2V）**

| 文件名 | 精度 | 说明 |
|--------|------|------|
| `wan2.1_i2v_480p_lightx2v_4step.safetensors` | BF16 | 4步模型原始精度 |
| `wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 4步模型FP8 量化 |
| `wan2.1_i2v_480p_int8_lightx2v_4step.safetensors` | INT8 | 4步模型INT8 量化 |
| `wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | FP8 | 4步模型ComfyUI 格式 |
| `wan2.1_i2v_720p_lightx2v_4step.safetensors` | BF16 | 4步模型原始精度 |
| `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 4步模型FP8 量化 |
| `wan2.1_i2v_720p_int8_lightx2v_4step.safetensors` | INT8 | 4步模型INT8 量化 |
| `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | FP8 | 4步模型ComfyUI 格式 |

**文生视频模型（T2V）**

| 文件名 | 精度 | 说明 |
|--------|------|------|
| `wan2.1_t2v_14b_lightx2v_4step.safetensors` | BF16 | 4步模型原始精度 |
| `wan2.1_t2v_14b_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 4步模型FP8 量化 |
| `wan2.1_t2v_14b_int8_lightx2v_4step.safetensors` | INT8 | 4步模型INT8 量化 |
| `wan2.1_t2v_14b_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | FP8 | 4步模型ComfyUI 格式 |

#### Wan2.2 单文件模型列表

**图生视频模型（I2V）- A14B 系列**

| 文件名 | 精度 | 说明 |
|--------|------|------|
| `wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors` | BF16 | 高噪声模型-4步原始精度 |
| `wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 高噪声模型-4步FP8量化 |
| `wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors` | INT8 | 高噪声模型-4步INT8量化 |
| `wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors` | BF16 | 低噪声模型-4步原始精度 |
| `wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 低噪声模型-4步FP8量化 |
| `wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors` | INT8 | 低噪声模型-4步INT8量化 |

> 💡 **使用提示**：
> - Wan2.2 模型采用双噪声架构，需要同时下载高噪声（high_noise）和低噪声（low_noise）模型
> - 详细的目录组织方式请参考上方"Wan2.2 单文件模型"部分

---

## 🗂️ 格式三：LightX2V LoRA 模型

LoRA（Low-Rank Adaptation）模型提供了一种轻量级的模型微调方案，可以在不修改基础模型的情况下实现特定效果的定制化。

### 模型仓库

- **Wan2.1 LoRA 模型**：[lightx2v/Wan2.1-Distill-Loras](https://huggingface.co/lightx2v/Wan2.1-Distill-Loras)
- **Wan2.2 LoRA 模型**：[lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)

### 使用方式

#### 方式一：离线合并

将 LoRA 权重离线合并到基础模型中，生成新的完整模型文件。

**操作步骤**：

参考 [模型转换文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md) 进行离线合并。

**优点**：
- ✅ 推理时无需额外加载 LoRA
- ✅ 性能更优

**缺点**：
- ❌ 需要额外存储空间
- ❌ 切换不同 LoRA 需要重新合并

#### 方式二：在线加载

在推理时动态加载 LoRA 权重，无需修改基础模型。

**LoRA 应用原理**：

```python
# LoRA 权重应用公式
# lora_scale = (alpha / rank)
# W' = W + lora_scale * B @ A
# 其中：B = up_proj (out_features, rank)
#      A = down_proj (rank, in_features)

if weights_dict["alpha"] is not None:
    lora_scale = weights_dict["alpha"] / lora_down.shape[0]
elif alpha is not None:
    lora_scale = alpha / lora_down.shape[0]
else:
    lora_scale = 1.0
```

**配置方法**：

**Wan2.1 LoRA 配置**：

```json
{
  "lora_configs": [
    {
      "path": "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors",
      "strength": 1.0,
      "alpha": null
    }
  ]
}
```

**Wan2.2 LoRA 配置**：

由于 Wan2.2 采用双模型架构（高噪声/低噪声），需要分别为两个模型配置 LoRA：

```json
{
  "lora_configs": [
    {
      "name": "low_noise_model",
      "path": "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step.safetensors",
      "strength": 1.0,
      "alpha": null
    },
    {
      "name": "high_noise_model",
      "path": "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step.safetensors",
      "strength": 1.0,
      "alpha": null
    }
  ]
}
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `path` | LoRA 模型文件路径 | 必填 |
| `strength` | LoRA 强度系数，范围 [0.0, 1.0] | 1.0 |
| `alpha` | LoRA 缩放因子，`null` 时使用模型内置值 | null |
| `name` | （仅 Wan2.2）指定应用到哪个模型 | 必填 |

**优点**：
- ✅ 灵活切换不同 LoRA
- ✅ 节省存储空间
- ✅ 可动态调整 LoRA 强度

**缺点**：
- ❌ 推理时需额外加载时间
- ❌ 略微增加显存占用

---

## 📚 相关资源

### 官方仓库
- [LightX2V GitHub](https://github.com/ModelTC/LightX2V)
- [LightX2V 单文件模型仓库](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan-AI 官方模型仓库](https://huggingface.co/Wan-AI)

### 模型下载链接

**Wan2.1 系列**
- [Wan2.1 Collection](https://huggingface.co/collections/Wan-AI/wan21-68ac4ba85372ae5a8e282a1b)

**Wan2.2 系列**
- [Wan2.2 Collection](https://huggingface.co/collections/Wan-AI/wan22-68ac4ae80a8b477e79636fc8)

**LightX2V 单文件模型**
- [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

### 文档链接
- [量化技术文档](../method_tutorials/quantization.md)
- [参数卸载文档](../method_tutorials/offload.md)
- [配置文件示例](https://github.com/ModelTC/LightX2V/tree/main/configs)

---

通过本文档，您应该能够：

✅ 理解 LightX2V 支持的所有模型格式
✅ 根据需求选择合适的模型和精度
✅ 正确下载和组织模型文件
✅ 配置启动参数并成功运行推理
✅ 解决常见的模型加载问题

如有其他问题，欢迎在 [GitHub Issues](https://github.com/ModelTC/LightX2V/issues) 中提问。
