# 模型结构介绍

## 📖 概述

本文档全面介绍 LightX2V 项目的模型目录结构，旨在帮助用户高效组织模型文件，实现便捷的使用体验。通过科学的目录组织方式，用户可以享受"一键启动"的便利，无需手动配置复杂的路径参数。同时，系统也支持灵活的手动路径配置，满足不同用户群体的多样化需求。

## 🗂️ 模型目录结构

### LightX2V 官方模型列表

查看所有可用模型：[LightX2V 官方模型仓库](https://huggingface.co/lightx2v)

### 标准目录结构

以 `Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V` 为例，标准文件结构如下：

```
Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/
├── fp8/                                          # FP8 量化版本 (DIT/T5/CLIP)
│   ├── block_xx.safetensors                      # DIT 模型 FP8 量化版本
│   ├── models_t5_umt5-xxl-enc-fp8.pth            # T5 编码器 FP8 量化版本
│   ├── clip-fp8.pth                              # CLIP 编码器 FP8 量化版本
│   ├── Wan2.1_VAE.pth                            # VAE 变分自编码器
│   ├── taew2_1.pth                               # 轻量级 VAE (可选)
│   └── config.json                               # 模型配置文件
├── int8/                                         # INT8 量化版本 (DIT/T5/CLIP)
│   ├── block_xx.safetensors                      # DIT 模型 INT8 量化版本
│   ├── models_t5_umt5-xxl-enc-int8.pth           # T5 编码器 INT8 量化版本
│   ├── clip-int8.pth                             # CLIP 编码器 INT8 量化版本
│   ├── Wan2.1_VAE.pth                            # VAE 变分自编码器
│   ├── taew2_1.pth                               # 轻量级 VAE (可选)
│   └── config.json                               # 模型配置文件
├── original/                                     # 原始精度版本 (DIT/T5/CLIP)
│   ├── distill_model.safetensors                 # DIT 模型原始精度版本
│   ├── models_t5_umt5-xxl-enc-bf16.pth           # T5 编码器原始精度版本
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP 编码器原始精度版本
│   ├── Wan2.1_VAE.pth                            # VAE 变分自编码器
│   ├── taew2_1.pth                               # 轻量级 VAE (可选)
│   └── config.json                               # 模型配置文件
```

### 💾 存储建议

**强烈建议将模型文件存储在 SSD 固态硬盘上**，此举可显著提升模型加载速度和推理性能。

**推荐存储路径**：
```bash
/mnt/ssd/models/          # 独立 SSD 挂载点
/data/ssd/models/         # 数据 SSD 目录
/opt/models/              # 系统优化目录
```

### 量化版本说明

每个模型均包含多个量化版本，适配不同硬件配置需求：
- **FP8 版本**：适用于支持 FP8 的 GPU（如 H100、A100、RTX 40系列），提供最佳性能表现
- **INT8 版本**：适用于大多数 GPU，在性能和兼容性间取得平衡，内存占用减少约50%
- **原始精度版本**：适用于对精度要求极高的应用场景，提供最高质量输出

## 🚀 使用方法

### 环境准备

#### 安装 Hugging Face CLI

在开始下载模型之前，请确保已正确安装 Hugging Face CLI：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 或者安装 huggingface-cli
pip install huggingface-cli

# 登录 Hugging Face（可选，但强烈推荐）
huggingface-cli login
```

### 方式一：完整模型下载（推荐）

**优势**：下载完整模型后，系统将自动识别所有组件路径，无需手动配置，使用体验更加便捷

#### 1. 下载完整模型

```bash
# 使用 Hugging Face CLI 下载完整模型
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V
```

#### 2. 启动推理

##### Bash 脚本启动

###### 场景一：使用全精度模型

修改[运行脚本](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_distill_4step_cfg.sh)中的配置：
- `model_path`：设置为下载的模型路径 `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V`
- `lightx2v_path`：设置为 `LightX2V` 项目根目录路径

###### 场景二：使用量化模型

当使用完整模型时，如需启用量化功能，请在[配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/distill/wan_i2v_distill_4step_cfg.json)中添加以下配置：

```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },                              // DIT 模型量化方案
    "t5_quantized": true,           // 启用 T5 量化
    "t5_quant_scheme": "fp8",       // T5 量化模式
    "clip_quantized": true,         // 启用 CLIP 量化
    "clip_quant_scheme": "fp8"      // CLIP 量化模式
}
```

> **重要提示**：各模型的量化配置可以灵活组合。量化路径无需手动指定，系统将自动定位各模型的量化版本。

有关量化技术的详细说明，请参考[量化文档](../method_tutorials/quantization.md)。

使用提供的 bash 脚本快速启动：

```bash
cd LightX2V/scripts
bash wan/run_wan_t2v_distill_4step_cfg.sh
```

##### Gradio 界面启动

通过 Gradio 界面进行推理时，只需在启动时指定模型根目录路径，轻量级 VAE 等可通过前端界面按钮灵活选择：

```bash
# 图像到视频推理 (I2V)
python gradio_demo_zh.py \
    --model_path ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --model_size 14b \
    --task i2v \
    --model_cls wan2.1_distill
```

### 方式二：选择性下载

**优势**：仅下载所需的版本（量化或非量化），有效节省存储空间和下载时间

#### 1. 选择性下载

```bash
# 使用 Hugging Face CLI 选择性下载非量化版本
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --include "original/*"
```

```bash
# 使用 Hugging Face CLI 选择性下载 FP8 量化版本
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --include "fp8/*"
```

```bash
# 使用 Hugging Face CLI 选择性下载 INT8 量化版本
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --include "int8/*"
```

> **重要提示**：当启动推理脚本或Gradio时，`model_path` 参数仍需要指定为不包含 `--include` 的完整路径。例如：`model_path=./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V`，而不是 `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/int8`。

#### 2. 启动推理

**以只下载了FP8版本的模型为例：**

##### Bash 脚本启动

###### 场景一：使用 FP8 DIT + FP8 T5 + FP8 CLIP

将[运行脚本](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_distill_4step_cfg.sh)中的 `model_path` 指定为您下载好的模型路径 `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/`，`lightx2v_path` 指定为您的 `LightX2V` 项目路径。

仅需修改配置文件中的量化模型配置如下：
```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },                              // DIT 的量化方案
    "t5_quantized": true,           // 是否使用 T5 量化版本
    "t5_quant_scheme": "fp8",       // T5 的量化模式
    "clip_quantized": true,         // 是否使用 CLIP 量化版本
    "clip_quant_scheme": "fp8",     // CLIP 的量化模式
}
```

> **重要提示**：此时各模型只能指定为量化版本。量化路径无需手动指定，系统将自动定位各模型的量化版本。

###### 场景二：使用 FP8 DIT + 原始精度 T5 + 原始精度 CLIP

将[运行脚本](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_distill_4step_cfg.sh)中的 `model_path` 指定为您下载好的模型路径 `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V`，`lightx2v_path` 指定为您的 `LightX2V` 项目路径。

由于仅下载了量化权重，需要手动下载 T5 和 CLIP 的原始精度版本，并在配置文件的 `t5_original_ckpt` 和 `clip_original_ckpt` 中配置如下：
```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },                              // DIT 的量化方案
    "t5_original_ckpt": "/path/to/models_t5_umt5-xxl-enc-bf16.pth",
    "clip_original_ckpt": "/path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
}
```

使用提供的 bash 脚本快速启动：

```bash
cd LightX2V/scripts
bash wan/run_wan_t2v_distill_4step_cfg.sh
```

##### Gradio 界面启动

通过 Gradio 界面进行推理时，启动时指定模型根目录路径：

```bash
# 图像到视频推理 (I2V)
python gradio_demo_zh.py \
    --model_path ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/ \
    --model_size 14b \
    --task i2v \
    --model_cls wan2.1_distill
```

> **重要提示**：由于模型根目录下仅包含各模型的量化版本，前端使用时，对于 DIT/T5/CLIP 模型的量化精度只能选择 fp8。如需使用非量化版本的T5/CLIP，请手动下载非量化权重并放置到gradio_demo的model_path目录（`./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/`）下，此时T5/CLIP的量化精度可以选择bf16/fp16。

### 方式三：手动配置

用户可根据实际需求灵活配置各个组件的量化选项和路径，实现量化与非量化组件的混合使用。请确保所需的模型权重已正确下载并放置在指定路径。

#### DIT 模型配置

```json
{
    "dit_quantized_ckpt": "/path/to/dit_quantized_ckpt",    // DIT 量化权重路径
    "dit_original_ckpt": "/path/to/dit_original_ckpt",      // DIT 原始精度权重路径
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"  // DIT 矩阵乘算子类型，非量化时指定为 "Default"
    }
}
```

#### T5 模型配置

```json
{
    "t5_quantized_ckpt": "/path/to/t5_quantized_ckpt",      // T5 量化权重路径
    "t5_original_ckpt": "/path/to/t5_original_ckpt",        // T5 原始精度权重路径
    "t5_quantized": true,                                   // 是否启用 T5 量化
    "t5_quant_scheme": "fp8"                                // T5 量化模式，仅在 t5_quantized 为 true 时生效
}
```

#### CLIP 模型配置

```json
{
    "clip_quantized_ckpt": "/path/to/clip_quantized_ckpt",  // CLIP 量化权重路径
    "clip_original_ckpt": "/path/to/clip_original_ckpt",    // CLIP 原始精度权重路径
    "clip_quantized": true,                                 // 是否启用 CLIP 量化
    "clip_quant_scheme": "fp8"                              // CLIP 量化模式，仅在 clip_quantized 为 true 时生效
}
```

#### VAE 模型配置

```json
{
    "vae_pth": "/path/to/Wan2.1_VAE.pth",                   // 原始 VAE 模型路径
    "use_tiny_vae": true,                                   // 是否使用轻量级 VAE
    "tiny_vae_path": "/path/to/taew2_1.pth"                 // 轻量级 VAE 模型路径
}
```

> **配置说明**：
> - 量化权重和原始精度权重可以灵活混合使用，系统将根据配置自动选择对应的模型
> - 量化模式的选择取决于您的硬件支持情况，建议在 H100/A100 等高端 GPU 上使用 FP8
> - 轻量级 VAE 可以显著提升推理速度，但可能略微影响生成质量

## 💡 最佳实践

### 推荐配置

**完整模型用户**：
- 下载完整模型，享受自动路径查找的便利
- 仅需配置量化方案和组件开关
- 推荐使用 bash 脚本快速启动

**存储空间受限用户**：
- 选择性下载所需的量化版本
- 灵活混合使用量化和原始精度组件
- 使用 bash 脚本简化启动流程

**高级用户**：
- 完全手动配置路径，实现最大灵活性
- 支持模型文件分散存储
- 可自定义 bash 脚本参数

### 性能优化建议

- **使用 SSD 存储**：显著提升模型加载速度和推理性能
- **选择合适的量化方案**：
  - FP8：适用于 H100/A100 等高端 GPU，精度高
  - INT8：适用于通用 GPU，内存占用小
- **启用轻量级 VAE**：`use_tiny_vae: true` 可提升推理速度
- **合理配置 CPU 卸载**：`t5_cpu_offload: true` 可节省 GPU 内存

### 下载优化建议

- **使用 Hugging Face CLI**：比 git clone 更稳定，支持断点续传
- **选择性下载**：仅下载所需的量化版本，节省时间和存储空间
- **网络优化**：使用稳定的网络连接，必要时使用代理
- **断点续传**：使用 `--resume-download` 参数支持中断后继续下载

## 🚨 常见问题

### Q: 模型文件过大，下载速度缓慢怎么办？
A: 建议使用选择性下载方式，仅下载所需的量化版本，或使用国内镜像源

### Q: 启动时提示模型路径不存在？
A: 请检查模型是否已正确下载，验证路径配置是否正确，确认自动查找机制是否正常工作

### Q: 如何切换不同的量化方案？
A: 修改配置文件中的 `mm_type`, `t5_quant_scheme`,`clip_quant_scheme`等参数，请参考[量化文档](../method_tutorials/quantization.md)

### Q: 如何混合使用量化和原始精度组件？
A: 通过 `t5_quantized` 和 `clip_quantized` 参数控制，并手动指定原始精度路径

### Q: 配置文件中的路径如何设置？
A: 推荐使用自动路径查找，如需手动配置请参考"手动配置"部分

### Q: 如何验证自动路径查找是否正常工作？
A: 查看启动日志，代码将输出实际使用的模型路径

### Q: bash 脚本启动失败怎么办？
A: 检查脚本中的路径配置是否正确，确保 `lightx2v_path` 和 `model_path` 变量已正确设置

## 📚 相关链接

- [LightX2V 官方模型仓库](https://huggingface.co/lightx2v)
- [Gradio 部署指南](./deploy_gradio.md)
- [配置文件示例](https://github.com/ModelTC/LightX2V/tree/main/configs)

---

通过科学的模型文件组织和灵活的配置选项，LightX2V 支持多种使用场景。完整模型下载提供最大的便利性，选择性下载节省存储空间，手动配置提供最大的灵活性。自动路径查找机制确保用户无需记忆复杂的路径配置，同时保持系统的可扩展性。
