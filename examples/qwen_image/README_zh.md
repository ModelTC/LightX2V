# Qwen Image Edit 2511 示例

本目录包含 Qwen Image Edit 2511 模型的使用示例，提供了两种不同的推理方式。

## 测速结果

Dit部分的推理耗时对比(不包含预热时间，数据更新于2025.12.23):

<div align="center">
  <img src="../../assets/figs/qwen/qwen-image-edit-2511.png" alt="Qwen-Image-Edit-2511" width="60%">
</div>

## 安装

首先克隆仓库并安装依赖：

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
pip install -v -e .
```

## 模型下载

在使用示例脚本之前，需要先下载相应的模型。所有模型都可以从以下地址下载：

**模型下载地址：** https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning

请确保将模型下载到正确的目录，以便脚本能够正确加载。

## 使用方法

进入示例目录：

```bash
cd examples/qwen_image/
```

### 方式一：使用步数蒸馏 + FP8 量化模型

运行 `qwen_2511_fp8.py` 脚本，该脚本使用步数蒸馏和 FP8 量化优化的模型：

```bash
python qwen_2511_fp8.py
```

该方式通过步数蒸馏技术减少推理步数，同时使用 FP8 量化降低模型大小和内存占用，实现更快的推理速度。

### 方式二：使用 Qwen-Image-Edit-2511 模型 + 蒸馏 LoRA

运行 `qwen_2511_with_distill_lora.py` 脚本，该脚本使用 Qwen-Image-Edit-2511 基础模型配合蒸馏 LoRA：

```bash
python qwen_2511_with_distill_lora.py
```

该方式使用完整的 Qwen-Image-Edit-2511 模型，并通过蒸馏 LoRA 进行模型优化，在保持模型性能的同时提升推理效率。
