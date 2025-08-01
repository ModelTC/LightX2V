# Windows Local Deployment Guide

## 📖 Overview

This document provides detailed instructions for deploying LightX2V locally on Windows environments, including batch file inference, Gradio Web interface inference, and other usage methods.

## 🚀 Quick Start

### Environment Requirements

#### Hardware Requirements
- **GPU**: NVIDIA GPU, recommended 8GB+ VRAM
- **Memory**: Recommended 16GB+ RAM
- **Storage**: Strongly recommended to use SSD solid-state drives, mechanical hard drives will cause slow model loading


## 🎯 Usage Methods

### Method 1: Using Batch File Inference

Refer to [Quick Start Guide](../getting_started/quickstart.md) to install environment, and use [batch files](https://github.com/ModelTC/LightX2V/tree/main/scripts/win) to run.

### Method 2: Using Gradio Web Interface Inference

#### Manual Gradio Configuration

Refer to [Quick Start Guide](../getting_started/quickstart.md) to install environment, refer to [Gradio Deployment Guide](./deploy_gradio.md)

#### One-Click Gradio Startup (Recommended)

**📦 Download Software Package**
- [Baidu Cloud](https://pan.baidu.com/s/14bMAbOuFFSj8yzt4Zdmyqw)
- [Quark Cloud](https://pan.quark.cn/s/97b4ad6bc76c)

**📁 Directory Structure**
After extraction, ensure the directory structure is as follows:

```
├── env/                        # LightX2V environment directory
├── LightX2V/                   # LightX2V project directory
├── start_lightx2v.bat          # One-click startup script
├── lightx2v_config.txt         # Configuration file
├── LightX2V使用说明.txt         # LightX2V usage instructions
├── outputs/                    # Generated video save directory
└── models/                     # Model storage directory
    ├── 说明.txt                       # Model documentation
    ├── Wan2.1-I2V-14B-480P-Lightx2v/  # Image-to-video model (480P)
    ├── Wan2.1-I2V-14B-720P-Lightx2v/  # Image-to-video model (720P)
    ├── Wan2.1-I2V-14B-480P-StepDistill-CfgDistil-Lightx2v/  # Image-to-video model (4-step distillation, 480P)
    ├── Wan2.1-I2V-14B-720P-StepDistill-CfgDistil-Lightx2v/  # Image-to-video model (4-step distillation, 720P)
    ├── Wan2.1-T2V-1.3B-Lightx2v/      # Text-to-video model (1.3B parameters)
    ├── Wan2.1-T2V-14B-Lightx2v/       # Text-to-video model (14B parameters)
    └── Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/      # Text-to-video model (4-step distillation)
```

**📥 Model Download**:

Refer to the [Model Structure Documentation](./model_structure.md) to download complete models (including quantized and non-quantized versions) or download only quantized/non-quantized versions.

**Download Options**:

- **Complete Model**: When downloading complete models with both quantized and non-quantized versions, you can freely choose the quantization precision for DIT/T5/CLIP in the advanced options of the `Gradio` Web frontend.

- **Non-quantized Version Only**: When downloading only non-quantized versions, in the `Gradio` Web frontend, the quantization precision for `DIT/T5/CLIP` can only be set to bf16/fp16. If you need to use quantized versions of models, please manually download quantized weights to the `i2v_model_path` or `t2v_model_path` directory where Gradio is started.

- **Quantized Version Only**: When downloading only quantized versions, in the `Gradio` Web frontend, the quantization precision for `DIT/T5/CLIP` can only be set to fp8 or int8 (depending on the weights you downloaded). If you need to use non-quantized versions of models, please manually download non-quantized weights to the `i2v_model_path` or `t2v_model_path` directory where Gradio is started.

- **Note**: Whether you download complete models or partial models, the values for `i2v_model_path` and `t2v_model_path` parameters should be the first-level directory paths. For example: `Wan2.1-I2V-14B-480P-Lightx2v/`, not `Wan2.1-I2V-14B-480P-Lightx2v/int8`.

**📋 Configuration Parameters**

Edit the `lightx2v_config.txt` file and modify the following parameters as needed:

```ini
# Task type (i2v: image-to-video, t2v: text-to-video)
task=i2v

# Interface language (zh: Chinese, en: English)
lang=en

# Server port
port=8032

# GPU device ID (0, 1, 2...)
gpu=0

# Model size (14b: 14B parameter model, 1.3b: 1.3B parameter model)
model_size=14b

# Model class (wan2.1: standard model, wan2.1_distill: distilled model)
model_cls=wan2.1

# Image-to-video model path
i2v_model_path=models/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v

# Text-to-video model path
t2v_model_path=models/Wan2.1-T2V-1.3B-Lightx2v
```

**⚠️ Important Note**: If using distilled models (model names containing StepDistill-CfgDistil field), please set `model_cls` to `wan2.1_distill`

**🚀 Start Service**

Double-click to run the `start_lightx2v.bat` file, the script will:
1. Automatically read configuration file
2. Verify model paths and file integrity
3. Start Gradio Web interface
4. Automatically open browser to access service

**💡 Usage Suggestion**: After opening the Gradio Web page, it's recommended to check "Auto-configure Inference Options", the system will automatically select appropriate optimization configurations for your machine. When reselecting resolution, you also need to re-check "Auto-configure Inference Options".

![Gradio English Interface](../../../../assets/figs/portabl_windows/pic_gradio_en.png)

**⚠️ Important Notes**:
- **First Run**: The system will automatically extract the environment file `env.zip`, which may take several minutes. Please be patient. Subsequent launches will skip this step.
- **Faster Startup**: You can also manually extract the `env.zip` file to the current directory to save time on first startup.
- **Display Issues**: If the webpage opens blank or displays abnormally, please run `pip install --upgrade gradio` to upgrade the Gradio version.

### Method 3: Using ComfyUI Inference

This guide will instruct you on how to download and use the portable version of the Lightx2v-ComfyUI environment, so you can avoid manual environment configuration steps. This is suitable for users who want to quickly start experiencing accelerated video generation with Lightx2v on Windows systems.

#### Download the Windows Portable Environment:

- [Baidu Cloud Download](https://pan.baidu.com/s/1FVlicTXjmXJA1tAVvNCrBw?pwd=wfid), extraction code: wfid

The portable environment already packages all Python runtime dependencies, including the code and dependencies for ComfyUI and LightX2V. After downloading, simply extract to use.

After extraction, the directory structure is as follows:

```shell
lightx2v_env
├──📂 ComfyUI                    # ComfyUI code
├──📂 portable_python312_embed   # Standalone Python environment
└── run_nvidia_gpu.bat            # Windows startup script (double-click to start)
```

#### Start ComfyUI

Directly double-click the run_nvidia_gpu.bat file. The system will open a Command Prompt window and run the program. The first startup may take a while, please be patient. After startup is complete, the browser will automatically open and display the ComfyUI frontend interface.

![i2v example workflow](../../../../assets/figs/portabl_windows/pic1.png)

The plugin used by LightX2V-ComfyUI is [ComfyUI-Lightx2vWrapper](https://github.com/ModelTC/ComfyUI-Lightx2vWrapper). Example workflows can be obtained from this project.

#### Tested Graphics Cards (offload mode)

- Tested model: `Wan2.1-I2V-14B-480P`

| GPU Model   | Task Type   | VRAM Capacity | Actual Max VRAM Usage | Actual Max RAM Usage |
|:-----------|:------------|:--------------|:---------------------|:---------------------|
| 3090Ti     | I2V         | 24G           | 6.1G                 | 7.1G                 |
| 3080Ti     | I2V         | 12G           | 6.1G                 | 7.1G                 |
| 3060Ti     | I2V         | 8G            | 6.1G                 | 7.1G                 |
| 4070Ti Super    | I2V        | 16G         | 6.1G        | 7.1G          |
| 4070    | I2V        | 12G         | 6.1G        | 7.1G          |
| 4060    | I2V        | 8G         | 6.1G        | 7.1G          |

#### Environment Packaging and Usage Reference
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Portable-Windows-ComfyUI-Docs](https://docs.comfy.org/zh-CN/installation/comfyui_portable_windows#portable-%E5%8F%8A%E8%87%AA%E9%83%A8%E7%BD%B2)
