# Gradio Deployment Guide

## 📖 Overview

Lightx2v is a lightweight video inference and generation engine that provides a web interface based on Gradio, supporting both Image-to-Video and Text-to-Video generation modes.

For Windows systems, we provide a convenient one-click deployment solution with automatic environment configuration and intelligent parameter optimization. Please refer to the [One-Click Gradio Startup (Recommended)](./deploy_local_windows.md/#one-click-gradio-startup-recommended) section for detailed instructions.

![Gradio English Interface](../../../../assets/figs/portabl_windows/pic_gradio_en.png)

## 📁 File Structure

```
LightX2V/app/
├── gradio_demo.py          # English interface demo
├── gradio_demo_zh.py       # Chinese interface demo
├── run_gradio.sh          # Startup script
├── README.md              # Documentation
├── outputs/               # Generated video save directory
└── inference_logs.log     # Inference logs
```

This project contains two main demo files:
- `gradio_demo.py` - English interface version
- `gradio_demo_zh.py` - Chinese interface version

## 🚀 Quick Start

### Environment Requirements

Follow the [Quick Start Guide](../getting_started/quickstart.md) to install the environment

#### Recommended Optimization Library Configuration

- ✅ [Flash attention](https://github.com/Dao-AILab/flash-attention)
- ✅ [Sage attention](https://github.com/thu-ml/SageAttention)
- ✅ [vllm-kernel](https://github.com/vllm-project/vllm)
- ✅ [sglang-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)
- ✅ [q8-kernel](https://github.com/KONAKONA666/q8_kernels) (only supports ADA architecture GPUs)

Install according to the project homepage tutorials for each operator as needed.

### 📥 Model Download

Refer to the [Model Structure Documentation](../getting_started/model_structure.md) to download complete models (including quantized and non-quantized versions) or download only quantized/non-quantized versions.

#### wan2.1 Model Directory Structure

```
models/
├── wan2.1_i2v_720p_lightx2v_4step.safetensors                   # Original precision
├── wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors   # FP8 quantization
├── wan2.1_i2v_720p_int8_lightx2v_4step.safetensors              # INT8 quantization
├── wan2.1_i2v_720p_int8_lightx2v_4step_split                    # INT8 quantization block storage directory
├── wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step_split         # FP8 quantization block storage directory
├── Other weights (e.g., t2v)
├── t5/clip/xlm-roberta-large/google    # text and image encoder
├── vae/lightvae/lighttae               # vae
└── config.json                         # Model configuration file
```

#### wan2.2 Model Directory Structure

```
models/
├── wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030.safetensors        # high noise original precision
├── wan2.2_i2v_A14b_high_noise_fp8_e4m3_lightx2v_4step_1030.safetensors    # high noise FP8 quantization
├── wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030.safetensors   # high noise INT8 quantization
├── wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split         # high noise INT8 quantization block storage directory
├── wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors         # low noise original precision
├── wan2.2_i2v_A14b_low_noise_fp8_e4m3_lightx2v_4step.safetensors     # low noise FP8 quantization
├── wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors    # low noise INT8 quantization
├── wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step_split          # low noise INT8 quantization block storage directory
├── t5/clip/xlm-roberta-large/google    # text and image encoder
├── vae/lightvae/lighttae               # vae
└── config.json                         # Model configuration file
```

**📝 Download Instructions**:

- Model weights can be downloaded from HuggingFace:
  - [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
  - [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- Text and Image Encoders can be downloaded from [Encoders](https://huggingface.co/lightx2v/Encoderss)
- VAE can be downloaded from [Autoencoders](https://huggingface.co/lightx2v/Autoencoders)
- For `xxx_split` directories (e.g., `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step_split`), which store multiple safetensors by block, suitable for devices with insufficient memory. For example, devices with 16GB or less memory should download according to their own situation.

### Startup Methods

#### Method 1: Using Startup Script (Recommended)

**Linux Environment:**
```bash
# 1. Edit the startup script to configure relevant paths
cd app/
vim run_gradio.sh

# Configuration items that need to be modified:
# - lightx2v_path: Lightx2v project root directory path
# - model_path: Model root directory path (contains all model files)

# 💾 Important note: Recommend pointing model paths to SSD storage locations
# Example: /mnt/ssd/models/ or /data/ssd/models/

# 2. Run the startup script
bash run_gradio.sh

# 3. Or start with parameters
bash run_gradio.sh --lang en --port 8032
bash run_gradio.sh --lang zh --port 7862
```

**Windows Environment:**
```cmd
# 1. Edit the startup script to configure relevant paths
cd app\
notepad run_gradio_win.bat

# Configuration items that need to be modified:
# - lightx2v_path: Lightx2v project root directory path
# - model_path: Model root directory path (contains all model files)

# 💾 Important note: Recommend pointing model paths to SSD storage locations
# Example: D:\models\ or E:\models\

# 2. Run the startup script
run_gradio_win.bat

# 3. Or start with parameters
run_gradio_win.bat --lang en --port 8032
run_gradio_win.bat --lang zh --port 7862
```

#### Method 2: Direct Command Line Startup

```bash
pip install -v git+https://github.com/ModelTC/LightX2V.git
```

**Linux Environment:**

**English Interface Version:**
```bash
python gradio_demo.py \
    --model_path /path/to/models \
    --server_name 0.0.0.0 \
    --server_port 7862
```

**Chinese Interface Version:**
```bash
python gradio_demo_zh.py \
    --model_path /path/to/models \
    --server_name 0.0.0.0 \
    --server_port 7862
```

**Windows Environment:**

**English Interface Version:**
```cmd
python gradio_demo.py ^
    --model_path D:\models ^
    --server_name 127.0.0.1 ^
    --server_port 7862
```

**Chinese Interface Version:**
```cmd
python gradio_demo_zh.py ^
    --model_path D:\models ^
    --server_name 127.0.0.1 ^
    --server_port 7862
```

**💡 Tip**: Model type (wan2.1/wan2.2), task type (i2v/t2v), and specific model file selection are all configured in the Web interface.

## 📋 Command Line Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--model_path` | str | ✅ | - | Model root directory path (directory containing all model files) |
| `--server_port` | int | ❌ | 7862 | Server port |
| `--server_name` | str | ❌ | 0.0.0.0 | Server IP address |
| `--output_dir` | str | ❌ | ./outputs | Output video save directory |

**💡 Note**: Model type (wan2.1/wan2.2), task type (i2v/t2v), and specific model file selection are all configured in the Web interface.

## 🎯 Features

### Model Configuration

- **Model Type**: Supports wan2.1 and wan2.2 model architectures
- **Task Type**: Supports Image-to-Video (i2v) and Text-to-Video (t2v) generation modes
- **Model Selection**: Frontend automatically identifies and filters available model files, supports automatic quantization precision detection
- **Encoder Configuration**: Supports selection of T5 text encoder, CLIP image encoder, and VAE decoder
- **Operator Selection**: Supports multiple attention operators and quantization matrix multiplication operators, system automatically sorts by installation status

### Input Parameters

- **Prompt**: Describe the expected video content
- **Negative Prompt**: Specify elements you don't want to appear
- **Input Image**: Upload input image required in i2v mode
- **Resolution**: Supports multiple preset resolutions (480p/540p/720p)
- **Random Seed**: Controls the randomness of generation results
- **Inference Steps**: Affects the balance between generation quality and speed (defaults to 4 steps for distilled models)

### Video Parameters

- **FPS**: Frames per second
- **Total Frames**: Video length
- **CFG Scale Factor**: Controls prompt influence strength (1-10, defaults to 1 for distilled models)
- **Distribution Shift**: Controls generation style deviation degree (0-10)

## 🔧 Auto-Configuration Feature

The system automatically configures optimal inference options based on your hardware configuration (GPU VRAM and CPU memory) without manual adjustment. The best configuration is automatically applied on startup, including:

- **GPU Memory Optimization**: Automatically enables CPU offloading, VAE tiling inference, etc. based on VRAM size
- **CPU Memory Optimization**: Automatically enables lazy loading, module unloading, etc. based on system memory
- **Operator Selection**: Automatically selects the best installed operators (sorted by priority)
- **Quantization Configuration**: Automatically detects and applies quantization precision based on model file names


### Log Viewing

```bash
# View inference logs
tail -f inference_logs.log

# View GPU usage
nvidia-smi

# View system resources
htop
```

Welcome to submit Issues and Pull Requests to improve this project!

**Note**: Please comply with relevant laws and regulations when using videos generated by this tool, and do not use them for illegal purposes.
