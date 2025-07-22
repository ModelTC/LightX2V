# Gradio Deployment Guide

## 📖 Overview

Lightx2v is a lightweight video inference and generation engine that provides a web interface based on Gradio, supporting both Image-to-Video and Text-to-Video generation modes.

## 📁 File Structure

```
LightX2V/app/
├── gradio_demo.py          # English interface demo
├── gradio_demo_zh.py       # Chinese interface demo
├── run_gradio.sh          # Startup script
├── README.md              # Documentation
├── saved_videos/          # Generated video save directory
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

Install according to the project homepage tutorials for each operator as needed

### 🤖 Supported Models

#### 🎬 Image-to-Video Models

| Model Name | Resolution | Parameters | Features | Recommended Use |
|------------|------------|------------|----------|-----------------|
| ✅ [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v) | 480p | 14B | Standard version | Balance speed and quality |
| ✅ [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v) | 720p | 14B | HD version | Pursue high-quality output |
| ✅ [Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v) | 480p | 14B | Distilled optimized version | Faster inference speed |
| ✅ [Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v) | 720p | 14B | HD distilled version | High quality + fast inference |

#### 📝 Text-to-Video Models

| Model Name | Parameters | Features | Recommended Use |
|------------|------------|----------|-----------------|
| ✅ [Wan2.1-T2V-1.3B-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-Lightx2v) | 1.3B | Lightweight | Fast prototyping and testing |
| ✅ [Wan2.1-T2V-14B-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-Lightx2v) | 14B | Standard version | Balance speed and quality |
| ✅ [Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) | 14B | Distilled optimized version | High quality + fast inference |

**💡 Model Selection Recommendations**:
- **First-time use**: Recommend choosing distilled versions (`wan2.1_distill`)
- **Pursuing quality**: Choose 720p resolution or 14B parameter models
- **Pursuing speed**: Choose 480p resolution or 1.3B parameter models, prioritize distilled versions
- **Resource-constrained**: Prioritize distilled versions and lower resolutions
- **Real-time applications**: Strongly recommend using distilled models (`wan2.1_distill`)

**🎯 Model Category Description**:
- **`wan2.1`**: Standard model, provides the best video generation quality, suitable for scenarios with extremely high quality requirements
- **`wan2.1_distill`**: Distilled model, optimized through knowledge distillation technology, significantly improves inference speed, maintains good quality while greatly reducing computation time, suitable for most application scenarios

### Startup Methods

#### Method 1: Using Startup Script (Recommended)

**Linux Environment:**
```bash
# 1. Edit the startup script to configure relevant paths
cd app/
vim run_gradio.sh

# Configuration items that need to be modified:
# - lightx2v_path: Lightx2v project root directory path
# - i2v_model_path: Image-to-video model path
# - t2v_model_path: Text-to-video model path

# 💾 Important note: Recommend pointing model paths to SSD storage locations
# Example: /mnt/ssd/models/ or /data/ssd/models/

# 2. Run the startup script
bash run_gradio.sh

# 3. Or start with parameters (recommended using distilled models)
bash run_gradio.sh --task i2v --lang en --model_cls wan2.1 --model_size 14b --port 8032
bash run_gradio.sh --task t2v --lang en --model_cls wan2.1 --model_size 1.3b --port 8032
bash run_gradio.sh --task i2v --lang en --model_cls wan2.1_distill --model_size 14b --port 8032
bash run_gradio.sh --task t2v --lang en --model_cls wan2.1_distill --model_size 1.3b --port 8032
```

**Windows Environment:**
```cmd
# 1. Edit the startup script to configure relevant paths
cd app\
notepad run_gradio_win.bat

# Configuration items that need to be modified:
# - lightx2v_path: Lightx2v project root directory path
# - i2v_model_path: Image-to-video model path
# - t2v_model_path: Text-to-video model path

# 💾 Important note: Recommend pointing model paths to SSD storage locations
# Example: D:\models\ or E:\models\

# 2. Run the startup script
run_gradio_win.bat

# 3. Or start with parameters (recommended using distilled models)
run_gradio_win.bat --task i2v --lang en --model_cls wan2.1 --model_size 14b --port 8032
run_gradio_win.bat --task t2v --lang en --model_cls wan2.1 --model_size 1.3b --port 8032
run_gradio_win.bat --task i2v --lang en --model_cls wan2.1_distill --model_size 14b --port 8032
run_gradio_win.bat --task t2v --lang en --model_cls wan2.1_distill --model_size 1.3b --port 8032
```

#### Method 2: Direct Command Line Startup

**Linux Environment:**

**Image-to-Video Mode:**
```bash
python gradio_demo.py \
    --model_path /path/to/Wan2.1-I2V-14B-480P-Lightx2v \
    --model_cls wan2.1 \
    --model_size 14b \
    --task i2v \
    --server_name 0.0.0.0 \
    --server_port 7862
```

**English Interface Version:**
```bash
python gradio_demo.py \
    --model_path /path/to/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v \
    --model_cls wan2.1_distill \
    --model_size 14b \
    --task t2v \
    --server_name 0.0.0.0 \
    --server_port 7862
```

**Windows Environment:**

**Image-to-Video Mode:**
```cmd
python gradio_demo.py ^
    --model_path D:\models\Wan2.1-I2V-14B-480P-Lightx2v ^
    --model_cls wan2.1 ^
    --model_size 14b ^
    --task i2v ^
    --server_name 127.0.0.1 ^
    --server_port 7862
```

**English Interface Version:**
```cmd
python gradio_demo.py ^
    --model_path D:\models\Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v ^
    --model_cls wan2.1_distill ^
    --model_size 14b ^
    --task t2v ^
    --server_name 127.0.0.1 ^
    --server_port 7862
```

## 📋 Command Line Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--model_path` | str | ✅ | - | Model folder path |
| `--model_cls` | str | ❌ | wan2.1 | Model class: `wan2.1` (standard model) or `wan2.1_distill` (distilled model, faster inference) |
| `--model_size` | str | ✅ | - | Model size: `14b (image-to-video or text-to-video)` or `1.3b (text-to-video)` |
| `--task` | str | ✅ | - | Task type: `i2v` (image-to-video) or `t2v` (text-to-video) |
| `--server_port` | int | ❌ | 7862 | Server port |
| `--server_name` | str | ❌ | 0.0.0.0 | Server IP address |

## 🎯 Features

### Basic Settings

#### Input Parameters
- **Prompt**: Describe the expected video content
- **Negative Prompt**: Specify elements you don't want to appear
- **Resolution**: Supports multiple preset resolutions (480p/540p/720p)
- **Random Seed**: Controls the randomness of generation results
- **Inference Steps**: Affects the balance between generation quality and speed

#### Video Parameters
- **FPS**: Frames per second
- **Total Frames**: Video length
- **CFG Scale Factor**: Controls prompt influence strength (1-10)
- **Distribution Shift**: Controls generation style deviation degree (0-10)

### Advanced Optimization Options

#### GPU Memory Optimization
- **Chunked Rotary Position Embedding**: Saves GPU memory
- **Rotary Embedding Chunk Size**: Controls chunk granularity
- **Clean CUDA Cache**: Promptly frees GPU memory

#### Asynchronous Offloading
- **CPU Offloading**: Transfers partial computation to CPU
- **Lazy Loading**: Loads model components on-demand, significantly reduces system memory consumption
- **Offload Granularity Control**: Fine-grained control of offloading strategies

#### Low-Precision Quantization
- **Attention Operators**: Flash Attention, Sage Attention, etc.
- **Quantization Operators**: vLLM, SGL, Q8F, etc.
- **Precision Modes**: FP8, INT8, BF16, etc.

#### VAE Optimization
- **Lightweight VAE**: Accelerates decoding process
- **VAE Tiling Inference**: Reduces memory usage

#### Feature Caching
- **Tea Cache**: Caches intermediate features to accelerate generation
- **Cache Threshold**: Controls cache trigger conditions
- **Key Step Caching**: Writes cache only at key steps

## 🔧 Auto-Configuration Feature

After enabling "Auto-configure Inference Options", the system will automatically optimize parameters based on your hardware configuration:

### GPU Memory Rules
- **80GB+**: Default configuration, no optimization needed
- **48GB**: Enable CPU offloading, offload ratio 50%
- **40GB**: Enable CPU offloading, offload ratio 80%
- **32GB**: Enable CPU offloading, offload ratio 100%
- **24GB**: Enable BF16 precision, VAE tiling
- **16GB**: Enable chunked offloading, rotary embedding chunking
- **12GB**: Enable cache cleaning, lightweight VAE
- **8GB**: Enable quantization, lazy loading

### CPU Memory Rules
- **128GB+**: Default configuration
- **64GB**: Enable DIT quantization
- **32GB**: Enable lazy loading
- **16GB**: Enable full model quantization

## ⚠️ Important Notes

### 🚀 Low-Resource Device Optimization Recommendations

**💡 For devices with insufficient VRAM or performance constraints**:

- **🎯 Model Selection**: Prioritize using distilled version models (`wan2.1_distill`)
- **⚡ Inference Steps**: Recommend setting to 4 steps
- **🔧 CFG Settings**: Recommend disabling CFG option to improve generation speed
- **🔄 Auto-Configuration**: Enable "Auto-configure Inference Options"
- **💾 Storage Optimization**: Ensure models are stored on SSD for optimal loading performance

## 🎨 Interface Description

### Basic Settings Tab
- **Input Parameters**: Prompts, resolution, and other basic settings
- **Video Parameters**: FPS, frame count, CFG, and other video generation parameters
- **Output Settings**: Video save path configuration

### Advanced Options Tab
- **GPU Memory Optimization**: Memory management related options
- **Asynchronous Offloading**: CPU offloading and lazy loading
- **Low-Precision Quantization**: Various quantization optimization options
- **VAE Optimization**: Variational Autoencoder optimization
- **Feature Caching**: Cache strategy configuration

## 🔍 Troubleshooting

### Common Issues

**💡 Tip**: Generally, after enabling "Auto-configure Inference Options", the system will automatically optimize parameter settings based on your hardware configuration, and performance issues usually won't occur. If you encounter problems, please refer to the following solutions:

1. **CUDA Memory Insufficient**
   - Enable CPU offloading
   - Reduce resolution
   - Enable quantization options

2. **System Memory Insufficient**
   - Enable CPU offloading
   - Enable lazy loading option
   - Enable quantization options

3. **Slow Generation Speed**
   - Reduce inference steps
   - Enable auto-configuration
   - Use lightweight models
   - Enable Tea Cache
   - Use quantization operators
   - 💾 **Check if models are stored on SSD**

4. **Slow Model Loading**
   - 💾 **Migrate models to SSD storage**
   - Enable lazy loading option
   - Check disk I/O performance
   - Consider using NVMe SSD

5. **Poor Video Quality**
   - Increase inference steps
   - Increase CFG scale factor
   - Use 14B models
   - Optimize prompts

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
