# LightX2V Quick Start Guide

Welcome to LightX2V! This guide will help you quickly set up the environment and start using LightX2V for video generation.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Linux Environment Setup](#linux-environment-setup)
  - [Docker Environment (Recommended)](#docker-environment-recommended)
  - [Conda Environment Setup](#conda-environment-setup)
- [Windows Environment Setup](#windows-environment-setup)
- [Inference Usage](#inference-usage)

## 🚀 System Requirements

- **Operating System**: Linux (Ubuntu 18.04+) or Windows 10/11
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support, at least 8GB VRAM
- **Memory**: 16GB or more recommended
- **Storage**: At least 50GB available space

## 🐧 Linux Environment Setup

### 🐳 Docker Environment (Recommended)

We strongly recommend using the Docker environment, which is the simplest and fastest installation method.

#### 1. Pull Image

Visit LightX2V's [Docker Hub](https://hub.docker.com/r/lightx2v/lightx2v/tags), select a tag with the latest date, such as `25111101-cu128`:

```bash
docker pull lightx2v/lightx2v:25111101-cu128
```

We recommend using the `cuda128` environment for faster inference speed. If you need to use the `cuda124` environment, you can use image versions with the `-cu124` suffix:

```bash
docker pull lightx2v/lightx2v:25101501-cu124
```

#### 2. Run Container

```bash
docker run --gpus all -itd --ipc=host --name [container_name] -v [mount_settings] --entrypoint /bin/bash [image_id]
```

#### 3. China Mirror Source (Optional)

For mainland China, if the network is unstable when pulling images, you can pull from Alibaba Cloud:

```bash
# cuda128
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/lightx2v:25111101-cu128

# cuda124
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/lightx2v:25101501-cu124
```

### 🐍 Conda Environment Setup

If you prefer to set up the environment yourself using Conda, please follow these steps:

#### Step 1: Clone Repository

```bash
# Download project code
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
```

#### Step 2: Create Conda Virtual Environment

```bash
# Create and activate conda environment
conda create -n lightx2v python=3.11 -y
conda activate lightx2v
```

#### Step 3: Install Dependencies

```bash
pip install -v -e .
```

#### Step 4: Install Attention Operators

**Option A: Flash Attention 2**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention && python setup.py install
```

**Option B: Flash Attention 3 (for Hopper architecture GPUs)**
```bash
cd flash-attention/hopper && python setup.py install
```

**Option C: SageAttention 2 (Recommended)**
```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention && CUDA_ARCHITECTURES="8.0,8.6,8.9,9.0,12.0" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 pip install -v -e .
```

**Option D: Q8 Kernels**
```bash
git clone https://github.com/KONAKONA666/q8_kernels.git
cd q8_kernels && git submodule init && git submodule update
python setup.py install
```

#### Step 5: Verify Installation

```python
import lightx2v
print(f"LightX2V Version: {lightx2v.__version__}")
```

## 🪟 Windows Environment Setup

Windows systems only support Conda environment setup. Please follow these steps:

### 🐍 Conda Environment Setup

#### Step 1: Check CUDA Version

First, confirm your GPU driver and CUDA version:

```cmd
nvidia-smi
```

Record the **CUDA Version** information in the output, which needs to be consistent in subsequent installations.

#### Step 2: Create Python Environment

```cmd
# Create new environment (Python 3.12 recommended)
conda create -n lightx2v python=3.12 -y

# Activate environment
conda activate lightx2v
```

> 💡 **Note**: Python 3.10 or higher is recommended for best compatibility.

#### Step 3: Install PyTorch Framework

**Method 1: Download Official Wheel Package (Recommended)**

1. Visit the [PyTorch Official Download Page](https://download.pytorch.org/whl/torch/)
2. Select the corresponding version wheel package, paying attention to matching:
   - **Python Version**: Consistent with your environment
   - **CUDA Version**: Matches your GPU driver
   - **Platform**: Select Windows version

**Example (Python 3.12 + PyTorch 2.6 + CUDA 12.4):**

```cmd
# Download and install PyTorch
pip install torch-2.6.0+cu124-cp312-cp312-win_amd64.whl

# Install supporting packages
pip install torchvision==0.21.0 torchaudio==2.6.0
```

**Method 2: Direct Installation via pip**

```cmd
# CUDA 12.4 version example
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: Install Windows Version vLLM

Download the corresponding wheel package from [vllm-windows releases](https://github.com/SystemPanic/vllm-windows/releases).

**Version Matching Requirements:**
- Python version matching
- PyTorch version matching
- CUDA version matching

```cmd
# Install vLLM (please adjust according to actual filename)
pip install vllm-0.9.1+cu124-cp312-cp312-win_amd64.whl
```

#### Step 5: Install Attention Mechanism Operators

**Option A: Flash Attention 2**

```cmd
pip install flash-attn==2.7.2.post1
```

**Option B: SageAttention 2 (Strongly Recommended)**

**Download Sources:**
- [Windows Special Version 1](https://github.com/woct0rdho/SageAttention/releases)
- [Windows Special Version 2](https://github.com/sdbds/SageAttention-for-windows/releases)

```cmd
# Install SageAttention (please adjust according to actual filename)
pip install sageattention-2.1.1+cu126torch2.6.0-cp312-cp312-win_amd64.whl
```

> ⚠️ **Note**: SageAttention's CUDA version doesn't need to be strictly aligned, but Python and PyTorch versions must match.

#### Step 6: Clone Repository

```cmd
# Clone project code
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V

# Install Windows-specific dependencies
pip install -r requirements_win.txt
```

## 🎯 Inference Usage

### 📥 Model Preparation

Before starting inference, you need to download the model files in advance. We recommend:

- **Download Source**: Download models from [LightX2V Official Hugging Face](https://huggingface.co/lightx2v/) or other open-source model repositories
- **Storage Location**: It's recommended to store models on SSD disks for better read performance
- **Available Models**: Including Wan2.1-I2V, Wan2.1-T2V, and other models supporting different resolutions and functionalities

### 📁 Configuration Files and Scripts

The configuration files used for inference are available [here](https://github.com/ModelTC/LightX2V/tree/main/configs), and scripts are available [here](https://github.com/ModelTC/LightX2V/tree/main/scripts).

You need to configure the downloaded model path in the run script. In addition to the input arguments in the script, there are also some necessary parameters in the configuration file specified by `--config_json`. You can modify them as needed.

### 🚀 Start Inference

#### Linux Environment

```bash
# Run after modifying the path in the script
bash scripts/wan/run_wan_t2v.sh
```

#### Windows Environment

```cmd
# Use Windows batch script
scripts\win\run_wan_t2v.bat
```

## 📞 Get Help

If you encounter problems during installation or usage, please:

1. Search for related issues in [GitHub Issues](https://github.com/ModelTC/LightX2V/issues)
2. Submit a new Issue describing your problem

---

🎉 **Congratulations!** You have successfully set up the LightX2V environment and can now start enjoying video generation!
