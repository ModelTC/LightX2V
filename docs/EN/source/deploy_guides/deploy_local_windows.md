# Local Windows Deployment Guide

This document provides detailed instructions for deploying LightX2V locally on Windows environments.

## System Requirements

Before getting started, please ensure your system meets the following requirements:

- **Operating System**: Windows 10/11
- **Graphics Card**: NVIDIA GPU (with CUDA support)
- **VRAM**: At least 8GB VRAM
- **Memory**: At least 16GB RAM
- **Storage**: 20GB+ available disk space
- **Environment Manager**: Anaconda or Miniconda installed
- **Network Tools**: Git (for cloning repositories)

## Deployment Steps

### Step 1: Check CUDA Version

First, verify your GPU driver and CUDA version by running the following command in Command Prompt:

```bash
nvidia-smi
```

Note the **CUDA Version** displayed in the output, as you'll need to match this version during subsequent installations.

### Step 2: Create Python Environment

Create an isolated conda environment, we recommend using Python 3.12:

```bash
# Create new environment (using Python 3.12 as example)
conda create -n lightx2v python=3.12 -y

# Activate environment
conda activate lightx2v
```

> 💡 **Tip**: Python 3.10 or higher is recommended for optimal compatibility.

### Step 3: Install PyTorch Framework

#### Method 1: Download Official Wheel Packages (Recommended)

1. Visit the [PyTorch Official Wheel Download Page](https://download.pytorch.org/whl/torch/)
2. Select the appropriate wheel package, ensuring you match:
   - **Python Version**: Must match your environment (cp312 means Python 3.12)
   - **CUDA Version**: Must match your GPU driver
   - **Platform**: Choose Windows version (win_amd64)

**Example for Python 3.12 + PyTorch 2.6 + CUDA 12.4:**

```
torch-2.6.0+cu124-cp312-cp312-win_amd64.whl
```

After downloading, install the packages:

```bash
# Install PyTorch (replace with actual file path)
pip install torch-2.6.0+cu124-cp312-cp312-win_amd64.whl

# Install accompanying vision and audio packages
pip install torchvision==0.21.0 torchaudio==2.6.0
```

#### Method 2: Direct pip Installation

If you prefer direct installation, use the following command:

```bash
# Example: CUDA 12.4 version
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Windows Version vLLM

Download the corresponding wheel package from the [vllm-windows releases page](https://github.com/SystemPanic/vllm-windows/releases).

**Version Matching Requirements:**
- Python version must match (e.g., cp312)
- PyTorch version must match
- CUDA version must match

**Recommended v0.9.1 Installation:**

```bash
pip install vllm-0.9.1+cu124-cp312-cp312-win_amd64.whl
```

> ⚠️ **Note**: Please select the appropriate wheel package filename based on your specific environment.

### Step 5: Install Attention Mechanism Operators

You can choose to install either Flash Attention 2 or SageAttention 2. **SageAttention 2 is strongly recommended**.

#### Option A: Flash Attention 2

```bash
pip install flash-attn==2.7.2.post1
```

#### Option B: SageAttention 2 (Recommended)

**Download Sources:**
- [Windows Version 1](https://github.com/woct0rdho/SageAttention/releases)
- [Windows Version 2](https://github.com/sdbds/SageAttention-for-windows/releases)

**Version Selection Guidelines:**
- Python version must match
- PyTorch version must match
- **CUDA version can be flexible** (SageAttention doesn't use breaking APIs yet)

**Recommended Installation Version:**

```bash
pip install sageattention-2.1.1+cu126torch2.6.0-cp312-cp312-win_amd64.whl
```

**Verify SageAttention Installation:**

After installation, we recommend running a verification script to ensure proper functionality:

> 📝 **Testing**: You can also run the [official test script](https://github.com/woct0rdho/SageAttention/blob/main/tests/test_sageattn.py) for more detailed functionality verification.

### Step 6: Get LightX2V Project Code

Clone the LightX2V project from GitHub and install Windows-specific dependencies:

```bash
# Clone project code
git clone https://github.com/ModelTC/LightX2V.git

# Enter project directory
cd LightX2V

# Install Windows-specific dependencies
pip install -r requirements_win.txt
```

> 🔍 **Note**: We use `requirements_win.txt` instead of the standard `requirements.txt` because Windows environments may require specific package versions or additional dependencies.


## Troubleshooting

### 1. CUDA Version Mismatch

**Symptoms**: CUDA-related errors occur

**Solutions**:
- Verify GPU driver supports required CUDA version
- Re-download matching wheel packages
- Use `nvidia-smi` to check maximum supported CUDA version

### 2. Dependency Conflicts

**Symptoms**: Package version conflicts or import errors

**Solutions**:
- Remove existing environment: `conda env remove -n lightx2v`
- Recreate environment and install dependencies strictly by version requirements
- Use virtual environments to isolate dependencies for different projects

### 3. Wheel Package Download Issues

**Symptoms**: Slow download speeds or connection failures

**Solutions**:
- Use download tools or browser for direct downloads
- Look for domestic mirror sources
- Check network connections and firewall settings

## Next Steps

After completing the environment setup, you can:

- 📚 Check the [Quick Start Guide](../getting_started/quickstart.md) (skip environment installation steps)
- 🌐 Use the [Gradio Web Interface](./deploy_gradio.md) for visual operations (skip environment installation steps)

## Version Compatibility Reference

| Component | Recommended Version |
|-----------|-------------------|
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| vLLM | 0.9.1+cu124 |
| SageAttention | 2.1.1+cu126torch2.6.0 |
| CUDA | 12.4+ |

---

💡 **Pro Tip**: If you encounter other issues, we recommend first checking whether all component versions match properly, as most problems stem from version incompatibilities.
