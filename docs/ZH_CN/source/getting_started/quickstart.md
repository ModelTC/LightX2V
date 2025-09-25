# LightX2V 快速入门指南

欢迎使用 LightX2V！本指南将帮助您快速搭建环境并开始使用 LightX2V 进行视频生成。

## 📋 目录

- [系统要求](#系统要求)
- [Linux 系统环境搭建](#linux-系统环境搭建)
  - [Docker 环境（推荐）](#docker-环境推荐)
  - [Conda 环境搭建](#conda-环境搭建)
- [Windows 系统环境搭建](#windows-系统环境搭建)
- [推理使用](#推理使用)

## 🚀 系统要求

- **操作系统**: Linux (Ubuntu 18.04+) 或 Windows 10/11
- **Python**: 3.10 或更高版本
- **GPU**: NVIDIA GPU，支持 CUDA，至少 8GB 显存
- **内存**: 建议 16GB 或更多
- **存储**: 至少 50GB 可用空间

## 🐧 Linux 系统环境搭建

### 🐳 Docker 环境（推荐）

我们强烈推荐使用 Docker 环境，这是最简单快捷的安装方式。

#### 1. 拉取镜像

访问 LightX2V 的 [Docker Hub](https://hub.docker.com/r/lightx2v/lightx2v/tags)，选择一个最新日期的 tag，比如 `25092201-cu128`：

```bash
docker pull lightx2v/lightx2v:25092201-cu128
```

我们推荐使用`cuda128`环境，以获得更快的推理速度，若需要使用`cuda124`环境，可以使用带`-cu124`后缀的镜像版本：

```bash
docker pull lightx2v/lightx2v:25092201-cu124
```

#### 2. 运行容器

```bash
docker run --gpus all -itd --ipc=host --name [容器名] -v [挂载设置] --entrypoint /bin/bash [镜像id]
```

#### 3. 中国镜像源（可选）

对于中国大陆地区，如果拉取镜像时网络不稳定，可以从阿里云上拉取：

```bash
# cuda128
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/lightx2v:25092201-cu128

# cuda124
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/lightx2v:25092201-cu124
```

### 🐍 Conda 环境搭建

如果您希望使用 Conda 自行搭建环境，请按照以下步骤操作：

#### 步骤 1: 克隆项目

```bash
# 下载项目代码
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
```

#### 步骤 2: 创建 conda 虚拟环境

```bash
# 创建并激活 conda 环境
conda create -n lightx2v python=3.11 -y
conda activate lightx2v
```

#### 步骤 3: 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt
```

> 💡 **提示**: 混元模型需要在 4.45.2 版本的 transformers 下运行，如果您不需要运行混元模型，可以跳过 transformers 版本限制。

#### 步骤 4: 安装注意力机制算子

**选项 A: Flash Attention 2**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention && python setup.py install
```

**选项 B: Flash Attention 3（用于 Hopper 架构显卡）**
```bash
cd flash-attention/hopper && python setup.py install
```

**选项 C: SageAttention 2（推荐）**
```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention && python setup.py install
```

## 🪟 Windows 系统环境搭建

Windows 系统仅支持 Conda 环境搭建方式，请按照以下步骤操作：

### 🐍 Conda 环境搭建

#### 步骤 1: 检查 CUDA 版本

首先确认您的 GPU 驱动和 CUDA 版本：

```cmd
nvidia-smi
```

记录输出中的 **CUDA Version** 信息，后续安装时需要保持版本一致。

#### 步骤 2: 创建 Python 环境

```cmd
# 创建新环境（推荐 Python 3.12）
conda create -n lightx2v python=3.12 -y

# 激活环境
conda activate lightx2v
```

> 💡 **提示**: 建议使用 Python 3.10 或更高版本以获得最佳兼容性。

#### 步骤 3: 安装 PyTorch 框架

**方法一：下载官方 wheel 包（推荐）**

1. 访问 [PyTorch 官方下载页面](https://download.pytorch.org/whl/torch/)
2. 选择对应版本的 wheel 包，注意匹配：
   - **Python 版本**: 与您的环境一致
   - **CUDA 版本**: 与您的 GPU 驱动匹配
   - **平台**: 选择 Windows 版本

**示例（Python 3.12 + PyTorch 2.6 + CUDA 12.4）：**

```cmd
# 下载并安装 PyTorch
pip install torch-2.6.0+cu124-cp312-cp312-win_amd64.whl

# 安装配套包
pip install torchvision==0.21.0 torchaudio==2.6.0
```

**方法二：使用 pip 直接安装**

```cmd
# CUDA 12.4 版本示例
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### 步骤 4: 安装 Windows 版 vLLM

从 [vllm-windows releases](https://github.com/SystemPanic/vllm-windows/releases) 下载对应的 wheel 包。

**版本匹配要求：**
- Python 版本匹配
- PyTorch 版本匹配
- CUDA 版本匹配

```cmd
# 安装 vLLM（请根据实际文件名调整）
pip install vllm-0.9.1+cu124-cp312-cp312-win_amd64.whl
```

#### 步骤 5: 安装注意力机制算子

**选项 A: Flash Attention 2**

```cmd
pip install flash-attn==2.7.2.post1
```

**选项 B: SageAttention 2（强烈推荐）**

**下载源：**
- [Windows 专用版本 1](https://github.com/woct0rdho/SageAttention/releases)
- [Windows 专用版本 2](https://github.com/sdbds/SageAttention-for-windows/releases)

```cmd
# 安装 SageAttention（请根据实际文件名调整）
pip install sageattention-2.1.1+cu126torch2.6.0-cp312-cp312-win_amd64.whl
```

> ⚠️ **注意**: SageAttention 的 CUDA 版本可以不严格对齐，但 Python 和 PyTorch 版本必须匹配。

#### 步骤 6: 克隆项目

```cmd
# 克隆项目代码
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V

# 安装 Windows 专用依赖
pip install -r requirements_win.txt
```

## 🎯 推理使用

### 📥 模型准备

在开始推理之前，您需要提前下载好模型文件。我们推荐：

- **下载源**: 从 [LightX2V 官方 Hugging Face](https://huggingface.co/lightx2v/)或者其他开源模型库下载模型
- **存储位置**: 建议将模型存储在 SSD 磁盘上以获得更好的读取性能
- **可用模型**: 包括 Wan2.1-I2V、Wan2.1-T2V 等多种模型，支持不同分辨率和功能

### 📁 配置文件与脚本

推理会用到的配置文件都在[这里](https://github.com/ModelTC/LightX2V/tree/main/configs)，脚本都在[这里](https://github.com/ModelTC/LightX2V/tree/main/scripts)。

需要将下载的模型路径配置到运行脚本中。除了脚本中的输入参数，`--config_json` 指向的配置文件中也会包含一些必要参数，您可以根据需要自行修改。

### 🚀 开始推理

#### Linux 环境

```bash
# 修改脚本中的路径后运行
bash scripts/wan/run_wan_t2v.sh
```

#### Windows 环境

```cmd
# 使用 Windows 批处理脚本
scripts\win\run_wan_t2v.bat
```

## 📞 获取帮助

如果您在安装或使用过程中遇到问题，请：

1. 在 [GitHub Issues](https://github.com/ModelTC/LightX2V/issues) 中搜索相关问题
2. 提交新的 Issue 描述您的问题

---

🎉 **恭喜！** 现在您已经成功搭建了 LightX2V 环境，可以开始享受视频生成的乐趣了！
