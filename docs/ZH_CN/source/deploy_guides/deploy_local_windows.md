# 本地Windows电脑部署指南

本文档将详细指导您在Windows环境下完成LightX2V的本地部署配置。

## 系统要求

在开始之前，请确保您的系统满足以下要求：

- **操作系统**: Windows 10/11
- **显卡**: NVIDIA GPU（支持CUDA）
- **显存**: 至少8GB显存
- **内存**: 至少16GB内存
- **存储空间**: 20GB以上可用硬盘空间
- **环境管理**: 已安装Anaconda或Miniconda
- **网络工具**: Git（用于克隆代码仓库）

## 部署步骤

### 步骤1：检查CUDA版本

首先确认您的GPU驱动和CUDA版本，在命令提示符中运行：

```bash
nvidia-smi
```

记录输出中显示的**CUDA Version**信息，后续安装时需要保持版本一致。

### 步骤2：创建Python环境

创建一个独立的conda环境，推荐使用Python 3.12：

```bash
# 创建新环境（以Python 3.12为例）
conda create -n lightx2v python=3.12 -y

# 激活环境
conda activate lightx2v
```

> 💡 **提示**: 建议使用Python 3.10或更高版本以获得最佳兼容性。

### 步骤3：安装PyTorch框架

#### 方法一：下载官方wheel包安装（推荐）

1. 访问 [PyTorch官方wheel包下载页面](https://download.pytorch.org/whl/torch/)
2. 选择对应版本的wheel包，注意匹配：
   - **Python版本**: 与您的环境一致（cp312表示Python 3.12）
   - **CUDA版本**: 与您的GPU驱动匹配
   - **平台**: 选择Windows版本（win_amd64）

**以Python 3.12 + PyTorch 2.6 + CUDA 12.4为例：**

```
torch-2.6.0+cu124-cp312-cp312-win_amd64.whl
```

下载完成后进行安装：

```bash
# 安装PyTorch（请替换为实际的文件路径）
pip install torch-2.6.0+cu124-cp312-cp312-win_amd64.whl

# 安装配套的vision和audio包
pip install torchvision==0.21.0 torchaudio==2.6.0
```

#### 方法二：使用pip直接安装

如果您偏好直接安装，可以使用以下命令：

```bash
# 示例：CUDA 12.4版本
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 步骤4：安装Windows版vLLM

从 [vllm-windows releases页面](https://github.com/SystemPanic/vllm-windows/releases) 下载对应的wheel包。

**版本匹配要求：**
- Python版本匹配（如cp312）
- PyTorch版本匹配
- CUDA版本匹配

**推荐安装v0.9.1版本：**

```bash
pip install vllm-0.9.1+cu124-cp312-cp312-win_amd64.whl
```

> ⚠️ **注意**: 请根据您的具体环境选择对应的wheel包文件名。

### 步骤5：安装注意力机制算子

您可以选择安装Flash Attention 2或SageAttention 2，**强烈推荐SageAttention 2**。

#### 选项A：Flash Attention 2

```bash
pip install flash-attn==2.7.2.post1
```

#### 选项B：SageAttention 2（推荐）

**下载源选择：**
- [Windows专用版本1](https://github.com/woct0rdho/SageAttention/releases)
- [Windows专用版本2](https://github.com/sdbds/SageAttention-for-windows/releases)

**版本选择要点：**
- Python版本必须匹配
- PyTorch版本必须匹配
- **CUDA版本可以不严格对齐**（SageAttention暂未使用破坏性API）

**推荐安装版本：**

```bash
pip install sageattention-2.1.1+cu126torch2.6.0-cp312-cp312-win_amd64.whl
```

**验证SageAttention安装：**

> 📝 **测试**: 您也可以运行[测试脚本](https://github.com/woct0rdho/SageAttention/blob/main/tests/test_sageattn.py)进行更详细的功能验证。

### 步骤6：获取LightX2V项目代码

从GitHub克隆LightX2V项目并安装Windows专用依赖：

```bash
# 克隆项目代码
git clone https://github.com/ModelTC/LightX2V.git

# 进入项目目录
cd LightX2V

# 安装Windows专用依赖包
pip install -r requirements_win.txt
```

> 🔍 **说明**: 这里使用`requirements_win.txt`而不是标准的`requirements.txt`，因为Windows环境可能需要特定的包版本或额外的依赖。


## 故障排除

### 1. CUDA版本不匹配

**问题现象**: 出现CUDA相关错误

**解决方案**:
- 确认GPU驱动支持所需CUDA版本
- 重新下载匹配的wheel包
- 可以通过`nvidia-smi`查看支持的最高CUDA版本

### 2. 依赖冲突

**问题现象**: 包版本冲突或导入错误

**解决方案**:
- 删除现有环境: `conda env remove -n lightx2v`
- 重新创建环境并严格按版本要求安装
- 使用虚拟环境隔离不同项目的依赖

### 3. wheel包下载问题

**问题现象**: 下载速度慢或失败

**解决方案**:
- 使用下载工具或浏览器直接下载
- 寻找国内镜像源
- 检查网络连接和防火墙设置


## 下一步操作

环境配置完成后，您可以：

- 📚 查看[快速开始指南](../getting_started/quickstart.md)（跳过环境安装步骤）
- 🌐 使用[Gradio Web界面](./deploy_gradio.md)进行可视化操作（跳过环境安装步骤）

## 版本兼容性参考

| 组件 | 推荐版本 |
|------|----------|
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| vLLM | 0.9.1+cu124 |
| SageAttention | 2.1.1+cu126torch2.6.0 |
| CUDA | 12.4+ |

---

💡 **小贴士**: 如果遇到其他问题，建议先检查各组件版本是否匹配，大部分问题都源于版本不兼容。
