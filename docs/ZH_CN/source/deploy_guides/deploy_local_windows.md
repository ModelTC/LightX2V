# Windows 本地部署指南

## 📖 概述

本文档将详细指导您在Windows环境下完成LightX2V的本地部署配置，包括批处理文件推理、Gradio Web界面推理等多种使用方式。

## 🚀 快速开始

### 环境要求

#### 硬件要求
- **GPU**: NVIDIA GPU，建议 8GB+ VRAM
- **内存**: 建议 16GB+ RAM
- **存储**: 强烈建议使用 SSD 固态硬盘，机械硬盘会导致模型加载缓慢

## 🎯 使用方式

### 方式一：使用批处理文件推理

参考[快速开始文档](../getting_started/quickstart.md)安装环境，并使用[批处理文件](https://github.com/ModelTC/LightX2V/tree/main/scripts/win)运行。

### 方式二：使用Gradio Web界面推理

#### 手动配置Gradio

参考[快速开始文档](../getting_started/quickstart.md)安装环境，参考[Gradio部署指南](./deploy_gradio.md)

#### 一键启动Gradio（推荐）

**📦 下载软件包**
- [百度云](https://pan.baidu.com/s/1ef3hEXyIuO0z6z9MoXe4nQ?pwd=7g4f)
- [夸克网盘](https://pan.quark.cn/s/36a0cdbde7d9)

**📁 目录结构**
解压后，确保目录结构如下：

```
├── env/                        # LightX2V 环境目录
├── LightX2V/                   # LightX2V 项目目录
├── start_lightx2v.bat          # 一键启动脚本
├── lightx2v_config.txt         # 配置文件
├── LightX2V使用说明.txt         # LightX2V使用说明
└── models/                     # 模型存放目录
    ├── 说明.txt                       # 模型说明文档
    ├── Wan2.1-I2V-14B-480P-Lightx2v/  # 图像转视频模型（480P）
    ├── Wan2.1-I2V-14B-720P-Lightx2v/  # 图像转视频模型（720P）
    ├── Wan2.1-I2V-14B-480P-StepDistill-CfgDistil-Lightx2v/  # 图像转视频模型（4步蒸馏，480P）
    ├── Wan2.1-I2V-14B-720P-StepDistill-CfgDistil-Lightx2v/  # 图像转视频模型（4步蒸馏，720P）
    ├── Wan2.1-T2V-1.3B-Lightx2v/      # 文本转视频模型（1.3B参数）
    ├── Wan2.1-T2V-14B-Lightx2v/       # 文本转视频模型（14B参数）
    └── Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/      # 文本转视频模型（4步蒸馏）
```

**📋 配置参数**

编辑 `lightx2v_config.txt` 文件，根据需要修改以下参数：

```ini
# 任务类型 (i2v: 图像转视频, t2v: 文本转视频)
task=i2v

# 界面语言 (zh: 中文, en: 英文)
lang=zh

# 服务器端口
port=8032

# GPU设备ID (0, 1, 2...)
gpu=0

# 模型大小 (14b: 14B参数模型, 1.3b: 1.3B参数模型)
model_size=14b

# 模型类别 (wan2.1: 标准模型, wan2.1_distill: 蒸馏模型)
model_cls=wan2.1
```

**⚠️ 重要提示**: 如果使用蒸馏模型（模型名称包含StepDistill-CfgDistil字段），请将`model_cls`设置为`wan2.1_distill`

**🚀 启动服务**

双击运行 `start_lightx2v.bat` 文件，脚本将：
1. 自动读取配置文件
2. 验证模型路径和文件完整性
3. 启动 Gradio Web 界面
4. 自动打开浏览器访问服务

**💡 使用建议**: 当打开Gradio Web页面后，建议勾选"自动配置推理选项"，系统会自动选择合适的优化配置针对您的机器。当重新选择分辨率后，也需要重新勾选"自动配置推理选项"。

**⚠️ 重要提示**: 首次运行时会自动解压环境文件 `env.zip`，此过程需要几分钟时间，请耐心等待。后续启动无需重复此步骤。您也可以手动解压 `env.zip` 文件到当前目录以节省首次启动时间。

### 方式三：使用ComfyUI推理

TODO - 待补充ComfyUI集成指南
