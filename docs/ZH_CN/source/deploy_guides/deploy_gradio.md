# Gradio 部署

## 📖 概述

Lightx2v 是一个轻量级的视频推理和生成引擎，提供了基于 Gradio 的 Web 界面，支持图像到视频（Image-to-Video）和文本到视频（Text-to-Video）两种生成模式。

本项目包含两个主要演示文件：
- `gradio_demo.py` - 英文界面版本
- `gradio_demo_zh.py` - 中文界面版本

## 🚀 快速开始

### 环境要求

- Python 3.10+ (推荐)
- CUDA 12.4+ (推荐)
- 至少 8GB GPU 显存
- 至少 16GB 系统内存（最好最少有 32G）
- 至少 128GB SSD固态硬盘 (**💾 强烈建议使用SSD固态硬盘存储模型文件！"延迟加载"启动时，显著提升模型加载速度和推理性能**)


### 安装依赖☀

```bash
# 安装基础依赖
pip install -r requirements.txt
pip install gradio
```
#### 推荐优化库配置

- ✅ [Flash attention](https://github.com/Dao-AILab/flash-attention)
- ✅ [Sage attention](https://github.com/thu-ml/SageAttention)
- ✅ [vllm-kernel](https://github.com/vllm-project/vllm)
- ✅ [sglang-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)
- ✅ [q8-kernel](https://github.com/KONAKONA666/q8_kernels) (只支持ADA架构的GPU)

### 🤖 支持的模型

#### 🎬 图像到视频模型 (Image-to-Video)

| 模型名称 | 分辨率 | 参数量 | 特点 | 推荐场景 |
|----------|--------|--------|------|----------|
| ✅ [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v) | 480p | 14B | 标准版本 | 平衡速度和质量 |
| ✅ [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v) | 720p | 14B | 高清版本 | 追求高质量输出 |
| ✅ [Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v) | 480p | 14B | 蒸馏优化版 | 更快的推理速度 |
| ✅ [Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v) | 720p | 14B | 高清蒸馏版 | 高质量+快速推理 |

#### 📝 文本到视频模型 (Text-to-Video)

| 模型名称 | 参数量 | 特点 | 推荐场景 |
|----------|--------|------|----------|
| ✅ [Wan2.1-T2V-1.3B-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-Lightx2v) | 1.3B | 轻量级 | 快速原型测试 |
| ✅ [Wan2.1-T2V-14B-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-Lightx2v) | 14B | 标准版本 | 平衡速度和质量 |
| ✅ [Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v) | 14B | 蒸馏优化版 | 高质量+快速推理 |


**💡 模型选择建议**:
- **首次使用**: 建议选择蒸馏版本
- **追求质量**: 选择720p分辨率或14B参数模型
- **追求速度**: 选择480p分辨率或1.3B参数模型
- **资源受限**: 优先选择蒸馏版本和较低分辨率



### 启动方式

#### 方式一：使用启动脚本（推荐）

```bash
# 1. 编辑启动脚本，配置相关路径
cd app/
vim run_gradio.sh

# 需要修改的配置项：
# - lightx2v_path: Lightx2v项目根目录路径
# - i2v_model_path: 图像到视频模型路径
# - t2v_model_path: 文本到视频模型路径

# 💾 重要提示：建议将模型路径指向SSD存储位置
# 例如：/mnt/ssd/models/ 或 /data/ssd/models/

# 2. 运行启动脚本
bash run_gradio.sh

# 3. 或使用参数启动（推荐）
bash run_gradio.sh --task i2v --lang zh --model_size 14b --port 8032
# bash run_gradio.sh --task i2v --lang zh --model_size 14b --port 8032
# bash run_gradio.sh --task i2v --lang zh --model_size 1.3b --port 8032
```

#### 方式二：直接命令行启动

**图像到视频模式：**
```bash
python gradio_demo_zh.py \
    --model_path /path/to/Wan2.1-I2V-14B-720P-Lightx2v \
    --model_size 14b \
    --task i2v \
    --server_name 0.0.0.0 \
    --server_port 7862
```

**文本到视频模式：**
```bash
python gradio_demo_zh.py \
    --model_path /path/to/Wan2.1-T2V-1.3B \
    --model_size 1.3b \
    --task t2v \
    --server_name 0.0.0.0 \
    --server_port 7862
```

**英文界面版本：**
```bash
python gradio_demo.py \
    --model_path /path/to/model \
    --model_size 14b \
    --task i2v \
    --server_name 0.0.0.0 \
    --server_port 7862
```

## 📋 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | ✅ | - | 模型文件夹路径 |
| `--model_cls` | str | ❌ | wan2.1 | 模型类别（目前仅支持wan2.1） |
| `--model_size` | str | ✅ | - | 模型大小：`14b（图像到视频或者文本到视频）` 或 `1.3b（文本到视频）` |
| `--task` | str | ✅ | - | 任务类型：`i2v`（图像到视频）或 `t2v`（文本到视频） |
| `--server_port` | int | ❌ | 7862 | 服务器端口 |
| `--server_name` | str | ❌ | 0.0.0.0 | 服务器IP地址 |

## 🎯 功能特性

### 基本设置

#### 输入参数
- **提示词 (Prompt)**: 描述期望的视频内容
- **负向提示词 (Negative Prompt)**: 指定不希望出现的元素
- **分辨率**: 支持多种预设分辨率（480p/540p/720p）
- **随机种子**: 控制生成结果的随机性
- **推理步数**: 影响生成质量和速度的平衡

#### 视频参数
- **FPS**: 每秒帧数
- **总帧数**: 视频长度
- **CFG缩放因子**: 控制提示词影响强度（1-10）
- **分布偏移**: 控制生成风格偏离程度（0-10）

### 高级优化选项

#### GPU内存优化
- **分块旋转位置编码**: 节省GPU内存
- **旋转编码块大小**: 控制分块粒度
- **清理CUDA缓存**: 及时释放GPU内存

#### 异步卸载
- **CPU卸载**: 将部分计算转移到CPU
- **延迟加载**: 按需加载模型组件，显著节省系统内存消耗
- **卸载粒度控制**: 精细控制卸载策略

#### 低精度量化
- **注意力算子**: Flash Attention、Sage Attention等
- **量化算子**: vLLM、SGL、Q8F等
- **精度模式**: FP8、INT8、BF16等

#### VAE优化
- **轻量级VAE**: 加速解码过程
- **VAE分块推理**: 减少内存占用

#### 特征缓存
- **Tea Cache**: 缓存中间特征加速生成
- **缓存阈值**: 控制缓存触发条件
- **关键步缓存**: 仅在关键步骤写入缓存

## 🔧 自动配置功能

启用"自动配置推理选项"后，系统会根据您的硬件配置自动优化参数：


### GPU内存规则
- **80GB+**: 默认配置，无需优化
- **48GB**: 启用CPU卸载，卸载比例50%
- **40GB**: 启用CPU卸载，卸载比例80%
- **32GB**: 启用CPU卸载，卸载比例100%
- **24GB**: 启用BF16精度、VAE分块
- **16GB**: 启用分块卸载、旋转编码分块
- **12GB**: 启用清理缓存、轻量级VAE
- **8GB**: 启用量化、延迟加载

### CPU内存规则
- **128GB+**: 默认配置
- **64GB**: 启用DIT量化
- **32GB**: 启用延迟加载
- **16GB**: 启用全模型量化

## ⚠️ 重要注意事项

### 🚀 低资源设备优化建议

**💡 针对显存不足或性能受限的设备**:

- **🎯 模型选择**: 优先使用蒸馏版本模型 (StepDistill-CfgDistill)
- **⚡ 推理步数**: 建议设置为 4 步
- **🔧 CFG设置**: 建议关闭CFG选项以提升生成速度
- **🔄 自动配置**: 启用"自动配置推理选项"


## 📁 文件结构

```
lightx2v/app/
├── gradio_demo.py          # 英文界面演示
├── gradio_demo_zh.py       # 中文界面演示
├── run_gradio.sh          # 启动脚本
├── README.md              # 说明文档
├── saved_videos/          # 生成视频保存目录
└── inference_logs.log     # 推理日志
```

## 🎨 界面说明

### 基本设置标签页
- **输入参数**: 提示词、分辨率等基本设置
- **视频参数**: FPS、帧数、CFG等视频生成参数
- **输出设置**: 视频保存路径配置

### 高级选项标签页
- **GPU内存优化**: 内存管理相关选项
- **异步卸载**: CPU卸载和延迟加载
- **低精度量化**: 各种量化优化选项
- **VAE优化**: 变分自编码器优化
- **特征缓存**: 缓存策略配置

## 🔍 故障排除

### 常见问题

**💡 提示**: 一般情况下，启用"自动配置推理选项"后，系统会根据您的硬件配置自动优化参数设置，通常不会出现性能问题。如果遇到问题，请参考以下解决方案：

1. **CUDA内存不足**
   - 启用CPU卸载
   - 降低分辨率
   - 启用量化选项

1. **系統内存不足**
   - 启用CPU卸载
   - 启用延迟加载选项
   - 启用量化选项

2. **生成速度慢**
   - 减少推理步数
   - 启用自动配置
   - 使用轻量级模型
   - 启用Tea Cache
   - 使用量化算子
   - 💾 **检查模型是否存放在SSD上**

3. **模型加载缓慢**
   - 💾 **将模型迁移到SSD存储**
   - 启用延迟加载选项
   - 检查磁盘I/O性能
   - 考虑使用NVMe SSD

4. **视频质量不佳**
   - 增加推理步数
   - 提高CFG缩放因子
   - 使用14B模型
   - 优化提示词

### 日志查看

```bash
# 查看推理日志
tail -f inference_logs.log

# 查看GPU使用情况
nvidia-smi

# 查看系统资源
htop
```


欢迎提交Issue和Pull Request来改进这个项目！


**注意**: 使用本工具生成的视频内容请遵守相关法律法规，不得用于非法用途。
