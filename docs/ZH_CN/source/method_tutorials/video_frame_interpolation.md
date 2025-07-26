# 视频帧插值 (VFI)

> **重要说明**: 视频帧插值功能通过配置文件启用，而不是通过命令行参数。请在配置 JSON 文件中添加 `video_frame_interpolation` 配置块来启用此功能。

## 概述

视频帧插值（VFI）是一种在现有帧之间生成中间帧的技术，用于提高帧率并创建更流畅的视频播放效果。LightX2V 集成了 RIFE（Real-Time Intermediate Flow Estimation）模型，提供高质量的帧插值能力。

## 什么是 RIFE？

RIFE 是一种最先进的视频帧插值方法，使用光流估计来生成中间帧。它能够有效地：

- 提高视频帧率（例如，从 16 FPS 提升到 32 FPS）
- 创建平滑的运动过渡
- 保持高视觉质量，最少伪影
- 实时处理视频

## 安装和设置

### 下载 RIFE 模型

首先，使用提供的脚本下载 RIFE 模型权重：

```bash
python tools/download_rife.py <目标目录>
```

例如，下载到指定位置：
```bash
python tools/download_rife.py /path/to/rife/train_log
```

此脚本将：
- 从 HuggingFace 下载 RIFEv4.26 模型
- 提取并将模型文件放置在正确的目录中
- 清理临时文件

## 使用方法

### 配置文件设置

视频帧插值功能通过配置文件启用。在你的配置 JSON 文件中添加 `video_frame_interpolation` 配置块：

```json
{
    "infer_steps": 50,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "fps": 16,
    "video_frame_interpolation": {
        "algo": "rife",
        "target_fps": 32,
        "model_path": "/path/to/rife/train_log"
    }
}
```

### 命令行使用

使用包含 VFI 配置的配置文件运行推理：

```bash
python lightx2v/infer.py \
    --model_cls wan2.1 \
    --task t2v \
    --model_path /path/to/model \
    --config_json ./configs/video_frame_interpolation/wan_t2v.json \
    --prompt "美丽的海上日落" \
    --save_video_path ./output.mp4
```

### 配置参数说明

在 `video_frame_interpolation` 配置块中：

- `algo`: 帧插值算法，目前支持 "rife"
- `target_fps`: 输出视频的目标帧率
- `model_path`: RIFE 模型路径，通常为 "train_log"

其他相关配置：
- `fps`: 源视频帧率（默认 16）

### 配置优先级

系统会自动处理视频帧率配置，优先级如下：
1. `video_frame_interpolation.target_fps` - 如果启用视频帧插值，使用此帧率作为输出帧率
2. `fps`（默认 16）- 如果未启用视频帧插值，使用此帧率；同时总是用作源帧率


## 工作原理

### 帧插值过程

1. **源视频生成**: 基础模型以源 FPS 生成视频帧
2. **帧分析**: RIFE 分析相邻帧以估计光流
3. **中间帧生成**: 在现有帧之间生成新帧
4. **时序平滑**: 插值帧创建平滑的运动过渡

### 技术细节

- **输入格式**: ComfyUI 图像张量 [N, H, W, C]，范围 [0, 1]
- **输出格式**: 插值后的 ComfyUI 图像张量 [M, H, W, C]，范围 [0, 1]
- **处理**: 自动填充和分辨率处理
- **内存优化**: 高效的 GPU 内存管理

## 示例配置

### 基础帧率翻倍

创建配置文件 `wan_t2v_vfi_32fps.json`：

```json
{
    "infer_steps": 50,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "seed": 42,
    "sample_guide_scale": 6,
    "enable_cfg": true,
    "fps": 16,
    "video_frame_interpolation": {
        "algo": "rife",
        "target_fps": 32,
        "model_path": "/path/to/rife/train_log"
    }
}
```

运行命令：
```bash
python lightx2v/infer.py \
    --model_cls wan2.1 \
    --task t2v \
    --model_path ./models/wan2.1 \
    --config_json ./wan_t2v_vfi_32fps.json \
    --prompt "一只小猫在花园里玩耍" \
    --save_video_path ./output_32fps.mp4
```

### 更高帧率增强

创建配置文件 `wan_i2v_vfi_60fps.json`：

```json
{
    "infer_steps": 30,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "seed": 42,
    "sample_guide_scale": 6,
    "enable_cfg": true,
    "fps": 16,
    "video_frame_interpolation": {
        "algo": "rife",
        "target_fps": 60,
        "model_path": "/path/to/rife/train_log"
    }
}
```

运行命令：
```bash
python lightx2v/infer.py \
    --model_cls wan2.1 \
    --task i2v \
    --model_path ./models/wan2.1 \
    --config_json ./wan_i2v_vfi_60fps.json \
    --image_path ./input.jpg \
    --prompt "平滑的相机运动" \
    --save_video_path ./output_60fps.mp4
```

## 性能考虑

### 内存使用

- RIFE 处理需要额外的 GPU 内存
- 内存使用量与视频分辨率和长度成正比
- 对于较长的视频，考虑使用较低的分辨率

### 处理时间

- 帧插值会增加处理开销
- 更高的目标帧率需要更多计算
- 处理时间大致与插值帧数成正比

### 质量与速度权衡

- 更高的插值比率可能引入伪影
- 最佳范围：2x 到 4x 帧率增加
- 对于极端插值（>4x），考虑多次处理

## 最佳实践

### 最佳使用场景

- **运动密集视频**: 从帧插值中受益最多
- **相机运动**: 更平滑的平移和缩放
- **动作序列**: 减少运动模糊感知
- **慢动作效果**: 创建流畅的慢动作视频

### 推荐设置

- **源 FPS**: 16-24 FPS（基础模型生成）
- **目标 FPS**: 32-60 FPS（2x 到 4x 增加）
- **分辨率**: 最高 720p 以获得最佳性能

### 故障排除

#### 常见问题

1. **内存不足**: 减少视频分辨率或目标 FPS
2. **输出中有伪影**: 降低插值比率
3. **处理缓慢**: 检查 GPU 内存并考虑使用 CPU 卸载

#### 解决方案

通过修改配置文件来解决问题：

```json
{
    // 内存问题解决：使用较低分辨率
    "target_height": 480,
    "target_width": 832,

    // 质量问题解决：使用适中的插值
    "video_frame_interpolation": {
        "target_fps": 24  // 而不是 60
    },

    // 性能问题解决：启用卸载
    "cpu_offload": true
}
```

## 技术实现

LightX2V 中的 RIFE 集成包括：

- **RIFEWrapper**: 与 ComfyUI 兼容的 RIFE 模型包装器
- **自动模型加载**: 与推理管道的无缝集成
- **内存优化**: 高效的张量管理和 GPU 内存使用
- **质量保持**: 在添加帧的同时保持原始视频质量
