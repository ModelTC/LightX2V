# EdenLive

[English](README.md) | [简体中文](README_zh.md)

EdenLive 是基于 Wan 的神经渲染模型，以 **proxy 视频或参考视频流**、**目标首帧图像**和**文本提示词**为输入，生成细化或风格化的视频。参考内容可以来自 2D/3D 场景、现有引擎或录屏。上游应用负责维护交互状态，EdenLive 负责视觉渲染。

控制信息通过**经过后训练的 DINOv3 视觉编码器**注入：编码器逐帧提取 proxy 特征，作为扩散 Transformer 的条件输入。proxy 提供场景布局、相机运动和时序，首帧与文本提示词指定目标画面外观。

[EdenLive 项目](https://vast-eden.ai/?p=edenlive)展示了超过 24 FPS 的生成速度、分钟级持续生成，以及 480p/720p 输出。5B 模型支持单卡 720p 部署。LightX2V 为双向和自回归版本提供推理加速，包括量化、流式解码、编译和 CUDA Graph。

## 支持的模型

| 基础模型 | 推理方式 | 权重目录 | 启动脚本 |
| --- | --- | --- | --- |
| Wan2.1 14B | 双向推理 | `dit-14b-bidir` | [wan21_14b_bidir.sh](wan21_14b_bidir.sh) |
| Wan2.1 14B | 四步自回归 | `dit-14b` | [wan21_14b_few_step.sh](wan21_14b_few_step.sh) |
| Wan2.2 5B | 双向推理 | `dit-5b-bidir` | [wan22_5b_bidir.sh](wan22_5b_bidir.sh) |
| Wan2.2 5B | 四步自回归 | `dit-5b` | [wan22_5b_few_step.sh](wan22_5b_few_step.sh) |

双向推理对完整片段联合去噪。自回归推理通过持久 KV 缓存连续生成视频块，支持流式输出解码后的视频。

## 环境与模型准备

按照 LightX2V [快速开始](../../docs/ZH_CN/source/getting_started/quickstart.md)准备环境。实时配置需要与目标 GPU 兼容的 SageAttention2、FlashInfer 和 SGL kernel。RTX 5090 配置还需要 comfy-kitchen 和 CUDA 编译器，用于 INT8 ConvRot。

以下命令均在 LightX2V 仓库根目录执行。

### 下载与转换

下载完整的 [VAST-AI/EdenLive](https://huggingface.co/VAST-AI/EdenLive) 模型，并转换为 LightX2V 格式：

```bash
hf download VAST-AI/EdenLive --local-dir ./models/EdenLive

export EDEN_RELEASE="$PWD/models/EdenLive"
export EDEN_MODEL="$PWD/models/EdenLive-x2v"

python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE" --output "$EDEN_MODEL" --all
```

完整转换会处理四种 DiT 和 T5/CLIP 编码器，并保留 DINO、VAE、tokenizer 和 LightTAE 文件。所有推理方式（包括量化推理）的 `MODEL_PATH` 均指向转换后的模型根目录。

## 推理

以 BF16 运行 Wan2.2 5B 自回归模型：

```bash
MODEL_PATH="$EDEN_MODEL" \
IMAGE_PATH=/path/to/cover.png \
VIDEO_PATH=/path/to/recording.mp4 \
PROMPT="A cinematic scene with detailed, realistic lighting." \
SAVE_RESULT_PATH="$PWD/save_results/edenlive_5b.mp4" \
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/wan22_5b_few_step.sh
```

其他版本使用模型表中对应的启动脚本。所有脚本均支持 `MODEL_PATH`、`IMAGE_PATH`、`VIDEO_PATH`、`PROMPT`、`SAVE_RESULT_PATH` 和 `CUDA_VISIBLE_DEVICES`。

四个基础脚本还支持通过 `CONFIG_JSON` 选择配置。额外推理参数会继续传递给 LightX2V，例如 `--seed 123`。基础配置位于 [`configs/edenlive`](../../configs/edenlive/)，其中包含 AR 动态提示词（prompt travel）配置。

## 实时部署

硬件预设面向 **Wan2.2 5B AR、1280×704、24 FPS**，每个视频块执行四步去噪，包含三个 latent 帧。两套配置均使用 GPU 常驻的 FP8 T5、LightTAE 同步流式解码、编译和预热。

| 硬件 | DiT | Proxy 编码器 | 启动脚本 |
| --- | --- | --- | --- |
| H100 / H200 | FP8-SGL | BF16 | [run_hopper.sh](run_hopper.sh) |
| RTX 5090 | INT8 ConvRot | INT8 ConvRot | [run_5090.sh](run_5090.sh) |

配置预设：[Hopper FP8-SGL](../../configs/edenlive/realtime/wan22_5b_ar_fp8_sgl_hopper.json) · [RTX 5090 INT8 ConvRot](../../configs/edenlive/realtime/wan22_5b_ar_int8_convrot_5090.json)。

### 准备量化权重

两套预设均使用由 LightX2V 格式模型转换得到的 FP8 T5 权重：

```bash
python tools/convert/converter.py \
  --source "$EDEN_MODEL/umt5/text_encoder/model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name t5-fp8 \
  --model_type wan_t5 --quantized --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 --device cuda:0 --single_file
```

对于 **H100/H200**，将 DiT 量化为 FP8：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8 \
  --quantize fp8 --device cuda:0
```

对于 **RTX 5090**，将 DiT 和 proxy 编码器量化为 INT8 ConvRot：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-int8-convrot \
  --quantize int8-convrot --device cuda:0

python tools/convert/convert_edenlive_proxy.py \
  --source "$EDEN_MODEL/vision-encoder/projection.safetensors" \
  --output "$EDEN_MODEL/quantized/proxy-dino-int8-convrot.safetensors" \
  --device cuda:0 --group-size 256
```

### 运行硬件预设

设置模型、输入和解码器路径：

```bash
export MODEL_PATH="$EDEN_MODEL"
export IMAGE_PATH=/path/to/cover.png
export VIDEO_PATH=/path/to/recording.mp4
export PROMPT="A cinematic scene with detailed, realistic lighting."
export TAE_PATH="$EDEN_MODEL/decoder-5b/lighttaew2_2.pth"
export CUDA_VISIBLE_DEVICES=0
```

在 **H100/H200** 上运行：

```bash
bash scripts/edenlive/run_hopper.sh
```

在 **RTX 5090** 上运行：

```bash
bash scripts/edenlive/run_5090.sh
```

转换后的模型包含示例所用的 LightTAE 权重。`TAE_PATH` 也支持指定微调后的 `.safetensors` 权重。原版 Wan VAE 负责目标首帧编码，LightTAE 负责生成视频块的解码。

启动脚本默认使用上述步骤生成的量化文件。权重存放在其他位置时，可通过以下环境变量指定：

| 变量 | 默认权重 |
| --- | --- |
| `DIT_QUANTIZED_CKPT` | Hopper 使用 `$MODEL_PATH/quantized/dit-5b-fp8.safetensors`；RTX 5090 使用 `dit-5b-int8-convrot.safetensors` |
| `T5_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/t5-fp8.safetensors` |
| `PROXY_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/proxy-dino-int8-convrot.safetensors`，仅 RTX 5090 |

`SAVE_RESULT_PATH` 指定输出 MP4 路径。`T5_CPU_OFFLOAD=true` 通过卸载 T5 降低 GPU 常驻显存；下方性能测试使用 `false`。

## LightX2V 加速

| 组件 | 优化 |
| --- | --- |
| 量化投影 | FP8-SGL 或 INT8 ConvRot 矩阵乘、共享激活量化、合并 QKV 投影，以及 GELU/量化融合。 |
| Attention 与小算子 | 面向 GPU 架构的 SageAttention2、FlashInfer RoPE、SGL RMSNorm，以及 Triton LayerNorm/modulation。 |
| AR 缓存 | 固定 sink 与环形 KV 存储、可复用的 attention 工作区，以及历史 RoPE 数据缓存。 |
| 条件计算 | 复用文本 K/V、文本与时间投影，以及重复的 proxy 特征插值。 |
| KV commit | 保留 clean forward，在完成最终所需缓存更新后，跳过不再使用的 attention、FFN 和输出计算。 |
| Proxy 编码器 | GPU 预处理、RoPE/归一化融合和 CUDA Graph 重放。 |
| 编译 | 编译 DiT block 和流式 LightTAE 解码器，预热所需形状与缓存状态。 |
| 流式处理 | 预取 proxy 帧，使用持久解码器状态逐块解码生成结果。 |

RTX 5090 预设支持 DiT 主干和 proxy 编码器的 CUDA Graph 执行。Hopper 预设使用编译后的 DiT block，以及 proxy 编码器的 CUDA Graph 执行。

## 性能

**Wan2.2 5B AR · 1280×704 · 24 FPS · batch size 1 · LightTAE 同步解码。** 两套预设均使用四步去噪，每块三个 latent 帧，attention 窗口为 18 个 latent 帧，sink 为六帧。

| GPU | Generation FPS ↑ | Inference FPS ↑ | 首块延迟 ms ↓ | 出块间隔 P95 ms ↓ | RTF@fps24 ↓ | Peak allocated GiB ↓ | Peak reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA H200 | 36.12 | 35.95 | 358.19 | 351.20 | 0.6676 | 29.72 | 32.23 |
| NVIDIA RTX 5090 | 24.87 | 24.69 | 500.02 | 492.63 | 0.9722 | 28.38 | 28.95 |

- **Generation FPS** 从 AR 生成开始计时，到最后一个解码块在 GPU 上就绪为止。**Inference FPS** 还包含请求的输入准备和编码。两者均不包含模型加载、编译、预热、最终视频拼接和 MP4 保存。
- **首块延迟**从请求开始计时。**出块间隔 P95** 衡量相邻解码块之间的间隔，H200 数值合并三次请求的间隔计算。FPS 按总帧数除以总测量时间计算。
- **RTF@fps24** 为推理耗时除以 24 FPS 下的视频播放时长，小于 1 表示生成速度快于播放速度。
- **显存峰值**覆盖完整请求，包含最终视频拼接和保存。
