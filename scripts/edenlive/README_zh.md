# EdenLive

[English](README.md) | [简体中文](README_zh.md)

EdenLive 根据代理录屏、目标首帧和文本提示生成视频。本文说明 Wan2.2 5B 自回归推理。公开任务统一为 `v2v`：`--video_path` 提供录屏，`--image_path` 提供目标首帧。

## 准备模型

按 LightX2V [快速开始](../../docs/ZH_CN/source/getting_started/quickstart.md)安装环境，下载 [EdenLive 权重](https://huggingface.co/VAST-AI/EdenLive)，然后从仓库根目录转换：

```bash
hf download VAST-AI/EdenLive --local-dir ./models/EdenLive
python tools/convert/convert_edenlive.py --source ./models/EdenLive --output ./models/EdenLive-x2v --all
```

RTX 5090 FP8 配置需要 DiT/T5 的 FP8-FP16 权重和 Proxy 的 INT8 ConvRot 权重：

```bash
EDEN_MODEL="$PWD/models/EdenLive-x2v" QUANT_DEVICE=cuda:0 bash scripts/edenlive/prepare_fp8_f16_accum_5090.sh
```

脚本会验证已有权重再复用。最终配置的 DiT/T5 使用 FP8-FP16 累加；Proxy 的 FP8 误差在生成中被放大，因此使用 INT8 ConvRot。该配置需要 SM120 FP8-FP16 算子，转换 Proxy 需要 comfy-kitchen。

## 运行

RTX 5090 保留两份配置：

- [最终 FP8 配置](../../configs/edenlive/realtime/wan22_5b_ar_fp8_f16_accum_sage_lighttae_async_5090.json)
- [BF16 对照配置](../../configs/edenlive/realtime/wan22_5b_ar_bf16_sage_lighttae_async_5090.json)

```bash
export MODEL_PATH="$PWD/models/EdenLive-x2v"
export IMAGE_PATH=/path/to/cover.png
export VIDEO_PATH=/path/to/recording.mp4
export PROMPT="A cinematic scene with detailed, realistic lighting."
export OUTPUT_PATH="$PWD/edenlive_output.mp4"
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_5090.sh
```

`run_5090.sh` 默认使用 FP8 配置。运行 BF16 时，在同一命令中将 `CONFIG_PATH` 设为 BF16 JSON。迁移模型时，需要修改复制后的 JSON 中模型、量化权重及 LightTAE 的路径。两份配置都开启 Sage、warmup、DiT compile/CUDA Graph、流式 LightTAE 和顶层异步解码；`vae_use_compile` 独立控制解码器编译。KV offload 关闭。

## RTX 5090 实测

完整官方 cryo-crossing case：717 个实际视频帧、1280×704、24 FPS、七段定时 prompt、四步去噪、每块三个 latent 帧、窗口 18、sink 6。各行均为排除预热后的一次正式请求。

| 实现 / GPU | Generation FPS ↑ | Inference FPS ↑ | 首块延迟 ms ↓ | 出块间隔 P95 ms ↓ | RTF@fps24 ↓ | Peak allocated GiB ↓ | Peak reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方源码 BF16 / RTX 5090 | 13.4663 | 11.6319 | 9083.1971 | 908.8465 | 2.0633 | 21.3601 | 25.3223 |
| LightX2V FP8 / RTX 5090 | **25.2811** | **25.1732** | **509.6956** | **511.9061** | **0.9534** | **23.3308** | **24.9082** |
