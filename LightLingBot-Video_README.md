---
license: apache-2.0
library_name: lightx2v
base_model:
  - robbyant/lingbot-video-moe-30b-a3b
pipeline_tag: text-to-video
tags:
  - video-generation
  - text-to-video
  - image-to-video
  - text-to-image
  - distillation
  - lora
---

# 🎬 LightLingBot-Video

> Few-step distilled LoRA checkpoints for LingBot-Video, accelerated by
> [LightX2V](https://github.com/ModelTC/LightX2V).

LightLingBot-Video provides distilled LoRA checkpoints for fast LingBot-Video
inference. The same LoRA configuration supports text-to-video (T2V),
text-to-image (T2I), and image-to-video (I2V) generation.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Model Preparation](#-model-preparation)
- [Inference](#-inference)
- [Generation Results](#-generation-results)
- [Notes](#-notes)
- [Acknowledgements](#-acknowledgements)

## ✨ Features

- ⚡ **4-step inference** with distilled LoRA checkpoints
- 🎬 **T2V, T2I, and I2V** generation
- 🔧 **One unified LoRA configuration** shared by all tasks
- 🚀 **LightX2V integration** for efficient inference and deployment
- 🎯 **Two distilled LoRA versions** for comparison

## 📦 Installation

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
python3 -m pip install -e .
```

Follow the
[LightX2V installation guide](https://github.com/ModelTC/LightX2V)
to install the optional acceleration dependencies required by your hardware.

## 📥 Model Preparation

Download the LingBot-Video base model and the distilled LoRA repository:

```bash
huggingface-cli download robbyant/lingbot-video-moe-30b-a3b \
  --local-dir /path/to/LingBot-Video

huggingface-cli download lightx2v/LightLingBot-Video \
  --local-dir /path/to/LightLingBot-Video
```

Before inference:

1. Set `model_path` in the corresponding script under
   `scripts/lingbot_video/`.
2. Set the LoRA checkpoint path in
   `configs/lingbot_video/lingbot_video_lora.json`.
3. Adjust `CUDA_VISIBLE_DEVICES` in the script if needed.
4. For I2V, also set `image_path` in the I2V script.

## 🚀 Inference

Run the following commands from the LightX2V repository root. These commands
use the shell scripts under `scripts/lingbot_video`; no `examples` entry point
is required.

### Text-to-Video

```bash
bash scripts/lingbot_video/run_lingbot_video_t2v.sh
```

### Text-to-Image

```bash
bash scripts/lingbot_video/run_lingbot_video_t2i.sh
```

### Image-to-Video

```bash
bash scripts/lingbot_video/run_lingbot_video_i2v.sh
```

### Recommended Settings

| Setting | Base model | Distilled LoRA |
|:--|:--:|:--:|
| Inference steps | 40 | 4 |
| LoRA strength | — | 1.0 |
| Resolution | 832 × 480 | 832 × 480 |
| Guidance scale | 3.0 | 1.0 |
| Sample shift | 3.0 | 3.0 |

## 🎬 Generation Results

### Text-to-Video (T2V)

| Method | Sample 1 | Sample 2 | Sample 3 |
|:--|:--:|:--:|:--:|
| 40 steps | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/aBfpevvzMCykZ2AN4nDoy.mp4" controls loop muted playsinline preload="metadata" width="280"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/GmroUGvhggG7Iwh3STSF0.mp4" controls loop muted playsinline preload="metadata" width="280"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/2hrGRrK8M89PqE6zU9uIH.mp4" controls loop muted playsinline preload="metadata" width="280"></video> |
| 4-step LoRA v1 | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/DCrdwVVelNhMXPR9l0TOb.mp4" controls loop muted playsinline preload="metadata" width="280"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/6yHNCWUHeW61cBxKbhIpt.mp4" controls loop muted playsinline preload="metadata" width="280"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/Nzr5O7KMHBUBgEkGtc-ps.mp4" controls loop muted playsinline preload="metadata" width="280"></video> |
| 4-step LoRA v2 | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/yTthqJbREftm0zLaHQ6YB.mp4" controls loop muted playsinline preload="metadata" width="280"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/MBTAs68pA6Crw7RUb1jRG.mp4" controls loop muted playsinline preload="metadata" width="280"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/nQVAyzryvoHjOk-531ibh.mp4" controls loop muted playsinline preload="metadata" width="280"></video> |

### Text-to-Image (T2I)

| Method | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|:--|:--:|:--:|:--:|:--:|
| 40 steps | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/heyC2-qQ0n4-UCnNwgAEn.png" alt="T2I 40-step sample 1" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/gf0j5YOtB0M16Jzu-rgu7.png" alt="T2I 40-step sample 2" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/yaTy3eiHIDU4whWXOS5CY.png" alt="T2I 40-step sample 3" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/IqA7K-XOiSbDqNP6jduus.png" alt="T2I 40-step sample 4" width="220"> |
| 4-step LoRA v1 | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/_6bWXNVVNYthLPTNfkcqn.png" alt="T2I LoRA v1 sample 1" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/iXqL-99pN_L59aSDBtpDl.png" alt="T2I LoRA v1 sample 2" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/t8qLxFwtwl3NBYkIpVkEh.png" alt="T2I LoRA v1 sample 3" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/ogGwg7x-erqwzvBuRfvJG.png" alt="T2I LoRA v1 sample 4" width="220"> |
| 4-step LoRA v2 | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/eqC8uWo86Ioi6SuCxOJFC.png" alt="T2I LoRA v2 sample 1" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/idiLpjm1ukZ10f4UIS0SU.png" alt="T2I LoRA v2 sample 2" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/a1f7l4Q5ylOSOHLmNf0lp.png" alt="T2I LoRA v2 sample 3" width="220"> | <img src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/3BCgc6ZQsi1_qM5wXXK0x.png" alt="T2I LoRA v2 sample 4" width="220"> |

### Image-to-Video (I2V)

| Method | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
|:--|:--:|:--:|:--:|:--:|
| 40 steps | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/uwIKVnhw0ZmYQ9W9h_2Bb.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/dBhmDQ1xg9VNc3_bFk9Y2.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/qyj46MCmyPRNjHGMtIjZg.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/bhjdpxkWY01iOtO1sdwWT.mp4" controls loop muted playsinline preload="metadata" width="210"></video> |
| 4-step LoRA v1 | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/iyeCPr8_jugefnGjQzVXy.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/n8nXH4iZteTIL9wvsCa7L.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/t6assZ-6RW85DMIOXkoUV.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/Fo_PsM3xHn6bdvPFSnkV0.mp4" controls loop muted playsinline preload="metadata" width="210"></video> |
| 4-step LoRA v2 | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/4M-jfTK6o0CCP2II1cFYd.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/j58fryiWhyQrT1rCc0WdX.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/E8KZPqL00qCjMqZ2951JD.mp4" controls loop muted playsinline preload="metadata" width="210"></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/680de13385293771bc57400b/_VP2MQaiQYqxdrtx9aB5J.mp4" controls loop muted playsinline preload="metadata" width="210"></video> |

## ⚠️ Notes

- The distilled LoRA configuration uses 4 inference steps and disables CFG.
- Merged LoRA inference requires original, non-quantized DiT weights.
- Prompts and negative prompts should follow the structured JSON format used
  in `scripts/lingbot_video/`.
- T2V, T2I, and I2V share
  `configs/lingbot_video/lingbot_video_lora.json`.
- Speed and memory usage depend on the GPU, attention backend, resolution, and
  video length.

## 🤝 Acknowledgements

- [LingBot-Video](https://huggingface.co/inclusionAI/LingBot-Video)
- [LightX2V](https://github.com/ModelTC/LightX2V)

If you find this project helpful, please give
[LightX2V](https://github.com/ModelTC/LightX2V) a ⭐.
