# 🚀 基准测试

> 本文档展示了LightX2V在不同硬件环境下的性能测试结果，包括H200和RTX 4090平台的详细对比数据。

---

## 🖥️ H200 环境 (~140GB显存)

### 📋 软件环境配置

| 组件 | 版本 |
|:-----|:-----|
| **Python** | 3.11 |
| **PyTorch** | 2.7.1+cu128 |
| **SageAttention** | 2.2.0 |
| **vLLM** | 0.9.2 |
| **sgl-kernel** | 0.1.8 |

---

### 🎬 480P 5s视频测试

**测试配置:**

- **模型**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **参数**: `infer_steps=40`, `seed=42`, `enable_cfg=True`

#### 📊 性能对比表

| 配置 | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:----------:|:---------------:|:------:|:--------:|
| **Wan2.1 Official** | 366 | 71 | 1.0x | <video src="https://github.com/user-attachments/assets/24fb112e-c868-4484-b7f0-d9542979c2c3" width="200px"></video> |
| **FastVideo** | 292 | 26 | **1.25x** | <video src="https://github.com/user-attachments/assets/26c01987-441b-4064-b6f4-f89347fddc15" width="200px"></video> |
| **LightX2V_1** | 250 | 53 | **1.46x** | <video src="https://github.com/user-attachments/assets/7bffe48f-e433-430b-91dc-ac745908ba3a" width="200px"></video> |
| **LightX2V_2** | 216 | 50 | **1.70x** | <video src="https://github.com/user-attachments/assets/0a24ca47-c466-433e-8a53-96f259d19841" width="200px"></video> |
| **LightX2V_3** | 191 | 35 | **1.92x** | <video src="https://github.com/user-attachments/assets/970c73d3-1d60-444e-b64d-9bf8af9b19f1" width="200px"></video> |
| **LightX2V_3-Distill** | 14 | 35 | **🏆 20.85x** | <video src="https://github.com/user-attachments/assets/b4dc403c-919d-4ba1-b29f-ef53640c0334" width="200px"></video> |
| **LightX2V_4** | 107 | 35 | **3.41x** | <video src="https://github.com/user-attachments/assets/49cd2760-4be2-432c-bf4e-01af9a1303dd" width="200px"></video> |

---

### 🎬 720P 5s视频测试

**测试配置:**

- **模型**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **参数**: `infer_steps=40`, `seed=1234`, `enable_cfg=True`

#### 📊 性能对比表

| 配置 | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:----------:|:---------------:|:------:|:--------:|
| **Wan2.1 Official** | 974 | 81 | 1.0x | <video src="https://github.com/user-attachments/assets/a28b3956-ec52-4a8e-aa97-c8baf3129771" width="200px"></video> |
| **FastVideo** | 914 | 40 | **1.07x** | <video src="https://github.com/user-attachments/assets/bd09a886-e61c-4214-ae0f-6ff2711cafa8" width="200px"></video> |
| **LightX2V_1** | 807 | 65 | **1.21x** | <video src="https://github.com/user-attachments/assets/a79aae87-9560-4935-8d05-7afc9909e993" width="200px"></video> |
| **LightX2V_2** | 751 | 57 | **1.30x** | <video src="https://github.com/user-attachments/assets/cb389492-9b33-40b6-a132-84e6cb9fa620" width="200px"></video> |
| **LightX2V_3** | 671 | 43 | **1.45x** | <video src="https://github.com/user-attachments/assets/71c3d085-5d8a-44e7-aac3-412c108d9c53" width="200px"></video> |
| **LightX2V_3-Distill** | 44 | 43 | **🏆 22.14x** | <video src="https://github.com/user-attachments/assets/9fad8806-938f-4527-b064-0c0b58f0f8c2" width="200px"></video> |
| **LightX2V_4** | 344 | 46 | **2.83x** | <video src="https://github.com/user-attachments/assets/c744d15d-9832-4746-b72c-85fa3b87ed0d" width="200px"></video> |

---

## 🖥️ RTX 4090 环境 (~24GB显存)

### 📋 软件环境配置

| 组件 | 版本 |
|:-----|:-----|
| **Python** | 3.9.16 |
| **PyTorch** | 2.5.1+cu124 |
| **SageAttention** | 2.1.0 |
| **vLLM** | 0.6.6 |
| **sgl-kernel** | 0.0.5 |
| **q8-kernels** | 0.0.0 |

---

### 🎬 480P 5s视频测试

**测试配置:**

- **模型**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **参数**: `infer_steps=40`, `seed=42`, `enable_cfg=True`

#### 📊 性能对比表

| 配置 | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:----------:|:---------------:|:------:|:--------:|
| **Wan2GP(profile=3)** | 779 | 20 | **1.0x** | <video src="https://github.com/user-attachments/assets/ba548a48-04f8-4616-a55a-ad7aed07d438" width="200px"></video> |
| **LightX2V_5** | 738 | 16 | **1.05x** | <video src="https://github.com/user-attachments/assets/ce72ab7d-50a7-4467-ac8c-a6ed1b3827a7" width="200px"></video> |
| **LightX2V_5-Distill** | 68 | 16 | **11.45x** | <video src="https://github.com/user-attachments/assets/5df4b8a7-3162-47f8-a359-e22fbb4d1836" width="200px"></video> |
| **LightX2V_6** | 630 | 12 | **1.24x** | <video src="https://github.com/user-attachments/assets/d13cd939-363b-4f8b-80d9-d3a145c46676" width="200px"></video> |
| **LightX2V_6-Distill** | 63 | 12 | **🏆 12.36x** | <video src="https://github.com/user-attachments/assets/f372bce4-3c2f-411d-aa6b-c4daeb467d90" width="200px"></video> |

---

### 🎬 720P 5s视频测试

**测试配置:**

- **模型**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **参数**: `infer_steps=40`, `seed=1234`, `enable_cfg=True`

#### 📊 性能对比表

| 配置 | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:----------:|:---------------:|:------:|:--------:|
| **Wan2GP(profile=3)** | -- | OOM | -- | <video src="--" width="200px"></video> |
| **LightX2V_5** | 2473 | 23 | -- | <video src="https://github.com/user-attachments/assets/0e83b146-3297-4c63-831c-8462cc657cad" width="200px"></video> |
| **LightX2V_5-Distill** | 183 | 23 | -- | <video src="https://github.com/user-attachments/assets/976d0af0-244c-4abe-b2cb-01f68ad69d3c" width="200px"></video> |
| **LightX2V_6** | 2169 | 18 | -- | <video src="https://github.com/user-attachments/assets/cf9edf82-53e1-46af-a000-79a88af8ad4a" width="200px"></video> |
| **LightX2V_6-Distill** | 171 | 18 | -- | <video src="https://github.com/user-attachments/assets/e3064b03-6cd6-4c82-9e31-ab28b3165798" width="200px"></video> |

---

## 📖 LightX2V 配置

所有 LightX2V 方案均使用 Wan2.1 I2V、SageAttention2 和 81 帧输出。配置文件位于 [configs/bench](https://github.com/ModelTC/LightX2V/tree/main/configs/bench)，用于设置推理步数、量化、缓存和 offload。计算精度通过环境变量设置：`DTYPE=BF16`，敏感层精度按下表设置 `SENSITIVE_LAYER_DTYPE`。

| 测试方案 | 配置文件 | 敏感层精度 |
|:---------|:---------|:-----------|
| **LightX2V_1** | [wan_i2v.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v.json) | `FP32` |
| **LightX2V_2** | [wan_i2v.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v.json) | `BF16` |
| **LightX2V_3** | [wan_i2v_fp8.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8.json) | `BF16` |
| **LightX2V_3-Distill** | [wan_i2v_fp8_distill.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8_distill.json) | `BF16` |
| **LightX2V_4** | [wan_i2v_fp8_teacache.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8_teacache.json) | `BF16` |
| **LightX2V_5** | [wan_i2v_fp8_offload.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8_offload.json) | `FP32` |
| **LightX2V_5-Distill** | [wan_i2v_fp8_offload_distill.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8_offload_distill.json) | `FP32` |
| **LightX2V_6** | [wan_i2v_fp8_offload.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8_offload.json) | `BF16` |
| **LightX2V_6-Distill** | [wan_i2v_fp8_offload_distill.json](https://github.com/ModelTC/LightX2V/blob/main/configs/bench/wan_i2v_fp8_offload_distill.json) | `BF16` |

- Distill：4 步 DMD2 蒸馏，关闭 CFG。
- FP8：使用 Tiling VAE。
- Offload：DiT 按 block 卸载到 CPU，同时开启 T5 CPU offload。
- TeaCache：阈值为 `0.2`。

## ▶️ 运行方式

根据测试方案选择配置文件和敏感层精度。`model_path` 指向对应分辨率的基础模型目录。FP8 方案需将 JSON 中的 `dit_quantized_ckpt` 替换为与 `fp8-sgl` 匹配的 DiT 量化权重文件或目录；蒸馏方案使用对应的 StepDistill-CfgDistill 权重。

| 分辨率 | 命令行尺寸 | 随机种子 |
|:-------|:-----------|:---------|
| 480P | `--size 480 832` | `--seed 42` |
| 720P | `--size 720 1280` | `--seed 1234` |

以下示例运行 H200 环境中的 **LightX2V_1 / 480P**：

```bash
lightx2v_path=path/to/LightX2V
model_path=path/to/Wan2.1-I2V-14B-480P-Lightx2v

export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task i2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/bench/wan_i2v.json" \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_2.jpg" \
  --prompt "A close-up cinematic view of a person cooking in a warm,sunlit kitchen, using a wooden spatula to stir-fry a colorful mix of freshvegetables—carrots, broccoli, and bell peppers—in a black frying pan on amodern induction stove. The scene captures the glistening texture of thevegetables, steam gently rising, and subtle reflections on the stove surface.In the background, soft-focus jars, fruits, and a window with natural daylightcreate a cozy atmosphere. The hand motions are smooth and rhythmic, with a realisticsense of motion blur and lighting." \
  --negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --size 480 832 \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/lightx2v_1.mp4"
```

运行其他方案时，按表设置 `--config_json` 和 `SENSITIVE_LAYER_DTYPE`，并选择相应的模型权重、尺寸、种子及输出路径。所有方案均使用 `--model_cls wan2.1 --task i2v`。

## 🔎 对比实现

| 实现 | 说明 |
|:-----|:-----|
| **Wan2.1 Official** | 基于 [Wan2.1 官方仓库](https://github.com/Wan-Video/Wan2.1)的原始实现 |
| **FastVideo** | 基于 [FastVideo 官方仓库](https://github.com/hao-ai-lab/FastVideo)，使用 SageAttention2 后端优化 |
| **Wan2GP(profile=3)** | 基于 [Wan2GP 仓库](https://github.com/deepbeepmeep/Wan2GP)，使用 MMGP 优化技术。profile=3 适用于至少 32GB 内存和 24GB 显存的 RTX 3090/4090 环境，通过增加显存占用来适应有限的内存资源。使用量化模型：[480P 模型](https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors)和 [720P 模型](https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/wan2.1_image2video_720p_14B_quanto_mbf16_int8.safetensors) |
