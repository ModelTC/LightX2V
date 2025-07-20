# 🚀 Benchmark

> This document showcases the performance test results of LightX2V across different hardware environments, including detailed comparison data for H200 and RTX 4090 platforms.

---

## 🖥️ H200 Environment (~140GB VRAM)

### 📋 Software Environment Configuration

| Component | Version |
|:----------|:--------|
| **Python** | 3.11 |
| **PyTorch** | 2.7.1+cu128 |
| **SageAttention** | 2.2.0 |
| **vLLM** | 0.9.2 |
| **sgl-kernel** | 0.1.8 |

---

### 🎬 480P 5s Video Test

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **Parameters**: `infer_steps=40`, `seed=42`, `enable_cfg=True`

#### 📊 Performance Comparison Table

| Configuration | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:-----------------:|:--------------:|:-------:|:------------:|
| **Wan2.1 Official** | 366 | 71 | 1.0x | <video src="https://github.com/user-attachments/assets/24fb112e-c868-4484-b7f0-d9542979c2c3" width="200px"></video> |
| **FastVideo** | 292 | 26 | **1.25x** | <video src="https://github.com/user-attachments/assets/26c01987-441b-4064-b6f4-f89347fddc15" width="200px"></video> |
| **LightX2V_1** | 250 | 53 | **1.46x** | <video src="https://github.com/user-attachments/assets/7bffe48f-e433-430b-91dc-ac745908ba3a" width="200px"></video> |
| **LightX2V_2** | 216 | 50 | **1.70x** | <video src="https://github.com/user-attachments/assets/0a24ca47-c466-433e-8a53-96f259d19841" width="200px"></video> |
| **LightX2V_3** | 191 | 35 | **1.92x** | <video src="https://github.com/user-attachments/assets/970c73d3-1d60-444e-b64d-9bf8af9b19f1" width="200px"></video> |
| **LightX2V_3-Distill** | 14 | 35 | **🏆 20.85x** | <video src="https://github.com/user-attachments/assets/b4dc403c-919d-4ba1-b29f-ef53640c0334" width="200px"></video> |
| **LightX2V_4** | 107 | 35 | **3.41x** | <video src="https://github.com/user-attachments/assets/49cd2760-4be2-432c-bf4e-01af9a1303dd" width="200px"></video> |

---

### 🎬 720P 5s Video Test

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **Parameters**: `infer_steps=40`, `seed=1234`, `enable_cfg=True`

#### 📊 Performance Comparison Table

| Configuration | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:-----------------:|:--------------:|:-------:|:------------:|
| **Wan2.1 Official** | 974 | 81 | 1.0x | <video src="https://github.com/user-attachments/assets/a28b3956-ec52-4a8e-aa97-c8baf3129771" width="200px"></video> |
| **FastVideo** | 914 | 40 | **1.07x** | <video src="https://github.com/user-attachments/assets/bd09a886-e61c-4214-ae0f-6ff2711cafa8" width="200px"></video> |
| **LightX2V_1** | 807 | 65 | **1.21x** | <video src="https://github.com/user-attachments/assets/a79aae87-9560-4935-8d05-7afc9909e993" width="200px"></video> |
| **LightX2V_2** | 751 | 57 | **1.30x** | <video src="https://github.com/user-attachments/assets/cb389492-9b33-40b6-a132-84e6cb9fa620" width="200px"></video> |
| **LightX2V_3** | 671 | 43 | **1.45x** | <video src="https://github.com/user-attachments/assets/71c3d085-5d8a-44e7-aac3-412c108d9c53" width="200px"></video> |
| **LightX2V_3-Distill** | 44 | 43 | **🏆 22.14x** | <video src="https://github.com/user-attachments/assets/9fad8806-938f-4527-b064-0c0b58f0f8c2" width="200px"></video> |
| **LightX2V_4** | 344 | 46 | **2.83x** | <video src="https://github.com/user-attachments/assets/c744d15d-9832-4746-b72c-85fa3b87ed0d" width="200px"></video> |

---

## 🖥️ RTX 4090 Environment (~24GB VRAM)

### 📋 Software Environment Configuration

| Component | Version |
|:----------|:--------|
| **Python** | 3.9.16 |
| **PyTorch** | 2.5.1+cu124 |
| **SageAttention** | 2.1.0 |
| **vLLM** | 0.6.6 |
| **sgl-kernel** | 0.0.5 |
| **q8-kernels** | 0.0.0 |

---

### 🎬 480P 5s Video Test

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **Parameters**: `infer_steps=40`, `seed=42`, `enable_cfg=True`

#### 📊 Performance Comparison Table

| Configuration | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:-----------------:|:--------------:|:-------:|:------------:|
| **Wan2GP(profile=3)** | 779 | 20 | **1.0x** | <video src="https://github.com/user-attachments/assets/ba548a48-04f8-4616-a55a-ad7aed07d438" width="200px"></video> |
| **LightX2V_5** | 738 | 16 | **1.05x** | <video src="https://github.com/user-attachments/assets/ce72ab7d-50a7-4467-ac8c-a6ed1b3827a7" width="200px"></video> |
| **LightX2V_5-Distill** | 68 | 16 | **11.45x** | <video src="https://github.com/user-attachments/assets/5df4b8a7-3162-47f8-a359-e22fbb4d1836" width="200px"></video> |
| **LightX2V_6** | 630 | 12 | **1.24x** | <video src="https://github.com/user-attachments/assets/d13cd939-363b-4f8b-80d9-d3a145c46676" width="200px"></video> |
| **LightX2V_6-Distill** | 63 | 12 | **🏆 12.36x** | <video src="https://github.com/user-attachments/assets/f372bce4-3c2f-411d-aa6b-c4daeb467d90" width="200px"></video>

---

### 🎬 720P 5s Video Test

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **Parameters**: `infer_steps=40`, `seed=1234`, `enable_cfg=True`

#### 📊 Performance Comparison Table

| Configuration | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:-----------------:|:--------------:|:-------:|:------------:|
| **Wan2GP(profile=3)** | -- | OOM | -- | <video src="--" width="200px"></video> |
| **LightX2V_5** | 2473 | 23 | -- | <video src="https://github.com/user-attachments/assets/0e83b146-3297-4c63-831c-8462cc657cad" width="200px"></video> |
| **LightX2V_5-Distill** | 183 | 23 | -- | <video src="https://github.com/user-attachments/assets/976d0af0-244c-4abe-b2cb-01f68ad69d3c" width="200px"></video> |
| **LightX2V_6** | 2169 | 18 | -- | <video src="https://github.com/user-attachments/assets/cf9edf82-53e1-46af-a000-79a88af8ad4a" width="200px"></video> |
| **LightX2V_6-Distill** | 171 | 18 | -- | <video src="https://github.com/user-attachments/assets/e3064b03-6cd6-4c82-9e31-ab28b3165798" width="200px"></video> |

---

## 📖 Configuration Descriptions

### 🖥️ H200 Environment Configuration Descriptions

| Configuration | Technical Features |
|:--------------|:------------------|
| **Wan2.1 Official** | Based on [Wan2.1 official repository](https://github.com/Wan-Video/Wan2.1) original implementation |
| **FastVideo** | Based on [FastVideo official repository](https://github.com/hao-ai-lab/FastVideo), using SageAttention2 backend optimization |
| **LightX2V_1** | Uses SageAttention2 to replace native attention mechanism, adopts DIT BF16+FP32 (partial sensitive layers) mixed precision computation, improving computational efficiency while maintaining precision |
| **LightX2V_2** | Unified BF16 precision computation, further reducing memory usage and computational overhead while maintaining generation quality |
| **LightX2V_3** | Introduces FP8 quantization technology to significantly reduce computational precision requirements, combined with Tiling VAE technology to optimize memory usage |
| **LightX2V_3-Distill** | Based on LightX2V_3 using 4-step distillation model(`infer_steps=4`, `enable_cfg=False`), further reducing inference steps while maintaining generation quality |
| **LightX2V_4** | Based on LightX2V_3 with TeaCache(teacache_thresh=0.2) caching reuse technology, achieving acceleration through intelligent redundant computation skipping |

### 🖥️ RTX 4090 Environment Configuration Descriptions

| Configuration | Technical Features |
|:--------------|:------------------|
| **Wan2GP(profile=3)** | Implementation based on [Wan2GP repository](https://github.com/deepbeepmeep/Wan2GP), using MMGP optimization technology. Profile=3 configuration is suitable for RTX 3090/4090 environments with at least 32GB RAM and 24GB VRAM, adapting to limited memory resources by sacrificing VRAM. Uses quantized models: [480P model](https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors) and [720P model](https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/wan2.1_image2video_720p_14B_quanto_mbf16_int8.safetensors) |
| **LightX2V_5** | Uses SageAttention2 to replace native attention mechanism, adopts DIT FP8+FP32 (partial sensitive layers) mixed precision computation, enables CPU offload technology, executes partial sensitive layers with FP32 precision, asynchronously offloads DIT inference process data to CPU, saves VRAM, with block-level offload granularity |
| **LightX2V_5-Distill** | Based on LightX2V_5 using 4-step distillation model(`infer_steps=4`, `enable_cfg=False`), further reducing inference steps while maintaining generation quality |
| **LightX2V_6** | Based on LightX2V_3 with CPU offload technology enabled, executes partial sensitive layers with FP32 precision, asynchronously offloads DIT inference process data to CPU, saves VRAM, with block-level offload granularity |
| **LightX2V_6-Distill** | Based on LightX2V_6 using 4-step distillation model(`infer_steps=4`, `enable_cfg=False`), further reducing inference steps while maintaining generation quality |

---

## 📁 Configuration Files Reference

Benchmark-related configuration files and execution scripts are available at:

| Type | Link | Description |
|:-----|:-----|:------------|
| **Configuration Files** | [configs/bench](https://github.com/ModelTC/LightX2V/tree/main/configs/bench) | Contains JSON files with various optimization configurations |
| **Execution Scripts** | [scripts/bench](https://github.com/ModelTC/LightX2V/tree/main/scripts/bench) | Contains benchmark execution scripts |

---

> 💡 **Tip**: It is recommended to choose the appropriate optimization solution based on your hardware configuration to achieve the best performance.
