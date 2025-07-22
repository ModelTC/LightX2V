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

## 📖 配置说明

### 🖥️ H200 环境配置说明

| 配置 | 技术特点 |
|:-----|:---------|
| **Wan2.1 Official** | 基于[Wan2.1官方仓库](https://github.com/Wan-Video/Wan2.1)的原始实现 |
| **FastVideo** | 基于[FastVideo官方仓库](https://github.com/hao-ai-lab/FastVideo)，使用SageAttention2后端优化 |
| **LightX2V_1** | 使用SageAttention2替换原生注意力机制，采用DIT BF16+FP32(部分敏感层)混合精度计算，在保持精度的同时提升计算效率 |
| **LightX2V_2** | 统一使用BF16精度计算，进一步减少显存占用和计算开销，同时保持生成质量 |
| **LightX2V_3** | 引入FP8量化技术显著减少计算精度要求，结合Tiling VAE技术优化显存使用 |
| **LightX2V_3-Distill** | 在LightX2V_3基础上使用4步蒸馏模型(`infer_steps=4`, `enable_cfg=False`)，进一步减少推理步数并保持生成质量 |
| **LightX2V_4** | 在LightX2V_3基础上加入TeaCache(teacache_thresh=0.2)缓存复用技术，通过智能跳过冗余计算实现加速 |

### 🖥️ RTX 4090 环境配置说明

| 配置 | 技术特点 |
|:-----|:---------|
| **Wan2GP(profile=3)** | 基于[Wan2GP仓库](https://github.com/deepbeepmeep/Wan2GP)实现，使用MMGP优化技术。profile=3配置适用于至少32GB内存和24GB显存的RTX 3090/4090环境，通过牺牲显存来适应有限的内存资源。使用量化模型：[480P模型](https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors)和[720P模型](https://huggingface.co/DeepBeepMeep/Wan2.1/blob/main/wan2.1_image2video_720p_14B_quanto_mbf16_int8.safetensors) |
| **LightX2V_5** | 使用SageAttention2替换原生注意力机制，采用DIT FP8+FP32(部分敏感层)混合精度计算，启用CPU offload技术，将部分敏感层执行FP32精度计算，将DIT推理过程中异步数据卸载到CPU上，节省显存，offload粒度为block级别 |
| **LightX2V_5-Distill** | 在LightX2V_5基础上使用4步蒸馏模型(`infer_steps=4`, `enable_cfg=False`)，进一步减少推理步数并保持生成质量 |
| **LightX2V_6** | 在LightX2V_3基础上启用CPU offload技术，将部分敏感层执行FP32精度计算，将DIT推理过程中异步数据卸载到CPU上，节省显存，offload粒度为block级别 |
| **LightX2V_6-Distill** | 在LightX2V_6基础上使用4步蒸馏模型(`infer_steps=4`, `enable_cfg=False`)，进一步减少推理步数并保持生成质量 |

---

## 📁 配置文件参考

基准测试相关的配置文件和运行脚本可在以下位置获取：

| 类型 | 链接 | 说明 |
|:-----|:-----|:-----|
| **配置文件** | [configs/bench](https://github.com/ModelTC/LightX2V/tree/main/configs/bench) | 包含各种优化配置的JSON文件 |
| **运行脚本** | [scripts/bench](https://github.com/ModelTC/LightX2V/tree/main/scripts/bench) | 包含基准测试的执行脚本 |

---

> 💡 **提示**: 建议根据您的硬件配置选择合适的优化方案，以获得最佳的性能表现。
