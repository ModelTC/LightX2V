# 基准测试

---

## H200 (~140GB显存)

**软件环境配置：**
- **Python**: 3.11
- **PyTorch**: 2.7.1+cu128
- **SageAttention**: 2.2.0
- **vLLM**: 0.9.2
- **sgl-kernel**: 0.1.8

### 480P 5s视频

**测试配置:**
- **模型**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **参数**: infer_steps=40, seed=42, enable_cfg=True

#### 性能对比

| 配置 | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:----------:|:---------------:|:------:|:--------:|
| **Wan2.1 Official** | 366 | 71 | 1.0x | <video src="https://github.com/user-attachments/assets/24fb112e-c868-4484-b7f0-d9542979c2c3" width="200px"></video> |
| **FastVideo** | 292 | 26 | **1.25x** | <video src="" width="200px"></video> |
| **LightX2V_1** | 250 | 53 | **1.46x** | <video src="https://github.com/user-attachments/assets/7bffe48f-e433-430b-91dc-ac745908ba3a" width="200px"></video> |
| **LightX2V_2** | 216 | 50 | **1.70x** | <video src="https://github.com/user-attachments/assets/0a24ca47-c466-433e-8a53-96f259d19841" width="200px"></video> |
| **LightX2V_3** | 191 | 35 | **1.92x** | <video src="https://github.com/user-attachments/assets/970c73d3-1d60-444e-b64d-9bf8af9b19f1" width="200px"></video> |
| **LightX2V_3-Distill** | 14 | 35 | **🏆 20.85x** | <video src="" width="200px"></video> |
| **LightX2V_4** | 107 | 35 | **3.41x** | <video src="https://github.com/user-attachments/assets/49cd2760-4be2-432c-bf4e-01af9a1303dd" width="200px"></video> |

### 720P 5s视频

**测试配置:**
- **模型**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **参数**: infer_steps=40, seed=1234, enable_cfg=True

#### 性能对比


| 配置 | 推理时间(s) | GPU显存占用(GB) | 加速比 | 视频效果 |
|:-----|:----------:|:---------------:|:------:|:--------:|
| **Wan2.1 Official** | 974 | 81 | 1.0x | <video src="" width="200px"></video> |
| **FastVideo** | 914 | 40 | **1.07x** | <video src="" width="200px"></video> |
| **LightX2V_1** | 807 | 65 | **1.21x** | <video src="" width="200px"></video> |
| **LightX2V_2** | 751 | 57 | **1.30x** | <video src="" width="200px"></video> |
| **LightX2V_3** | 671 | 43 | **1.45x** | <video src="" width="200px"></video> |
| **LightX2V_3-Distill** | 44 | 43 | **🏆 22.14x** | <video src="" width="200px"></video> |
| **LightX2V_4** | 344 | 46 | **2.83x** | <video src="" width="200px"></video> |

---

## RTX 4090 (~24GB显存)

### 480P 5s视频

*即将更新...*

### 720P 5s视频

*即将更新...*

---

## 配置说明

- **Wan2.1 Official**: 基于[Wan2.1官方仓库](https://github.com/Wan-Video/Wan2.1)
- **FastVideo**: 基于[FastVideo官方仓库](https://github.com/hao-ai-lab/FastVideo)，使用SageAttention后端
- **LightX2V_1**: 使用SageAttention2替换原生注意力机制，采用DIT BF16+FP32(部分敏感层)混合精度计算，在保持精度的同时提升计算效率
- **LightX2V_2**: 统一使用BF16精度计算，进一步减少显存占用和计算开销，同时保持生成质量
- **LightX2V_3**: 引入FP8量化技术显著减少计算精度要求，结合Tiling VAE技术优化显存使用
- **LightX2V_3-Distill**: 在LightX2V_3基础上使用4步蒸馏模型(`infer_step=4`, `enable_cfg=False`)，进一步减少推理步数并保持生成质量。
- **LightX2V_4**: 在LightX2V_3基础上加入TeaCache(teacache_thresh=0.2)缓存复用技术，通过智能跳过冗余计算实现加速
