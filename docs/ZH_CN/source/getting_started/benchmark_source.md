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
| **FastVideo** | 292 | 26 | **1.25x** | <video src="https://github.com/user-attachments/assets/26c01987-441b-4064-b6f4-f89347fddc15" width="200px"></video> |
| **LightX2V_1** | 250 | 53 | **1.46x** | <video src="https://github.com/user-attachments/assets/7bffe48f-e433-430b-91dc-ac745908ba3a" width="200px"></video> |
| **LightX2V_2** | 216 | 50 | **1.70x** | <video src="https://github.com/user-attachments/assets/0a24ca47-c466-433e-8a53-96f259d19841" width="200px"></video> |
| **LightX2V_3** | 191 | 35 | **1.92x** | <video src="https://github.com/user-attachments/assets/970c73d3-1d60-444e-b64d-9bf8af9b19f1" width="200px"></video> |
| **LightX2V_3-Distill** | 14 | 35 | **🏆 20.85x** | <video src="https://github.com/user-attachments/assets/b4dc403c-919d-4ba1-b29f-ef53640c0334" width="200px"></video> |
| **LightX2V_4** | 107 | 35 | **3.41x** | <video src="https://github.com/user-attachments/assets/49cd2760-4be2-432c-bf4e-01af9a1303dd" width="200px"></video> |

### 720P 5s视频

**测试配置:**
- **模型**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **参数**: infer_steps=40, seed=1234, enable_cfg=True

#### 性能对比


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

## RTX 4090 (~24GB显存)

### 480P 5s视频

*即将更新...*

### 720P 5s视频

*即将更新...*

---

## 表格说明

- **Wan2.1 Official**: 基于[Wan2.1官方仓库](https://github.com/Wan-Video/Wan2.1)
- **FastVideo**: 基于[FastVideo官方仓库](https://github.com/hao-ai-lab/FastVideo)，使用SageAttention后端
- **LightX2V_1**: 使用SageAttention2替换原生注意力机制，采用DIT BF16+FP32(部分敏感层)混合精度计算，在保持精度的同时提升计算效率
- **LightX2V_2**: 统一使用BF16精度计算，进一步减少显存占用和计算开销，同时保持生成质量
- **LightX2V_3**: 引入FP8量化技术显著减少计算精度要求，结合Tiling VAE技术优化显存使用
- **LightX2V_3-Distill**: 在LightX2V_3基础上使用4步蒸馏模型(`infer_step=4`, `enable_cfg=False`)，进一步减少推理步数并保持生成质量。
- **LightX2V_4**: 在LightX2V_3基础上加入TeaCache(teacache_thresh=0.2)缓存复用技术，通过智能跳过冗余计算实现加速
- **配置文件参考**: 基准测试相关的配置文件和运行脚本可在以下位置获取：
  - [配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/bench) - 包含各种优化配置的JSON文件
  - [运行脚本](https://github.com/ModelTC/LightX2V/tree/main/scripts/bench) - 包含基准测试的执行脚本
