# Benchmark

---

## H200 (~140GB VRAM)

**Software Environment:**
- **Python**: 3.11
- **PyTorch**: 2.7.1+cu128
- **SageAttention**: 2.2.0
- **vLLM**: 0.9.2
- **sgl-kernel**: 0.1.8

### 480P 5s Video

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-480P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-Lightx2v)
- **Parameters**: infer_steps=40, seed=42, enable_cfg=True

#### Performance Comparison

| Configuration | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:-----------------:|:--------------:|:-------:|:------------:|
| **Wan2.1 Official** | 366 | 71 | 1.0x | <video src="https://github.com/user-attachments/assets/24fb112e-c868-4484-b7f0-d9542979c2c3" width="200px"></video> |
| **FastVideo** | 292 | 26 | **1.25x** | <video src="" width="200px"></video> |
| **LightX2V_1** | 250 | 53 | **1.46x** | <video src="https://github.com/user-attachments/assets/7bffe48f-e433-430b-91dc-ac745908ba3a" width="200px"></video> |
| **LightX2V_2** | 216 | 50 | **1.70x** | <video src="https://github.com/user-attachments/assets/0a24ca47-c466-433e-8a53-96f259d19841" width="200px"></video> |
| **LightX2V_3** | 191 | 35 | **1.92x** | <video src="https://github.com/user-attachments/assets/970c73d3-1d60-444e-b64d-9bf8af9b19f1" width="200px"></video> |
| **LightX2V_3-Distill** | 14 | 35 | **🏆 20.85x** | <video src="" width="200px"></video> |
| **LightX2V_4** | 107 | 35 | **3.41x** | <video src="https://github.com/user-attachments/assets/49cd2760-4be2-432c-bf4e-01af9a1303dd" width="200px"></video> |

### 720P 5s Video

**Test Configuration:**
- **Model**: [Wan2.1-I2V-14B-720P-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-Lightx2v)
- **Parameters**: infer_steps=40, seed=42, enable_cfg=True

#### Performance Comparison

| Configuration | Inference Time(s) | GPU Memory(GB) | Speedup | Video Effect |
|:-------------|:-----------------:|:--------------:|:-------:|:------------:|
| **Wan2.1 Official** | 974 | 81 | 1.0x | <video src="" width="200px"></video> |
| **FastVideo** | 914 | 40 | **1.07x** | <video src="" width="200px"></video> |
| **LightX2V_1** | 807 | 65 | **1.21x** | <video src="" width="200px"></video> |
| **LightX2V_2** | 751 | 57 | **1.30x** | <video src="" width="200px"></video> |
| **LightX2V_3** | 671 | 43 | **1.45x** | <video src="" width="200px"></video> |
| **LightX2V_3-Distill** | 44 | 43 | **🏆 22.14x** | <video src="" width="200px"></video> |
| **LightX2V_4** | 344 | 46 | **2.83x** | <video src="" width="200px"></video> |

---

## RTX 4090 (~24GB VRAM)

### 480P 5s Video

*Coming soon...*

### 720P 5s Video

*Coming soon...*

---

## Configuration Descriptions

- **Wan2.1 Official**: Based on [Wan2.1 official repository](https://github.com/Wan-Video/Wan2.1)
- **FastVideo**: Based on [FastVideo official repository](https://github.com/hao-ai-lab/FastVideo), using SageAttention backend
- **LightX2V_1**: Uses SageAttention2 to replace native attention mechanism, adopts DIT BF16+FP32 (partial sensitive layers) mixed precision computation, improving computational efficiency while maintaining precision
- **LightX2V_2**: Unified BF16 precision computation, further reducing memory usage and computational overhead while maintaining generation quality
- **LightX2V_3**: Introduces FP8 quantization technology to significantly reduce computational precision requirements, combined with Tiling VAE technology to optimize memory usage
- **LightX2V_3-Distill**: Based on LightX2V_3 using 4-step distillation model(`infer_step=4`, `enable_cfg=False`), further reducing inference steps while maintaining generation quality.
- **LightX2V_4**: Based on LightX2V_3 with TeaCache(teacache_thresh=0.2) caching reuse technology, achieving acceleration through intelligent redundant computation skipping
