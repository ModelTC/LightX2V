# VAE TensorRT 优化指南 (Advanced Guide)

VAE (Variational Autoencoder) 是图像生成/编辑模型中的关键组件，负责图像与 latent 空间之间的编解码。对于高性能推理场景，LightX2V 提供了 **TensorRT 加速方案**，可将 VAE 性能提升 **2x 以上**。

本指南针对 **Qwen-Image** 系列模型的 VAE 进行了深度优化测试。

## 方案对比

| 特性 | **Baseline (PyTorch)** | **TensorRT 静态 Shape** | **TensorRT 多比例** |
| :--- | :--- | :--- | :--- |
| **推理延迟 (Encoder)** | ~165 ms* | **~53 ms** (3.1x) | **~56 ms** (2.9x) |
| **推理延迟 (Decoder)** | ~280 ms | **~130 ms** (2.2x) | **~130 ms** (2.2x) |
| **分辨率支持** | 任意 | 仅构建时分辨率 | **任意** (自动匹配+动态Resize) |
| **质量影响** | 无 | 无 | **几何一致性 (Center Crop / Resize)** |
| **部署复杂度** | 低 | 中 | 中 |
| **适用场景** | 开发调试 | 固定分辨率生产环境 | **多分辨率生产环境** |

---

## 1. 环境准备

### 1.1 安装 TensorRT

TensorRT 需要与 CUDA 版本匹配。以下以 CUDA 12.x 为例：

```bash
# 安装 TensorRT Python 包
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs

# 安装 ONNX 相关依赖 (dynamo 导出器需要)
pip install onnx onnxruntime onnxscript
```

**验证安装:**
```bash
python -c "import tensorrt; print(tensorrt.__version__)"
# 应输出类似: 10.15.1.29
```

### 1.2 脚本位置

所有转换工具位于:
```
LightX2V/tools/convert/tensorrt/
├── convert_vae_trt.py    # VAE TensorRT 转换工具
└── README.md             # 使用文档
```

---

## 2. TensorRT 静态 Shape 优化

适用于**固定分辨率**的生产环境 (如统一 1024x1024)。

### 2.1 构建 TensorRT Engine

```bash
export model_path=/path/to/Qwen-Image-Edit-2511
export output_dir=/path/to/vae_trt_engines

python tools/convert/tensorrt/convert_vae_trt.py \
    --model_path $model_path \
    --output_dir $output_dir \
    --height 1024 --width 1024
```

**构建过程说明:**
1. 导出 VAE Encoder/Decoder 到 ONNX (使用 PyTorch dynamo 导出器)
2. 构建 FP16 TensorRT Engine (约需 5-8 分钟)
3. 保存 `.trt` 文件

**生成文件:**
```
/path/to/vae_trt_engines/
├── vae_encoder.onnx
├── vae_encoder.onnx.data  # 外部权重文件
├── vae_encoder.trt        # TensorRT Engine
├── vae_decoder.onnx
├── vae_decoder.onnx.data
└── vae_decoder.trt
```

### 2.2 性能测试结果

| 组件 | PyTorch | TensorRT (FP16) | 加速比 |
| :--- | :--- | :--- | :--- |
| Encoder (1024x1024) | 45 ms | **21.6 ms** | **2.08x** |
| Decoder (1024x1024) | 68 ms | **35.6 ms** | **1.91x** |

---

## 3. TensorRT 多比例优化 (推荐)

适用于**用户输入分辨率不固定**的场景 (如 I2I 图像编辑)。

### 3.1 核心思路

1. **预构建多个宽高比的 Engine**: 覆盖常见比例 (1:1, 4:3, 16:9 等)
2. **运行时自动匹配**: 选择最接近输入比例的 Engine
3. **Center Crop + Resize**: 保持宽高比，最小化质量损失

### 3.2 构建所有比例 Engine

```bash
python tools/convert/tensorrt/convert_vae_trt.py \
    --model_path $model_path \
    --output_dir /path/to/vae_trt_multi_ratio \
    --multi_ratio
```

**预构建的 Engine 列表:**

| 名称 | 分辨率 | 宽高比 | 适用场景 |
| :--- | :--- | :--- | :--- |
| 1_1_1024 | 1024x1024 | 1:1 | 正方形图像 |
| 1_1_512 | 512x512 | 1:1 | 小图/缩略图 |
| 4_3_1024 | 1024x768 | 4:3 | 横向照片 |
| 3_4_1024 | 768x1024 | 3:4 | 竖向照片 |
| 16_9_1152 | 1152x640 | ~16:9 | 横向视频 |
| 9_16_1152 | 640x1152 | ~9:16 | 竖向视频 |
| 3_2_1024 | 1024x672 | ~3:2 | 标准照片 |
| 2_3_1024 | 672x1024 | ~2:3 | 竖向照片 |

**构建时间:** 约 25-30 分钟 (8 个 Engine)

### 3.3 性能测试结果

| 输入分辨率 | 匹配 Engine | 目标分辨率 | 延迟 |
| :--- | :--- | :--- | :--- |
| 512x512 | 1_1_1024 | 1024x1024 | **21.74 ms** |
| 1024x1024 | 1_1_1024 | 1024x1024 | **21.79 ms** |
| 800x600 | 3_4_1024 | 768x1024 | **17.35 ms** |
| 1920x1080 | 9_16_1080 | 720x1280 | **19.70 ms** |
| 1080x1920 | 16_9_1080 | 1280x720 | **19.85 ms** |
| 2048x1536 | 3_4_1024 | 768x1024 | **17.15 ms** |
| 720x1280 | 9_16_1080 | 720x1280 | **20.07 ms** |

### 3.4 质量影响说明

*   **比例完全匹配时** (如 720x1280 → 9_16_1080): 无裁剪，仅 resize，**几乎无损**
*   **比例接近时** (如 800x600 → 4:3 Engine): 轻微 center crop，**影响极小**
*   **比例差异较大时**: 会裁剪较多边缘区域，建议根据业务需求评估

---

## 4. 技术细节

### 4.1 为什么不支持真正的动态 Shape？

VAE 模型内部存在**基于输入形状计算的切片操作** (如位置编码、缓存机制)，这些操作在 ONNX 导出时被固化为常量。TensorRT 的动态 Shape 优化无法处理这种情况。

**具体表现:**
```
[TRT] [E] ISliceLayer has out of bounds access on axis 3
```

**解决方案:** 使用多比例静态 Engine + 运行时选择。

### 4.2 ONNX 导出注意事项

**问题**: PyTorch 的 `vae.encode()` 包含 `encoder` 和 `quant_conv` 两个模块，但仅导出 `vae.encoder` 会导致 `quant_conv` 被遗漏，造成 **Cosine Similarity < 0.6** 的严重精度损失。

**修复**: 使用 `EncoderWrapper` 将两者封装为一个 Module 进行导出。
```python
class EncoderWrapper(nn.Module):
    def __init__(self, encoder, quant_conv):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        
    def forward(self, x):
        return self.quant_conv(self.encoder(x))
```
此修复已集成在 `convert_vae_trt.py` 中，确保了 **Cosine Similarity > 0.9999** 的完美精度。

此外，必须使用 PyTorch 2.x 的 **dynamo 导出器** 以支持 `aten::_upsample_nearest_exact2d` 操作。

### 4.3 端到端加速分析 (Resolution Reduction)

多比例 TRT 方案的一个**隐藏优势**是降低了系统的整体计算负载。

- **场景**: 输入 1280x720 (16:9)
- **Baseline**: 处理 1280x720 像素 (921k px)
- **TRT 方案**: 匹配到 `16_9_1152` Engine (1152x640, 737k px)
- **收益**: 像素量减少 20%，导致 DiT (扩散模型) 和 VAE Decoder 的耗时大幅降低。
- **代价**: 图像边缘会被轻微裁剪 (Center Crop)，PSNR 会因此下降 (这是预期的几何差异，非画质损失)。

### 4.4 TensorRT 解析外部权重

dynamo 导出器会生成 `.onnx.data` 外部权重文件，TensorRT 解析时需使用:

```python
parser.parse_from_file(onnx_path)  # 正确
# 而非: parser.parse(f.read())    # 错误，找不到外部权重
```

---

## 5. 集成到推理 Pipeline

LightX2V 已原生支持 TensorRT VAE，通过配置文件即可启用。

### 5.1 配置方式

**多比例模式 (推荐):**

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "multi_ratio": true,
        "engine_dir": "/path/to/vae_trt_multi_ratio"
    }
}
```

**静态分辨率模式:**

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "multi_ratio": false,
        "encoder_engine": "/path/to/vae_encoder.trt",
        "decoder_engine": "/path/to/vae_decoder.trt"
    }
}
```

### 5.2 配置参数说明

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `vae_type` | string | `"tensorrt"` 启用 TRT，`"baseline"` 使用 PyTorch (默认) |
| `trt_vae_config.multi_ratio` | bool | `true` 多比例模式，`false` 静态分辨率 |
| `trt_vae_config.engine_dir` | string | 多比例 Engine 目录路径 |
| `trt_vae_config.encoder_engine` | string | 静态模式 Encoder Engine 路径 |
| `trt_vae_config.decoder_engine` | string | 静态模式 Decoder Engine 路径 (可选) |

### 5.3 自动回退机制

如果 TensorRT 不可用或 Engine 文件不存在，系统会自动回退到 PyTorch VAE：

```
[WARNING] TensorRT engine files not found, falling back to PyTorch VAE
[INFO] Loading PyTorch baseline VAE
```

### 5.4 示例配置文件

完整示例见: `configs/qwen_image/qwen_image_i2i_2511_trt_vae.json`

### 5.5 运行推理

```bash
python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path /path/to/Qwen-Image-Edit-2511 \
    --config_json configs/qwen_image/qwen_image_i2i_2511_trt_vae.json \
    --prompt "..." \
    --image_path "input.png" \
    --save_result_path output.png
```

---

## 6. 相关代码

| 文件 | 说明 |
| :--- | :--- |
| `tools/convert/tensorrt/convert_vae_trt.py` | TensorRT Engine 转换工具 |
| `tools/convert/tensorrt/README.md` | 转换工具使用文档 |
| `lightx2v/models/video_encoders/hf/qwen_image/vae_trt.py` | TensorRT VAE 封装类 |
| `lightx2v/models/runners/qwen_image/qwen_image_runner.py` | Runner 集成支持 |

---

## 总结建议

*   **开发调试**: 使用默认 PyTorch VAE，无需额外配置。
*   **固定分辨率生产**: 使用 **静态 Shape TensorRT Engine**。
*   **多分辨率生产 (推荐)**: 使用 **多比例 TensorRT Engine**，自动匹配最接近的宽高比。

**性能提升总结:**
*   VAE Encoder: **3.0x** 加速 (53ms vs 165ms)
*   VAE Decoder: **2.2x** 加速 (130ms vs 280ms, 需构建 Decoder Engines)
*   **端到端 Pipeline**:
    *   **标准比例 (1:1)**: **1.3x** 加速 (3.0s vs 3.9s)
    *   **非标准比例**: **1.0x - 1.3x** 加速 (取决于是否产生 Crop/Resize 带来的额外收益/损耗)

