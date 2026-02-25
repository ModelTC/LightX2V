# VAE TensorRT 优化指南 (Multi-Profile Edition)

VAE (Variational Autoencoder) 是图像生成/编辑模型中的关键组件。LightX2V 提供了基于 **TensorRT Multi-Profile** 的高性能加速方案，支持动态分辨率且性能提升显著。

## 方案对比

| 特性 | **Baseline (PyTorch)** | **TensorRT (Multi-Profile)** |
| :--- | :--- | :--- |
| **推理延迟 (Encoder)** | ~165 ms | **~28-56 ms** (3x-6x) |
| **推理延迟 (Decoder)** | ~280 ms | **~45-90 ms** (3x-6x) |
| **分辨率支持** | 任意 | **任意** (64x64 ~ 1920x1920) |
| **部署文件** | 无 | 单个 Engine 文件 |
| **显存占用** | 较低 | 略高 (用于 Workspace) |

---

## 1. 环境准备

### 1.1 安装 TensorRT

需安装 TensorRT Python 包（推荐 TensorRT 10.x + CUDA 12.x）：

```bash
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs
pip install onnx onnxruntime onnxscript
```

验证安装: `python -c "import tensorrt; print(tensorrt.__version__)"`

### 1.2 脚本位置

```
LightX2V/tools/convert/tensorrt/
├── convert_vae_trt.py    # 转换工具
```

---

## 2. 构建 Multi-Profile TensorRT Engine (推荐)

该模式构建**单个 Engine 文件**，内部包含多个优化 Profile，支持动态分辨率输入。

### 2.1 执行构建

```bash
export model_path=/path/to/Qwen-Image-Edit-2511
export output_dir=/path/to/vae_trt_mp

# 使用 --multi_profile 标志
python tools/convert/tensorrt/convert_vae_trt.py \
    --model_path $model_path \
    --output_dir $output_dir \
    --multi_profile \
    --build_decoder
```

**构建耗时**: 约 20-30 分钟 (包含 Encoder 和 Decoder)。

**产出文件**:
```
/path/to/vae_trt_mp/
├── vae_encoder_multi_profile.trt  # 单文件，支持所有分辨率
└── vae_decoder_multi_profile.trt  # (可选)
```

### 2.2 性能与优势

**Multi-Profile 优势**:
1.  **全动态分辨率**: 支持 64x64 到 1920x1920 范围内的任意分辨率。
2.  **无损画幅**: 无需 Center Crop，完整保留输入图像内容。
3.  **极速切换**: 运行时根据输入自动选择最佳 Profile，切换零开销。

**实测数据 (H100)**:

| 分辨率 | Encoder (ms) | Decoder (ms) | Total E2E (ms) | 加速比 (vs PT) |
| :--- | :--- | :--- | :--- | :--- |
| **512x512** | 8.5 | 13.1 | **21.5** | **~15x** |
| **1024x1024** | 27.9 | 44.9 | **72.9** | **~4x** |
| **1280x720** | 25.3 | 40.5 | **65.8** | **~4x** |
| **1920x1088** | 56.3 | 90.3 | **146.6** | **~3x** |

---

## 3. 集成到推理 Pipeline

### 3.1 配置文件
 
 在 `configs/*.json` 中添加 `trt_vae_config`：
 
 ```json
 {
     "vae_type": "tensorrt",
     "trt_vae_config": {
         "multi_profile": true,
         "trt_engine_path": "/path/to/vae_trt_mp"
     }
 }
 ```
 
 **参数说明**:
 *   `trt_engine_path`: 统一引擎路径配置。
     *   **目录模式 (推荐)**: 指向包含引擎的目录。脚本会自动查找 `vae_encoder_multi_profile.trt` 和 `vae_decoder_multi_profile.trt` (I2I模式) 或各分辨率子目录 (T2I模式)。
     *   **文件模式**: 直接指向 Encoder Engine 文件 (旧方式)。
 *   `multi_profile`: I2I 模式需设为 `true`。T2I/Static 模式设为 `false`。
 
 ...
 


### 3.2 架构限制说明 (重要)

Qwen-Image 模型架构采用 Patch Size = 2x2 的设计。为保证 Latent 空间对齐：
*   **输入分辨率的长和宽必须是 16 的倍数**。
*   例如: `1920x1080` (1088 是 16 倍数，1080 不是) **不支持**，会导致 Pipeline 报错。
*   **建议**: 将 1080p 图像 Resize/Pad 到 `1920x1088` 后再输入。

---

## 4. 常见问题

### 4.1 为什么不使用以前的"多文件夹"多比例模式？
旧的 Multi-Ratio 模式需要预先生成大量独立的 `.trt` 文件（如 `vae_encoder_1_1_1024.trt`），部署繁琐且不支持任意非标准分辨率。Multi-Profile 模式完全替代了旧方案，部署更简单，覆盖更全面。

### 4.2 精度损失？
Multi-Profile 模式经过验证，Encoder/Decoder 的重构误差 (MSE) 低于 0.001，肉眼不可见差异。Cosine Similarity > 0.9995。

### 4.3 显存占用
TensorRT Engine 加载后会占用一定的显存（约 200MB-500MB，取决于 Workspace 配置）。相比 PyTorch 略有增加，但换来了巨大的速度提升。

### 4.4 ONNX 导出细节
工具内部使用了 `EncoderWrapper` 和 `DecoderWrapper` 封装，确保了 VAE `quant_conv` 和 `post_quant_conv` 层被包含在 Engine 中，这是保证精度的关键。同时使用 Dynamo 导出器以支持动态 Shape 操作。

---

## 5. Qwen T2I VAE 专项优化 (New)

针对 Text-to-Image (T2I) 任务中分辨率相对固定的特性（如 16:9, 9:16 等），我们提供了进一步的优化方案：**静态 Shape (Static)** 引擎。

### 5.1 方案选择建议

| 场景 | 推荐方案 | 理由 |
| :--- | :--- | :--- |
| **I2I (图生图)** | Multi-Profile | 输入分辨率不可控，需动态支持任意尺寸。此模式需设置 `"multi_profile": true`。 |
| **T2I (文生图)** | **Static Shape** | 生成分辨率通常固定 (如 1024x1024, 720p)，静态引擎无额外 Overhead，性能更极致。无需设置 `multi_profile` (默认 False)。 |

### 5.2 构建 T2I 静态引擎

我们提供了专用脚本，一次性为 5 种主流 T2I 分辨率构建独立的静态引擎：

```bash
# 5 种预设分辨率: 16:9, 9:16, 1:1, 4:3, 3:4
python tools/convert/tensorrt/convert_vae_trt_t2i_static.py \
    --vae_path /path/to/models/Qwen/Qwen-Image-2512/vae \
    --output_dir /path/to/output/vae_trt_t2i_static
```

**产出结构**:
```
/path/to/output/vae_trt_t2i_static/
├── 16_9/  (1664x928)
│   ├── vae_encoder.trt
│   └── vae_decoder.trt
├── 9_16/  (928x1664)
├── 1_1/   (1328x1328)
├── 4_3/   (1472x1140)
└── 3_4/   (768x1024)
```

> **注意**: 特别支持了 **1140** 分辨率 (4:3)，无需强制对齐到 16 的倍数，这是 Multi-Profile 模式难以做到的。

### 5.3 T2I 性能对比 (Baseline vs Static)

我们在 H100 上对比了 T2I 场景下 PyTorch 原生实现、Multi-Profile 引擎与 Static 引擎的性能。

**测试数据**:

| Resolution | Type | PyTorch (ms) | Multi-Profile (ms) | Static (ms) | 加速比 (Static vs PT) |
|------------|------|--------------|--------------------|-------------|------------------------|
| **1664x928** | ENC | 67.66 | 48.09 | **33.31** | **2.03x** |
| | DEC | 102.50 | 94.48 | **51.52** | **1.99x** |
| **928x1664** | ENC | 67.24 | 49.52 | **34.18** | **1.97x** |
| | DEC | 102.55 | 94.46 | **50.05** | **2.05x** |
| **1328x1328**| ENC | 79.86 | 60.12 | **41.68** | **1.92x** |
| | DEC | 121.01 | 110.57 | **61.16** | **1.98x** |
| **1472x1140**| ENC | 75.89 | 53.06 | **36.67** | **2.07x** |
| | DEC | 114.96 | 101.36 | **55.14** | **2.08x** |
| **768x1024** | ENC | 32.58 | 25.46 | **17.35** | **1.88x** |
| | DEC | 50.85 | 48.63 | **26.60** | **1.91x** |

**结论**: 
*   **Static Engine** 相比 PyTorch 原生实现提供了 **~2倍** 的端到端加速。
*   相比 Multi-Profile 引擎，Static 引擎在 T2I 固定分辨率场景下进一步提升了 **~30-45%** 的性能。
*   对于 T2I 生产环境，强烈建议使用 **Static Engine**。

### 5.4 T2I 配置文件示例

针对 T2I 任务，无需指定 `multi_profile` (默认 false)，直接将 `trt_engine_path` 指向静态引擎的**根目录**，程序会在推理时**按需动态加载 (Lazy Loading)** 对应分辨率的引擎，以节省显存。

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "trt_engine_path": "/path/to/vae_trt_t2i_static"
    }
}
```

> **注意**:
> 1. `trt_engine_path` 必须配置为包含子目录（`16_9`, `1_1` 等）的**根目录路径**。
> 2. 程序采用 **Lazy Loading** 策略，仅在首次遇到某种分辨率时加载对应的 Engine，避免一次性加载所有引擎导致 OOM。
> 3. 目录结构需符合 `convert_vae_trt_t2i_static.py` 的产出规范。

### 5.5 I2I 场景性能分析 (综合对比)

I2I 任务通常涉及多种非固定分辨率。我们以标准 **1024x1024** 为基准，对比了 PyTorch 原生、torch.compile、TensorRT Multi-Profile 和 TensorRT Static 四种方案的性能差异。

**测试数据 (H100, 1024x1024)**:

| 方案 | Encoder (ms) | Decoder (ms) | 端到端加速比 (vs PT) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Eager)** | 44.1 | 68.4 | 1.00x | Baseline |
| **torch.compile** | 35.1 | 108.7 | **0.78x (Regress)** | Decoder 显著变慢，不推荐 |
| **TensorRT (Multi-Profile)** | 27.3 | 44.2 | **1.57x** | 支持任意分辨率，**推荐用于 I2I** |
| **TensorRT (Static)** | **21.8** | **36.5** | **1.93x** | 性能最佳，但需固定分辨率 |

**不同分辨率下的扩展性对比 (Total Latency)**:

| 分辨率 | PyTorch | TRT Multi-Profile | 加速比 |
| :--- | :--- | :--- | :--- |
| **512x512** | 29.2 ms | ~20.0 ms | ~1.46x |
| **1024x1024** | 112.5 ms | 71.5 ms | 1.57x |
| **1920x1088 (1080p)** | 235.8 ms | 145.0 ms | **1.63x** |

**结论**:
1.  **关于 torch.compile**: 虽然 Encoder 有 ~1.2x 加速，但 Decoder 性能出现倒退。对于包含大量卷积和上下采样的 VAE 模型，编译收益不稳定。
2.  **Multi-Profile vs Static**: Multi-Profile 相比 Static 引擎仅有约 **20-25%** 的性能损耗，但换来了对任意分辨率（如 1080p, 720p 等非标尺寸）的完整支持。
3.  **建议**: 在 **I2I (图生图)** 场景下，由于用户上传图片尺寸不一，**强烈推荐使用 TensorRT Multi-Profile 方案**，它在保持 1.6x 稳定加速的同时提供了最大的灵活性。

---

## 6. torch.compile 与 CUDA Graphs 对比评估

除 TensorRT 外，PyTorch 原生的 `torch.compile` 和 CUDA Graphs 也是常见的推理加速手段。我们针对 Qwen Image VAE 进行了对比测试。

### 6.1 VAE 组件测试结果

**测试环境**: H100, PyTorch 2.x, 1024x1024 RGB Input
**Unit**: Latency (ms)

| 组件 | Baseline (Eager) | torch.compile (Default) | Speedup | TensorRT (Static) | TRT Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder** | 44.1 ms | 35.1 ms | **1.26x** | ~17.4 ms | **2.53x** |
| **Decoder** | 67.9 ms | 108.7 ms (Regress) | 0.62x | ~26.6 ms | **2.55x** |

**结论**: 
*   `torch.compile` 对于 Qwen VAE 模型（包含大量小卷积和上采样操作）效果不佳，甚至在 Decoder 上导致显著变慢。
*   **强烈建议使用 TensorRT**，可获得稳定且显著的加速（~2.5x）。

### 6.2 端到端加速效果 (End-to-End)

我们在 Qwen Image I2I (4-Steps) 任务上测试了 `torch.compile` 的实际收益。

**测试环境**: H100, 1024x1024, 4 Steps
**配置**: `torch.compile(mode='reduce-overhead')` 仅应用于 **DiT** (Transformer)。

| 编译对象 | E2E Latency (Total) | Speedup | Latency Reduction |
| :--- | :--- | :--- | :--- |
| **Baseline** | ~4.75 s | 1.00x | - |
| **Compile DiT Only** | ~4.24 s | **1.12x** | **511 ms** |
| **Compile VAE** | - | - | *不推荐 (见上文)* |

**分析**:
*   DiT 模型作为 Transformer 架构，非常适合 `torch.compile` (算子融合+CUDA Graphs)，贡献了约 0.5s 的加速。
*   VAE 部分建议保持 Eager 模式或使用 TensorRT，不要对其使用 `torch.compile`。

### 6.3 分析

1.  **torch.compile**: 
    - 适合 Transformer 类模型 (DiT, LLM)。
    - 对复杂 CNN (如 VAE Decoder) 可能出现性能倒退。

2.  **TensorRT**:
    - 对 CNN 和 Transformer 均有极佳优化效果。
    - 是 VAE 加速的首选方案。

### 6.4 选择建议

| 场景 | 推荐方案 |
| :--- | :--- |
| **快速原型 (DiT)** | torch.compile |
| **生产部署 (全流程)** | TensorRT VAE + torch.compile DiT (或 TensorRT DiT) |
