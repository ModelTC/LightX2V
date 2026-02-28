# Qwen Image VAE TensorRT Acceleration Guide

To significantly improve the inference speed of the Qwen Image model, we have introduced TensorRT optimizations for both the VAE Encoder and Decoder.

Given the different input shape characteristics of Text-to-Image (T2I) and Image-to-Image (I2I) tasks, we have designed two acceleration strategies: **Static Shape Optimization** and **Dynamic Shape Optimization (Multi-Profile)**.

---

## 1. T2I: Static Shape Approach

In T2I tasks, the output image resolution is typically fixed (e.g., 16:9, 1:1, 4:3, etc.), meaning the inference shape is known in advance. Therefore, **Static Shape engines** are the optimal choice, as they completely eliminate the overhead of dynamic shape inference at the underlying level.

### 1.1 Key Advantages & Performance
*   **Peak Performance**: In standalone VAE-only testing, Encoder and Decoder achieve an average **~2.0x** speedup; in end-to-end service mode, VAE Decoder averages **~1.8x** speedup.
*   **On-Demand Loading (Lazy Load)**: Engines are loaded per-resolution on first request (~5GB VRAM per engine pair). When the resolution changes, old engines are automatically released before loading new ones. Compared to eager loading all engines (~25GB), this significantly reduces VRAM usage and is compatible with end-to-end inference scenarios.

#### Standalone VAE-Only Benchmark (H100, Pure VAE Inference)

| Aspect Ratio | Size (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc Speedup | PT Dec (ms) | TRT Dec (ms) | Dec Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16:9** | 1664x928 | 66.53 | **32.70** | **2.03x** | 103.65 | **49.66** | **2.09x** |
| **9:16** | 928x1664 | 65.72 | **32.22** | **2.04x** | 103.02 | **50.71** | **2.03x** |
| **1:1** | 1328x1328 | 78.16 | **41.95** | **1.86x** | 121.91 | **61.52** | **1.98x** |
| **4:3** | 1472x1140 | 73.99 | **37.23** | **1.99x** | 114.45 | **54.75** | **2.09x** |
| **3:4** | 768x1024 | 31.74 | **17.33** | **1.83x** | 50.77 | **26.86** | **1.89x** |

> Overall average speedup: Encoder ~1.95x, Decoder ~2.02x

#### End-to-End Service Mode Benchmark (H100, Qwen-Image-2512, 5 steps, VAE Decoder)

> Note: T2I tasks do not involve VAE Encoder (no input image), so only VAE Decoder is measured.

| Aspect Ratio | Size (WxH) | PT Dec (ms) | TRT Dec (ms) | Dec Speedup | First Load (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **16:9** | 1664x928 | 189.3 | **88.4** | **2.14x** | 343.9 |
| **9:16** | 928x1664 | 179.6 | **85.6** | **2.10x** | 226.4 |
| **1:1** | 1328x1328 | 157.6 | **106.2** | **1.48x** | 304.1 |
| **4:3** | 1472x1140 | 148.7 | **94.7** | **1.57x** | 238.0 |
| **3:4** | 768x1024 | 70.4 | **46.1** | **1.53x** | 178.2 |

> Overall average Decoder speedup: ~1.8x. The "First Load" column shows the one-time Lazy Load overhead when switching resolutions (includes engine deserialization). Subsequent requests at the same resolution incur no loading cost.

#### Performance Gap Between Standalone and E2E

The higher absolute latency in E2E mode is due to:
1. **Post-processing overhead**: E2E `decode()` includes `image_processor.postprocess()` (tensor → PIL image conversion), which is absent in standalone tests that return raw tensors.
2. **GPU resource contention**: In E2E, the Transformer (~30GB) + Text Encoder (~15GB) remain resident in VRAM. L2 cache is populated by the large model weights after DiT inference, forcing VAE to re-fetch from HBM.
3. **Memory fragmentation**: PyTorch's dynamic allocator and TRT's fixed execution context alternate CUDA memory usage. PyTorch is more affected by fragmentation (hence its larger absolute gap).

### 1.2 Engine Configuration
*   **Default engine directory**: `path/to/vae_trt_t2i_static`
*   **Directory structure requirements**:
    The root engine directory must contain sub-directories named exactly as follows, each holding the corresponding `.trt` engine files (the loading logic relies on these preset directory names for Lazy Load discovery):

    ```text
    vae_trt_t2i_static/
    ├── 16_9/        # Resolution 1664x928 (WxH)
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    ├── 9_16/        # Resolution 928x1664
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    ├── 1_1/         # Resolution 1328x1328
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    ├── 4_3/         # Resolution 1472x1140
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    └── 3_4/         # Resolution 768x1024
        ├── vae_encoder.trt
        └── vae_decoder.trt
    ```

### 1.3 Usage
In the corresponding JSON config file, set `vae_type` to `tensorrt`, point `trt_engine_path` to the root directory containing the resolution sub-folders, and set `multi_profile` to `false`:

```json
{
    "task": "t2i",
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "trt_engine_path": "path/to/vae_trt_t2i_static",
        "multi_profile": false
    }
}
```

---

## 2. I2I: Dynamic Shape (Multi-Profile) Approach

For I2I tasks, input images can have arbitrary dimensions, so the VAE must handle a dynamic range of input shapes. A purely static engine is insufficient.
To balance speed and flexibility, we use a **Multi-Profile engine** that bundles multiple optimization profiles (Opt Shapes) into a single engine file.

### 2.1 Key Advantages & Performance
*   **Strong Performance Gains**: In standalone VAE-only testing, Encoder and Decoder achieve an average **~1.45x** speedup. In end-to-end service mode, VAE Encoder averages **~1.66x** speedup, Decoder averages **~1.08x** speedup.
*   **Dynamic Optimal Matching**: At runtime, the engine automatically selects the closest optimization profile to the current input dimensions for the best memory layout and kernel execution plan.

#### Standalone VAE-Only Benchmark (H100, Pure VAE Inference, 10-iter avg)

**Encoder**:

| Resolution | Size (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 11.00 | **8.53** | **1.29x** |
| **1024x1024** | 1024x1024 | 42.85 | **27.56** | **1.55x** |
| **480p 16:9** | 848x480 | 17.25 | **12.00** | **1.44x** |
| **720p 16:9** | 1280x720 | 38.00 | **25.35** | **1.50x** |
| **768p 4:3** | 1024x768 | 31.98 | **21.76** | **1.47x** |

> Encoder average speedup: ~1.45x

**Decoder**:

| Resolution | Size (WxH) | PT Dec (ms) | TRT Dec (ms) | Dec Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 17.60 | **12.78** | **1.38x** |
| **1024x1024** | 1024x1024 | 68.16 | **44.93** | **1.52x** |
| **480p 16:9** | 848x480 | 27.67 | **18.85** | **1.47x** |
| **720p 16:9** | 1280x720 | 60.24 | **40.80** | **1.48x** |
| **768p 4:3** | 1024x768 | 51.14 | **34.92** | **1.46x** |

> Decoder average speedup: ~1.46x. Overall Encoder + Decoder average: ~1.45x

#### End-to-End Service Mode Benchmark (H100, qwen-image-edit-251130, 4 steps)

**VAE Encoder**:

| Resolution | Size (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 48.5 | **28.8** | **1.68x** |
| **1024x1024** | 1024x1024 | 48.2 | **28.4** | **1.70x** |
| **480p 16:9** | 848x480 | 48.7 | **29.6** | **1.64x** |
| **720p 16:9** | 1280x720 | 48.6 | **30.1** | **1.62x** |
| **768p 4:3** | 1024x768 | 49.2 | **29.8** | **1.65x** |

> Encoder average speedup: ~1.66x

**VAE Decoder**:

| Resolution | Size (WxH) | PT Dec (ms) | TRT Dec (ms) | Dec Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 138.4 | **134.0** | **1.03x** |
| **1024x1024** | 1024x1024 | 152.7 | **133.3** | **1.15x** |
| **480p 16:9** | 848x480 | 140.4 | **134.4** | **1.04x** |
| **720p 16:9** | 1280x720 | 139.0 | **134.2** | **1.04x** |
| **768p 4:3** | 1024x768 | 152.8 | **134.8** | **1.13x** |

> Decoder average speedup: ~1.08x

#### Performance Gap Between Standalone and E2E

**Encoder is faster in E2E (1.66x > standalone 1.45x)**:
- The I2I pipeline's `resize_mode: "adaptive"` normalizes all input images to a fixed total pixel count (`CONDITION_IMAGE_SIZE: 147456`). Therefore, the E2E Encoder always processes a **fixed resolution** regardless of the test image size. This particular resolution happens to hit a well-optimized TRT Multi-Profile slot.
- GPU resource contention impacts PyTorch's dynamic memory allocator more than TRT's fixed context, further widening the gap.

**Decoder speedup is significantly diluted in E2E (1.08x < standalone 1.46x)**:
- The `decode()` method includes `image_processor.postprocess(output_type="pil")` which performs tensor → PIL image conversion (denormalize → clamp → permute → to(uint8) → ToPILImage). This is a **CPU-bound operation** (~80-90ms) that TRT cannot accelerate.
- Standalone tests use `return_result_tensor=True` which skips this overhead; E2E service mode defaults to `return_result_tensor=False`.
- Adding a constant overhead inevitably dilutes the speedup ratio: e.g., for 1024x1024, standalone PT=68ms / TRT=45ms → 1.52x; adding ~85ms gives PT=153ms / TRT=130ms → 1.18x, matching the actual E2E measurement of 1.15x.

> **Conclusion**: The lower E2E Decoder speedup does not mean TRT is slower — it means the profiling scope includes CPU post-processing that TRT cannot optimize. For accurate TRT kernel speedup measurements, refer to the standalone benchmark data.

### 2.2 Engine Configuration
*   **Default engine directory**: `path/to/vae_trt_extended_mp`
*   **Built-in Optimization Profiles**:
    1.  `1_1_512` (512x512)
    2.  `1_1_1024` (1024x1024)
    3.  `16_9_480p` (480x848)
    4.  `16_9_720p` (720x1280)
    5.  `16_9_1080p` (1080x1920)
    6.  `9_16_720p` (1280x720)
    7.  `9_16_1080p` (1920x1080)
    8.  `4_3_768p` (768x1024)
    9.  `3_2_1080p` (1088x1620)

*(Maximum supported height/width is 1920 pixels)*

### 2.3 Usage
Since the engine is a single file containing multiple profiles (e.g., `vae_encoder_multi_profile.trt`), set `multi_profile` to `true`:

```json
{
    "task": "i2i",
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "trt_engine_path": "path/to/vae_trt_extended_mp",
        "multi_profile": true
    }
}
```

---

## 3. Deployment Prerequisites

Ensure that TensorRT dependencies are installed in your environment (these are typically pre-installed in our custom Docker images):

```bash
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs
```
