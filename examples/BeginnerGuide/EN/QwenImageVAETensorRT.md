# Qwen Image VAE TensorRT Acceleration Guide

To significantly improve the inference speed of the Qwen Image model, we have introduced TensorRT optimizations for both the VAE Encoder and Decoder.

Given the different input shape characteristics of Text-to-Image (T2I) and Image-to-Image (I2I) tasks, we have designed two acceleration strategies: **Static Shape Optimization** and **Dynamic Shape Optimization (Multi-Profile)**.

---

## 1. T2I: Static Shape Approach

In T2I tasks, the output image resolution is typically fixed (e.g., 16:9, 1:1, 4:3, etc.), meaning the inference shape is known in advance. Therefore, **Static Shape engines** are the optimal choice, as they completely eliminate the overhead of dynamic shape inference at the underlying level.

### 1.1 Key Advantages & Performance
*   **Peak Performance**: Achieves an average **~2.0x** speedup over native PyTorch (covering both Encoder and Decoder).
*   **Eager Loading**: All pre-built resolution engines are automatically loaded at service startup, ensuring low latency and high stability during inference.

**Benchmark Results (H100)**:

| Aspect Ratio | Size (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc Speedup | PT Dec (ms) | TRT Dec (ms) | Dec Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16:9** | 1664x928 | 66.53 | **32.70** | **2.03x** | 103.65 | **49.66** | **2.09x** |
| **9:16** | 928x1664 | 65.72 | **32.22** | **2.04x** | 103.02 | **50.71** | **2.03x** |
| **1:1** | 1328x1328 | 78.16 | **41.95** | **1.86x** | 121.91 | **61.52** | **1.98x** |
| **4:3** | 1472x1140 | 73.99 | **37.23** | **1.99x** | 114.45 | **54.75** | **2.09x** |
| **3:4** | 768x1024 | 31.74 | **17.33** | **1.83x** | 50.77 | **26.86** | **1.89x** |

> Overall average speedup: Encoder ~1.95x, Decoder ~2.02x

### 1.2 Engine Configuration
*   **Default engine directory**: `path/to/vae_trt_t2i_static`
*   **Directory structure requirements**:
    The root engine directory must contain sub-directories named exactly as follows, each holding the corresponding `.trt` engine files (the loading logic relies on these preset directory names for Eager Loading):

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
*   **Strong Performance Gains**: Achieves an average **~1.6x** speedup over PyTorch while maintaining compatibility with arbitrary input resolutions.
*   **Dynamic Optimal Matching**: At runtime, the engine automatically selects the closest optimization profile to the current input dimensions for the best memory layout and kernel execution plan.

**Benchmark Results (H100, Opt Shape inputs)**:

| Resolution | Size (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc Speedup | PT Dec (ms) | TRT Dec (ms) | Dec Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 12.01 | 12.28 | 0.98x | 22.06 | 14.54 | **1.52x** |
| **1024x1024** | 1024x1024 | 43.82 | 24.47 | **1.79x** | 69.61 | 43.08 | **1.62x** |
| **480p 16:9** | 848x480 | 17.78 | - | - | 28.83 | - | - |
| **720p 16:9** | 1280x720 | 38.71 | - | - | 61.04 | - | - |

> Overall average speedup ~1.6x (best results when input matches an Opt Shape profile)

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
