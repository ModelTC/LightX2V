# VAE TensorRT Optimization Guide (Advanced Guide)

VAE (Variational Autoencoder) is a critical component in image generation/editing models, responsible for encoding and decoding between image and latent space. For high-performance inference scenarios, LightX2V provides **TensorRT acceleration**, achieving **2x+ speedup** for VAE operations.

This guide covers deep optimization testing for **Qwen-Image** series model VAE.

## Comparison

| Feature | **Baseline (PyTorch)** | **TensorRT Static Shape** | **TensorRT Multi-Ratio** |
| :--- | :--- | :--- | :--- |
| **Encoder Latency** | ~45 ms | **~22 ms** (2.0x) | **17-22 ms** (2.0-2.6x) |
| **Decoder Latency** | ~280 ms | **~130 ms** (2.2x) | **~130 ms** (2.2x) |
| **Resolution Support** | Any | Build-time resolution only | **Any** (auto-match + resize) |
| **Quality Impact** | None | None | **Geometric Consistency (Center Crop / Resize)** |
| **Deployment Complexity** | Low | Medium | Medium |
| **Use Case** | Development | Fixed resolution production | **Multi-resolution production** |

---

## 1. Environment Setup

### 1.1 Install TensorRT

TensorRT must match your CUDA version. Example for CUDA 12.x:

```bash
# Install TensorRT Python packages
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs

# Install ONNX dependencies (required for dynamo exporter)
pip install onnx onnxruntime onnxscript
```

**Verify installation:**
```bash
python -c "import tensorrt; print(tensorrt.__version__)"
# Should output something like: 10.15.1.29
```

### 1.2 Script Locations

All conversion tools are located at:
```
LightX2V/tools/convert/tensorrt/
├── convert_vae_trt.py    # VAE TensorRT conversion tool
└── README.md             # Usage documentation
```

---

## 2. TensorRT Static Shape Optimization

Suitable for **fixed resolution** production environments (e.g., unified 1024x1024).

### 2.1 Build TensorRT Engine

```bash
export model_path=/path/to/Qwen-Image-Edit-2511
export output_dir=/path/to/vae_trt_engines

python tools/convert/tensorrt/convert_vae_trt.py \
    --model_path $model_path \
    --output_dir $output_dir \
    --height 1024 --width 1024
```

**Build process:**
1. Export VAE Encoder/Decoder to ONNX (using PyTorch dynamo exporter)
2. Build FP16 TensorRT Engine (~5-8 minutes)
3. Save `.trt` files

**Generated files:**
```
/path/to/vae_trt_engines/
├── vae_encoder.onnx
├── vae_encoder.onnx.data  # External weights file
├── vae_encoder.trt        # TensorRT Engine
├── vae_decoder.onnx
├── vae_decoder.onnx.data
└── vae_decoder.trt
```

### 2.2 Performance Results

| Component | PyTorch | TensorRT (FP16) | Speedup |
| :--- | :--- | :--- | :--- |
| Encoder (1024x1024) | 45 ms | **21.6 ms** | **2.08x** |
| Decoder (1024x1024) | 68 ms | **35.6 ms** | **1.91x** |

---

## 3. TensorRT Multi-Ratio Optimization (Recommended)

Suitable for **variable input resolution** scenarios (e.g., I2I image editing).

### 3.1 Core Approach

1. **Pre-build engines for multiple aspect ratios**: Cover common ratios (1:1, 4:3, 16:9, etc.)
2. **Runtime auto-matching**: Select the closest matching engine for input
3. **Center Crop + Resize**: Preserve aspect ratio, minimize quality loss

### 3.2 Build All Ratio Engines

```bash
python tools/convert/tensorrt/convert_vae_trt.py \
    --model_path $model_path \
    --output_dir /path/to/vae_trt_multi_ratio \
    --multi_ratio
```

**Pre-built Engine List:**

| Name | Resolution | Aspect Ratio | Use Case |
| :--- | :--- | :--- | :--- |
| 1_1_1024 | 1024x1024 | 1:1 | Square images |
| 1_1_512 | 512x512 | 1:1 | Thumbnails |
| 4_3_1024 | 1024x768 | 4:3 | Landscape photos |
| 3_4_1024 | 768x1024 | 3:4 | Portrait photos |
| 16_9_1152 | 1152x640 | ~16:9 | Landscape video |
| 9_16_1152 | 640x1152 | ~9:16 | Portrait video |
| 3_2_1024 | 1024x672 | ~3:2 | Standard photos |
| 2_3_1024 | 672x1024 | ~2:3 | Portrait photos |

**Build time:** ~25-30 minutes (8 engines)

### 3.3 Performance Results

| Input Resolution | Matched Engine | Target Resolution | Latency |
| :--- | :--- | :--- | :--- |
| 512x512 | 1_1_1024 | 1024x1024 | **21.74 ms** |
| 1024x1024 | 1_1_1024 | 1024x1024 | **21.79 ms** |
| 800x600 | 3_4_1024 | 768x1024 | **17.35 ms** |
| 1920x1080 | 9_16_1080 | 720x1280 | **19.70 ms** |
| 1080x1920 | 16_9_1080 | 1280x720 | **19.85 ms** |
| 2048x1536 | 3_4_1024 | 768x1024 | **17.15 ms** |
| 720x1280 | 9_16_1080 | 720x1280 | **20.07 ms** |

### 3.4 Quality Impact

*   **Exact ratio match** (e.g., 720x1280 → 9_16_1080): No cropping, resize only, **nearly lossless**
*   **Close ratio** (e.g., 800x600 → 4:3 Engine): Minor center crop, **minimal impact**
*   **Large ratio difference**: Significant edge cropping, evaluate based on use case

---

## 4. Technical Details

### 4.1 Why True Dynamic Shape Is Not Supported

The VAE model contains **shape-dependent slicing operations** (e.g., positional encoding, caching mechanisms) that are baked as constants during ONNX export. TensorRT's dynamic shape optimization cannot handle this case.

**Error message:**
```
[TRT] [E] ISliceLayer has out of bounds access on axis 3
```

**Solution:** Use multi-ratio static engines + runtime selection.

### 4.2 ONNX Export Notes (Critical Fix)
Updated 2026-02-02: Fixed VAE Encoder precision issue.

**Issue**: PyTorch's `vae.encode()` includes `encoder` and `quant_conv` modules. Exporting only `vae.encoder` misses `quant_conv`, causing severe precision loss (**Cosine Similarity < 0.6**).

**Fix**: Use `EncoderWrapper` to encapsulate both modules during export.
```python
class EncoderWrapper(nn.Module):
    def __init__(self, encoder, quant_conv):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def forward(self, x):
        return self.quant_conv(self.encoder(x))
```
This fix is integrated into `convert_vae_trt.py`, ensuring perfect precision (**Cosine Similarity > 0.9999**).

Also, use PyTorch 2.x **dynamo exporter**:

### 4.4 End-to-End Speedup Analysis (Resolution Reduction)

The multi-ratio TRT scheme offers a **hidden advantage**: reduced computational load.

- **Scenario**: Input 1280x720 (16:9)
- **Baseline**: Processes 1280x720 pixels (921k px)
- **TRT Scheme**: Matches `16_9_1152` Engine (1152x640, 737k px)
- **Benefit**: 20% fewer pixels -> Faster DiT and VAE Decoder.
- **Trade-off**: Minor center crop (geometric consistency maintained, PSNR drop is expected due to crop).

### 4.3 TensorRT External Weights

The dynamo exporter generates `.onnx.data` external weight files. When parsing in TensorRT:

```python
parser.parse_from_file(onnx_path)  # Correct
# Not: parser.parse(f.read())      # Wrong, can't find external weights
```

---

## 5. Pipeline Integration

LightX2V natively supports TensorRT VAE through configuration files.

### 5.1 Configuration

**Multi-ratio mode (recommended):**

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "multi_ratio": true,
        "engine_dir": "/path/to/vae_trt_multi_ratio"
    }
}
```

**Static resolution mode:**

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

### 5.2 Configuration Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `vae_type` | string | `"tensorrt"` enables TRT, `"baseline"` uses PyTorch (default) |
| `trt_vae_config.multi_ratio` | bool | `true` for multi-ratio mode, `false` for static resolution |
| `trt_vae_config.engine_dir` | string | Multi-ratio engine directory path |
| `trt_vae_config.encoder_engine` | string | Static mode encoder engine path |
| `trt_vae_config.decoder_engine` | string | Static mode decoder engine path (optional) |

### 5.3 Automatic Fallback

If TensorRT is unavailable or engine files don't exist, the system automatically falls back to PyTorch VAE:

```
[WARNING] TensorRT engine files not found, falling back to PyTorch VAE
[INFO] Loading PyTorch baseline VAE
```

### 5.4 Example Configuration

See: `configs/qwen_image/qwen_image_i2i_2511_trt_vae.json`

### 5.5 Run Inference

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

## 6. Related Code

| File | Description |
| :--- | :--- |
| `tools/convert/tensorrt/convert_vae_trt.py` | TensorRT engine conversion tool |
| `tools/convert/tensorrt/README.md` | Conversion tool documentation |
| `lightx2v/models/video_encoders/hf/qwen_image/vae_trt.py` | TensorRT VAE wrapper class |
| `lightx2v/models/runners/qwen_image/qwen_image_runner.py` | Runner integration support |

---

## Summary

*   **Development/Debug**: Use default PyTorch VAE, no extra configuration needed.
*   **Fixed Resolution Production**: Use **Static Shape TensorRT Engine**.
*   **Multi-Resolution Production (Recommended)**: Use **Multi-Ratio TensorRT Engine**, auto-matches closest aspect ratio.

**Performance Summary:**
*   VAE Encoder: **3.0x** speedup (53ms vs 165ms)
*   VAE Decoder: **2.2x** speedup (130ms vs 280ms, requires Decoder Engines)
*   **End-to-End Pipeline**:
    *   **Standard (1:1)**: **1.3x** speedup (3.0s vs 3.9s)
    *   **Non-Standard**: **1.0x - 1.3x** speedup (depends on crop/resize benefits/overheads)
