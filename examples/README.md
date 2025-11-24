# LightX2V Usage Examples

This document introduces how to use LightX2V for video generation, including basic usage and advanced configurations.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Basic Usage Examples](#basic-usage-examples)
- [Model Path Configuration](#model-path-configuration)
- [Creating Generator](#creating-generator)
- [Advanced Configurations](#advanced-configurations)
  - [Parameter Offloading](#parameter-offloading)
  - [Model Quantization](#model-quantization)
  - [Parallel Inference](#parallel-inference)
  - [Feature Caching](#feature-caching)
  - [Lightweight VAE](#lightweight-vae)

## 🔧 Environment Setup

Please refer to the main project's [Quick Start Guide](../docs/EN/source/getting_started/quickstart.md) for environment setup.

## 🚀 Basic Usage Examples

A minimal code example can be found in `examples/wan_t2v.py`:

```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-T2V-14B",
    model_cls="wan2.1",
    task="t2v",
)

pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=50,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    sample_shift=5.0,
)

seed = 42
prompt = "Your prompt here"
negative_prompt = ""
save_result_path="/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
```

## 📁 Model Path Configuration

### Basic Configuration

Pass the model path to `LightX2VPipeline`:

```python
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",  # For wan2.1, use "wan2.1"
    task="i2v",
)
```

### Specifying Multiple Model Weight Versions

When there are multiple versions of bf16 precision DIT model safetensors files in the `model_path` directory, you need to use the following parameters to specify which weights to use:

- **`dit_original_ckpt`**: Used to specify the original DIT weight path for models like wan2.1 and hunyuan15
- **`low_noise_original_ckpt`**: Used to specify the low noise branch weight path for wan2.2 models
- **`high_noise_original_ckpt`**: Used to specify the high noise branch weight path for wan2.2 models

**Usage Example:**

```python
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",
    task="i2v",
    low_noise_original_ckpt="/path/to/low_noise_model.safetensors",
    high_noise_original_ckpt="/path/to/high_noise_model.safetensors",
)
```

## 🎛️ Creating Generator

### Loading from Configuration File

The generator can be loaded directly from a JSON configuration file. Configuration files are located in the `configs` directory:

```python
pipe.create_generator(config_json="../configs/wan/wan_t2v.json")
```

### Creating Generator Manually

You can also create the generator manually and configure multiple parameters:

```python
pipe.create_generator(
    attn_mode="flash_attn2",  # Options: flash_attn2, flash_attn3, sage_attn2, sage_attn3 (B-architecture GPUs)
    infer_steps=50,           # Number of inference steps
    num_frames=81,            # Number of video frames
    height=480,               # Video height
    width=832,                # Video width
    guidance_scale=5.0,       # CFG guidance strength (CFG disabled when =1)
    sample_shift=5.0,         # Sample shift
    fps=16,                   # Frame rate
    aspect_ratio="16:9",      # Aspect ratio
    boundary=0.900,           # Boundary value
    boundary_step_index=2,    # Boundary step index
    denoising_step_list=[1000, 750, 500, 250],  # Denoising step list
)
```

**Parameter Description:**
- **Resolution**: Specified via `height` and `width`
- **CFG**: Specified via `guidance_scale` (set to 1 to disable CFG)
- **FPS**: Specified via `fps`
- **Video Length**: Specified via `num_frames`
- **Inference Steps**: Specified via `infer_steps`
- **Sample Shift**: Specified via `sample_shift`
- **Attention Mode**: Specified via `attn_mode`, options include `flash_attn2`, `flash_attn3`, `sage_attn2`, `sage_attn3` (for B-architecture GPUs)

## ⚙️ Advanced Configurations

**⚠️ Important: When manually creating a generator, you can configure some advanced options. All advanced configurations must be specified before `create_generator()`, otherwise they will not take effect!**

### Parameter Offloading

Significantly reduces memory usage with almost no impact on inference speed. Suitable for RTX 30/40/50 series GPUs.

```python
pipe.enable_offload(
    cpu_offload=True,              # Enable CPU offloading
    offload_granularity="block",   # Offload granularity: "block" or "phase"
    text_encoder_offload=False,    # Whether to offload text encoder
    image_encoder_offload=False,   # Whether to offload image encoder
    vae_offload=False,             # Whether to offload VAE
)
```

**Notes:**
- For Wan models, `offload_granularity` supports both `"block"` and `"phase"`
- For HunyuanVideo-1.5, only `"block"` is currently supported

### Model Quantization

Quantization can significantly reduce memory usage and accelerate inference.

```python
pipe.enable_quantize(
    dit_quantized=False,                    # Whether to use quantized DIT model
    text_encoder_quantized=False,           # Whether to use quantized text encoder
    image_encoder_quantized=False,          # Whether to use quantized image encoder
    dit_quantized_ckpt=None,                # DIT quantized weight path (required when model_path doesn't contain quantized weights or has multiple weight files)
    low_noise_quantized_ckpt=None,          # Wan2.2 low noise branch quantized weight path
    high_noise_quantized_ckpt=None,         # Wan2.2 high noise branch quantized weight path
    text_encoder_quantized_ckpt=None,       # Text encoder quantized weight path (required when model_path doesn't contain quantized weights or has multiple weight files)
    image_encoder_quantized_ckpt=None,      # Image encoder quantized weight path (required when model_path doesn't contain quantized weights or has multiple weight files)
    quant_scheme="fp8-sgl",                 # Quantization scheme
)
```

**Parameter Description:**
- **`dit_quantized_ckpt`**: When the `model_path` directory doesn't contain quantized weights, or has multiple weight files, you need to specify the specific DIT quantized weight path
- **`text_encoder_quantized_ckpt`** and **`image_encoder_quantized_ckpt`**: Similarly, used to specify encoder quantized weight paths
- **`low_noise_quantized_ckpt`** and **`high_noise_quantized_ckpt`**: Used to specify dual-branch quantized weights for Wan2.2 models

**Quantized Model Downloads:**

- **Wan-2.1 Quantized Models**: Download from [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- **Wan-2.2 Quantized Models**: Download from [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- **HunyuanVideo-1.5 Quantized Models**: Download from [Hy1.5-Quantized-Models](https://huggingface.co/lightx2v/Hy1.5-Quantized-Models)
  - `hy15_qwen25vl_llm_encoder_fp8_e4m3_lightx2v.safetensors` is the quantized weight for the text encoder

**Usage Examples:**

```python
# HunyuanVideo-1.5 Quantization Example
pipe.enable_quantize(
    quant_scheme='fp8-sgl',
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/hy15_720p_i2v_fp8_e4m3_lightx2v.safetensors",
    text_encoder_quantized=True,
    image_encoder_quantized=False,
    text_encoder_quantized_ckpt="/path/to/hy15_qwen25vl_llm_encoder_fp8_e4m3_lightx2v.safetensors",
)

# Wan2.1 Quantization Example
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
)

# Wan2.2 Quantization Example
pipe.enable_quantize(
    dit_quantized=True,
    low_noise_quantized_ckpt="/path/to/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    high_noise_quantized_ckpt="/path/to/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_1030.safetensors",
)
```

**Quantization Scheme Reference:** For detailed information, please refer to the [Quantization Documentation](../docs/EN/source/method_tutorials/quantization.md)

### Parallel Inference

Supports multi-GPU parallel inference. Requires running with `torchrun`:

```python
pipe.enable_parallel(
    seq_p_size=4,                    # Sequence parallel size
    seq_p_attn_type="ulysses",       # Sequence parallel attention type
)
```

**Running Method:**
```bash
torchrun --nproc_per_node=4 your_script.py
```

### Feature Caching

You can specify the cache method as Mag or Tea, using MagCache and TeaCache methods:

```python
pipe.enable_cache(
    cache_method='Tea',  # Cache method: 'Tea' or 'Mag'
    coefficients=[-3.08907507e+04, 1.67786188e+04, -3.19178643e+03,
                  2.60740519e+02, -8.19205881e+00, 1.07913775e-01],  # Coefficients
    teacache_thresh=0.15,  # TeaCache threshold
)
```

**Coefficient Reference:** Refer to configuration files in `configs/caching` or `configs/hunyuan_video_15/cache` directories

### Lightweight VAE

Using lightweight VAE can accelerate decoding and reduce memory usage.

```python
pipe.enable_lightvae(
    use_lightvae=False,    # Whether to use LightVAE
    use_tae=False,         # Whether to use LightTAE
    vae_path=None,         # Path to LightVAE
    tae_path=None,         # Path to LightTAE
)
```

**Support Status:**
- **LightVAE**: Currently only supports wan2.1, wan2.2 moe
- **LightTAE**: Currently only supports wan2.1, wan2.2-ti2v, wan2.2 moe, HunyuanVideo-1.5

**Model Downloads:** Lightweight VAE models can be downloaded from [Autoencoders](https://huggingface.co/lightx2v/Autoencoders)

- LightVAE for Wan-2.1: [lightvaew2_1.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lightvaew2_1.safetensors)
- LightTAE for Wan-2.1: [lighttaew2_1.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaew2_1.safetensors)
- LightTAE for Wan-2.2-ti2v: [lighttaew2_2.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaew2_2.safetensors)
- LightTAE for HunyuanVideo-1.5: [lighttaehy1_5.safetensors](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaehy1_5.safetensors)

**Usage Example:**

```python
# Using LightTAE for HunyuanVideo-1.5
pipe.enable_lightvae(
    use_tae=True,
    tae_path="/path/to/lighttaehy1_5.safetensors",
    use_lightvae=False,
    vae_path=None
)
```

## 📚 More Resources

- [Full Documentation](https://lightx2v-en.readthedocs.io/en/latest/)
- [GitHub Repository](https://github.com/ModelTC/LightX2V)
- [HuggingFace Model Hub](https://huggingface.co/lightx2v)
