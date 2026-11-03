# Model Format and Loading Guide

## 📖 Overview

LightX2V is a flexible video generation inference framework that supports multiple model sources and formats, providing users with rich options:

- ✅ **Wan Official Models**: Directly compatible with officially released complete models from Wan2.1 and Wan2.2
- ✅ **Single-File Models**: Supports single-file format models released by LightX2V (including quantized versions)
- ✅ **LoRA Models**: Supports loading distilled LoRAs released by LightX2V

This document provides detailed instructions on how to use various model formats, configuration parameters, and best practices.

---

## 🗂️ Format 1: Wan Official Models

### Model Repositories
- [Wan2.1 Collection](https://huggingface.co/collections/Wan-AI/wan21-68ac4ba85372ae5a8e282a1b)
- [Wan2.2 Collection](https://huggingface.co/collections/Wan-AI/wan22-68ac4ae80a8b477e79636fc8)

### Model Features
- **Official Guarantee**: Complete models officially released by Wan-AI with highest quality
- **Complete Components**: Includes all necessary components (DIT, T5, CLIP, VAE)
- **Original Precision**: Uses BF16/FP32 precision with no quantization loss
- **Strong Compatibility**: Fully compatible with Wan official toolchain

### Wan2.1 Official Models

#### Directory Structure

Using [Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) as an example:

```
Wan2.1-I2V-14B-720P/
├── diffusion_pytorch_model-00001-of-00007.safetensors   # DIT model shard 1
├── diffusion_pytorch_model-00002-of-00007.safetensors   # DIT model shard 2
├── diffusion_pytorch_model-00003-of-00007.safetensors   # DIT model shard 3
├── diffusion_pytorch_model-00004-of-00007.safetensors   # DIT model shard 4
├── diffusion_pytorch_model-00005-of-00007.safetensors   # DIT model shard 5
├── diffusion_pytorch_model-00006-of-00007.safetensors   # DIT model shard 6
├── diffusion_pytorch_model-00007-of-00007.safetensors   # DIT model shard 7
├── diffusion_pytorch_model.safetensors.index.json       # Shard index file
├── models_t5_umt5-xxl-enc-bf16.pth                      # T5 text encoder
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP encoder
├── Wan2.1_VAE.pth                                       # VAE encoder/decoder
├── config.json                                          # Model configuration
├── xlm-roberta-large/                                   # CLIP tokenizer
├── google/                                              # T5 tokenizer
├── assets/
└── examples/
```

#### Usage

Download the complete model from the repository root:

```bash
hf download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir ./models/Wan2.1-I2V-14B-720P
```

In `scripts/wan/run_wan_i2v.sh`, point `model_path` to the downloaded directory, set `lightx2v_path`, and change the default `--size 480 832` to `--size 720 1280` for this 720P example. Then run from the repository root:

```bash
bash scripts/wan/run_wan_i2v.sh
```

### Wan2.2 Official Models

#### Directory Structure

Using [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) as an example:

```
Wan2.2-I2V-A14B/
├── high_noise_model/                                    # High-noise model directory
│   ├── diffusion_pytorch_model-00001-of-00009.safetensors
│   ├── diffusion_pytorch_model-00002-of-00009.safetensors
│   ├── ...
│   ├── diffusion_pytorch_model-00009-of-00009.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json
├── low_noise_model/                                     # Low-noise model directory
│   ├── diffusion_pytorch_model-00001-of-00009.safetensors
│   ├── diffusion_pytorch_model-00002-of-00009.safetensors
│   ├── ...
│   ├── diffusion_pytorch_model-00009-of-00009.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json
├── models_t5_umt5-xxl-enc-bf16.pth                      # T5 text encoder
├── Wan2.1_VAE.pth                                       # VAE encoder/decoder
├── configuration.json                                   # Model configuration
├── google/                                              # T5 tokenizer
├── assets/                                              # Example assets (optional)
└── examples/                                            # Example files (optional)
```

#### Usage

Download the complete model from the repository root:

```bash
hf download Wan-AI/Wan2.2-I2V-A14B \
    --local-dir ./models/Wan2.2-I2V-A14B
```

Set `model_path` and `lightx2v_path` inside `scripts/wan22/run_wan22_moe_i2v.sh`, then run from the repository root:

```bash
bash scripts/wan22/run_wan22_moe_i2v.sh
```

### Available Model List

#### Wan2.1 Official Model List

| Model Name | Download Link |
|---------|----------|
| Wan2.1-I2V-14B-720P | [Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| Wan2.1-I2V-14B-480P | [Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1-T2V-14B | [Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1-T2V-1.3B | [Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-FLF2V-14B-720P | [Link](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P) |
| Wan2.1-VACE-14B | [Link](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) |
| Wan2.1-VACE-1.3B | [Link](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) |

#### Wan2.2 Official Model List

| Model Name | Download Link |
|---------|----------|
| Wan2.2-I2V-A14B | [Link](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| Wan2.2-T2V-A14B | [Link](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Wan2.2-TI2V-5B | [Link](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Wan2.2-Animate-14B | [Link](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) |

### Usage Tips

> 💡 **Quantized Model Usage**: To use quantized models, refer to the [Model Conversion Script](https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme_zh.md) for conversion, or directly use pre-converted quantized models in Format 2 below
>
> 💡 **Memory Optimization**: For devices with RTX 4090 24GB or smaller memory, it's recommended to combine quantization techniques with CPU offload features:
> - Quantization Configuration: Refer to [Quantization Documentation](../method_tutorials/quantization.md)
> - CPU Offload: Refer to [Parameter Offload Documentation](../method_tutorials/offload.md)
> - Wan2.1 Configuration: Refer to [offload config files](https://github.com/ModelTC/LightX2V/tree/main/configs/offload)
> - Wan2.2 Configuration: Refer to [wan22 config files](https://github.com/ModelTC/LightX2V/tree/main/configs/wan22) with `4090` suffix

---

## 🗂️ Format 2: LightX2V Single-File Models (Recommended)

### Model Repositories
- [Wan2.1-LightX2V](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan2.2-LightX2V](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

### Model Features
- **Single-File Management**: Single safetensors file, easy to manage and deploy
- **Multi-Precision Support**: Provides original precision, FP8, INT8, and other precision versions
- **Distillation Acceleration**: Supports 4-step fast inference
- **Tool Compatibility**: Compatible with ComfyUI and other tools

**Examples**:
- `wan2.1_i2v_720p_lightx2v_4step.safetensors` - 720P I2V original precision
- `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors` - 720P I2V FP8 quantization
- `wan2.1_i2v_480p_int8_lightx2v_4step.safetensors` - 480P I2V INT8 quantization
- ...

### Wan2.1 Single-File Models

The Wan2.1 single-file examples here download 720P weights. Change the script's `--size 480 832` to `--size 720 1280`, or remove that argument and set `[720, 1280]` in JSON. The CLI size takes precedence over JSON.

#### Scenario A: Download Single Model File

**Step 1: Select and Download Model**

```bash
# Create model directory
mkdir -p ./models/wan2.1_i2v_720p

# Download the 720P I2V original-precision distilled model
hf download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p \
    --include "wan2.1_i2v_720p_lightx2v_4step.safetensors"
```

**Step 2: Manually Organize Other Components**

Directory structure as follows:
```
wan2.1_i2v_720p/
├── wan2.1_i2v_720p_lightx2v_4step.safetensors                    # Original precision
└── t5/clip/vae/config.json/xlm-roberta-large/google and other components       # Need manual organization
```

**Step 3: Configure Launch Script**

Set `model_path` and `lightx2v_path` inside `scripts/wan/inference/distill/run_wan_i2v_distill_single_file.sh`, then run from the repository root:

```bash
bash scripts/wan/inference/distill/run_wan_i2v_distill_single_file.sh
```

> 💡 **Tip**: When there's only one model file in the directory, LightX2V will automatically load it.

#### Scenario B: Download Multiple Model Files

When you download multiple models with different precisions to the same directory, you need to explicitly specify which model to use in the configuration file.

**Step 1: Download Multiple Models**

```bash
# Create model directory
mkdir -p ./models/wan2.1_i2v_720p_multi

# Download original precision model
hf download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p_multi \
    --include "wan2.1_i2v_720p_lightx2v_4step.safetensors"

# Download FP8 quantized model
hf download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p_multi \
    --include "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors"

# Download INT8 quantized model
hf download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models/wan2.1_i2v_720p_multi \
    --include "wan2.1_i2v_720p_int8_lightx2v_4step.safetensors"
```

**Step 2: Manually Organize Other Components**

Directory structure as follows:

```
wan2.1_i2v_720p_multi/
├── wan2.1_i2v_720p_lightx2v_4step.safetensors                    # Original precision
├── wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors   # FP8 quantization
└── wan2.1_i2v_720p_int8_lightx2v_4step.safetensors              # INT8 quantization
└── t5/clip/vae/config.json/xlm-roberta-large/google and other components       # Need manual organization
```

**Step 3: Specify Model in Configuration File**

Edit configuration file (e.g., `configs/distill/wan_i2v_distill_4step_cfg.json`):

Choose one of these three alternatives and merge its fields into the complete model config:

Original precision:

```json
{
    "dit_quantized": false,
    "dit_original_ckpt": "./models/wan2.1_i2v_720p_multi/wan2.1_i2v_720p_lightx2v_4step.safetensors"
}
```

FP8:

```json
{
    "dit_quantized_ckpt": "./models/wan2.1_i2v_720p_multi/wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-vllm"
}
```

INT8:

```json
{
    "dit_quantized_ckpt": "./models/wan2.1_i2v_720p_multi/wan2.1_i2v_720p_int8_lightx2v_4step.safetensors",
    "dit_quantized": true,
    "dit_quant_scheme": "int8-vllm"
}
```
### Usage Tips

> 💡 **Configuration Parameter Description**:
> - **dit_original_ckpt**: Used to specify the path to original precision models (BF16/FP32/FP16)
> - **dit_quantized_ckpt**: Used to specify the path to quantized models (FP8/INT8), must be used with `dit_quantized` and `dit_quant_scheme` parameters

**Step 4: Start Inference**

Set `model_path` and `lightx2v_path` inside `scripts/wan/inference/distill/run_wan_i2v_distill_single_file.sh`, then run from the repository root:

```bash
bash scripts/wan/inference/distill/run_wan_i2v_distill_single_file.sh
```

> 💡 **Tip**: Other components (T5, CLIP, VAE, tokenizer, etc.) need to be manually organized into the model directory

### Wan2.2 Single-File Models

#### Directory Structure Requirements

When using Wan2.2 single-file models, you need to manually create a specific directory structure:

```
wan2.2_models/
├── high_noise_model/                                    # High-noise model directory (required)
│   └── wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors
├── low_noise_model/                                     # Low-noise model directory (required)
│   └── wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors
└── t5/clip/vae/config.json/...                          # Other components (manually organized)
```

#### Scenario A: Only One Model File Per Directory

Download the original-precision distilled weights matching the directory layout above:

```bash
mkdir -p ./models/wan2.2_models/high_noise_model ./models/wan2.2_models/low_noise_model
hf download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models/high_noise_model \
    --include "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"
hf download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models/low_noise_model \
    --include "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"
```

These downloads contain only DiT weights. Also prepare T5, VAE, tokenizer, and configuration files following the official model layout, and point `model_path` to the complete model directory.

Keep the script's `configs/distill/wan22/wan_moe_i2v_distill_model.json` and set its `high_noise_original_ckpt` and `low_noise_original_ckpt` to the two downloaded files.

Set `model_path` and `lightx2v_path` inside `scripts/wan22/inference/distill/run_wan22_moe_i2v_distill.sh`, then run from the repository root:

```bash
bash scripts/wan22/inference/distill/run_wan22_moe_i2v_distill.sh
```

> 💡 **Tip**: For quantized weights, use `configs/distill/wan22/wan_moe_i2v_distill_quant.json` and configure the matching quantized paths, backend, and T5 weights; do not use original-precision files as quantized checkpoints.

#### Scenario B: Multiple Model Files Per Directory

When you place multiple models with different precisions in both `high_noise_model/` and `low_noise_model/` directories, you need to explicitly specify them in the configuration file.

```bash
# Create directories
mkdir -p ./models/wan2.2_models_multi/high_noise_model
mkdir -p ./models/wan2.2_models_multi/low_noise_model

# Download multiple versions of high-noise model
hf download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models_multi/high_noise_model \
    --include "wan2.2_i2v_A14b_high_noise_*.safetensors"

# Download multiple versions of low-noise model
hf download lightx2v/Wan2.2-Distill-Models \
    --local-dir ./models/wan2.2_models_multi/low_noise_model \
    --include "wan2.2_i2v_A14b_low_noise_*.safetensors"
```

**Directory Structure**:

```
wan2.2_models_multi/
├── high_noise_model/
│   ├── wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors        # Original precision
│   ├── wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors    # FP8 quantization
│   └── wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors   # INT8 quantization
└── low_noise_model/
│    ├── wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors         # Original precision
│    ├── wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors     # FP8 quantization
│    └── wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors    # INT8 quantization
└── t5/vae/config.json/xlm-roberta-large/google and other components       # Need manual organization
```

**Configuration File Settings**:

Choose one of these three alternatives and merge its fields into the complete model config:

Original precision:

```json
{
    "dit_quantized": false,
    "high_noise_original_ckpt": "./models/wan2.2_models_multi/high_noise_model/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors",
    "low_noise_original_ckpt": "./models/wan2.2_models_multi/low_noise_model/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"
}
```

FP8:

```json
{
    "high_noise_quantized_ckpt": "./models/wan2.2_models_multi/high_noise_model/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "low_noise_quantized_ckpt": "./models/wan2.2_models_multi/low_noise_model/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-vllm"
}
```

INT8:

```json
{
    "high_noise_quantized_ckpt": "./models/wan2.2_models_multi/high_noise_model/wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors",
    "low_noise_quantized_ckpt": "./models/wan2.2_models_multi/low_noise_model/wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors",
    "dit_quantized": true,
    "dit_quant_scheme": "int8-vllm"
}
```

### Usage Tips

> 💡 **Configuration Parameter Description**:
> - **high_noise_original_ckpt** / **low_noise_original_ckpt**: Used to specify the path to original precision models (BF16/FP32/FP16)
> - **high_noise_quantized_ckpt** / **low_noise_quantized_ckpt**: Used to specify the path to quantized models (FP8/INT8), must be used with `dit_quantized` and `dit_quant_scheme` parameters


### Available Model List

#### Wan2.1 Single-File Model List

**Image-to-Video Models (I2V)**

| Filename | Precision | Description |
|--------|------|------|
| `wan2.1_i2v_480p_lightx2v_4step.safetensors` | BF16 | 4-step model original precision |
| `wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 4-step model FP8 quantization |
| `wan2.1_i2v_480p_int8_lightx2v_4step.safetensors` | INT8 | 4-step model INT8 quantization |
| `wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | FP8 | 4-step model ComfyUI format |
| `wan2.1_i2v_720p_lightx2v_4step.safetensors` | BF16 | 4-step model original precision |
| `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 4-step model FP8 quantization |
| `wan2.1_i2v_720p_int8_lightx2v_4step.safetensors` | INT8 | 4-step model INT8 quantization |
| `wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | FP8 | 4-step model ComfyUI format |

**Text-to-Video Models (T2V)**

| Filename | Precision | Description |
|--------|------|------|
| `wan2.1_t2v_14b_lightx2v_4step.safetensors` | BF16 | 4-step model original precision |
| `wan2.1_t2v_14b_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | 4-step model FP8 quantization |
| `wan2.1_t2v_14b_int8_lightx2v_4step.safetensors` | INT8 | 4-step model INT8 quantization |
| `wan2.1_t2v_14b_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors` | FP8 | 4-step model ComfyUI format |

#### Wan2.2 Single-File Model List

**Image-to-Video Models (I2V) - A14B Series**

| Filename | Precision | Description |
|--------|------|------|
| `wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors` | BF16 | High-noise model - 4-step original precision |
| `wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | High-noise model - 4-step FP8 quantization |
| `wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step.safetensors` | INT8 | High-noise model - 4-step INT8 quantization |
| `wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors` | BF16 | Low-noise model - 4-step original precision |
| `wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors` | FP8 | Low-noise model - 4-step FP8 quantization |
| `wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors` | INT8 | Low-noise model - 4-step INT8 quantization |

> 💡 **Usage Tips**:
> - Wan2.2 models use a dual-noise architecture, requiring both high-noise and low-noise models to be downloaded
> - Refer to the "Wan2.2 Single-File Models" section above for detailed directory organization

---

## 🗂️ Format 3: LightX2V LoRA Models

LoRA (Low-Rank Adaptation) models provide a lightweight model fine-tuning solution that enables customization for specific effects without modifying the base model.

### Model Repositories

- **Wan2.1 LoRA Models**: [lightx2v/Wan2.1-Distill-Loras](https://huggingface.co/lightx2v/Wan2.1-Distill-Loras)
- **Wan2.2 LoRA Models**: [lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)

### Usage

LoRA files require a matching base model and cannot run independently. Set the weight path and strength through `lora_configs`; Wan2.2 MoE also needs `name` to select the high-noise or low-noise model.

See the [LoRA deployment guide](../deploy_guides/lora_deploy.md#configuration-file-method) for configuration examples, static merging versus dynamic loading, and supported combinations. To produce a merged model file, use the [conversion guide](https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme.md).

---

## 📚 Related Resources

### Official Repositories
- [LightX2V GitHub](https://github.com/ModelTC/LightX2V)
- [LightX2V Single-File Model Repository](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan-AI Official Model Repository](https://huggingface.co/Wan-AI)

### Model Download Links

**Wan2.1 Series**
- [Wan2.1 Collection](https://huggingface.co/collections/Wan-AI/wan21-68ac4ba85372ae5a8e282a1b)

**Wan2.2 Series**
- [Wan2.2 Collection](https://huggingface.co/collections/Wan-AI/wan22-68ac4ae80a8b477e79636fc8)

**LightX2V Single-File Models**
- [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

### Documentation Links
- [Quantization Documentation](../method_tutorials/quantization.md)
- [Parameter Offload Documentation](../method_tutorials/offload.md)
- [Configuration File Examples](https://github.com/ModelTC/LightX2V/tree/main/configs)

---

Through this document, you should be able to:

✅ Understand all model formats supported by LightX2V
✅ Select appropriate models and precisions based on your needs
✅ Correctly download and organize model files
✅ Configure launch parameters and successfully run inference
✅ Resolve common model loading issues

If you have other questions, feel free to ask in [GitHub Issues](https://github.com/ModelTC/LightX2V/issues).
