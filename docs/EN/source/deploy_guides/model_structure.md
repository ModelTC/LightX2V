# Model Structure Guide

## 📖 Overview

This document provides a comprehensive introduction to the model directory structure of the LightX2V project, designed to help users efficiently organize model files and achieve a convenient user experience. Through scientific directory organization, users can enjoy the convenience of "one-click startup" without manually configuring complex path parameters. Meanwhile, the system also supports flexible manual path configuration to meet the diverse needs of different user groups.

## 🗂️ Model Directory Structure

### LightX2V Official Model List

View all available models: [LightX2V Official Model Repository](https://huggingface.co/lightx2v)

### Standard Directory Structure

Using `Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V` as an example, the standard file structure is as follows:

```
Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/
├── fp8/                                          # FP8 quantized version (DIT/T5/CLIP)
│   ├── block_xx.safetensors                      # DIT model FP8 quantized version
│   ├── models_t5_umt5-xxl-enc-fp8.pth            # T5 encoder FP8 quantized version
│   ├── clip-fp8.pth                              # CLIP encoder FP8 quantized version
│   ├── Wan2.1_VAE.pth                            # VAE variational autoencoder
│   ├── taew2_1.pth                               # Lightweight VAE (optional)
│   └── config.json                               # Model configuration file
├── int8/                                         # INT8 quantized version (DIT/T5/CLIP)
│   ├── block_xx.safetensors                      # DIT model INT8 quantized version
│   ├── models_t5_umt5-xxl-enc-int8.pth           # T5 encoder INT8 quantized version
│   ├── clip-int8.pth                             # CLIP encoder INT8 quantized version
│   ├── Wan2.1_VAE.pth                            # VAE variational autoencoder
│   ├── taew2_1.pth                               # Lightweight VAE (optional)
│   └── config.json                               # Model configuration file
├── original/                                     # Original precision version (DIT/T5/CLIP)
│   ├── distill_model.safetensors                 # DIT model original precision version
│   ├── models_t5_umt5-xxl-enc-bf16.pth           # T5 encoder original precision version
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP encoder original precision version
│   ├── Wan2.1_VAE.pth                            # VAE variational autoencoder
│   ├── taew2_1.pth                               # Lightweight VAE (optional)
│   └── config.json                               # Model configuration file
```

### 💾 Storage Recommendations

**It is strongly recommended to store model files on SSD solid-state drives**, as this can significantly improve model loading speed and inference performance.

**Recommended storage paths**:
```bash
/mnt/ssd/models/          # Independent SSD mount point
/data/ssd/models/         # Data SSD directory
/opt/models/              # System optimization directory
```

### Quantization Version Description

Each model includes multiple quantized versions to adapt to different hardware configuration requirements:
- **FP8 Version**: Suitable for GPUs that support FP8 (such as H100, A100, RTX 40 series), providing optimal performance
- **INT8 Version**: Suitable for most GPUs, balancing performance and compatibility, reducing memory usage by approximately 50%
- **Original Precision Version**: Suitable for applications with extremely high precision requirements, providing highest quality output

## 🚀 Usage Methods

### Environment Setup

#### Installing Hugging Face CLI

Before starting to download models, please ensure that Hugging Face CLI is properly installed:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Or install huggingface-cli
pip install huggingface-cli

# Login to Hugging Face (optional, but strongly recommended)
huggingface-cli login
```

### Method 1: Complete Model Download (Recommended)

**Advantage**: After downloading the complete model, the system will automatically identify all component paths without manual configuration, providing a more convenient user experience

#### 1. Download Complete Model

```bash
# Use Hugging Face CLI to download complete model
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V
```

#### 2. Start Inference

##### Bash Script Startup

###### Scenario 1: Using Full Precision Model

Modify the configuration in the [run script](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_distill_4step_cfg.sh):
- `model_path`: Set to the downloaded model path `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V`
- `lightx2v_path`: Set to the LightX2V project root directory path

###### Scenario 2: Using Quantized Model

When using the complete model, if you need to enable quantization, please add the following configuration to the [configuration file](https://github.com/ModelTC/LightX2V/tree/main/configs/distill/wan_i2v_distill_4step_cfg.json):

```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },                              // DIT model quantization scheme
    "t5_quantized": true,           // Enable T5 quantization
    "t5_quant_scheme": "fp8",       // T5 quantization mode
    "clip_quantized": true,         // Enable CLIP quantization
    "clip_quant_scheme": "fp8"      // CLIP quantization mode
}
```

> **Important Note**: Quantization configurations for each model can be flexibly combined. Quantization paths do not need to be manually specified, as the system will automatically locate the quantized versions of each model.

For detailed explanation of quantization technology, please refer to the [Quantization Documentation](../method_tutorials/quantization.md).

Use the provided bash script for quick startup:

```bash
cd LightX2V/scripts
bash wan/run_wan_t2v_distill_4step_cfg.sh
```

##### Gradio Interface Startup

When performing inference through the Gradio interface, simply specify the model root directory path at startup, and lightweight VAE can be flexibly selected through frontend interface buttons:

```bash
# Image-to-video inference (I2V)
python gradio_demo.py \
    --model_path ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --model_size 14b \
    --task i2v \
    --model_cls wan2.1_distill
```

### Method 2: Selective Download

**Advantage**: Only download the required versions (quantized or non-quantized), effectively saving storage space and download time

#### 1. Selective Download

```bash
# Use Hugging Face CLI to selectively download non-quantized version
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --include "original/*"
```

```bash
# Use Hugging Face CLI to selectively download FP8 quantized version
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --include "fp8/*"
```

```bash
# Use Hugging Face CLI to selectively download INT8 quantized version
huggingface-cli download lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --local-dir ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V \
    --include "int8/*"
```

> **Important Note**: When starting inference scripts or Gradio, the `model_path` parameter still needs to be specified as the complete path without the `--include` parameter. For example: `model_path=./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V`, not `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/int8`.

#### 2. Start Inference

**Taking the model with only FP8 version downloaded as an example:**

##### Bash Script Startup

###### Scenario 1: Using FP8 DIT + FP8 T5 + FP8 CLIP

Set the `model_path` in the [run script](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_distill_4step_cfg.sh) to your downloaded model path `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/`, and set `lightx2v_path` to your LightX2V project path.

Only need to modify the quantized model configuration in the configuration file as follows:
```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },                              // DIT quantization scheme
    "t5_quantized": true,           // Whether to use T5 quantized version
    "t5_quant_scheme": "fp8",       // T5 quantization mode
    "clip_quantized": true,         // Whether to use CLIP quantized version
    "clip_quant_scheme": "fp8",     // CLIP quantization mode
}
```

> **Important Note**: At this time, each model can only be specified as a quantized version. Quantization paths do not need to be manually specified, as the system will automatically locate the quantized versions of each model.

###### Scenario 2: Using FP8 DIT + Original Precision T5 + Original Precision CLIP

Set the `model_path` in the [run script](https://github.com/ModelTC/LightX2V/tree/main/scripts/wan/run_wan_i2v_distill_4step_cfg.sh) to your downloaded model path `./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V`, and set `lightx2v_path` to your LightX2V project path.

Since only quantized weights were downloaded, you need to manually download the original precision versions of T5 and CLIP, and configure them in the configuration file's `t5_original_ckpt` and `clip_original_ckpt` as follows:
```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },                              // DIT quantization scheme
    "t5_original_ckpt": "/path/to/models_t5_umt5-xxl-enc-bf16.pth",
    "clip_original_ckpt": "/path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
}
```

Use the provided bash script for quick startup:

```bash
cd LightX2V/scripts
bash wan/run_wan_t2v_distill_4step_cfg.sh
```

##### Gradio Interface Startup

When performing inference through the Gradio interface, specify the model root directory path at startup:

```bash
# Image-to-video inference (I2V)
python gradio_demo.py \
    --model_path ./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/ \
    --model_size 14b \
    --task i2v \
    --model_cls wan2.1_distill
```

> **Important Note**: Since the model root directory only contains quantized versions of each model, when using the frontend, the quantization precision for DIT/T5/CLIP models can only be selected as fp8. If you need to use non-quantized versions of T5/CLIP, please manually download non-quantized weights and place them in the gradio_demo model_path directory (`./Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-LightX2V/`). In this case, the T5/CLIP quantization precision can be set to bf16/fp16.

### Method 3: Manual Configuration

Users can flexibly configure quantization options and paths for each component according to actual needs, achieving mixed use of quantized and non-quantized components. Please ensure that the required model weights have been correctly downloaded and placed in the specified paths.

#### DIT Model Configuration

```json
{
    "dit_quantized_ckpt": "/path/to/dit_quantized_ckpt",    // DIT quantized weights path
    "dit_original_ckpt": "/path/to/dit_original_ckpt",      // DIT original precision weights path
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"  // DIT matrix multiplication operator type, specify as "Default" when not quantized
    }
}
```

#### T5 Model Configuration

```json
{
    "t5_quantized_ckpt": "/path/to/t5_quantized_ckpt",      // T5 quantized weights path
    "t5_original_ckpt": "/path/to/t5_original_ckpt",        // T5 original precision weights path
    "t5_quantized": true,                                   // Whether to enable T5 quantization
    "t5_quant_scheme": "fp8"                                // T5 quantization mode, only effective when t5_quantized is true
}
```

#### CLIP Model Configuration

```json
{
    "clip_quantized_ckpt": "/path/to/clip_quantized_ckpt",  // CLIP quantized weights path
    "clip_original_ckpt": "/path/to/clip_original_ckpt",    // CLIP original precision weights path
    "clip_quantized": true,                                 // Whether to enable CLIP quantization
    "clip_quant_scheme": "fp8"                              // CLIP quantization mode, only effective when clip_quantized is true
}
```

#### VAE Model Configuration

```json
{
    "vae_pth": "/path/to/Wan2.1_VAE.pth",                   // Original VAE model path
    "use_tiny_vae": true,                                   // Whether to use lightweight VAE
    "tiny_vae_path": "/path/to/taew2_1.pth"                 // Lightweight VAE model path
}
```

> **Configuration Notes**:
> - Quantized weights and original precision weights can be flexibly mixed and used, and the system will automatically select the corresponding model based on the configuration
> - The choice of quantization mode depends on your hardware support, it is recommended to use FP8 on high-end GPUs like H100/A100
> - Lightweight VAE can significantly improve inference speed but may slightly affect generation quality

## 💡 Best Practices

### Recommended Configurations

**Complete Model Users**:
- Download complete models to enjoy the convenience of automatic path discovery
- Only need to configure quantization schemes and component switches
- Recommended to use bash scripts for quick startup

**Storage Space Limited Users**:
- Selectively download required quantized versions
- Flexibly mix and use quantized and original precision components
- Use bash scripts to simplify startup process

**Advanced Users**:
- Completely manual path configuration for maximum flexibility
- Support scattered storage of model files
- Can customize bash script parameters

### Performance Optimization Recommendations

- **Use SSD Storage**: Significantly improve model loading speed and inference performance
- **Choose Appropriate Quantization Schemes**:
  - FP8: Suitable for high-end GPUs like H100/A100, high precision
  - INT8: Suitable for general GPUs, small memory footprint
- **Enable Lightweight VAE**: `use_tiny_vae: true` can improve inference speed
- **Reasonable CPU Offload Configuration**: `t5_cpu_offload: true` can save GPU memory

### Download Optimization Recommendations

- **Use Hugging Face CLI**: More stable than git clone, supports resume download
- **Selective Download**: Only download required quantized versions, saving time and storage space
- **Network Optimization**: Use stable network connections, use proxy when necessary
- **Resume Download**: Use `--resume-download` parameter to support continuing download after interruption

## 🚨 Frequently Asked Questions

### Q: Model files are too large and download speed is slow, what should I do?
A: It is recommended to use selective download method, only download required quantized versions, or use domestic mirror sources

### Q: Model path does not exist when starting up?
A: Please check if the model has been correctly downloaded, verify if the path configuration is correct, and confirm if the automatic discovery mechanism is working properly

### Q: How to switch between different quantization schemes?
A: Modify parameters such as `mm_type`, `t5_quant_scheme`, `clip_quant_scheme` in the configuration file, please refer to the [Quantization Documentation](../method_tutorials/quantization.md)

### Q: How to mix and use quantized and original precision components?
A: Control through `t5_quantized` and `clip_quantized` parameters, and manually specify original precision paths

### Q: How to set paths in configuration files?
A: It is recommended to use automatic path discovery, for manual configuration please refer to the "Manual Configuration" section

### Q: How to verify if automatic path discovery is working properly?
A: Check the startup logs, the code will output the actual model paths being used

### Q: What should I do if bash script startup fails?
A: Check if the path configuration in the script is correct, ensure that `lightx2v_path` and `model_path` variables are correctly set

## 📚 Related Links

- [LightX2V Official Model Repository](https://huggingface.co/lightx2v)
- [Gradio Deployment Guide](./deploy_gradio.md)
- [Configuration File Examples](https://github.com/ModelTC/LightX2V/tree/main/configs)

---

Through scientific model file organization and flexible configuration options, LightX2V supports multiple usage scenarios. Complete model download provides maximum convenience, selective download saves storage space, and manual configuration provides maximum flexibility. The automatic path discovery mechanism ensures that users do not need to remember complex path configurations while maintaining system scalability.
