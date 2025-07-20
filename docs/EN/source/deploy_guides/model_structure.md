# Model Structure Introduction

## 📖 Overview

This document introduces the model directory structure of the Lightx2v project, helping users correctly organize model files for a convenient user experience. Through proper directory organization, users can enjoy the convenience of "one-click startup" without manually configuring complex path parameters.

## 🗂️ Model Directory Structure

### Lightx2v Official Model List

View all available models: [Lightx2v Official Model Repository](https://huggingface.co/lightx2v)

### Standard Directory Structure

Using `Wan2.1-I2V-14B-480P-Lightx2v` as an example:

```
Model Root Directory/
├── Wan2.1-I2V-14B-480P-Lightx2v/
│   ├── config.json                                    # Model configuration file
│   ├── Wan2.1_VAE.pth                                # VAE variational autoencoder
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP visual encoder (FP16)
│   ├── models_t5_umt5-xxl-enc-bf16.pth               # T5 text encoder (BF16)
│   ├── taew2_1.pth                                   # Lightweight VAE (optional)
│   ├── fp8/                                          # FP8 quantized version (DIT/T5/CLIP)
│   ├── int8/                                         # INT8 quantized version (DIT/T5/CLIP)
│   ├── original/                                     # Original precision version (DIT)
│   ├── xlm-roberta-large/
│   └── google/
```

### 💾 Storage Recommendations

**Strongly recommend storing model files on SSD solid-state drives** to significantly improve model loading speed and inference performance.

**Recommended storage paths**:
```bash
/mnt/ssd/models/          # Independent SSD mount point
/data/ssd/models/         # Data SSD directory
/opt/models/              # System optimization directory
```

### Quantized Version Directories

Each model contains multiple quantized versions for different hardware configurations:

```
Model Directory/
├── fp8/                         # FP8 quantized version (H100/A100 high-end GPUs)
├── int8/                        # INT8 quantized version (general GPUs)
└── original/                    # Original precision version (DIT)
```

**💡 Using Full Precision Models**: To use full precision models, simply copy the official weight files to the `original/` directory.

## 🚀 Usage Methods

### Gradio Interface Startup

When using the Gradio interface, simply specify the model root directory path:

```bash
# Image to Video (I2V)
python gradio_demo_zh.py \
    --model_path /path/to/Wan2.1-I2V-14B-480P-Lightx2v \
    --model_size 14b \
    --task i2v

# Text to Video (T2V)
python gradio_demo_zh.py \
    --model_path /path/to/Wan2.1-T2V-14B-Lightx2v \
    --model_size 14b \
    --task t2v
```

### Configuration File Startup

When starting with configuration files, such as [configuration file](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/disk/wan_i2v_phase_lazy_load_480p.json), the following path configurations can be omitted:

- `dit_quantized_ckpt`: No need to specify, code will automatically search in the model directory
- `tiny_vae_path`: No need to specify, code will automatically search in the model directory
- `clip_quantized_ckpt`: No need to specify, code will automatically search in the model directory
- `t5_quantized_ckpt`: No need to specify, code will automatically search in the model directory

**💡 Simplified Configuration**: After organizing model files according to the recommended directory structure, most path configurations can be omitted as the code will handle them automatically.

### Manual Download

1. Visit the [Hugging Face Model Page](https://huggingface.co/lightx2v)
2. Select the required model version
3. Download all files to the corresponding directory

**💡 Download Recommendations**: It is recommended to use SSD storage and ensure stable network connection. For large files, you can use `git lfs` or download tools such as `aria2c`.

## 💡 Best Practices

- **Use SSD Storage**: Significantly improve model loading speed and inference performance
- **Unified Directory Structure**: Facilitate management and switching between different model versions
- **Reserve Sufficient Space**: Ensure adequate storage space (recommended at least 200GB)
- **Regular Cleanup**: Delete unnecessary model versions to save space
- **Network Optimization**: Use stable network connections and download tools

## 🚨 Common Issues

### Q: Model files are too large and download is slow?
A: Use domestic mirror sources, download tools such as `aria2c`, or consider using cloud storage services

### Q: Model path not found when starting?
A: Check if the model has been downloaded correctly and verify the path configuration

### Q: How to switch between different model versions?
A: Modify the model path parameter in the startup command, supports running multiple model instances simultaneously

### Q: Model loading is very slow?
A: Ensure models are stored on SSD, enable lazy loading, and use quantized version models

### Q: How to set paths in configuration files?
A: After organizing according to the recommended directory structure, most path configurations can be omitted as the code will handle them automatically

## 📚 Related Links

- [Lightx2v Official Model Repository](https://huggingface.co/lightx2v)
- [Gradio Deployment Guide](./deploy_gradio.md)

---

Through proper model file organization, users can enjoy the convenience of "one-click startup" without manually configuring complex path parameters. It is recommended to organize model files according to the structure recommended in this document and fully utilize the advantages of SSD storage.
