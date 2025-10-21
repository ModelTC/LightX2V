# Model Quantization Techniques

## 📖 Overview

LightX2V supports quantized inference for DIT, T5, and CLIP models, reducing memory usage and improving inference speed by lowering model precision.

---

## 🔧 Quantization Modes

| Quantization Mode | Weight Quantization | Activation Quantization | Compute Kernel | Supported Hardware |
|--------------|----------|----------|----------|----------|
| `fp8-vllm` | FP8 channel symmetric | FP8 channel dynamic symmetric | [VLLM](https://github.com/vllm-project/vllm) | H100/H200/H800, RTX 40 series, etc. |
| `int8-vllm` | INT8 channel symmetric | INT8 channel dynamic symmetric | [VLLM](https://github.com/vllm-project/vllm) | A100/A800, RTX 30/40 series, etc.  |
| `fp8-sgl` | FP8 channel symmetric | FP8 channel dynamic symmetric | [SGL](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) | H100/H200/H800, RTX 40 series, etc. |
| `int8-sgl` | INT8 channel symmetric | INT8 channel dynamic symmetric | [SGL](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) | A100/A800, RTX 30/40 series, etc.  |
| `fp8-q8f` | FP8 channel symmetric | FP8 channel dynamic symmetric | [Q8-Kernels](https://github.com/KONAKONA666/q8_kernels) | RTX 40 series, L40S, etc. |
| `int8-q8f` | INT8 channel symmetric | INT8 channel dynamic symmetric | [Q8-Kernels](https://github.com/KONAKONA666/q8_kernels) | RTX 40 series, L40S, etc. |
| `int8-torchao` | INT8 channel symmetric | INT8 channel dynamic symmetric | [TorchAO](https://github.com/pytorch/ao) | A100/A800, RTX 30/40 series, etc. |
| `int4-g128-marlin` | INT4 group symmetric | FP16 | [Marlin](https://github.com/IST-DASLab/marlin) | H200/H800/A100/A800, RTX 30/40 series, etc. |
| `fp8-b128-deepgemm` | FP8 block symmetric | FP8 group symmetric | [DeepGemm](https://github.com/deepseek-ai/DeepGEMM) | H100/H200/H800, RTX 40 series, etc.|

---

## 🔧 Obtaining Quantized Models

### Method 1: Download Pre-Quantized Models

Download pre-quantized models from LightX2V model repositories:

**DIT Models**

Download pre-quantized DIT models from [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models):

```bash
# Download DIT FP8 quantized model
huggingface-cli download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models \
    --include "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors"
```

**Encoder Models**

Download pre-quantized T5 and CLIP models from [Encoders-LightX2V](https://huggingface.co/lightx2v/Encoders-Lightx2v):

```bash
# Download T5 FP8 quantized model
huggingface-cli download lightx2v/Encoders-Lightx2v \
    --local-dir ./models \
    --include "models_t5_umt5-xxl-enc-fp8.pth"

# Download CLIP FP8 quantized model
huggingface-cli download lightx2v/Encoders-Lightx2v \
    --local-dir ./models \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8.pth"
```

### Method 2: Self-Quantize Models

For detailed quantization tool usage, refer to: [Model Conversion Documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)

---

## 🚀 Using Quantized Models

### DIT Model Quantization

#### Supported Quantization Modes

DIT quantization modes (`dit_quant_scheme`) support: `fp8-vllm`, `int8-vllm`, `fp8-sgl`, `int8-sgl`, `fp8-q8f`, `int8-q8f`, `int8-torchao`, `int4-g128-marlin`, `fp8-b128-deepgemm`

#### Configuration Example

```json
{
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-sgl",
    "dit_quantized_ckpt": "/path/to/dit_quantized_model"  // Optional
}
```

> 💡 **Tip**: When there's only one DIT model in the script's `model_path`, `dit_quantized_ckpt` doesn't need to be specified separately.

### T5 Model Quantization

#### Supported Quantization Modes

T5 quantization modes (`t5_quant_scheme`) support: `int8-vllm`, `fp8-sgl`, `int8-q8f`, `fp8-q8f`, `int8-torchao`

#### Configuration Example

```json
{
    "t5_quantized": true,
    "t5_quant_scheme": "fp8-sgl",
    "t5_quantized_ckpt": "/path/to/t5_quantized_model"  // Optional
}
```

> 💡 **Tip**: When a T5 quantized model exists in the script's specified `model_path` (such as `models_t5_umt5-xxl-enc-fp8.pth` or `models_t5_umt5-xxl-enc-int8.pth`), `t5_quantized_ckpt` doesn't need to be specified separately.

### CLIP Model Quantization

#### Supported Quantization Modes

CLIP quantization modes (`clip_quant_scheme`) support: `int8-vllm`, `fp8-sgl`, `int8-q8f`, `fp8-q8f`, `int8-torchao`

#### Configuration Example

```json
{
    "clip_quantized": true,
    "clip_quant_scheme": "fp8-sgl",
    "clip_quantized_ckpt": "/path/to/clip_quantized_model"  // Optional
}
```

> 💡 **Tip**: When a CLIP quantized model exists in the script's specified `model_path` (such as `models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8.pth` or `models_clip_open-clip-xlm-roberta-large-vit-huge-14-int8.pth`), `clip_quantized_ckpt` doesn't need to be specified separately.

### Performance Optimization Strategy

If memory is insufficient, you can combine parameter offloading to further reduce memory usage. Refer to [Parameter Offload Documentation](../method_tutorials/offload.md):

> - **Wan2.1 Configuration**: Refer to [offload config files](https://github.com/ModelTC/LightX2V/tree/main/configs/offload)
> - **Wan2.2 Configuration**: Refer to [wan22 config files](https://github.com/ModelTC/LightX2V/tree/main/configs/wan22) with `4090` suffix

---

## 📚 Related Resources

### Configuration File Examples
- [INT8 Quantization Config](https://github.com/ModelTC/LightX2V/blob/main/configs/quantization/wan_i2v.json)
- [Q8F Quantization Config](https://github.com/ModelTC/LightX2V/blob/main/configs/quantization/wan_i2v_q8f.json)
- [TorchAO Quantization Config](https://github.com/ModelTC/LightX2V/blob/main/configs/quantization/wan_i2v_torchao.json)

### Run Scripts
- [Quantization Inference Scripts](https://github.com/ModelTC/LightX2V/tree/main/scripts/quantization)

### Tool Documentation
- [Quantization Tool Documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)
- [LightCompress Quantization Documentation](https://github.com/ModelTC/llmc/blob/main/docs/zh_cn/source/backend/lightx2v.md)

### Model Repositories
- [Wan2.1-LightX2V Quantized Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan2.2-LightX2V Quantized Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- [Encoders Quantized Models](https://huggingface.co/lightx2v/Encoders-Lightx2v)

---

Through this document, you should be able to:

✅ Understand quantization schemes supported by LightX2V
✅ Select appropriate quantization strategies based on hardware
✅ Correctly configure quantization parameters
✅ Obtain and use quantized models
✅ Optimize inference performance and memory usage

If you have other questions, feel free to ask in [GitHub Issues](https://github.com/ModelTC/LightX2V/issues).
