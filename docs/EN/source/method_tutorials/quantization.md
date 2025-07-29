# Model Quantization

LightX2V supports quantization inference for linear layers in `Dit`, supporting `w8a8-int8`, `w8a8-fp8`, `w8a8-fp8block`, `w8a8-mxfp8`, and `w4a4-nvfp4` matrix multiplication. Additionally, LightX2V also supports quantization of T5 and CLIP encoders to further improve inference performance.

## 📊 Quantization Scheme Overview

### DIT Model Quantization

LightX2V supports multiple DIT matrix multiplication quantization schemes, configured through the `mm_type` parameter:

#### Supported mm_type Types

| mm_type | Weight Quantization | Activation Quantization | Compute Kernel |
|---------|-------------------|------------------------|----------------|
| `Default` | No Quantization | No Quantization | PyTorch |
| `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm` | FP8 Channel Symmetric | FP8 Channel Dynamic Symmetric | VLLM |
| `W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm` | INT8 Channel Symmetric | INT8 Channel Dynamic Symmetric | VLLM |
| `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F` | FP8 Channel Symmetric | FP8 Channel Dynamic Symmetric | Q8F |
| `W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F` | INT8 Channel Symmetric | INT8 Channel Dynamic Symmetric | Q8F |
| `W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm` | FP8 Block Symmetric | FP8 Channel Group Symmetric | DeepGEMM |
| `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl` | FP8 Channel Symmetric | FP8 Channel Dynamic Symmetric | SGL |

#### Detailed Quantization Scheme Description

**FP8 Quantization Scheme**:
- **Weight Quantization**: Uses `torch.float8_e4m3fn` format with per-channel symmetric quantization
- **Activation Quantization**: Dynamic quantization supporting per-token and per-channel modes
- **Advantages**: Provides optimal performance on FP8-supported GPUs with minimal precision loss (typically <1%)
- **Compatible Hardware**: H100, A100, RTX 40 series and other FP8-supported GPUs

**INT8 Quantization Scheme**:
- **Weight Quantization**: Uses `torch.int8` format with per-channel symmetric quantization
- **Activation Quantization**: Dynamic quantization supporting per-token mode
- **Advantages**: Best compatibility, suitable for most GPU hardware, reduces memory usage by ~50%
- **Compatible Hardware**: All INT8-supported GPUs

**Block Quantization Scheme**:
- **Weight Quantization**: FP8 quantization by 128x128 blocks
- **Activation Quantization**: Quantization by channel groups (group size 128)
- **Advantages**: Particularly suitable for large models with higher memory efficiency, supports larger batch sizes

### T5 Encoder Quantization

T5 encoder supports the following quantization schemes:

#### Supported quant_scheme Types

| quant_scheme | Quantization Precision | Compute Kernel |
|--------------|----------------------|----------------|
| `int8` | INT8 | VLLM |
| `fp8` | FP8 | VLLM |
| `int8-torchao` | INT8 | TorchAO |
| `int8-q8f` | INT8 | Q8F |
| `fp8-q8f` | FP8 | Q8F |

### CLIP Encoder Quantization

CLIP encoder supports the same quantization schemes as T5

## 🚀 Producing Quantized Models

Download quantized models from the [LightX2V Official Model Repository](https://huggingface.co/lightx2v), refer to the [Model Structure Documentation](../deploy_guides/model_structure.md) for details.

Use LightX2V's convert tool to convert models into quantized models. Refer to the [documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme.md).

## 📥 Loading Quantized Models for Inference

### DIT Model Configuration

Write the path of the converted quantized weights to the `dit_quantized_ckpt` field in the [configuration file](https://github.com/ModelTC/lightx2v/blob/main/configs/quantization).

```json
{
    "dit_quantized_ckpt": "/path/to/dit_quantized_ckpt",
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    }
}
```

### T5 Encoder Configuration

```json
{
    "t5_quantized": true,
    "t5_quant_scheme": "fp8",
    "t5_quantized_ckpt": "/path/to/t5_quantized_ckpt"
}
```

### CLIP Encoder Configuration

```json
{
    "clip_quantized": true,
    "clip_quant_scheme": "fp8",
    "clip_quantized_ckpt": "/path/to/clip_quantized_ckpt"
}
```

### Complete Configuration Example

```json
{
    "dit_quantized_ckpt": "/path/to/dit_quantized_ckpt",
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },
    "t5_quantized": true,
    "t5_quant_scheme": "fp8",
    "t5_quantized_ckpt": "/path/to/t5_quantized_ckpt",
    "clip_quantized": true,
    "clip_quant_scheme": "fp8",
    "clip_quantized_ckpt": "/path/to/clip_quantized_ckpt"
}
```

By specifying `--config_json` to the specific config file, you can load the quantized model for inference.

[Here](https://github.com/ModelTC/lightx2v/tree/main/scripts/quantization) are some running scripts for use.

## 💡 Quantization Scheme Selection Recommendations

### Hardware Compatibility

- **H100/A100 GPU/RTX 4090/RTX 4060**: Recommended to use FP8 quantization schemes
  - DIT: `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm`
  - T5/CLIP: `fp8`
- **A100/RTX 3090/RTX 3060**: Recommended to use INT8 quantization schemes
  - DIT: `W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm`
  - T5/CLIP: `int8`
- **Other GPUs**: Choose based on hardware support

### Performance Optimization

- **Memory Constrained**: Choose INT8 quantization schemes
- **Speed Priority**: Choose FP8 quantization schemes
- **High Precision Requirements**: Use FP8 or mixed precision schemes

### Mixed Quantization Strategy

You can choose different quantization schemes for different components:

```json
{
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    },
    "t5_quantized": true,
    "t5_quant_scheme": "int8",
    "clip_quantized": true,
    "clip_quant_scheme": "fp8"
}
```

## 🔧 Advanced Quantization Features

For details, please refer to the documentation of the quantization tool [LLMC](https://github.com/ModelTC/llmc/blob/main/docs/en/source/backend/lightx2v.md)

### Custom Quantization Kernels

LightX2V supports custom quantization kernels that can be extended in the following ways:

1. **Register New mm_type**: Add new quantization classes in `mm_weight.py`
2. **Implement Quantization Functions**: Define quantization methods for weights and activations
3. **Integrate Compute Kernels**: Use custom matrix multiplication implementations

## 🚨 Important Notes

1. **Hardware Requirements**: FP8 quantization requires FP8-supported GPUs (such as H100, RTX 40 series)
2. **Precision Impact**: Quantization will bring certain precision loss, which needs to be weighed based on application scenarios

## 📚 Related Resources

- [Quantization Tool Documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme.md)
- [Running Scripts](https://github.com/ModelTC/lightx2v/tree/main/scripts/quantization)
- [Configuration File Examples](https://github.com/ModelTC/lightx2v/blob/main/configs/quantization)
- [LLMC Quantization Documentation](https://github.com/ModelTC/llmc/blob/main/docs/en/source/backend/lightx2v.md)
