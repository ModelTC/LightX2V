# 模型量化

LightX2V支持对`Dit`中的线性层进行量化推理，支持`w8a8-int8`、`w8a8-fp8`、`w8a8-fp8block`、`w8a8-mxfp8`和`w4a4-nvfp4`的矩阵乘法。同时，LightX2V也支持对T5和CLIP编码器进行量化，以进一步提升推理性能。

## 📊 量化方案概览

### DIT 模型量化

LightX2V支持多种DIT矩阵乘法量化方案，通过配置文件中的`mm_type`参数进行配置：

#### 支持的 mm_type 类型

| mm_type | 权重量化 | 激活量化 | 计算内核 | 适用场景 |
|---------|----------|----------|----------|----------|
| `Default` | 无量化 | 无量化 | PyTorch | 精度优先 |
| `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm` | FP8 通道对称 | FP8 通道动态对称 | VLLM | H100/A100高性能 |
| `W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm` | INT8 通道对称 | INT8 通道动态对称 | VLLM | 通用GPU兼容 |
| `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F` | FP8 通道对称 | FP8 通道动态对称 | Q8F | 高性能推理 |
| `W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F` | INT8 通道对称 | INT8 通道动态对称 | Q8F | 高性能推理 |
| `W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm` | FP8 块对称 | FP8 通道组对称 | DeepGEMM | 大模型优化 |
| `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl` | FP8 通道对称 | FP8 通道动态对称 | SGL | 流式推理 |

#### 量化方案详细说明

**FP8 量化方案**：
- **权重量化**：使用 `torch.float8_e4m3fn` 格式，按通道进行对称量化
- **激活量化**：动态量化，支持 per-token 和 per-channel 模式
- **优势**：在支持 FP8 的 GPU 上提供最佳性能，精度损失最小（通常<1%）
- **适用硬件**：H100、A100、RTX 40系列等支持FP8的GPU

**INT8 量化方案**：
- **权重量化**：使用 `torch.int8` 格式，按通道进行对称量化
- **激活量化**：动态量化，支持 per-token 模式
- **优势**：兼容性最好，适用于大多数 GPU 硬件，内存占用减少约50%
- **适用硬件**：所有支持INT8的GPU

**块量化方案**：
- **权重量化**：按 128x128 块进行 FP8 量化
- **激活量化**：按通道组（组大小128）进行量化
- **优势**：特别适合大模型，内存效率更高，支持更大的batch size

### T5 编码器量化

T5编码器支持以下量化方案：

#### 支持的 quant_scheme 类型

| quant_scheme | 量化精度 | 计算内核 | 适用场景 |
|--------------|----------|----------|----------|
| `int8` | INT8 | VLLM | 通用GPU |
| `fp8` | FP8 | VLLM | H100/A100 GPU |
| `int8-torchao` | INT8 | TorchAO | 兼容性优先 |
| `int8-q8f` | INT8 | Q8F | 高性能推理 |
| `fp8-q8f` | FP8 | Q8F | 高性能推理 |

#### T5量化特性

- **线性层量化**：量化注意力层和FFN层中的线性变换
- **动态量化**：激活在推理过程中动态量化，无需预计算
- **精度保持**：通过对称量化和缩放因子保持数值精度

### CLIP 编码器量化

CLIP编码器支持与T5相同的量化方案：

#### CLIP量化特性

- **视觉编码器量化**：量化Vision Transformer中的线性层
- **文本编码器量化**：量化文本编码器中的线性层
- **多模态对齐**：保持视觉和文本特征之间的对齐精度

## 🚀 生产量化模型

可通过[LightX2V 官方模型仓库](https://huggingface.co/lightx2v)下载量化模型，具体可参考[模型结构文档](../deploy_guides/model_structure.md)。

使用LightX2V的convert工具，将模型转换成量化模型，参考[文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)。

## 📥 加载量化模型进行推理

### DIT 模型配置

将转换后的量化权重的路径，写到[配置文件](https://github.com/ModelTC/lightx2v/blob/main/configs/quantization)中的`dit_quantized_ckpt`中。

```json
{
    "dit_quantized_ckpt": "/path/to/dit_quantized_ckpt",
    "mm_config": {
        "mm_type": "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"
    }
}
```

### T5 编码器配置

```json
{
    "t5_quantized": true,
    "t5_quant_scheme": "fp8",
    "t5_quantized_ckpt": "/path/to/t5_quantized_ckpt"
}
```

### CLIP 编码器配置

```json
{
    "clip_quantized": true,
    "clip_quant_scheme": "fp8",
    "clip_quantized_ckpt": "/path/to/clip_quantized_ckpt"
}
```

### 完整配置示例

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

通过指定`--config_json`到具体的config文件，即可以加载量化模型进行推理。

[这里](https://github.com/ModelTC/lightx2v/tree/main/scripts/quantization)有一些运行脚本供使用。

## 💡 量化方案选择建议

### 硬件兼容性

- **H100/A100 GPU/RTX 4090/RTX 4060**：推荐使用 FP8 量化方案
  - DIT: `W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm`
  - T5/CLIP: `fp8`
- **A100/RTX 3090/RTX 3060**：推荐使用 INT8 量化方案
  - DIT: `W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm`
  - T5/CLIP: `int8`
- **其他 GPU**：根据硬件支持情况选择

### 性能优化

- **内存受限**：选择 INT8 量化方案
- **速度优先**：选择 FP8 量化方案
- **精度要求高**：使用 FP8 或混合精度方案

### 混合量化策略

可以针对不同组件选择不同的量化方案：

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

## 🔧 高阶量化功能

### 量化算法调优

具体可参考量化工具[LightCompress的文档](https://github.com/ModelTC/llmc/blob/main/docs/zh_cn/source/backend/lightx2v.md)

### 自定义量化内核

LightX2V支持自定义量化内核，可以通过以下方式扩展：

1. **注册新的 mm_type**：在 `mm_weight.py` 中添加新的量化类
2. **实现量化函数**：定义权重和激活的量化方法
3. **集成计算内核**：使用自定义的矩阵乘法实现


## 🚨 重要注意事项

1. **硬件要求**：FP8 量化需要支持 FP8 的 GPU（如 H100、RTX40系）
2. **精度影响**：量化会带来一定的精度损失，需要根据应用场景权衡
3. **模型兼容性**：确保量化模型与推理代码版本兼容
4. **内存管理**：量化模型加载时注意内存使用情况
5. **量化校准**：建议使用代表性数据集进行量化校准以获得最佳效果

## 📚 相关资源

- [量化工具文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)
- [运行脚本](https://github.com/ModelTC/lightx2v/tree/main/scripts/quantization)
- [配置文件示例](https://github.com/ModelTC/lightx2v/blob/main/configs/quantization)
- [LightCompress 量化文档](https://github.com/ModelTC/llmc/blob/main/docs/zh_cn/source/backend/lightx2v.md)
