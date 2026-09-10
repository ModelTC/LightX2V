# 模型量化技术

## 📖 概述

LightX2V 支持对 DIT、VAE、T5 和 CLIP 模型进行量化推理，通过降低模型精度来减少显存占用并提升推理速度。

---

## 🔧 量化模式

| 量化模式 | 权重量化 | 激活量化 | 计算内核 | 适用硬件 |
|--------------|----------|----------|----------|----------|
| `fp8-vllm` | FP8 通道对称 | FP8 通道动态对称 | [VLLM](https://github.com/vllm-project/vllm) | H100/H200/H800, RTX 40系等 |
| `int8-vllm` | INT8 通道对称 | INT8 通道动态对称 | [VLLM](https://github.com/vllm-project/vllm) | A100/A800, RTX 30/40系等  |
| `fp8-sgl` | FP8 通道对称 | FP8 通道动态对称 | [SGL](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) | H100/H200/H800, RTX 40系等 |
| `fp8-f16-accum` | FP8 通道对称 | FP8 行动态对称 | CUTLASS FP16 累加 | RTX 5090（SM120） |
| `int8-sgl` | INT8 通道对称 | INT8 通道动态对称 | [SGL](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) | A100/A800, RTX 30/40系等  |
| `fp8-q8f` | FP8 通道对称 | FP8 通道动态对称 | [Q8-Kernels](https://github.com/KONAKONA666/q8_kernels) | RTX 40系, L40S等 |
| `int8-q8f` | INT8 通道对称 | INT8 通道动态对称 | [Q8-Kernels](https://github.com/KONAKONA666/q8_kernels) | RTX 40系, L40S等 |
| `int8-torchao` | INT8 通道对称 | INT8 通道动态对称 | [TorchAO](https://github.com/pytorch/ao) | A100/A800, RTX 30/40系等 |
| `int4-g128-marlin` | INT4 分组对称 | FP16 | [Marlin](https://github.com/IST-DASLab/marlin) | H200/H800/A100/A800, RTX 30/40系等 |
| `fp8-b128-deepgemm` | FP8 分块对称 | FP8 分组对称 | [DeepGemm](https://github.com/deepseek-ai/DeepGEMM) | H100/H200/H800, RTX 40系等|

---

## 🔧 量化模型获取

### 方式一：下载预量化模型

从 LightX2V 模型仓库下载预量化的模型：

**DIT 模型**

从 [Wan2.1-Distill-Models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models) 下载预量化的 DIT 模型：

```bash
# 下载 DIT FP8 量化模型
huggingface-cli download lightx2v/Wan2.1-Distill-Models \
    --local-dir ./models \
    --include "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors"
```

**Encoder 模型**

从 [Encoders-LightX2V](https://huggingface.co/lightx2v/Encoders-Lightx2v) 下载预量化的 T5 和 CLIP 模型：

```bash
# 下载 T5 FP8 量化模型
huggingface-cli download lightx2v/Encoders-Lightx2v \
    --local-dir ./models \
    --include "models_t5_umt5-xxl-enc-fp8.pth"

# 下载 CLIP FP8 量化模型
huggingface-cli download lightx2v/Encoders-Lightx2v \
    --local-dir ./models \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8.pth"
```

### 方式二：自行量化模型

详细量化工具使用方法请参考：[模型转换文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)

---

## 🚀 量化模型使用

### DIT 模型量化

#### 支持的量化模式

DIT 量化模式（`dit_quant_scheme`）支持：`fp8-vllm`、`int8-vllm`、`fp8-sgl`、`fp8-f16-accum`、`int8-sgl`、`fp8-q8f`、`int8-q8f`、`int8-torchao`、`int4-g128-marlin`、`fp8-b128-deepgemm`

#### 配置示例

```json
{
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-sgl",
    "dit_quantized_ckpt": "/path/to/dit_quantized_model"  // 可选
}
```

> 💡 **提示**：当运行脚本的 `model_path` 中只有一个 DIT 模型时，`dit_quantized_ckpt` 可以不用单独指定。

#### MiniMax-H3 FP8 FP16 累加

RTX 5090 上的 MiniMax-H3 可以通过 `fp8-f16-accum` 使用 FP8 输入和 FP16 累加。权重需要用
`h3-fp8-f16-accum` profile 转换；普通 `fp8-sgl` checkpoint 不兼容。DiT 和 Video VAE decoder
分别转换，profile 会独立选择使用 qmax 14 的投影层、保留其他层的标准 FP8 量化，并把策略写入
safetensors metadata。

```bash
python tools/convert/converter.py \
    --source /path/to/MiniMax-H3/transformer \
    --output /path/to/h3_quantized \
    --output_name minimax_h3_dit_fp8_f16_accum \
    --output_ext .safetensors \
    --model_type h3 \
    --device cuda \
    --quantized \
    --bits 8 \
    --linear_type fp8 \
    --quantization_profile h3-fp8-f16-accum \
    --single_file

python tools/convert/converter.py \
    --source /path/to/MiniMax-H3/vae \
    --output /path/to/h3_quantized \
    --output_name minimax_h3_video_vae_fp8_f16_accum \
    --output_ext .safetensors \
    --model_type h3_video_vae_decoder \
    --device cuda \
    --quantized \
    --bits 8 \
    --linear_type fp8 \
    --quantization_profile h3-fp8-f16-accum \
    --single_file
```

```json
{
  "dit_quantized": true,
  "dit_quant_scheme": "fp8-f16-accum",
  "dit_quantized_ckpt": "/path/to/minimax_h3_dit_fp8_f16_accum.safetensors",
  "video_vae_quantized": true,
  "video_vae_quant_scheme": "fp8-f16-accum",
  "video_vae_quantized_ckpt": "/path/to/minimax_h3_video_vae_fp8_f16_accum.safetensors"
}
```

激活按行动态量化，`scale = max(abs(x)) / qmax`。减小 qmax 会扩大 scale，从而降低 FP16
累加器中的原始数值范围，但也会减少 FP8 有效量化级数。实测中 DiT 的 qmax 14 和 12 会在 FFN-out
产生非有限值，qmax 7 可完成全部去噪步骤；Video VAE decoder 在 qmax 14 下保持有限且误差最小。因此
当前 H3 策略固定使用 DiT activation qmax 7 和 Video VAE activation qmax 14，避免运行配置与
checkpoint 的转换策略错配。

DiT 仅对 Q/K/V、attention output 和 FFN projection 启用该内核，Video VAE 仅对 packed QKV、
attention output 和 FFN projection 启用。扩展不可用或设备不是 SM120 时会回退到 `fp8-sgl`；
DiT tensor parallel 当前也回退到 `fp8-sgl`。初始化日志会打印实际启用范围或回退原因。
其他 pipeline 配置与该量化模式相互独立。

该内核会按精确 GEMM shape 自动调优 CUTLASS tile 和 swizzle。首次遇到新 shape 时在 C++ 内遍历
内置候选，winner 保存在当前进程的 C++ cache 中；同一进程的后续调用只执行 cache 查询。若 warmup
覆盖正式请求的 shape，首次调优开销会在请求前完成。进程重启后会重新调优一次，不写用户目录。

### MiniMax-H3 Video VAE Encoder Conv3D

MiniMax-H3 Video VAE Encoder 支持默认 PyTorch 路径和三种可选的 Conv3D 模式：

| mode | 实现 | 权重 qmax | 激活 qmax | 累加类型 | checkpoint 约束 |
| --- | --- | ---: | ---: | --- | --- |
| `torch`（默认） | PyTorch Conv3D | - | - | 由后端决定 | 原始 Encoder 权重 |
| `torch_channels_last` | PyTorch Conv3D + `channels_last_3d` | - | - | 由后端决定 | 原始 Encoder 权重 |
| `cutlass_fp8_f32_accum` | CUTLASS FP8 Conv3D | 448 | 448 | FP32 | 标准 FP8 权重与 scale |
| `cutlass_fp8_f16_accum` | CUTLASS FP8 Conv3D | 21 | 21 | FP16 | `h3-vae-encoder-fp8-f16-accum` profile |

这些模式只影响会调用 Video VAE Encoder 的任务，例如 I2AV 和 REF2AV；T2AV 不执行 Encoder。
`torch_channels_last` 使用相同的 PyTorch 算子和 dtype 策略，该配置只改变 tensor layout。

FP32 累加使用标准 E4M3 全量程，不需要额外的命名 profile。FP16 累加的 qmax 是权重量化与运行时
激活量化共同遵守的数值约束，因此 checkpoint 必须携带对应 profile。它是更激进的模式，部署前应完成
输出质量准出。

转换器的输入必须是运行时兼容的 H3 Video VAE FP8 混合 checkpoint：Decoder Linear 的权重和
scale 已分别是 FP8 和 FP32，对应 bias 为 FP16；所有 Encoder tensor 则保持原始 FP32 dtype。转换器会校验
但不会修改 Decoder，只量化 33 个 Encoder Conv3D 中已经验证的 27 个；其余 6 个敏感权重和所有
Encoder bias 保持 FP32。转换器不负责把 BF16 Decoder 转换为 FP8。离线转换和运行时必须使用相同的
Encoder FP8 mode。Decoder tensor 和 metadata 均原样保留，因此 `video_vae_quant_scheme` 必须与源
Decoder 一致（`fp8-sgl` 或 `fp8-f16-accum`）。下例以满足上述 dtype 约束的 FP8-F16 Decoder checkpoint 为输入，
追加 Encoder qmax 21 量化，不改变 Decoder 的 qmax 14 策略：

```bash
python tools/convert/converter.py \
    --source /path/to/minimax_h3_video_vae_fp8_f16_accum.safetensors \
    --output /path/to/h3_quantized \
    --output_name minimax_h3_video_vae_fp8_f16_accum_encoder_conv_q21 \
    --output_ext .safetensors \
    --model_type h3_video_vae_encoder \
    --vae_encoder_conv_mode cutlass_fp8_f16_accum \
    --device cuda \
    --quantized \
    --bits 8 \
    --linear_type fp8 \
    --single_file
```

运行时使用相同的 mode：

```json
{
    "video_vae_quantized": true,
    "video_vae_quant_scheme": "fp8-f16-accum",
    "video_vae_quantized_ckpt": "/path/to/minimax_h3_video_vae_fp8_f16_accum_encoder_conv_q21.safetensors",
    "vae_encoder_conv_mode": "cutlass_fp8_f16_accum"
}
```

FP8 Conv3D 当前仅支持 SM120，并要求安装由包含该算子的代码版本构建的 `lightx2v_kernel` wheel。kernel
会在新 Conv3D shape 首次出现时自动调优，并在当前进程内复用最优配置，不需要额外的离线调优步骤。

### T5 模型量化

#### 支持的量化模式

T5 量化模式（`t5_quant_scheme`）支持：`int8-vllm`、`fp8-sgl`、`int8-q8f`、`fp8-q8f`、`int8-torchao`

#### 配置示例

```json
{
    "t5_quantized": true,
    "t5_quant_scheme": "fp8-sgl",
    "t5_quantized_ckpt": "/path/to/t5_quantized_model"  // 可选
}
```

> 💡 **提示**：当运行脚本指定的 `model_path` 中存在 T5 量化模型（如 `models_t5_umt5-xxl-enc-fp8.pth` 或 `models_t5_umt5-xxl-enc-int8.pth`）时，`t5_quantized_ckpt` 可以不用单独指定。

### CLIP 模型量化

#### 支持的量化模式

CLIP 量化模式（`clip_quant_scheme`）支持：`int8-vllm`、`fp8-sgl`、`int8-q8f`、`fp8-q8f`、`int8-torchao`

#### 配置示例

```json
{
    "clip_quantized": true,
    "clip_quant_scheme": "fp8-sgl",
    "clip_quantized_ckpt": "/path/to/clip_quantized_model"  // 可选
}
```

> 💡 **提示**：当运行脚本指定的 `model_path` 中存在 CLIP 量化模型（如 `models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8.pth` 或 `models_clip_open-clip-xlm-roberta-large-vit-huge-14-int8.pth`）时，`clip_quantized_ckpt` 可以不用单独指定。

### 性能优化策略

如果显存不够，可以结合参数卸载来进一步减少显存占用，参考[参数卸载文档](../method_tutorials/offload.md)：

> - **Wan2.1 配置**：参考 [offload 配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/offload)
> - **Wan2.2 配置**：参考 [wan22 配置文件](https://github.com/ModelTC/LightX2V/tree/main/configs/wan22) 中以 `4090` 结尾的配置

---

## 📚 相关资源

### 配置文件示例
- [INT8 量化配置](https://github.com/ModelTC/LightX2V/blob/main/configs/quantization/wan_i2v.json)
- [Q8F 量化配置](https://github.com/ModelTC/LightX2V/blob/main/configs/quantization/wan_i2v_q8f.json)
- [TorchAO 量化配置](https://github.com/ModelTC/LightX2V/blob/main/configs/quantization/wan_i2v_torchao.json)

### 运行脚本
- [量化推理脚本](https://github.com/ModelTC/LightX2V/tree/main/scripts/quantization)

### 工具文档
- [量化工具文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)
- [LightCompress 量化文档](https://github.com/ModelTC/llmc/blob/main/docs/zh_cn/source/backend/lightx2v.md)

### 模型仓库
- [Wan2.1-LightX2V 量化模型](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)
- [Wan2.2-LightX2V 量化模型](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)
- [Encoders 量化模型](https://huggingface.co/lightx2v/Encoders-Lightx2v)

---

通过本文档，您应该能够：

✅ 理解 LightX2V 支持的量化方案
✅ 根据硬件选择合适的量化策略
✅ 正确配置量化参数
✅ 获取和使用量化模型
✅ 优化推理性能和显存使用

如有其他问题，欢迎在 [GitHub Issues](https://github.com/ModelTC/LightX2V/issues) 中提问。
