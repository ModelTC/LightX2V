# EdenLive 部署指南

[English](README.md) | [简体中文](README_zh.md)

EdenLive 根据首帧图片、录屏视频和提示词生成视频。示例支持 Wan2.2 5B 与 Wan2.1 14B，提供双向和自回归（AR）推理。以下命令均在 LightX2V 仓库根目录执行。

## 1. 环境与模型文件

按仓库的[快速开始](../../docs/ZH_CN/source/getting_started/quickstart.md)准备运行环境。Attention 与 linear kernel 需支持目标 GPU：实时配置使用 SageAttention2、FlashInfer、SGL kernel，RTX 5090 还使用 comfy-kitchen INT8 ConvRot。ConvRot GELU 融合首次构建需要 CUDA 编译器。

原始模型与转换后模型放在不同目录：

```bash
export EDEN_RELEASE=/path/to/EdenLive
export EDEN_MODEL=/path/to/EdenLive-x2v
export CUDA_VISIBLE_DEVICES=0
export PROFILING_DEBUG_LEVEL=0
```

完整目录转换需要四种 DiT 和两种编码器：

```text
EdenLive/
├── dit-5b/
├── dit-5b-bidir/
├── dit-14b/
├── dit-14b-bidir/
├── umt5/text_encoder/
├── umt5/tokenizer/
├── clip/image_encoder/
├── vision-encoder/backbone/
├── vision-encoder/projection.safetensors
├── vae-5b/
└── vae-14b/
```

DiT 子目录需包含 `config.json` 和 safetensors 权重，并保留原始模型的 tokenizer 与辅助文件。即使单独加载量化 DiT，`model_path` 仍指向包含各组件的完整转换后模型根目录。

## 2. 模型格式转换与量化

### 2.1 转换原始模型

转换四种 DiT 及 T5/CLIP 编码器的参数名称，同时复制 DINO、VAE、tokenizer 和其他辅助文件：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE" --output "$EDEN_MODEL" --all
```

`--all` 需要上述完整模型目录。输出目录不能与输入相同，也不能位于输入目录内部。转换后 DiT 权重为 BF16，配置示例也会复制到输出目录的 `configs/` 下。

只转换一个 DiT 时：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE/dit-5b" --output "$EDEN_MODEL/dit-5b"
```

此命令不会准备编码器和其他模型文件。转换 `dit-5b-bidir`、`dit-14b` 或 `dit-14b-bidir` 时，同步调整输入和输出子目录。

### 2.2 DiT 量化

Hopper FP8-SGL 配置使用的 FP8 权重：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8 \
  --quantize fp8 --device cuda:0
```

RTX 5090 配置使用的 INT8 ConvRot 权重：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-int8-convrot \
  --quantize int8-convrot --device cuda:0
```

转换参数使用 `fp8`，运行时量化方案使用 `fp8-sgl`。对应产物为 `dit-5b-fp8.safetensors` 和 `dit-5b-int8-convrot.safetensors`。修改基础配置时，需要同时设置 `dit_quantized`、`dit_quant_scheme` 和 `dit_quantized_ckpt`。

### 2.3 T5 与 ProxyDINO 量化

从已完成参数名称转换的 native T5 权重生成 FP8 文件：

```bash
python tools/convert/converter.py \
  --source "$EDEN_MODEL/umt5/text_encoder/model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name t5-fp8 \
  --model_type wan_t5 --quantized --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 --device cuda:0 --single_file
```

运行时设置 `t5_quantized=true`、`t5_quant_scheme="fp8-sgl"` 和生成的 `t5_quantized_ckpt`。两个实时配置均使用 `t5_cpu_offload=false`，让 T5 常驻 GPU。

RTX 5090 实时配置还使用独立的 INT8 ConvRot ProxyDINO 权重：

```bash
python tools/convert/convert_edenlive_proxy.py \
  --source "$EDEN_MODEL/vision-encoder/projection.safetensors" \
  --output "$EDEN_MODEL/quantized/proxy-dino-int8-convrot.safetensors" \
  --device cuda:0 --group-size 256
```

这里的 `--output` 是文件路径。工具量化 linear 权重，保留其他张量，并在权重旁生成量化误差报告。默认拒绝覆盖已有权重或报告，确需替换时显式传入 `--overwrite`。Hopper 配置使用原始 ProxyDINO 权重。

## 3. 推理脚本

样本目录需包含 `cover.png`、`recording.mp4` 和带有 `staticPrompt` 字段的 `prompts.json`。指定模型与输入路径后运行：

```bash
MODEL_PATH="$EDEN_MODEL" \
SAMPLE_DIR=/path/to/cryo-crossing \
SAVE_RESULT_PATH=/path/to/output.mp4 \
CUDA_VISIBLE_DEVICES=0 bash scripts/ed/wan22_5b_few_step.sh
```

| 脚本 | 模型 | 推理方式 / 默认配置 |
| --- | --- | --- |
| `wan22_5b_bidir.sh` | Wan2.2 5B | 双向推理，基础 BF16 配置 |
| `wan22_5b_few_step.sh` | Wan2.2 5B | 四步 AR，基础 BF16 配置 |
| `wan21_14b_bidir.sh` | Wan2.1 14B | 双向推理，基础 BF16 配置 |
| `wan21_14b_few_step.sh` | Wan2.1 14B | 四步 AR，基础 BF16 配置 |
| `run_5090.sh` | Wan2.2 5B | RTX 5090 单卡，INT8 ConvRot 实时配置 |
| `run_hopper.sh` | Wan2.2 5B | H100/H200 单卡，FP8-SGL 实时配置 |

六个脚本均支持 `MODEL_PATH`、`SAMPLE_DIR`、`SAVE_RESULT_PATH` 和 `CUDA_VISIBLE_DEVICES`。四个基础脚本还支持 `CONFIG_JSON`，可选择保留的其他基础配置，包括 prompt travel。额外 CLI 参数会继续传给推理入口，例如 `--seed 123` 或 `--prompt 'A frozen cavern'`。

两个实时脚本额外支持以下环境变量：

| 变量 | 默认值 |
| --- | --- |
| `DIT_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/dit-5b-int8-convrot.safetensors`（5090），或 `dit-5b-fp8.safetensors`（Hopper） |
| `T5_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/t5-fp8.safetensors` |
| `PROXY_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/proxy-dino-int8-convrot.safetensors`，仅 5090 |
| `TAE_PATH` | 仓库同级目录的 `lighttaew2_2.safetensors` |
| `T5_CPU_OFFLOAD` | `false`；设为 `true` 可对比降低常驻显存后的表现 |

运行示例：

```bash
MODEL_PATH="$EDEN_MODEL" SAMPLE_DIR=/path/to/cryo-crossing \
TAE_PATH=/path/to/lighttaew2_2.safetensors \
CUDA_VISIBLE_DEVICES=0 bash scripts/ed/run_5090.sh

MODEL_PATH="$EDEN_MODEL" SAMPLE_DIR=/path/to/cryo-crossing \
TAE_PATH=/path/to/lighttaew2_2.safetensors \
CUDA_VISIBLE_DEVICES=0 bash scripts/ed/run_hopper.sh
```

脚本会自动定位仓库，并使用临时配置覆盖权重路径，原始实时 JSON 不变。Profiling 默认为 `0`；需要诊断计时时可设置 `PROFILING_DEBUG_LEVEL=1`。

## 4. 实时配置与 LightTAE

[`configs/edenlive/realtime`](../../configs/edenlive/realtime/) 中两个 JSON 均使用 1280×704 输出、24 FPS、四步去噪，每块三个 latent 帧，local attention 为 18、sink 为 6、KV 按浮点存储，并使用 LightTAE 同步流式解码。

这里的 24 FPS 指视频播放和录屏采样设置，生成吞吐需单独计时测量。

| 配置 | DiT attention：self / cross 1 / cross 2 | ProxyDINO |
| --- | --- | --- |
| `wan22_5b_ar_int8_convrot_5090.json` | SM120 cached / SM120 cached / Sage2 | INT8 ConvRot，Sage2 |
| `wan22_5b_ar_fp8_sgl_hopper.json` | SM90 fused / SM90 fused / SM90 fused | 原始权重，Sage2 |

按 GPU 架构选择对应配置。Attention 算子由 `self_attn_1_type`、`cross_attn_1_type` 和 `cross_attn_2_type` 显式指定。

配套 Wan2.2 LightTAE 权重使用以下解码设置：

```json
{
  "use_tae": true,
  "tae_path": "/path/to/lighttaew2_2.safetensors",
  "need_scaled": true,
  "streaming_vae": true,
  "use_cuda_graph": true
}
```

首帧仍由原版 Wan VAE 编码。`use_cuda_graph` 同时控制 DiT 和 ProxyDINO 的 Graph；执行条件不满足时使用原有 compiled/eager 路径。

LightTAE 的流式解码张量计算跟随 `use_compile` 开关，保留跨块缓存和首尾块处理，进度条不进入编译区域。使用现有 `vae_use_compile=false` 可单独对比 eager 解码；新形状首次执行包含编译开销，需在预热阶段完成。不带持久缓存的整段视频解码保持 eager；编译形状缓存达到上限时也回退到 eager。

## 5. 优化项

| 范围 | 优化内容与适用场景 |
| --- | --- |
| DiT linear kernel | 使用 FP8-SGL 或 INT8 ConvRot linear，非线性张量使用 BF16。 |
| QKV 投影 | 共享输入激活量化；满足条件时将 self-attention QKV pack 为一次 GEMM，cross-attention K/V 也可共享量化。 |
| FFN | 融合 GELU 与激活量化，保留支持路径中的 BF16 舍入边界。 |
| AR KV 存储 | 固定 sink 加最近帧 ring，避免窗口滚动时整体搬移保留的历史。 |
| Attention / RoPE | 跨步复用历史 K/V 工作区、持久转置 V 与固定窗口 RoPE 数据。 |
| Cross-attention | 缓存文本 K/V 投影；SM120 cached Sage2 还可复用预处理后的量化 K/V。 |
| 条件计算 | 在适用请求内缓存固定 text/time 投影和重复的 Proxy 特征插值。 |
| KV commit | 最后一层仅更新缓存的 pass 只计算所需 K/V，省去未使用的 attention、FFN 与输出后缀。 |
| ProxyDINO | GPU bilinear 预处理、融合 Q/K RoPE 与归一化、模型 dtype 残差、batch dense attention，以及可选共享 INT8 QKV 量化。 |
| 录屏输入 | 顺序解码，CPU worker 提前读取下一块 RGB uint8 数据。 |
| Compile / warmup | 编译满足条件的 DiT block 与 LightTAE 顺序解码器，并在正式计时前预热对应形状和流式缓存状态。 |
| CUDA Graph | 复用满足固定形状条件的 DiT 与 bilinear ProxyDINO Graph，减少 kernel launch 开销。 |
| 解码与显存 | 逐块 LightTAE 解码；拼接完整视频前释放已结束请求引用与 KV pool。 |

共享投影与 GELU/量化融合自动选择支持的 `fp8-sgl` 和 `int8-convrot` 路径。Shape、dtype、GPU、adapter、offload 或并行执行条件不满足时回退到原算子。INT8 共享路径面向 SM120 常驻 group-256 投影；INT8 GELU 融合要求 BF16 输入、输入宽度 14336、至少三个 token，且输出宽度能被 16 整除。FP8 GELU 融合支持输入宽度为 1–16384 的二维 BF16 张量。FFN 融合需要保留精度转换的 compiled 路径（`use_compile=true`、默认 compile mode、`emulate_precision_casts=true`）。

| Attention 实现 | 持久转置 V | 预处理量化 cross-KV | DiT 整段 CUDA Graph |
| --- | --- | --- | --- |
| SM120 cached Sage2 | 支持 | 支持 | 常驻单卡实时配置满足条件时启用 |
| SM90 fused Sage2 | 支持 | 暂不支持 | 回退到 compiled block；ProxyDINO Graph 仍可用 |

Hopper 当前缺少 DiT 整段 Graph 所需的 prepared cross-KV 接口。量化、SageAttention、ProxyDINO dtype 调整和 LightTAE 均可能影响输出质量，需使用代表性输入评估。上述配置保留既定去噪步数和 attention 上下文。
