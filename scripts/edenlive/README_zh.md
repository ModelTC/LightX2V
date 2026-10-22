# EdenLive

[English](README.md) | [简体中文](README_zh.md)

EdenLive 是基于 Wan 的神经渲染模型，以 **proxy 视频或参考视频流**、**目标首帧图像**和**文本提示词**为输入，生成细化或风格化的视频。参考内容可以来自 2D/3D 场景、现有引擎或录屏。上游应用负责维护交互状态，EdenLive 负责视觉渲染。

控制信息通过**经过后训练的 DINOv3 视觉编码器**注入：编码器逐帧提取 proxy 特征，作为扩散 Transformer 的条件输入。proxy 提供场景布局、相机运动和时序，首帧与文本提示词指定目标画面外观。

[EdenLive 项目](https://vast-eden.ai/?p=edenlive)展示了超过 24 FPS 的生成速度、分钟级持续生成，以及 480p/720p 输出。5B 模型支持单卡 720p 部署。LightX2V 为双向和自回归版本提供推理加速，包括量化、流式解码、编译和 CUDA Graph。

## 支持的模型

| 基础模型 | 推理方式 | 权重目录 | 启动脚本 |
| --- | --- | --- | --- |
| Wan2.1 14B | 双向推理 | `dit-14b-bidir` | [wan21_14b_bidir.sh](wan21_14b_bidir.sh) |
| Wan2.1 14B | 四步自回归 | `dit-14b` | [wan21_14b_few_step.sh](wan21_14b_few_step.sh) |
| Wan2.2 5B | 双向推理 | `dit-5b-bidir` | [wan22_5b_bidir.sh](wan22_5b_bidir.sh) |
| Wan2.2 5B | 四步自回归 | `dit-5b` | [wan22_5b_few_step.sh](wan22_5b_few_step.sh) |

双向推理对完整片段联合去噪。自回归推理通过持久 KV 缓存连续生成视频块，支持流式输出解码后的视频。

## 环境与模型准备

按照 LightX2V [快速开始](../../docs/ZH_CN/source/getting_started/quickstart.md)准备环境。实时配置需要与目标 GPU 兼容的 SageAttention2、FlashInfer 和 SGL kernel。RTX 5090 配置还需要 comfy-kitchen 和 CUDA 编译器，用于 INT8 ConvRot。

以下命令均在 LightX2V 仓库根目录执行。

### 下载与转换

下载完整的 [VAST-AI/EdenLive](https://huggingface.co/VAST-AI/EdenLive) 模型，并转换为 LightX2V 格式：

```bash
hf download VAST-AI/EdenLive --local-dir ./models/EdenLive

export EDEN_RELEASE="$PWD/models/EdenLive"
export EDEN_MODEL="$PWD/models/EdenLive-x2v"

python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE" --output "$EDEN_MODEL" --all
```

完整转换会处理四种 DiT 和 T5/CLIP 编码器，并保留 DINO、VAE、tokenizer 和 LightTAE 文件。所有推理方式（包括量化推理）都应将启动脚本中的 `model_path` 设为转换后的模型根目录。

## 推理

参照 [`scripts/wan`](../wan) 的写法，先修改启动脚本顶部的 `lightx2v_path`、`model_path`，再设置命令中的输入图像/视频、提示词与输出路径。运行 BF16 Wan2.2 5B 自回归模型：

```bash
IMAGE_PATH=/path/to/cover.png VIDEO_PATH=/path/to/recording.mp4 bash scripts/edenlive/wan22_5b_few_step.sh
```

其他版本使用模型表中对应的启动脚本。修改 `--config_json` 参数可选择其他配置；基础配置位于 [`configs/edenlive`](../../configs/edenlive/)，包含 AR 动态提示词配置。`--seed` 等推理参数直接在脚本中修改。

### 本机 cryo-crossing 推理

`wan22_5b_few_step.sh` 使用本机转换后的 BF16 模型、seed 2026092901 和 cryo-crossing 提示词。运行前通过 `IMAGE_PATH`、`VIDEO_PATH` 指定本地输入。对应 JSON 开启 compile、warmup、公共 CUDA Graph 与 RoPE/cache 融合。

```bash
IMAGE_PATH=/path/to/cover.png VIDEO_PATH=/path/to/recording.mp4 bash scripts/edenlive/wan22_5b_few_step.sh
```

脚本将视频保存到当前仓库的 `save_results`。profiling=1 便于查看 `RUN pipeline`，但逐 step 同步会影响测速；`Total Cost` 还包含加载和 warmup。严谨比较稳态速度及完整 latent/pixel，当前代码可运行 [pipeline benchmark](../self_forcing/benchmark_edenlive_pipeline.py)，使用 `--execution compiled_graph --requests 3`，启动时显式设置 `PROFILING_DEBUG_LEVEL=0`，它会保存完整 tensor。

## 实时部署

硬件预设面向 **Wan2.2 5B AR、1280×704、24 FPS**，每个视频块执行四步去噪，包含三个 latent 帧。两套配置均使用 GPU 常驻的 FP8 T5、LightTAE 同步流式解码、编译和预热。

| 硬件 | DiT | Proxy 编码器 | 启动脚本 |
| --- | --- | --- | --- |
| H100 / H200 | FP8-SGL | BF16 | [run_hopper.sh](run_hopper.sh) |
| RTX 5090 | INT8 ConvRot | INT8 ConvRot | [run_5090.sh](run_5090.sh) |

配置预设：[Hopper FP8-SGL](../../configs/edenlive/realtime/wan22_5b_ar_fp8_sgl_hopper.json) · [RTX 5090 INT8 ConvRot](../../configs/edenlive/realtime/wan22_5b_ar_int8_convrot_5090.json) · [SM120 FP8-F16 累加（待实卡验证）](../../configs/edenlive/realtime/wan22_5b_ar_fp8_f16_accum_5090.json)。

### 准备量化权重

两套预设均使用由 LightX2V 格式模型转换得到的 FP8 T5 权重：

```bash
python tools/convert/converter.py \
  --source "$EDEN_MODEL/umt5/text_encoder/model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name t5-fp8 \
  --model_type wan_t5 --quantized --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 --device cuda:0 --single_file
```

对于 **H100/H200**，将 DiT 量化为 FP8：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8 \
  --quantize fp8 --device cuda:0
```

对于 **SM120 FP8-F16 累加**，需要单独生成缩小量化范围的 DiT checkpoint。普通 FP8-SGL checkpoint 的量化范围不同，不能直接复用：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8-f16-accum \
  --quantize fp8-f16-accum --device cuda:0
```

在启动脚本中选择 `wan22_5b_ar_fp8_f16_accum_5090.json` 并填写权重路径。此路径尚未在 SM120 上测速。

对于 **RTX 5090**，将 DiT 和 proxy 编码器量化为 INT8 ConvRot：

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-int8-convrot \
  --quantize int8-convrot --device cuda:0

python tools/convert/convert_edenlive_proxy.py \
  --source "$EDEN_MODEL/vision-encoder/projection.safetensors" \
  --output "$EDEN_MODEL/quantized/proxy-dino-int8-convrot.safetensors" \
  --device cuda:0 --group-size 256
```

### 运行硬件预设

设置对应启动脚本中的 `lightx2v_path`、`model_path` 和输入/输出参数，再将该脚本所选 JSON 中的权重路径改为上一步准备的文件。

在 **H100/H200** 上运行：

```bash
bash scripts/edenlive/run_hopper.sh
```

在 **RTX 5090** 上运行：

```bash
bash scripts/edenlive/run_5090.sh
```

转换后的模型包含本示例使用的 LightTAE 权重。JSON 字段 `tae_path` 也支持微调后的 `.safetensors` 权重。原始 Wan VAE 编码目标首帧，LightTAE 解码生成的 chunk。

在 `--config_json` 指向的 JSON 中设置下列字段；启动脚本直接使用该配置，不再生成临时 JSON：

| JSON 字段 | 转换后模型目录中的权重 |
| --- | --- |
| `dit_quantized_ckpt` | Hopper：`quantized/dit-5b-fp8.safetensors`；INT8 ConvRot：`quantized/dit-5b-int8-convrot.safetensors`；SM120 FP8-F16：`quantized/dit-5b-fp8-f16-accum.safetensors` |
| `t5_quantized_ckpt` | `quantized/t5-fp8.safetensors` |
| `proxy_quantized_ckpt` | `quantized/proxy-dino-int8-convrot.safetensors`，仅 RTX 5090 |
| `tae_path` | `decoder-5b/lighttaew2_2.pth` |

将 JSON 中的 `/path/to/EdenLive-x2v` 前缀改为实际模型目录；脚本中的 `model_path` 不会自动改写 JSON 权重路径。设置 JSON 中的 `t5_cpu_offload=true` 可降低 GPU 常驻显存，下方性能测试使用 `false`。输出 MP4 路径由脚本中的 `--save_result_path` 指定。

## LightX2V 加速

| 组件 | 优化 |
| --- | --- |
| 量化投影 | FP8-SGL 或 INT8 ConvRot 矩阵乘、共享激活量化、合并 QKV 投影，以及 GELU/量化融合。 |
| Attention 与小算子 | 面向 GPU 架构的 SageAttention2、FlashInfer RoPE、SGL RMSNorm，以及 Triton LayerNorm/modulation。 |
| AR 缓存 | 固定 sink 与环形 KV 存储、可复用的 attention 工作区，以及历史 RoPE 数据缓存。 |
| 条件计算 | 复用文本 K/V、文本与时间投影，以及重复的 proxy 特征插值。 |
| KV commit | 保留 clean forward，在完成最终所需缓存更新后，跳过不再使用的 attention、FFN 和输出计算。 |
| Proxy 编码器 | GPU 预处理、RoPE/归一化融合和 CUDA Graph 重放。 |
| 编译 | 编译 DiT block 和流式 LightTAE 解码器，预热所需形状与缓存状态。 |
| 流式处理 | 预取 proxy 帧，使用持久解码器状态逐块解码生成结果。 |

此前常驻 BF16 / FA3 和标准 SageAttention2 专项已验证统一 DiT CUDA Graph。其中 Sage2 的完整 717 帧 cryo-crossing 与原始 main 的 latent 和视频逐字节一致，预热后整栈 replay 275 次。本轮 FA3 SP2/offload 结果在下文单独记录。硬件预设继续使用各自的编译 block 和 proxy Graph；Sage2 通过普通有序 RoPE workspace 运行，不需要专用 prepared-KV 缓存。

### 历史 BF16 FlashAttention-3 验证（2026-10-19/20）

已验证的 5B AR 精确路径使用常驻 BF16 权重、原生 `flash_attn3` self/cross attention，并同时开启 `use_compile=true`、`use_cuda_graph=true`、`warmup=true`。保留 `ar_config.kv_cache_scheme="rolling"`：继续存储 raw K、按原逻辑位置计算 RoPE、复用历史工作区并执行原 dense FA3 归约，同时共用[公共自回归 CUDA Graph stack](../../lightx2v/common/transformer_infer/autoregressive_cuda_graph.py)。

真实 cryo-crossing 取原 case 前 249 帧，832×480、24 FPS，保留原前三段动态 prompt。独立导出的修改前 HEAD `d48a6173` 三请求平均 **9.347 秒**，最终共享图版本平均 **9.042 秒，快 3.27%**；旧/新六个请求的完整 latent 与未压缩 pixel tensor 逐值一致。DiT event 从 3208.50 降至 2903.71 ms。原版 BF16 FA3 支持 compile，但当时的 guard 不启用 DiT 手动图；新版每正式请求 replay 80 次，无正式 capture、重编译或 fallback。保留三个原始生产 warmup shape。单次启动加首请求旧/新为 57.151 / 57.619 秒，因此这里证明正式请求提速，不宣称冷启动更快。[完整流水线基准](../self_forcing/benchmark_edenlive_pipeline.py)测试此 dense 路径。

常驻 FA3 / interleaved FlashInfer RoPE 路径自动融合历史 gather、RoPE 和有序 workspace 写入；另一个 kernel 处理当前 Q/K RoPE 与尾部写入，保留 raw K/V 给 clean commit。同一 chunk 的历史只准备一次，去噪期间复用，workspace 与 dense attention 的逻辑顺序保持原样。Clean commit 的末层 K/V-only 自动启用；能力检查读取实际 adapter 状态，不再将 RMSNorm 默认的零值 `weight_diff` 误判为已加载 adapter。其他 RoPE 格式通过普通 workspace 路径计算。已移除融合开关及 Sage 专用的转置 V、prepared/量化 cross-KV 分支；普通文本 cross-KV 缓存保留。

2026-10-20 的同源码融合开关对照使用真实 BF16 5B cryo-crossing、480×832 / 249 帧，在同一张 H200 三开，各测三次正式请求。DiT CUDA event 均值 **2881.911 → 2846.562 ms，减少 1.227%**；完整 pipeline **8.948430 → 8.922167 秒，减少 0.293%**，三对对应请求均更快。原始 main、融合前完整源码和融合后共 32 个完整 latent/pixel tensor 比较逐字节一致；eager 与 compiled 各自对齐原始 main 的相同执行模式。正式每请求 80 次 graph replay，无新增编译、capture 或 fallback。融合匹配 FlashInfer 的 FP32 cosine FMA 与 BF16/FP16 舍入。本 case 仍保留 **30 层合计 2.4101 GiB 历史 workspace**，收益来自减少准备过程的读写轮次和 kernel launch，没有消除历史 copy 或 workspace 显存；启动和 warmup 不计入稳态收益。

EdenLive 的 [`RoPEKVWorkspace`](../../lightx2v/common/kvcache/attn/rope.py) 负责可复用的有序历史工作区，`prepare_history` / `update_rope` 接收可见范围与频率。SF 的物理 ring view 由 rolling pool 提供，attention 后端与此 workspace 独立；位置策略与提交时机仍由模型决定。

EdenLive AR 组合 `EdenLiveTransformerInfer` 与 `WanCausalTransformerInfer`；后者继承 `WanAutoregressiveTransformerInfer`，与 SF 共用 KV-only 编译分发、请求清理与 Graph 生命周期。专用 RoPE、proxy 和 scheduler 流程保留在 EdenLive；runner 共用预热尺寸排序和请求清理。接口约定见 [SF 公共执行接口](../self_forcing/README_zh.md#14-公共执行接口)。

2026-10-20 的默认 KV-only 修正与公共执行层验收：真实 5B 三开，两轮同卡分组 DiT 减少约 1.3–1.9%，clean 减少约 3.7–4.2%；受控反向顺序复测 pipeline 减少 1.4%，输出回传存在时段波动。48 项完整 latent/video 字节比较均为零差；正式 80 replay、0 compile/capture/fallback。增加独立 clean 图后 allocated 峰值增加约 618 MiB。

## 性能

下表保留此前硬件预设的性能记录；上方注明日期的 BF16 / FA3 结果对应此前缓存清理；当前公共执行结果见下文。

**Wan2.2 5B AR · 1280×704 · 24 FPS · batch size 1 · LightTAE 同步解码。** 两套预设均使用四步去噪，每块三个 latent 帧，attention 窗口为 18 个 latent 帧，sink 为六帧。

| GPU | Generation FPS ↑ | Inference FPS ↑ | 首块延迟 ms ↓ | 出块间隔 P95 ms ↓ | RTF@fps24 ↓ | Peak allocated GiB ↓ | Peak reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA H200 | 36.12 | 35.95 | 358.19 | 351.20 | 0.6676 | 29.72 | 32.23 |
| NVIDIA RTX 5090 | 24.87 | 24.69 | 500.02 | 492.63 | 0.9722 | 28.38 | 28.95 |

- **Generation FPS** 从 AR 生成开始计时，到最后一个解码块在 GPU 上就绪为止。**Inference FPS** 还包含请求的输入准备和编码。两者均不包含模型加载、编译、预热、最终视频拼接和 MP4 保存。
- **首块延迟**从请求开始计时。**出块间隔 P95** 衡量相邻解码块之间的间隔，H200 数值合并三次请求的间隔计算。FPS 按总帧数除以总测量时间计算。
- **RTF@fps24** 为推理耗时除以 24 FPS 下的视频播放时长，小于 1 表示生成速度快于播放速度。
- **显存峰值**覆盖完整请求，包含最终视频拼接和保存。

## 当前 AR 缓存与验证范围

FA3 与标准 `sage_attn2` 共用 `RoPERingAttention`：raw K/V ring → 按原逻辑位置计算 RoPE 并写入复用的有序 K/V workspace → dense attention。EdenLive 不使用已删除的分段 Sage/LSE 缓存。它减少去噪各步重复准备历史的开销，仍保留历史拷贝与 workspace 显存；常驻层各自保留 workspace，KV offload 在各层间共享一组、加载后刷新。常驻、权重 block/KV offload、SP 和 TP 可同时开启 compile、CUDA Graph 与 warmup。当前仅预热配置的目标尺寸。

2026-10-21 的 FA3 SP2 + 权重 block/KV offload cryo-crossing 测试生成 1280×704、717 帧。完整 latent 与浮点视频 SHA 对齐相同模式的未修改 main；正式请求 **0 次 capture、9000 次逐层 replay**。完整 pipeline 从 132.736 降至 127.807 秒，DiT 从 91.046 降至 89.410 秒。每侧只有一次预热后的正式请求，属于回归测量，不是稳定吞吐统计。

另一个 85 帧 TP2 常驻测试的完整 latent/视频 SHA 也一致；其独立 main oracle 仅修正原 runner 的 TP cache-head 数量。原样 main 在此模式推理前失败，此前 Sage2 常驻 717 帧使用 275 次整栈 replay，属于另一后端/执行模式，不能代替全部 Sage2 多卡/offload 验收。本轮缓存改动没有 RTX 40/50 硬件验收，上方硬件预设表属于历史记录；本机 FA2 runtime 无缓存长序列也产生 NaN，仍是验证限制。安装和编译边界见[Sage 说明](../self_forcing/SAGE_ATTENTION.md)。
