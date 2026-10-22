# Matrix-Game-4

原生 Wan2.2 5B Control 自回归视频重渲染，输入为低模 RGB 视频和独立 sink 图像。游戏引擎的键鼠控制、HUD 和音频在模型外处理。核心推理使用 LightX2V 权重、算子、scheduler 和 AR runner，不调用官方模型或 pipeline。

## 启动

本机模型目录为 `/data/nvme3/gushiqiao/models/Matrix-Game-4.0`，当前下载 Castle 场景。`base/` 中的 T5、tokenizer、Wan2.2 VAE 与官方文件 SHA256 一致，复用现有本地权重。目录布局对应官方 resource package：

```text
Matrix-Game-4.0/
  release-config.json
  base/{models_t5_umt5-xxl-enc-bf16.pth,Wan2.2_VAE.pth,taew2_2.pth}
  base/google/umt5-xxl/
  models/castle/{model.safetensors,inference.json,sink.png}
```

[run_matrix_game4.sh](run_matrix_game4.sh) 使用 `VIDEO_PATH` 指定 Castle 录屏；试用下文其他配置时，先导出该变量：

```bash
VIDEO_PATH=/path/to/castle-recording.mp4 bash scripts/matrix_game4/run_matrix_game4.sh
```

启动参数为 `--model_cls matrix_game4 --model-variant castle --task v2v`。`--video_path` 为游戏录屏，`--image_path` 为独立参考图，`--prompt` 使用对应场景提示词。输出保留输入 FPS 和实际帧数；末尾不足四帧时重复最后一帧编码，解码后裁掉补帧。

[quality.json](../../configs/matrix_game4/quality.json) 使用原版 Wan VAE、两步 `[1000,500]`、480×864、FA3，并同时开启 compile、CUDA Graph 和 warmup。Sage2 / FA2 可通过 `self_attn_1_type` 和 `cross_attn_1_type` 切换。Sage2 的 GPU 内核适配和安装要求见 [Sage Attention](../self_forcing/SAGE_ATTENTION.md)。FA3 需要兼容的 GPU 与安装环境。

[realtime.json](../../configs/matrix_game4/realtime.json) 使用官方 TAEHV 编码及解码、一步 `[1000]`；DiT 同样开启 compile、CUDA Graph 和 warmup，codec 保持已验证的 eager 运算次序。

Racing 的首输出就是参考帧，sink 保持绝对位置 RoPE，使用 [quality_racing.json](../../configs/matrix_game4/quality_racing.json) 和 `--model-variant racing`；对应权重与输入需另行准备。Castle/Egypt 则先单独预填参考 KV。当前完整视频验证使用 Castle，不把其他场景列为已实测。

## KV 与继承

```text
WanMatrixGame4Runner -> WanAutoregressiveRunnerMixin + Wan22DenseRunner
WanMatrixGame4Model -> WanAutoregressiveModel
WanMatrixGame4TransformerInfer -> WanCausalTransformerInfer
RollingKVCachePool + RingAttention / SageRingAttention
SinkRoPE -> raw sink key and fixed/absolute position policy
```

Castle/Egypt 的窗口共 21 个 latent 帧（1 sink + 20 recent），每 chunk 为 1 latent，对应首个 1 RGB、之后每组 4 RGB。Recent K 写入前完成绝对位置 RoPE，环形回绕后直接读物理 KV view。固定相对位置的 sink 额外保存 raw K，窗口移动时只重算 sink 的 RoPE 并原位更新；Sage2 同时刷新 sink 的量化缓存。KV 读取不拼接、不整理全窗口 workspace。

写入槽位和帧位置通过 GPU tensor 传入 kernel，填满窗口后不同槽位复用同一 CUDA Graph。缓存创建与请求重置由 Transformer Infer 持有，runner 只提交 latent shape。编译、graph、跨帧 cross-attention KV 缓存和 clean commit 复用 SF 的公共实现。MG4 保留官方 FP64 RoPE、scheduler RNG/运算次序和 VAE 归一化，避免改动影响 SF/EdenLive。

默认保持单 GPU、resident KV、无 CFG。现已接入 KV offload、权重 block offload、SP 和 TP，仍使用同一 rolling storage 与公共 causal 执行器；MG4 不使用逐步 KV bank。

- [realtime_offload.json](../../configs/matrix_game4/realtime_offload.json) 同时开启 KV 与权重 block offload，输入 VAE 与 T5 保持驻 GPU，确保与 resident 配置的 conditioning 一致。KV 只上传已提交的有效前缀；动态 sink 重锚后，当前写槽和 sink K 会持久化回 CPU。reference prefill、增长/满窗口和 clean commit 按层捕获不同 Graph variant，复用双权重 buffer；CPU/GPU 传输留在图外。
- [realtime_sp2.json](../../configs/matrix_game4/realtime_sp2.json) 使用两卡 SP：KV 保留全序列、仅分片 attention heads；RoPE 和 sink 仍按全局位置计算。padding 不进入缓存。
- [realtime_tp2.json](../../configs/matrix_game4/realtime_tp2.json) 使用实验性两卡 TP：column projection 继续切分，row projection 保留完整权重并聚齐输入后执行原始 dense GEMM；self KV 按 TP×SP 切 heads，cross KV 只按 TP 切分。默认 Torch RMSNorm 在全局 FP32 统计后保留 BF16 舍入边界。reference prefill 在编译 block 内保留原生 TP reduction 及归约后计算的舍入，denoise 继续使用原有 compiled 运算顺序。

以上配置保留 compile、CUDA Graph 和 warmup；常驻 SP/TP 复用整栈 Graph，叠加权重/KV staging 后使用逐层 Graph。40/50 系仍需用户实机验证。双卡并行并不保证比本机单卡 5B 更快；offload 主要交换显存与传输时间。

```bash
CONFIG_PATH=configs/matrix_game4/realtime_offload.json bash scripts/matrix_game4/run_matrix_game4.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 CONFIG_PATH=configs/matrix_game4/realtime_sp2.json bash scripts/matrix_game4/run_matrix_game4.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 CONFIG_PATH=configs/matrix_game4/realtime_tp2.json bash scripts/matrix_game4/run_matrix_game4.sh
```

Python 接口使用同一启动配置：

```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(model_path="/data/nvme3/gushiqiao/models/Matrix-Game-4.0",
                       model_cls="matrix_game4", task="v2v", model_variant="castle")
pipe.create_generator(config_json="configs/matrix_game4/quality.json")
pipe.generate(video_path="lowpoly.mp4", image_path="sink.png", prompt="...",
              seed=42, save_result_path="output.mp4")
```

## 首次原生接入的实测

环境为 H200、PyTorch 2.11.0+cu130、Castle BF16，输入 480×864、241 帧、24 FPS，扩散 seed 为 42。两种 LightX2V 预设都同时开启 compile、CUDA Graph、warmup；先排除一次完整请求，再平均三次正式请求。下表 DiT 耗时包含 denoise 与 clean KV commit，不含参考帧预填。

| 后端 / 预设 | 对照 DiT | LightX2V DiT | DiT 耗时变化 | 视频 PSNR |
| --- | ---: | ---: | ---: | ---: |
| FA3 quality | 8.982 秒 | 7.865 秒 | 减少 12.43% | 19.75 dB |
| FA3 realtime | 6.003 秒 | 2.648 秒 | 减少 55.88% | 18.13 dB |
| Sage2 quality | 8.168 秒 | 8.944 秒 | 增加 9.50% | 16.70 dB |
| Sage2 realtime | 5.914 秒 | 3.386 秒 | 减少 42.75% | 18.18 dB |

FA3 对照为官方 native FA3 路径；Sage2 对照在同一官方模型中将 self/cross attention 直接接 dense Sage2，使用 H200 BF16，不代表官方 RTX5090 FP8 优化预设。Sage2 quality 在当前形状和异步 Wan VAE 配置下更慢，因此默认保持 FA3。

两个 FA3 预设的 control、文本和参考 latent 完全一致，前 20 个输出 latent chunk 也完全一致。环形回绕后物理顺序改变 attention 浮点归约次序，后段爆炸、烟雾和光照出现差异，整段视频并非逐 bit 一致。MG4 默认的 `torch_opaque` RMSNorm 在 compile 中保留原生均方归约，将 FA3 quality 的视频 PSNR 从 18.30 提升至 19.75 dB；原有 RMSNorm 后端及 SF/Eden 默认配置保持不变。

FA3 quality / realtime 的本框架完整请求耗时分别为 14.06 / 7.36 秒，包含视频读取、control/sink 编码、T5、DiT、输出解码。官方 push 耗时从预处理和文本/参考编码完成后开始，两者不能作为同范围端到端加速比。异步 VAE 与 DiT 的 CUDA event 有重叠，分阶段耗时不可直接求和。

每组四次请求的 latent SHA 一致，warmup 后无新增 graph capture；最终 135 项集成及回归测试通过。完整视频仅在 H200 实测，FA2 完成缓存/内核回归，尚未完成 MG4 整段视频验证。当前 resident FA3 流程 quality / realtime 峰值 allocated 显存约 33.69 / 26.83 GB，未验证 24 GB 显卡。迁移本机模型目录时，需要同时准备 T5/VAE 软链所指向的组件。



## 历史 offload 与并行验证

此前 offload 接入沿用 Castle BF16、480×864、241 帧 case，比较该阶段修改前后的原生 LightX2V 实现。resident、SP2 和 KV+权重双 offload 的完整 latent 与解码视频逐字节一致；该阶段 TP2 与单卡的跨模式比较中，完整视频 PSNR 为 17.49 dB、MP4 SSIM 为 0.7508，后段烟雾、火光仍有时序漂移。这是历史跨模式结果，不代表下文 prefill RMS 修正后的同模式 TP2 精度。

同一 H200、同一 CPU affinity，各排除一次完整请求、平均三次正式请求：resident FA3 的 DiT 从 2.632 秒变为 2.622 秒，完整请求从 7.832 秒变为 7.643 秒。变化较小，主要结论是保持原来的速度。MG4 的短 chunk 在调用线程发射 VAE kernel，VAE 仍在独立 CUDA stream 并发解码，避免后台 VAE 线程与逐层编译调用争用 Python 调度；其他模型保留原有策略。

SP2 的单次正式 DiT 为 3.314 秒；实验性 TP2 为 3.424 秒、22.14 GB；双 offload 为 29.484 秒，显存从 resident 的 26.83 GB 降至 14.57 GB。这两项提供多卡及低显存运行能力，在该 5B case 上不比单卡 resident 快。resident/SP2 分别复用 4 个捕获，双 offload 复用 60 个分层捕获；正式请求均无新增 capture。FP64 RoPE、动态 sink、奇数 405 token 的 SP padding 与缓存重置均有回归覆盖。


## 当前同模式验证（2026-10-21）

上游 LightX2V main 尚无 MG4，本轮回归基准是此前冻结的 MG4 实现。TP2 与旧实现的 **TP2** 对比，不能用单卡输出代替同模式基准。按阶段选择的 RMS hook 在编译 block 内保留原生 reference-prefill 舍入，denoise 运算顺序保持原样；整栈及逐层 staging 的 Graph variant 都包含 reference-prefill 标记。

Castle realtime、FA3、480×864、241 帧、TP2 常驻：两 rank 的两次请求完整 latent SHA 一致；两 rank 第 0 次请求的完整浮点视频 SHA 一致。compile、Graph、warmup 全开，每请求 **0 次新 capture、184 次 replay**。每侧一次正式 rank-0 测量的 DiT 为 3.581 → 2.021 秒、pipeline 为 8.422 → 6.511 秒；属于回归测量，不是稳定吞吐统计。

SP2 + 权重 block/KV 双 offload 也对齐旧实现相同模式的完整 latent/视频；稳态耗时基本持平，差异在 0.5% 内，冷启动预热更长。最终 TP2 + 权重 block/KV 双 offload 测试中，两 rank 的两次请求完整 latent 与浮点视频 SHA 也对齐同一个正确 TP 基准；每请求 0 次新 capture/compile、3690 次逐层 replay。每 rank allocated 峰值约 6.80 GB，已核验源权重在 CPU、矩阵 buffer 已 pin。这是功能/显存验证，不与常驻 TP 作速度比较，输入 VAE 权重仍需常驻。本轮未验证 FA2/Sage2 整段 TP 视频精度、其他 TP RMS 后端、混合 SP×TP 或 RTX 40/50 硬件。
