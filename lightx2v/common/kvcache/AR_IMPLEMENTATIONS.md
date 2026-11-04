# 当前自回归视频模型的 KV cache 实现

本表沿用本轮对话的叫法：**pre-RoPE** 指 K 写入 cache 前已旋转；**post-RoPE** 指 cache 保存原始 K、读出后再旋转。
本表按 2026-10-26 的实际代码整理，覆盖当前讨论的流式视频模型，并补充几条仍有独立缓存实现的路径。这里直接写“写入前旋转 K”或“读出时旋转 K”，避免 pre/post-RoPE 叫法混淆。模型权重的 BF16、FP8-FP16、INT8 精度，与下表的 KV cache 存储精度是两件事。

## 主要模型对照

下表的“原始 K/V”指推理 dtype 的张量，常用 BF16；“量化 V”在 RTX 5090 上是 FP8。在 FlashAttention 路径里，K/V 继续采用原始推理精度。

| 模型 | 缓存组织、step cache | K 的旋转与刷新 | Sage 实际保存什么 | FlashAttention 如何读 | 主要入口 |
| --- | --- | --- | --- | --- | --- |
| **EdenLive AR** | sink＋recent 的 rolling ring；不同去噪 step 共用历史；每个输出 chunk 有 clean commit | ring 保存未旋转 K；在量化时按原模型局部坐标做 RoPE。每个新 chunk 第一次 forward 刷新历史量化 K，后续 forward 复用。当前 chunk K/V 每次去噪和 clean commit 都更新 | **原始 K＋INT8 K＋唯一量化 V**；没有第二份原始 BF16 V 历史。历史 V 跨 chunk 复用；SM120 的固定分段为2640 | 原始 K/V ring；全窗口：把 K 按物理 slot 对应的逻辑坐标旋转到 K workspace，V 直接物理 view，一次 FA；较窄且不连续的窗口另用可复用 KV workspace | [Eden infer](../../models/networks/wan/infer/edenlive/transformer_infer.py)、[Sage RoPE](attn/sage_rope.py)、[FA RoPE](attn/rope.py) |
| **Seko AR，4-step** | `StepRollingKVCachePool`；每个 diffusion step 各有历史；通常局部窗口21帧、sink2帧；没有 Eden 那种额外 clean commit | 缓存未旋转 K，读出时按局部窗口坐标旋转；每个 step bank 每次 forward 重建旋转后的量化 K，当前 K/V 每 forward 更新 | **原始 K＋唯一原生 V** 按层、step bank 持久保存；INT8 K/scale/mean 是同几何层与 bank 共用的临时工作区，每次 apply 在图内重建。无原始 BF16 V 历史；不保留4份量化 K | 与 Eden 共用物理 RoPE K workspace＋V view；不连续子窗口走可复用 KV workspace | [Audio AR infer](../../models/networks/wan/infer/audio/transformer_infer.py)、[480p 配置](../../../configs/seko_talk/ar/seko_talk_ar_fp8_sgl_480p_5090_tp2_sp4.json) |
| **Seko AR，2-step hybrid** | `HybridStepRollingKVCachePool`；每个 step 独立，层窗口模式 `W2-W2-W2-W2-W21`，同容量层分 bucket 分配 | 原始 K 按全局位置旋转，位置上限40帧；不改变原坐标。每 forward 重建量化 K；当前 K/V 每 forward 更新 | 与4-step共用临时量化 K 工作区策略；原始 K、原生 V 仍按层/step 独立，W2/W21 不同几何使用不同工作区，无原始 BF16 V 历史 | 同一物理 RoPE adapter，按该层实际窗口选择物理 spans；跨 spans 时使用 workspace | [Audio AR infer](../../models/networks/wan/infer/audio/transformer_infer.py)、[hybrid pool](rolling.py) |
| **Wan Self-Forcing / LingbotFast** | 共用普通 rolling ring；默认 step 间共享历史，可由通用存储策略选择 step bank；sink、recent 由配置决定 | **当前 K 在写入 ring 前已经做 causal RoPE**；普通历史 K 不随新 chunk 重新旋转。当前 chunk 在每次去噪和 clean commit 时更新 | **resident rolling Sage 只保存 INT8 K＋平台原生 V＋统计量/有效位**；原始 K/V 历史均不分配。只编码当前完整物理写入段，历史量化 K/V 跨 forward、跨 chunk 复用。启用 KV offload 时保留原始 CPU K/V＋GPU staging/量化 companion | 继续保留原始 K/V ring；全 ring 直接把物理 K/V view 送一次 FA，不需要先排成逻辑顺序；窄窗口多 spans 时使用 workspace | [Wan causal infer](../../models/networks/wan/infer/autoregressive/causal_transformer_infer.py)、[Sage ring](attn/sage.py)、[FA ring](attn/ring.py) |
| **Krea RealtimeVideo，`wan2.1_realtime`** | 每个输出 chunk 先重建3帧 clean context，再生成3帧；总容量6帧，step 间共享本 chunk 的 context。窗口移动后，context 首 latent 由最近9帧 decoded RGB 的最早一帧重新编码 | clean context 在局部坐标0–2重跑模型，当前 chunk 使用3–5；每个新 chunk 的 context **内容和 K/V 都重新计算**，不能把上一 chunk 的量化历史直接当成新 context | 使用相同 resident **仅编码 K/V** 存储；每 chunk reset 有效位并 refill，随后本 chunk 的多次去噪复用这3帧量化 context。没有额外 SF clean commit | 原始 K/V 的同一6帧缓存；reset/refill 与局部坐标策略相同 | [Krea runner](../../models/runners/wan/wan_realtime_video_runner.py)、[Krea infer](../../models/networks/wan/infer/realtime_video/transformer_infer.py)、[Krea scheduler](../../models/schedulers/wan/realtime_video/scheduler.py) |
| **Matrix-Game-v4** | rolling ring，模型要求去噪 step 共享历史；一个 sink latent frame、每 chunk 一个 latent frame | 普通 K 写入前完成动态 causal RoPE。另保留 **raw sink K**，由 `SinkRoPE` 按 fixed/absolute 模式处理；sink 坐标变化时仅重编码对应 sink，不刷新普通 recent K/V | resident Sage 的普通历史同样**只有 INT8 K＋平台原生 V**；仅 sink 保留小份原始 K 和旋转 scratch，sink V 仍只有编码副本。KV offload 保留原始 backing/staging | 与 Wan causal 共用原始物理 ring view；需要时才 gather 子窗口到 workspace | [Matrix infer](../../models/networks/wan/infer/matrix_game4/transformer_infer.py)、[SinkRoPE](../../models/networks/wan/infer/matrix_game4/rope.py) |
| **MiniMax-H3 Causal** | 原始 K/V 的 **per-step rolling**；condition、音频、视频写入长度不同；step 间各自维护历史 | 原始 K 在读出时做 partial split-half RoPE，默认旋转128维中的96维；每次 forward 在图内刷新共享的旋转/量化 K 工作区 | **原始 K/V＋量化 V** 按层、step bank 独立保存；INT8 K/scale/mean 由同几何的层与 bank 共用。使用专属 `H3SageRoPERingAttention(share_step_keys=True)`；保留原始 V 以支持不规则音视频写入 | `H3PhysicalRoPERingAttention`：同几何的层与 step bank 共用一个旋转 K 工作区，写入图内每次重算；原始 V 仍用物理 view，不连续子窗口按需 gather。GPU eager/compile/Graph oracle 与独立工作区逐位相同 | [H3 causal model](../../models/networks/minimax_h3/causal/model.py)、[H3 attention](../../models/networks/minimax_h3/infer/autoregressive/causal/attention.py)、[H3 infer](../../models/networks/minimax_h3/infer/autoregressive/causal/transformer_infer.py) |
| **TaoMate H3** | clean AV rolling 历史，step 间共享；首 chunk 后先保留完整 AV，再把首 chunk 的视频提升为 sink，保留两个最近完整 AV chunk；condition/current/padding 有独立可见性 | **clean K 写入前已经完成 QK norm 和 RoPE**，历史坐标不重排；仅在 clean commit 改变历史时刷新对应 history bank | 仍保留原始 clean K/V＋量化 companion；历史、condition、当前 media、padding 分成不同 bank。历史量化结果在多次 forward 间复用 | 对不同 query 可见性分别执行 FA；media query 的不连续 banks gather 到可复用 workspace 后一次 FA；不把 condition/padding 的可见性混在一起 | [clean cache](../../models/networks/minimax_h3/infer/autoregressive/taomate/clean_kv.py)、[TaoMate attention](../../models/networks/minimax_h3/infer/autoregressive/taomate/attention.py) |
| **StreamRender H3** | **每个 guidance branch、每个 timestep 独立 prefix bank**；固定 prefix＋每次覆盖的 current suffix；不是不断滚动整个历史的普通 ring 语义 | prefix K 写入前已旋转；固定 prefix 后续不变，current K 每 forward 更新 | **完整 prefix 只常驻编码 K/V**；层与 bank 共用原始 K/V scratch。每个 prefix 仅保留不足一整段的原始尾部，长度为 `prefix_length % segment_tokens`，供 current 改写边界段时重编码；完整编码历史跨窗口复用 | 独立原始 K/V buffer；prefix＋current 连续物理 view，一次 FA。首次 prefix capture 分开处理 prefix query 与 current query，以保持 prefix 看不到 current | [prefix bank](../../models/networks/minimax_h3/infer/autoregressive/streamrender/prefix_cache.py)、[StreamRender infer](../../models/networks/minimax_h3/infer/autoregressive/streamrender/transformer_infer.py)、[bank 生命周期](../../models/networks/minimax_h3/streamrender/model.py) |

Eden/Seko 的 V-only 路径要求物理段覆盖完整的 V 更新单位，以便历史 V 不必反量化后重新统计 scale。H3 Causal 的音视频、condition 长度使共同对齐单位可能只有1～2个 token；直接照搬会产生大量 attention 分段和 partial output，因此目前明确保留 H3 的原始 V adapter。H3 的当前实现是模型专属路径。Causal FA 原来按层/step 各分配旋转 K 工作区，在原生画布预热时使异步 VAE 拼接 OOM；现在按相同 cache 几何顺序复用，减少约5.79 GiB 的工作区预算。这个节省属于临时旋转工作区，不表示 per-step 原始 KV 历史也已共享。完整吞吐须看正式请求表，不能由小型 oracle 推算。

Wan resident Sage 的“仅编码 K/V”沿用同样的完整写入对齐约束，额外要求 K 的 RoPE 已经完成且在有效历史内不变。pool 只保留 slot、长度和 ring metadata，不分配原始 K/V 张量；当前投影产生的原始 K/V 是本次 forward 的临时输入。量化层仍保存 K centering 均值、K/V scale 和有效位，它们不等于另一套原始 K/V 历史。这个存储选择由实际 attention/backend 与 offload 策略自动决定，没有新增用户开关。

Krea 的新 runner 与旧的“用 `wan2.1_sf` 加载同一14B checkpoint”需要区分：旧 SF 路径保留全局 causal RoPE 和普通 clean commit，不会重编码 decoded pixels，因而不是官方 Krea context 策略。使用原完整 VAE 时，Krea 的 temporal-resample adapter 保留官方 tensor-cache 行为。可选 LightTAE decoder 时，仍由原 Wan2.1 encoder 重新编码反馈帧，但显示和反馈的像素均来自 LightTAE，因此更换 decoder 会影响后续 context 内容，需单列速度和精度。README 九个 latent chunk 的实际输出是6+8×12=102帧，配置中的105帧仅用于保留27个 latent 的请求几何。

## 公共继承与模型差异

`WanAutoregressiveModel` 在 `WanModel` 上复用分块预测及条件缓存生命周期；SF、LingbotFast、MG4 和 Krea 分别选择自己的 pre-infer、scheduler 及小型策略子类。执行侧的 `WanCausalTransformerInfer` 继承 `WanAutoregressiveTransformerInfer`，统一持有 `KVCacheManager`、attention backend、普通 block/KV-only commit、权重 block offload 和 KV offload；`WanSFTransformerInfer`、`WanLingbotFastTransformerInfer`、`WanMatrixGame4TransformerInfer` 只补模型相关的输入、camera 或 sink 处理。Krea 的 context reset/refill 仍由自己的策略控制。

runner 通过 `WanAutoregressiveRunnerMixin` 复用 warmup、分块去噪、模型允许的 clean commit 和流式 VAE 生命周期，实际异步工作交给同一个 `AsyncVAEChunkDecoder`。Eden 保留 proxy 和专属 scheduler，通过 `AutoregressiveRunnerMixin` 共享请求清理；没有为了统一继承而改变各模型的 RoPE、step bank 或 context 语义。存储布局和量化编码在 `common/kvcache` 内，模型不再各自复制普通 ring attention。

公共 [AutoregressiveRunnerMixin](../../models/runners/autoregressive_runner.py) 统一管理本地 MP4 流式写盘。Wan、固定音频 Seko、Eden 与 H3 runner 都将解码完成的 chunk 交给同一个 [StreamingVideoWriter](../../utils/streaming_video.py)：CPU 队列最多等待两个 chunk，后台 FFmpeg 持续写入临时 fragmented MP4，成功后原子发布最终文件；音频在结束时附加。只保存文件时不再拼接完整 GPU 视频，显式请求 tensor 返回仍保留原接口。MG4 在解码 chunk 上裁掉输入补齐产生的尾帧，文件帧率使用输入实际 FPS。Seko 实时音频输入保留 VAController 的可截断、回滚发布缓冲；它与追加写入的普通文件 sink 语义不同。H3 通过薄的 `H3AutoregressiveRunnerMixin` 追加模型状态清理；TaoMate/StreamRender 的实现直接位于各自 `runner.py`，旧 native runner 和注册别名已删除。解码异步与编码写盘是两个独立的生命周期。

Seko step bank 每个输出 chunk 各执行一次 forward，没有 Eden 同一共享 bank 内多次去噪复用量化 K 的机会。`SageRoPERingAttention(share_step_keys=True)` 因而只共享临时 INT8 K/scale/mean；`key_valid.zero_()` 位于每次 apply 的图内，保证其他层/bank覆盖工作区后 Graph replay 仍会重建本次 K。raw K、原生 V 与有效位没有共享，RoPE、窗口、分段及 LSE 数学不变。Eden 默认关闭该共享，继续复用同 chunk 历史量化 K。720p 的最终配置显式启用 raw-K CPU offload。

## Sage 的调用与 LSE

“分段”不等于每一段都发一次 Python/CUDA attention 调用。普通完整窗口把多个物理段作为 Sage 的 batch，通常一次 native attention 计算各段的 output/LSE，再用 merge kernel 合并，并校正分段 K-centering 的 LSE。尾部不足一整段时可能另有一次 native 调用；非常多段时还会按32位地址上限拆 batch。多个不连续量化 bank 也会分别计算再合并。因此应分别统计段数、native 调用次数和 LSE merge 次数。

Eden 当前标准5090 case 的2640-token段对应一个 latent chunk，整窗口是6段。这个值来自已完成的离线搜索；不是所有分辨率、chunk 或显卡通用的最优值。几何改变时会选择满足实际写入对齐约束的固定值。详见 [Sage policy](attn/sage_policy.py) 和 [Sage 执行/合并](attn/sage.py)。

## KV offload 与 CUDA Graph

`ar_config.kv_offload` 缺省为 `false`。启用时选择相应 offloaded rolling/step/hybrid pool：原始历史在 CPU，GPU 使用逐层 staging；slot metadata、物理写入 spans、传输事件决定正确的读写次序。Eden 的量化 companion 按层保留；Seko 原生 V 按层/step 留在 GPU，量化 K 使用同几何共享临时工作区。Eden/Seko V-only 情况下 CPU backing 和 GPU staging 都不再分配原始 V，仅保存/传输原始 K。Wan 的仅编码 K/V 路径适用于 resident rolling；启用 KV offload 时自动使用原始 K/V backing＋量化 companion。不能把“启用 KV offload”理解为量化 V 也一起挪到 CPU。

Wan SF/LingbotFast、TaoMate 部分对齐场景可以在量化历史有效时跳过原始历史传输，但这是各自 adapter 的资格检查，不是所有模型无条件启用。模型权重 offload 与 KV offload 独立。实现见 [offload](offload.py)、[rolling offload](rolling_offload.py)、[manager](manager.py)。

rolling 的 slot 写入不需要每次新分配/移动历史；captured graph 绑定该路径实际需要的稳定原始/编码缓存、频率表和 metadata。仅编码 K/V 路径直接绑定编码张量及其实际写入 slices，Krea reset/refill 保留存储地址。cache identity、几何或存储地址变化时应重新捕获。Graph 不要求把 ring 的逻辑 token 顺序物理整理成连续数组。

## 仍有独立缓存实现的模型

| 模型 | 当前代码行为 | 与本次 Eden 路径的关系 |
| --- | --- | --- |
| **Lingbot-VA** | 默认 `FIFOKVCachePool`，区分临时/predicted slots 与保留的 observation cache；K 写入前做 RoPE。读缓存时排序有效 slots 并用索引取出 K/V，再送模型 attention | 使用通用 manager/FIFO，但不是 rolling 物理段量化 adapter；没有继承 Eden V-only 优化。见 [Lingbot-VA infer](../../models/networks/wan/infer/lingbot_va/transformer_infer.py)、[FIFO](fifo.py) |
| **DreamZero** | 模型自有每层连续 K/V buffer，按 cache name 分支；满窗时保留最近历史并复制，attention 用 scratch K/V 拼接历史、当前 video、action。K 在入缓存前做 RoPE | 独立布局，没有接本次 `SageRoPERingAttention` 或增量量化 companion。见 [DreamZero infer](../../models/networks/wan/infer/dreamzero/transformer_infer.py) |
| **WorldPlay AR** | 模型自有 text KV＋vision KV；text 一次缓存，vision 用 `torch.cat` 累积；attention 再组合 text/history/current，ProPE 有自己的额外分支 | 尚未统一到物理 rolling slots；没有本次缓存量化复用。见 [WorldPlay AR infer](../../models/networks/worldplay/infer/ar_transformer_infer.py) |
| **FastWAM / RealtimeWAM** | video prefill 后每层保存 K/V，action 分支把 video K/V 与当前 action K/V `torch.cat`；RealtimeWAM 另有跨层 KV fusion | 这是 video-prefix 条件缓存，不是 Eden 的输出 chunk rolling 历史。见 [FastWAM](../../models/networks/wan/infer/fastwam/transformer_infer.py)、[RealtimeWAM](../../models/networks/wan/infer/realtimewam/transformer_infer.py) |

这些模型仍使用自身的 dtype 和 attention 后端；配置为 Sage 不代表自动具有“原始 K＋唯一量化 V”的存储方式。本轮没有对这些附录模型做完整权重速度或质量验证。

## 异步解码字段的位置

所有模型统一使用**顶层** `async_vae_decode`，与 `warmup`、`use_compile`、`use_cuda_graph` 同级：

```json
{
  "warmup": true,
  "use_compile": true,
  "use_cuda_graph": true,
  "async_vae_decode": true,
  "ar_config": {
    "kv_cache_scheme": "rolling",
    "kv_offload": false
  }
}
```

异步解码只读取顶层 `async_vae_decode`，缺省关闭。各流式 runner 通过同一个 [AsyncVAEChunkDecoder](../../utils/async_vae.py) 解析。异步提交使用独立 CUDA stream；解码器内部仍按 chunk 顺序维护状态。

## 验证边界

上表记录实际存储和更新语义。Wan resident Sage 的仅编码 K/V、Eden 的量化 V 复用、Seko 的逐 step 临时量化 K，以及 H3 的独立 bank 均已在各自模型路径验证。完整视频吞吐还受权重精度、并行拓扑、编译、Graph、输入编码和 VAE 解码影响，不能从缓存结构单独推断。

SM80/86 的原生量化 V 使用 FP16，SM89/90/120 使用 FP8。其它架构的 dispatch/转换检查不等于已在相应 GPU 上完成端到端硬件验收。Seko 2-step hybrid 的小尺寸随机权重验证也不等于完整模型的速度或画质验收。
