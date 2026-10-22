
# Self-Forcing (Wan2.1-T2V-1.3B)

本文档介绍如何在 LightX2V 中运行 `Wan2.1-T2V-1.3B` 的 Self-Forcing 加速。

## 准备环境

请参考[环境安装](../../docs/ZH_CN/source/getting_started/quickstart.md)

## 下载

### 1. BF16 原始模型 (Wan2.1-T2V-1.3B)

从 Wan-AI 的 Hugging Face 仓库下载 BF16 原始模型：

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```

### 2. Self-Forcing checkpoint

下载 Self-Forcing 的 checkpoint（`self_forcing_dmd.pt`）：

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### 3. 量化模型 (FP8 / NVFP4)

从 LightX2V 的 HuggingFace Collection 下载量化模型：

https://huggingface.co/collections/lightx2v/ar-lightx2v

## 开始运行

我们提供三个脚本：

1. `scripts/self_forcing/run_wan_t2v_sf.sh`：BF16 原始模型 + Self-Forcing checkpoint
2. `scripts/self_forcing/run_wan_t2v_sf_fp8.sh`：FP8 量化模型
3. `scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh`：NVFP4 量化模型

运行下面任意脚本之前，你需要先把脚本中的路径变量填好。

### 1) BF16 + Self-Forcing checkpoint

修改并运行：

```
bash scripts/self_forcing/run_wan_t2v_sf.sh
```

`run_wan_t2v_sf.sh` 中关键字段：

- `lightx2v_path`：你的 LightX2V 仓库路径
- `model_path`：原始 BF16 `Wan2.1-T2V-1.3B` 模型目录路径
- 配置文件中的 `dit_original_ckpt`：`self_forcing_dmd.pt` 的路径（从 `gdhe17/Self-Forcing` 下载）

该脚本实际会执行：

- `python -m lightx2v.infer`
- `--model_cls wan2.1_sf`：使用 Wan2.1 的 Self-Forcing pipeline
- `--task t2v`：文生视频
- `--config_json configs/self_forcing/wan_t2v_sf.json`：Self-Forcing 的运行配置

注意：`configs/self_forcing/wan_t2v_sf.json` 中默认 `enable_cfg=false`。

`ar_config.kv_cache_scheme` 默认为 `"rolling"`，表示固定 sink 加 recent ring 的缓存策略。原先显式配置 `"fp"` 的地方改为 `"rolling"`；该字段不决定模型或缓存张量的 dtype。

### 2) FP8 量化模型

修改并运行：

```
bash scripts/self_forcing/run_wan_t2v_sf_fp8.sh
```

关于 FP8 的额外说明：

- 该脚本使用 `configs/self_forcing/wan_t2v_sf_fp8.json`。
- 你需要把该配置文件中的 `dit_quantized_ckpt` 修改为从 HuggingFace collection 下载的 FP8 权重文件的**绝对路径**。
- `dit_quant_scheme` 为 `fp8-vllm`。

### 3) NVFP4 量化模型

修改并运行：

```
bash scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh
```

关于 NVFP4 的额外说明：

- 该脚本使用 `configs/self_forcing/wan_t2v_sf_nvfp4.json`。
- 你需要把该配置文件中的 `dit_quantized_ckpt` 修改为从 HuggingFace collection 下载的 NVFP4 权重文件的**绝对路径**。
- `dit_quant_scheme` 为 `nvfp4`。

### 4) FA2/FA3 物理 ring 与 Sage2 分段缓存

使用 `configs/self_forcing/wan_t2v_sf_ring_fa3.json`，填入 `dit_original_ckpt` 后传给 `--config_json`。配置使用 FA3 self/cross attention、12 帧 local window、3 帧 sink、每 chunk 3 帧；这里均指 **latent 帧**。

持久 ring 保存已经完成 RoPE 的 K，GPU slot kernel 写入当前 K/V。attention 覆盖有效物理 ring 时，dense FA2/FA3 直接读取 view，无有序 workspace、历史 gather 或 KV 拼接。K/V 同步换序且可见集合不变时，数学结果等价，浮点归约顺序仍可能改变，需要用视频和数值比较验收。窄于存储区间的窗口把有效 spans gather 到可复用 K/V buffer，不能称零 copy。两种路径都仍有当前 KV 写入和 metadata 上传。

`use_cuda_graph=true` 为首 chunk、增长/满窗口、reference prefill 和 clean commit 捕获不同 variant。slot 和 RoPE 首帧保存在固定 GPU 地址，兼容 variant 可跨绕环和请求重放。常驻计算捕获整栈；权重 block offload 或 KV staging 在数据到达后捕获单层，均可同时开启 `use_compile=true`。SP/TP、step bank、image cross-attention 和窄窗口不再切到另一条未优化 CUDA executor。

验收必须区分后端和配置：本轮 H200 多卡/offload 真实模型记录使用 FA3；第 16 节 SF Sage2 视频来自此前常驻专项。Sage2 缓存/Graph 合成测试不能代替全部模型的 SP/TP/offload 视频与性能验收。本机安装的 FA2 runtime 在无缓存长序列基线也产生 NaN，因此本轮 Flash 多卡结果使用 FA3。RTX 40/50 尚无硬件实测。当前范围见第 19 节；第 9–16 节保留对应日期的历史实验。

`WanSFModel` 继承 `WanAutoregressiveModel`。SF、LingbotFast、MG4 和 Audio AR 的 infer 共用 `WanCausalTransformerInfer`，后者继承 `WanAutoregressiveTransformerInfer`；SF 只保留输入投影 hook，LingbotFast 保留相机注入。公共代码负责 ring 规划、cross-attention KV 生命周期与 graph 执行；模型保留位置编码与 clean commit 时机。slot 写入统一在 [`common/kvcache/kernel.py`](../../lightx2v/common/kvcache/kernel.py)；[`RollingKVOffload`](../../lightx2v/common/kvcache/offload.py) 单独负责 CPU pinned buffer、staging、prefetch 和 writeback。

完整 DiT benchmark 的执行模式是 `--mode eager` 或 `--mode graph`：

```bash
PYTHONPATH=. python scripts/self_forcing/benchmark_ring_cache.py \
    --model-path /path/to/Wan2.1-T2V-1.3B \
    --checkpoint /path/to/self_forcing_dmd.pt \
    --mode graph --compile --warmup \
    --reference-latents /path/to/old/latents.pt \
    --output-dir /path/to/benchmark/graph
```

默认 480×832、12 chunks、3 次重复，每 chunk 计入 4 步去噪与一次 clean rerun。参考 latent 用于比较首次完整 rollout 的数值误差，再开始正式重复；物理换序不要求逐 bit 一致。启动、warmup、正式 DiT 与可选解码分别记录。

### 5) 启动 warmup

在启动 JSON 中设置 `"warmup": true`。Self-Forcing 的 warmup 在 runner 初始化结束后、首个正式请求之前执行一次；不通过 `generate()` 或 `lightx2v.infer --warmup` 开启。

warmup 复用真实文本编码、全部去噪步骤、clean context rerun 和带缓存的 VAE decode。SF 与 LingbotFast 现在只预热配置的目标尺寸。12 帧窗口下执行 5 chunks，覆盖首 chunk、窗口增长、满窗和绕环，以便正式请求复用已预热的 KV allocation 与 Graph 绑定。

结束后重置 KV 有效长度和 ring 位置，清除 scheduler、latent、输入和 generator，恢复 CPU/CUDA RNG；兼容的缓存分配、compiled callable 和 Graph 继续复用。权重 block offload、KV offload 与 SP/TP 使用同一生命周期，具体后端/模型的实测范围单独记录。Seko AR 的 `lazy_load` warmup 仍未支持，MG4 流式输入编码仍要求 VAE 常驻。整模型卸载重载会改变 GPU 权重地址，因此需要重新捕获 Graph。

benchmark 的 `--warmup` 可测试该生命周期；`warmup_seconds` 单独记录启动预热，`first_request_seconds` 与重复请求只计正式 DiT。该选项属于 benchmark 脚本，与产品入口的启动 JSON 分开。

### 6) torch.compile 与 CUDA Graph 同时使用

`configs/self_forcing/wan_t2v_sf_ring_fa3_compile.json` 启用 `warmup`、`use_compile` 和 `use_cuda_graph`。关闭 `use_cuda_graph` 可单独 compile，无需额外的 cache 顺序开关。

编译边界是单个 transformer block。SF 常驻执行按层缓存 callable，staging 执行绑定复用的 GPU block buffer。ring 规划、metadata 上传和 CPU/GPU 传输在编译计算外；FA2/FA3 使用物理 view 或可复用窄窗口 buffer，Sage2 在完整可见 ring 路径绑定持久量化段。公共 [`AutoregressiveCUDAGraphStack`](../../lightx2v/common/transformer_infer/autoregressive_cuda_graph.py) 为增长/满窗口和 prefill 保存独立 variant，支持整栈和逐层 staging 两种执行。

Wan causal 基类为默认 Inductor 启用 `emulate_precision_casts=true` 保留 BF16 中间舍入。动态 RoPE、slot 写入和原生 LayerNorm 保留算子边界。手动图设置 `triton.cudagraphs=false`，避免 Inductor 同时管理图。生产 block API 不强制 `fullgraph=True`；是否实际编译和 replay，须以编译及 capture/replay 计数确认。下方真实权重记录覆盖各自配置的正式无重编译、无 recapture 验收。

SF 的原始 KV 只由 [`RollingKVCachePool`](../../lightx2v/common/kvcache/rolling.py) 保存；`prepare_attention` 提供稳定的物理 view 和 slot 参数。`RingAttention` 负责写槽并调用 FA2/FA3，`SageRingAttention` 在同一存储上维护分段量化状态。EdenLive 的后 RoPE 有序工作区独立放在 [`RoPEKVWorkspace`](../../lightx2v/common/kvcache/attn/rope.py)，不再与 SF ring 后端混合继承。

### 7) 自回归 KV 方法分类与当前范围

能否复用同一优化，取决于缓存生命周期和位置编码：

| 分类 / 模型 | 当前实现与优化范围 |
| --- | --- |
| K 已做 RoPE：Self-Forcing / realtime-video 14B / LingbotFast | 持久 sink/ring 存储；slot kernel 写当前 KV，FA2/FA3 直接读取物理 view，Sage2 使用持久量化段和 LSE 合并；完整可见 ring 无历史 gather、cat 或有序 workspace；窄窗口使用可复用 gather buffer。换序与量化误差须用实际视频/数值验收。 |
| raw K、按窗口本地位置重锚：EdenLive | ring 保存 raw K/V；每 chunk 一次融合历史 gather、RoPE 和有序 workspace 写入，去噪各步复用历史、仅更新当前尾部。常驻时每层保留历史 workspace；KV offload 时各层共享 workspace，加载新层后重建历史。 |
| raw K：Wan Audio / Seko AR | step/hybrid rolling pool 保存 raw K，`PhysicalRoPERingAttention` 把 K 旋转到物理顺序 workspace，V 直接读 ring view。2 步的 global/capped 与 4 步的 local 位置策略分别保留，包括 SP 语义。 |
| raw K：MiniMax H3 causal | 保留 step/layer bank 与全历史 RoPE，以及 condition prefix、首 chunk sink 和视频/音频 joint attention。 |
| FIFO：LingbotVA | 保留原有效 token gather 和 dense attention。 |
| 连续历史与临时 action token：DreamZero | 保留原历史搬移和 KV 拼接；仅提交视频历史。 |
| 每 chunk 重建 memory window：WorldPlay AR / AR Distill | 保留原 text prefill、context 重建与 `[text, context, current]` 拼接，含 normal/ProPE 两路。 |

SF 的 FA2/FA3 与 EdenLive 使用 dense attention；完整 ring 的 SF Sage2 批量计算分段后合并 LSE。EdenLive 保留逻辑窗口顺序，Audio/Seko 使用物理顺序 K workspace；H3、LingbotVA、DreamZero 和 WorldPlay 继续保留各自位置策略。共享 compile/Graph executor 不代表这些缓存布局可以直接互换。真实精度与性能结论以各报告明确记录的模型和配置为准。

### 8) 普通 cross-attention K/V 缓存（P1）

`ar_config.use_cross_attn_kv_cache` 默认为 `true`，设为 `false` 可做不缓存的对照。缓存保存普通投影 K/V（包含该路径需要的 K norm），独立于 Sage 的量化或 prepared-KV。公共 [`ConditionKVCache`](../../lightx2v/common/kvcache/condition.py) 按 source、CFG/命名分支、模态、层和 variant 隔离；source 张量的存储范围、版本、dtype 或 device 变化会使旧值失效，权重/LoRA 更新须通过模型的 cache reset 生命周期清除旧值。不能跟踪原地修改版本的 inference tensor 不复用缓存。

这项可选缓存用于 Self-Forcing/realtime-video、LingbotFast、Wan Audio AR 和 EdenLive。动态 audio adapter 条件仍逐次计算。H3 使用 joint-attention condition prefix，没有单独的 cross-attention KV；LingbotVA、DreamZero 和 WorldPlay 恢复各自原有的条件/文本缓存行为。

保留的接入路径中，`AutoregressiveModelMixin` 负责权重变化时的失效，`AutoregressiveTransformerInferMixin` 管理条件 KV 生命周期。Self-Forcing 实现 `WanAutoregressiveModel`，其 infer 与 LingbotFast 共用 `WanCausalTransformerInfer`；Audio 和 EdenLive 保留各自的 attention 与调度计算。

无权重文件评测脚本为 [`benchmark_cross_kv_cache.py`](benchmark_cross_kv_cache.py)，默认复现 SF 形状：

```bash
PYTHONPATH=. python scripts/self_forcing/benchmark_cross_kv_cache.py \
    --output-dir /path/to/benchmark/cross-kv
```

2026-10-19，H200 / BF16 / FA3，seed=42，随机权重与输入，调用实际 MM、RMSNorm、FA3、`ConditionKVCache` 和 Wan KV helper。每请求 12 chunks，每 chunk 4 步去噪与 clean rerun；预热 3 请求后测 3 轮 × 30 请求，首次生成 KV 的成本计入。以下为**单层 Q/K/V/O projection + Q/K RMSNorm + FA3 子路径**的 CUDA event 中位耗时，不是完整 DiT。这里只保留 Wan 代表形状的算子 microbenchmark，不代表其他架构已接入，也不代表整模收益。

| 合成代表形状 | 完全不缓存 | 首 chunk 每步投影、之后复用 | 每条件/variant 一次 | 比完全不缓存减少 |
| --- | ---: | ---: | ---: | ---: |
| Wan SF 1536 | 13.911 ms | 10.212 ms | 10.399 ms | 25.25% |
| Wan 3072（Eden5B 形状） | 14.636 ms | 11.838 ms | 11.648 ms | 20.41% |
| Wan 5120 text | 57.078 ms | 52.042 ms | 51.684 ms | 9.45% |
| Wan 5120 image，双 CFG | 107.167 ms | 100.201 ms | 100.264 ms | 6.44% |

已测输出全部与不缓存路径逐 bit 一致。单分支静态条件每层 K/V 投影对数由 60 降至 1。SF 等原先已缓存后续 chunk 的路径，新增收益不显著，部分 case 略慢；不能把相对完全不缓存的减少幅度当作它们的新增收益，也不能外推整 DiT。表中“首 chunk”列是策略 reference，未重放旧完整模型。

普通 KV 容量为 `2 × context_tokens × heads × head_dim × element_bytes × layers × branches × variants`，不含权重、self KV 和临时 tensor。SF 原有 text KV 已占用的空间不能算作新增开销。


第 9–14 节保留物理直读改动之前的历史验收；其中逐字节一致和有序 scratch 的描述对应当时版本。当前物理直读的精度与性能见第 15 节。

### 9) 精确验收：Self-Forcing 1.3B

2026-10-19，真正改前X2V源码与最终logical-order三开路径在同一张 **H200 GPU1** 上顺序运行。真实30层SF DMD、BF16/FA3、相同prompt/seed42、12/3/3 latent帧窗口/sink/chunk，生成832×480、165帧、16fps、10.3125秒视频。每组先完整执行一次请求，再测3次重复；旧runner没有启动warmup，因此该首次请求用于预热其正式signature。

| 实现 | 14个chunk请求中位 | 满窗chunk中位 |
| --- | ---: | ---: |
| 真正改前X2V HEAD源码 | 3.84615 s | 290.417 ms |
| 最终logical-order compile + warmup + CUDA Graph | 3.69740 s | 278.172 ms |

预热后的正式DiT请求下降 **3.87%**，满窗chunk下降 **4.22%**。完整42个latent帧与旧源码 **逐字节一致**，max abs/RMSE均为0，同seed重复也一致。保留30个compiled callable和累计2次capture，每正式请求新增55次replay，无额外编译或recapture。Warmup为9.499 s，启动与首次请求另计；该表不含文本、VAE和MP4编码，不代表一次性冷启动更快。早期GPU7异常时延不参与本次同卡性能结论。


### 10) 精确验收：Krea Realtime 14B

给定`/data/nvme4/models/realtime-video/krea-realtime-video-14b.safetensors`确认为 **40层、hidden5120、40×128 heads、142.88亿参数的T2V模型**，不是1.3B。现有`model.`前缀权重可直接加载；仅复用stock Wan的tokenizer/T5/VAE。旧源快照1775个tracked源码/config文件全部匹配HEAD `d48a6173bc15a88b06cdcfb211183b43d8038846`。

两组均在同一张空闲 **H200 GPU5** 顺序运行，BF16/FA3、相同prompt/seed42、832×480、14chunks、12/3/3 latent帧cache。这里控制相同attention backend和可见集合，不与原示例不同的Sage2/无限窗默认参数混比。

| 实现 | 14个chunk请求中位 | 满窗chunk中位 |
| --- | ---: | ---: |
| 真正改前X2V，已核对HEAD快照 | 21.84255 s | 1626.146 ms |
| 最终logical-order compile + warmup + CUDA Graph | 21.33125 s | 1588.597 ms |

预热后的正式DiT耗时分别下降 **2.34% / 2.31%**，新3轮均快于旧3轮中最快的一轮。完整latent对HEAD逐字节一致，max abs/RMSE为0，完整MP4 SHA256也相同。实际40个compiled callable、4个Dynamo graph、累计2次capture，每正式请求55次replay，无正式重编译。仅一对365.625 MiB logical KV scratch跨层复用；**仍有历史拷贝，不是零搬运**。

加载含warmup为47.041 s，其中warmup21.566 s；首次请求21.223 s。正式allocated/reserved峰值为52.954/58.570 GiB。最终VAE解码3.543 s，保存完整视频与预览两份MP4另耗2.001 s，均未计入表中；该稳态验收不包含启动成本，也不宣称一次性冷启动不退化。


以上为 2026-10-19 真实权重验收记录；后续公共 cache 接入由下方单独的 SF 回归验证。


### 11) 精确验收：Lingbot World Fast cam

2026-10-19，使用真实 16 分片 `lingbot-world-base-cam/lingbot_world_fast` 权重、40 层、BF16 常驻、FA3、480×832、14 chunks / 165 帧。输入为官方 example 03 的照片、相机 poses 和 intrinsics，seed=42，local window=21、sink=3。compile、生产 warmup 与 CUDA Graph 同时实际运行。

70 次 noisy input、70 次 noise prediction、完整 latent 和全部视频帧 tensor 均与原 HEAD `d48a6173` 逐 bit 一致，三次重复请求的完整 latent 也一致。同一张 H200 GPU6 的稳态 pipeline **41.500 → 35.844 秒，降低 13.63%**；GPU DiT **30.798 → 29.898 秒，降低 2.92%**。正式每请求增加 40 次 graph replay，保留两次 warmup capture，编译计数不变。pipeline 收益还包括将每 chunk 无条件清理 allocator 改为现有压力感知策略，不能全部归因于 KV kernel。

两侧使用同一个外部测试适配器固定 VAE 输入为 480×832；普通 CLI 仍有 resize 舍入可能得到 464 高度的既有问题，本验收不代表已修复。base-act MoE 权重没有作为 fast AR 模型验收。

LingbotFast 现在用同一套整数对齐规则计算 VAE 编码和预热的 latent 网格。普通 480×832 请求得到 60×104 latent，并复用预热 Graph；上面的历史验收仍是当时使用外部适配器的结果。

2026-10-22 在 H200 GPU1 使用官方 example 03 生成 165 帧：修复后首请求与热请求的 DiT 分别为 28.465／28.545 秒，40 个编译块、21 个保留的 Graph capture、每次请求 84 次 replay，正式请求无新捕获。相同精确尺寸 VAE 输入的原始 main `d48a6173` 为 31.768／29.944 秒，热请求 DiT 快 4.7%。前 7 个 chunk 的 latent 逐 bit 相同；窗口滚动后，最终视频的 YUV PSNR 为 31.37 dB、SSIM 为 0.9565。环形缓存 oracle 验证跨环绕的逻辑 K/V 完全一致；物理顺序 FA3 的归约次序会产生小幅单层舍入差异。

### 12) 公共 cache 插件回归

2026-10-20，SF1.3B 真实权重在 H200 GPU6 对照冻结的插件接入前源码与公共 `AttentionKVCache` 接入，480×832、14 chunks、local window=12 / sink=3，compile、warmup、CUDA Graph 三开。两组均先跑首请求，再测三次正式重复；每个完整 latent 均与已保存的原始 main 参考逐字节一致。DiT 均值 **3.719586 → 3.719567 秒**，按持平解释，不宣称接口统一带来新增收益。正式每请求 55 次 graph replay，无新编译或 capture。此回归未解码视频。该轮未重新跑 realtime14B 和 LingbotFast，它们的整模证据为上方较早验收。

### 13) Clean commit 的末层 K/V-only

SF1.3B、realtime-video14B 和 LingbotFast 共用此优化，对支持的缓存更新自动启用，不再提供配置开关。只在 scheduler 的 `is_rerun=true` 缓存更新调用中生效，正常去噪保持完整计算。

前 N−1 层必须完整执行，以得到末层正确输入。末层保留原 norm/modulation、K/V 投影、K norm/RoPE 和 ring 写入；跳过 Q、自注意力、cross-attention、FFN、输出 head/post，以及不会被消费的 clean noise prediction 写入。普通 SF 的 `before_proj` 保留；Lingbot 前面各层的 camera 注入保持原样。

常驻 denoise 使用完整 block-stack 图；clean 使用独立的前 N−1 层 variant 和单独编译的末层 K/V-only。权重/KV staging 通过逐层 Graph 执行相同策略。仅启用 Graph 时也传入 GPU RoPE offset。reference prefill 和需要末层 post-adapter 输出的模型保留完整计算，但仍进入公共优化 executor；Audio AR 因此设置 `cache_commit_enabled=False`。并行、offload 与多 step bank 不会关闭 compile 或 Graph。

2026-10-20，H200 / BF16 / FA3，480×832、14 chunks，各自同卡改前与改后均三次正式请求，compile、生产 warmup、CUDA Graph 实际启用：

| 模型 | 全请求 DiT 均值 ms | clean 合计均值 ms | DiT 耗时减少 | clean 耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| SF1.3B，local12/sink3 | 3681.887 → 3658.085 | 735.914 → 711.734 | 0.65% | 3.29% |
| realtime-video14B，local12/sink3 | 21225.324 → 21132.581 | 4244.832 → 4149.106 | 0.44% | 2.26% |
| LingbotFast cam，local21/sink3 | 29655.453 → 29527.478 | 5903.611 → 5773.922 | 0.43% | 2.20% |

三模型逐 chunk 的完整末层有效 K/V 与 ring/cursor 元数据逐字节一致，所有首请求及三次正式请求的完整 latent 对齐原始 main。Lingbot 的完整视频、70次输入、56次正常去噪预测也全部对齐；clean 输出已不再消费，因此不比较其 prediction。Lingbot 两侧仍使用外部尺寸适配器固定480高，base-act未验收。本轮 SF/14B 未重复解码视频。

正式请求无新 compile/capture；SF/14B 每请求55次 replay，Lingbot40次。独立 clean 图会增加预热和显存成本，不能将这项优化视为冷启动提速。

### 14) 公共执行接口

`WanCausalTransformerInfer` 继承 `WanAutoregressiveTransformerInfer`；`EdenLiveARTransformerInfer` 组合 `EdenLiveTransformerInfer` 与该 causal 基类。它们共用条件 KV 缓存、KV-only 编译分发和 CUDA Graph 生命周期。causal 基类负责 rolling manager/backend、缓存初始化与重置、Wan block/cross-attention/FFN/head 计算和权重/KV offload。SF、LingbotFast、MG4、Audio AR 在此基础上实现差异；EdenLive 保留独立的 RoPE/workspace 计算。普通 block 使用 `infer_block` / `run_block`，末层缓存计算使用 `infer_kv_block` / `run_kv_block`，提交统一命名为 `update_self_attn_cache`；模型继续实现各自的 RoPE 与写入时机。`can_use_kv_only` 选择符合模型语义的 clean shortcut；Graph 不再要求常驻权重、满窗或单卡执行。

SF/LingbotFast runner 通过 `WanAutoregressiveRunnerMixin` 共用分块去噪、clean commit、流式解码和 warmup；EdenLive 保留 proxy 与专用 scheduler 流程，通过 `AutoregressiveRunnerMixin` 共用预热尺寸排序和请求清理。请求结束保留已编译 block、Graph 与可复用 buffer，清除条件缓存；更换权重或存储时调用 `reset_cuda_graph`。`use_compile`、`use_cuda_graph`、`warmup` 仍控制执行模式；可精确复用的缓存和 clean commit 优化自动启用。

2026-10-20 的公共接口整理通过 201 项去重回归，并重跑 SF1.3B、realtime14B、LingbotFast 和 EdenLive 真实三开。完整输出逐字节对齐，SF/Lingbot 吞吐持平。

### 15) 2026-10-20 物理 KV 直读验收

原生 FA3 直接读取 post-RoPE ring 的有效物理 view；删除 SF 有序 workspace 与历史 gather，没有新增顺序开关。EdenLive 路径未改。该性质并非 FA3 独有，但该阶段只接入和验证原生 FA3；窄窗口等原 fallback 不属于本次零 cat 范围。

H200、BF16、832×480、14 chunks、每 chunk 4 步加 clean，实际开启 compile、warmup 和 CUDA Graph。每侧首次精度请求后测三次正式请求，下表为 CUDA event 的 DiT GPU 总耗时（含 clean，不含 VAE）。SF/14B local=12、sink=3，Lingbot local=21、sink=3；完整视频165帧、16fps。

| 模型 | 改前 | 改后 | 耗时减少 | 峰值 allocated 减少 | 视频 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| SF 1.3B | 3.656 s | 3.565 s | 2.51% | 109.69 MiB | 28.27 dB |
| Realtime14B | 21.186 s | 20.716 s | 2.22% | 365.62 MiB | 24.65 dB |
| LingbotFast cam/fast | 29.491 s | 28.908 s | 1.98% | 639.84 MiB | 28.15 dB |

三模型各自正式请求均无新 compile/capture，SF/14B 每请求55次 replay、Lingbot40次。ring attention workspace 改后为0。主体、场景和整体运动相近，但绕环后的浮点差异会累积，后段姿态、纹理与局部边界有可见变化，不能称精度完全对齐。PSNR 来自完整归一化视频 tensor，不是 MP4。每模型只测一个场景/种子；Lingbot 使用外部固定480高适配器，仅验证 cam/fast。

本次不宣称冷启动改善；GPU 回归219项通过且无跳过。

### 16) FA2 与 SageAttention2

SF1.3B、Realtime14B、LingbotFast 现在允许 FA2、FA3、Sage2 self/cross attention 组合，同时开启 `use_compile=true`、`use_cuda_graph=true`、`warmup=true`。完整可见 ring 中，FA2/FA3 复用物理 KV view；选择 Sage2 自注意力时自动使用分段量化缓存和 LSE 合并，不增加配置开关，不拼接历史 KV。窄窗口使用第 4 节说明的可复用 gather buffer。EdenLive 使用 raw-K ring 和有序 RoPE workspace 调用标准 sage_attn2；普通投影后的条件 KV 缓存保留，后端专用的 prepare_kv / apply_prepared 缓存已移除。

Sage 量化插件与原始 rolling 存储分开：`SageRingAttention` 保存分段 INT8 K、K mean/scale、FP8 V 和 V scale；GPU slot 写入后只更新受影响段。clean 最后一层只算 KV 时也更新相应量化段。SM89、SM90、SM120 按各自 native kernel、量化粒度和 FP8 scale 工作。常驻 rolling pool 已去除 offload 字段/分支；独立的 `OffloadedRollingKVCachePool` 及 step/hybrid 子类复用传输生命周期。

此前 2026-10-20 常驻 Sage2 对照的主精度基准是原始 main（commit `d48a6173`），self/cross 都直接使用 `sage_attn2`。H200、480×832、14 chunks，三次正式 GPU DiT 均值：SF1.3B 4.195→3.704s（耗时减少11.71%）、Realtime14B 23.095→20.843s（9.75%）、LingbotFast 32.986→28.437s（13.79%）。main 使用 eager，首次完整请求预热并排除计时；新路径实际开启 compile、Graph 与生产 warmup。此对照包含本轮及之前公共执行优化，不能归因成纯量化缓存收益。辅助的本轮修改前/mixed-backend结果另外保留，不作为主精度结论。

完整165帧新旧视频 PSNR：SF1.3B 16.06dB、Realtime14B 16.76dB、LingbotFast 16.36dB。INT8/FP8 分段与整窗量化不同，主体/场景/整体运动接近，但姿态、水线及 Lingbot 局部几何有可见漂移；不能称精度完全对齐或质量等价。量化伴随缓存还会增加显存。本机只有 H200，RTX40/50仅完成实现和布局验证，由对应硬件继续验收。安装、current-stream 补丁、显存和支持边界见 [Sage说明](SAGE_ATTENTION.md)。

### 17) EdenLive runner 生命周期

普通 `EdenLiveRunner` 继承 `RunnerLifecycleMixin`，复用 warmup 分辨率排序和请求状态清理；`EdenLiveARRunner` 显式增加 `AutoregressiveRunnerMixin`。注册的 Wan2.1/Wan2.2 普通 runner 的 MRO 因此不再包含 AR mixin。本次继承修正不改变 EdenLive attention 或 KV 计算；62 项生命周期、warmup 和 runner 聚焦测试通过。

FA2完整视频验证通过，使用健康Torch2.8/cu128，并仅在验证进程回植upstream BF16 compiler精度修复。SF1.3B/Realtime14B/LingbotFast的DiT耗时下降4.88%/3.83%/4.19%，视频PSNR26.60/28.81/29.57dB；首次绕环前4/4/7个chunk与旧eager逐bit一致。这与原始main Sage2表使用不同基线和runtime。编译边界见[Sage说明](SAGE_ATTENTION.md)。

### 18) 缓存初始化与后端选择

SF、Realtime14B、LingbotFast、MG4 的缓存由 Transformer Infer 创建和持有。Runner 只调用 `initialize_kv_cache(latent_shape)` 并更新 scheduler 的输出帧数；预热清理通过 `reset_kv_cache()`，不再创建 manager 或绑定 pool。

| self-attention 配置 | 原始存储 | attention 实现 |
| --- | --- | --- |
| `flash_attn2` / `flash_attn3` | `RollingKVCachePool` | `RingAttention`，直接读取物理 view |
| `sage_attn2` | 同一个 `RollingKVCachePool` | `SageRingAttention`，增量量化和 LSE 合并 |

表中直读/分段路线适用于完整可见 ring；窄窗口按第 4 节说明 gather 到可复用 buffer。无需单独选择缓存后端。正常配置省略 `kv_cache_scheme` 与 `step_kv_cache`，默认 rolling、跨步复用。SF 系列已移除 `sage_attn2_sm90_fused`、`sage_attn2_sm120_cached`，统一使用 `sage_attn2` 自动按 SM89/SM90/SM120 分发；EdenLive 同样使用标准 `sage_attn2`，从 raw-K ring 构造有序 RoPE workspace 后直接调用后端。MG4 只在 `rope.py` 维护 sink 位置策略，复用上述存储和 attention 后端。

### 19) 当前公共执行与验证范围

causal executor 对常驻、权重 block offload、KV offload、SP/TP、prefill 和多 step bank 共用 prepared-cache 分发；计算仍可 compile + Graph，staging 与缓存规划在图外。SF、LingbotFast、MG4、Audio/Seko AR 与 EdenLive AR 共用 causal 基类，通过模型 adapter 保留不同位置策略。非 CUDA 兼容路径仍保留。

2026-10-21 的验收区分 FA3 真实多卡/offload、Sage2 缓存测试，以及此前 SF Sage2 常驻视频，不代表所有后端/模型/并行组合都已验收。本轮 EdenLive FA3 SP2 + 权重 block/KV offload 的 717 帧完整 latent/视频 SHA 对齐相同模式 main，正式请求 0 次 capture、9000 次 replay；[Sage 说明](SAGE_ATTENTION.md) 中此前常驻 Sage2 的 717 帧为 275 次 replay，属于另一组实验。

本机当前 FA2 runtime 的四项长序列数值测试连无 KV cache 基线也失败，物理 view 实现保留，本轮 Flash 整模多卡结果使用 FA3。此前 Torch 2.8/cu128 FA2 结果依赖已记录的进程内编译器修复。RTX 40/50 没有硬件实测。本轮多数性能对照每侧只有一次预热后的正式请求，应视为回归检查，不是稳定吞吐统计。
