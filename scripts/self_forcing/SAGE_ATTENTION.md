# SF 系列 SageAttention2 ring cache

自注意力选择 `sage_attn2` 后，SF 1.3B、LingbotFast、Matrix-Game-v4 和 Krea RealtimeVideo 共用 `SageRingAttention`。普通 resident rolling 路径只保存各段 INT8 K、K mean/scale、平台原生 V 和 V scale，完整原始 BF16/FP16 K/V 历史为0字节。当前 K 做完原模型 causal RoPE 后，与当前 V 一起直接编码到物理写入段；普通历史 K/V 跨 forward、跨 chunk 复用。Matrix-Game-v4 另保留少量 raw sink K；Krea 按源码每 chunk 重算 context，不能复用过期的 context 内容。启用 KV offload 的兼容路径仍保留原始 CPU K/V backing 和 GPU staging。

全部等长段通过一次 native batched attention 计算，用 LSE 和 `q · Kmean` 修正合并；不足一段的有效前缀额外用一次 tail batch，padding K 不参与 softmax。完整可见 ring 路径不拼接 K/V，也不分配逻辑有序 K/V workspace；窄窗口的处理见下文。

| GPU | Native kernel | Q/K granularity | PV accumulation |
| --- | --- | --- | --- |
| Ampere / SM80、SM86 | Sage `_qattn_sm80`，FP16 V | per-thread INT8 | FP32 |
| RTX 40 / SM89 | Sage `_qattn_sm89` | per-thread INT8 | FP32 + FP16 |
| H100/H200 / SM90 | Sage `_qattn_sm90` | per-thread INT8 | FP32 + FP32 |
| RTX 50 / SM120 | Sage `_qattn_sm89` | per-warp INT8 | FP32 + FP16 |

SM89 的新缓存采用官方 CUDA **FP8 V** 路线。LightX2V 原有 `sage_attn.py` 在 SM89 上覆盖为 `sageattn_qk_int8_pv_fp16_triton`，使用 per-block INT8 Q/K 和 **FP16 V**；本次 SF 缓存使用 per-thread INT8 Q/K 和 FP8 V，内核及量化方式均有变化。[SageAttention 官方 dispatcher](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py) 对 SM89 默认选择 `sageattn_qk_int8_pv_fp8_cuda`，其默认参数为 `qk_quant_gran="per_thread"`、`pv_accum_dtype="fp32+fp16"`、`smooth_v=False`，与本次选择一致。MG4 参考实现面向 SM120，支持分段量化和 LSE 合并的设计参考，不构成 RTX 40 系列实测依据。

此前 Sage2 专项在 H200 验证了接口和数值参考；本轮另在 RTX 5090 验证了部分真实模型的分段缓存、CUDA Graph 和 offload 组合，具体边界和结果应以对应实测记录为准。RTX 40 系列仍需单独验收。40 系列应以 LightX2V 原有 FP16 V 路径作优化前基线，检查实际视频及耗时。

V 不减均值。native kernel 的 LSE 是 log2，合并时转换成自然 log，再补每段减去 K mean 所造成的 score 偏移。INT8/FP8 引入量化误差，分段量化也与整窗量化有不同的 scale；视频对比应使用误差指标和实际画面。分段量化与整窗量化会改变长链 AR 轨迹，不能因为单算子接近就声称最终视频逐位一致。

## 启用

在原模型配置中同时开启执行选项，保留原来的 checkpoint、task、RoPE 和其他模型字段：

```json
{
    "self_attn_1_type": "sage_attn2",
    "cross_attn_1_type": "sage_attn2",
    "use_compile": true,
    "use_cuda_graph": true,
    "warmup": true,
    "ar_config": {
        "local_attn_size": 12,
        "sink_size": 3,
        "num_frame_per_chunk": 3
    }
}
```

SF 系列只使用 `sage_attn2`，按 GPU 选择上表内核；已移除 `sage_attn2_sm90_fused` 和 `sage_attn2_sm120_cached` 入口。cross attention 可与 FA2/FA3 混用。当前 causal executor 为常驻、权重 block offload、KV offload、SP/TP、reference prefill 和 step/hybrid cache bank 使用同一 prepared-cache 接口。常驻计算捕获整栈；staging 路径在权重和 KV 到达后捕获单层，传输与 cache 规划留在图外。首 chunk、增长窗口、prefill 和 clean commit 均可同时使用 compile 与 CUDA Graph，不再要求满窗或关闭 image cross-attention。

零 cat 与零历史拷贝的范围是完整可见 ring：FA2/FA3 直接读物理 view，Sage2 使用对应量化分段。若 attention 窗口窄于存储区间，Flash 将不连续的有效物理 views 收集到可复用 KV workspace，再执行一次 attention；Sage2 在量化时直接读取目标 slots，并保留分段 attention＋LSE 合并，不回退 dense attention。

resident causal Wan 的量化缓存已经替代完整原始 K/V 历史；pool 只保留 slot metadata、长度及零字节原始占位张量。当前投影得到的原始 K/V 是本次 forward 的临时输入。分段长度必须整除物理容量、chunk 写入、reference/sink 等边界，避免覆盖半段后需要恢复原始值；未新增用户分段参数或运行时搜索。启用 KV offload、以及保留模型专属原始存储的 H3 adapter，仍使用量化 companion。持久量化存储之外仍有分段 attention 临时输出、Q 量化缓冲和 Graph 内存池。请求清理失效量化 valid 标记和条件 KV 内容，同时保留可复用存储；兼容请求可复用 Graph。动态 slot 不触发重新捕获；更换 shape、外部存储或整模型卸载重载会使对应 Graph 失效。

原生 `flash_attn2` / `flash_attn3` 共用物理 view；窄窗口的不连续 views 使用可复用 KV workspace，执行一次 attention，不保存 Sage 量化缓存。读取时才应用 RoPE 的模型使用物理 K workspace，见下文。Flash 分段 view＋LSE 和分段 RoPE＋LSE 路线已移除。Flash 和 Sage 均接通 compile、warmup、CUDA Graph；代码接通与特定 runtime 的数值验收需要分别看待。

| 验证范围 | 当前证据与限制 |
| --- | --- |
| FA3 多卡与 offload | 本轮 H200 真实模型覆盖 SP2/TP2 及权重 block/KV offload 的具体组合。不是所有模型、所有组合都逐一验收。 |
| SF Sage2 | 下述 SF/Realtime14B/LingbotFast 真实视频与收益来自此前 H200 常驻专项；分段缓存、slot、compile/Graph 的合成测试不等同于 Sage2 全模型多卡/offload 性能验收，不能套用 FA3 的实测结果。 |
| FA2 | 此前真实视频验证使用 Torch 2.8/cu128 和下述外部编译器精度修复。本轮安装的 FA2 runtime 在不使用 KV cache 的长序列基线也产生 NaN，四项长序列数值测试因此未计为通过；本轮 Flash 多卡真实结果使用 FA3。 |
| RTX 40/50 | RTX 5090 已完成 SF1.3B、MG4 的完整源码对照和实际 compile/Graph/异步流式解码；RTX 40 仅有分派/布局检查，仍待实卡验收。 |

2026-10-25 单卡5090的新结果见 [SF1.3B 报告](../../save_results/wan_causal_unification_20261025/self_forcing_1_3b/README.md) 和 [MG4 报告](../../save_results/wan_causal_unification_20261025/matrix_game4/README.md)。这里保留的 H200/FA2 记录属于各自历史配置，不能与本轮结果拼成同一组测量。当前所有模型的存储、RoPE、FA/Sage及step bank对照见 [KV实现汇总](../../lightx2v/common/kvcache/AR_IMPLEMENTATIONS.md)。

## CUDA Graph 与 compile

此路径使用 `torch.library.custom_op` 保持编译边界，支持 `torch.compile(fullgraph=True)` 与外层 CUDA Graph 同时开启。slot metadata 是 GPU tensor，切换环形槽位无需重新捕获。clean commit 也会更新量化缓存。

标准 Torch 2.8 缺少本轮验证涉及的两项上游编译精度修复：Inductor 没有把 `emulate_precision_casts` 对应的禁用 FMA 选项传给 Triton；`addmm` 优化也没有阻止 BF16/FP16 的 MM 与 bias 被拆开，导致舍入顺序改变。仅设置 `emulate_precision_casts=True` 不足以修复这两点。它们影响通用 FFN 和逐元素计算，并非 FA2 专属问题。Torch 2.11 已原生包含相应处理，模型代码无需增加版本分支；此前 SF/Eden Sage2 专项均在原生 Torch 2.11 上测试，无需外部 hook。

此前 FA2 专项实验在健康的 cu128 环境中，通过进程内编译器修复验证了真实视频。这是此前 FA2 专项实验的复现入口，不是通用安装方案，也不能用于声称未经修复的 Torch 2.8 在首个 wrap 前与 eager 完全一致。

更换 Torch/CUDA 或 GPU architecture 后，需要使用匹配该环境构建的 FA2/Sage 原生扩展并重新验证。V3 hook 不修复后端二进制兼容性；Sage 的 current-stream 发射要求还需按下文处理。

上游 SageAttention2 的部分 CUDA native kernel 默认在 CUDA default stream 发射。CUDA Graph 捕获和用户自建 stream 必须使用 PyTorch current stream。编译 SageAttention 前运行下面的幂等源码补丁，然后重新安装；脚本不会改动已经显式指定 stream 的 launch。

```bash
python scripts/self_forcing/patch_sageattention_streams.py /path/to/SageAttention
cd /path/to/SageAttention
pip install . --no-build-isolation
```

Blackwell 需要 CUDA 12.8 或更新版本；Ada FP8 需要 CUDA 12.4，Hopper FP8 需要 CUDA 12.3。安装时保留对应 GPU architecture 的 Sage native 扩展。

安装后从 LightX2V 根目录验证量化布局、有效长度、LSE 修正、动态 slot、compile 和 Graph：

```bash
pytest tests/test_sage_ring_rope.py tests/test_sage_ring_merge.py tests/test_kv_slot_metadata.py tests/test_rope_ring_commit.py -q
```

## EdenLive AR

EdenLive 使用与 SF 相同的 `sage_attn2` 后端名，raw K 按物理槽位存储。`SageRoPERingAttention` 把重定位 RoPE 融入分段 K 量化，V 保持独立的量化有效位；没有逻辑有序 BF16 KV workspace，也没有 dense Sage self-attention 回退。增长窗口沿用历史量化，滚动窗口改变 anchor 时只使 K 失效，当前写入段的 K/V 始终更新。窄窗口在量化中直接选取物理 slots。

FA2/FA3 使用物理顺序 RoPE K workspace，完整 ring 的 V 直接读取 raw ring；窄窗口的不连续 views 可额外将选中的旋转后 K/V 收集到可复用 workspace，执行一次 attention。5090 本轮只测试 FA2 与 Sage2，不测试 FA3。旧 gather+RoPE 融合算子已删除，Flash 不再分段执行 attention 或合并 LSE。完整窗口与尾 chunk 的量化存储分别复用，避免下一请求替换绑定地址导致重新 capture。

以下为改成分段 Sage 之前的历史有序路线记录，不能作为当前分段量化与 main 逐位等价的声明：

有序路线保持原始 main 的整窗 K mean、V 量化尺度和 attention 输出舍入顺序。此前 H200 Sage2 专项的 cryo-crossing 85 帧和 717 帧测试同时开启 compile、CUDA Graph、warmup，完整 latent 与解码像素均与原 main 直接 Sage2 逐字节一致；该次 717 帧常驻请求为 275 次 Graph replay，没有正式请求捕获或重新编译。

本轮 2026-10-21 的 717 帧多卡主结果使用 **FA3、SP2、权重 block offload + KV offload**：完整 latent 与浮点解码视频均对齐原 main 的相同模式，正式请求 **0 次 capture、9000 次 replay**。逐层 staging 与常驻整栈的 Graph 计数不同；275 与 9000 属于不同后端、执行模式和测试轮次。性能按各自同卡、同配置的记录评估。

## Seko AR 4step / 2step

两者共用 `WanAudioARTransformerInfer`：rolling self-KV 选择 Sage2 时使用 `SageRoPERingAttention`，读取时把 RoPE 融入分段 K 量化并用 LSE 合并；FA2/FA3 统一使用 `PhysicalRoPERingAttention`，不再提供 Flash 分段模式配置。step cache 为各 denoising step 保存独立 bank，2step hybrid 的窄窗口仍按有效物理 slots 读取；不连续 views 使用可复用 KV workspace，执行一次 Flash attention。audio perceiver adapter 和 text/image cross attention 各自采用原来的 attention，不属于这项 self-KV 路径约束。

`seko_talk_ar_fp8_f16_accum_5090.json` 与 `seko_talk_ar_hybrid_2step_fp8_f16_accum_5090.json` 是需要替换权重和 adapter 路径的配置模板。截至 2026-10-23，[官方 SekoTalk 仓库](https://github.com/OpenSenseNova/SekoTalk)只公开 README，未发布这两个 AR checkpoint；本机也缺少对应权重，因此没有本轮真实视频、速度或完整多卡/offload 验收。2/4step、local/global RoPE、FA2 物理 workspace 及 Sage2＋compile＋CUDA Graph 的模型接口有合成测试，不能据此声称真实模型的收益或画质。

参考源码：

- [Matrix-Game-4 quantized ring](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/matrix_game/inference/rtx5090_sage/sage_attention.py)
- [MG4 current-stream patch](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/matrix_game/inference/rtx5090_sage/sageattention.patch)
- [SageAttention2 dispatch and LSE units](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py)
