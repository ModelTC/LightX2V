# SF 系列 SageAttention2 ring cache

自注意力选择 `sage_attn2` 后，SF 1.3B、Realtime14B 和 LingbotFast 共用 `SageRingAttention` 插件。原始 BF16/FP16 K/V 仍保留在物理 ring；旁路缓存各段 INT8 K、K mean/scale、native permuted FP8 V 和 V scale。当前 slot 写入后，只重新量化受影响的段。全部等长段通过一次 native batched attention 计算，用 LSE 和 `q · Kmean` 修正合并；不足一段的有效前缀额外用一次 tail batch，padding K 不参与 softmax。完整可见 ring 路径不拼接 K/V，也不分配逻辑有序 K/V workspace；窄窗口的处理见下文。

| GPU | Native kernel | Q/K granularity | PV accumulation |
| --- | --- | --- | --- |
| RTX 40 / SM89 | Sage `_qattn_sm89` | per-thread INT8 | FP32 + FP16 |
| H100/H200 / SM90 | Sage `_qattn_sm90` | per-thread INT8 | FP32 + FP32 |
| RTX 50 / SM120 | Sage `_qattn_sm89` | per-warp INT8 | FP32 + FP16 |

SM89 的新缓存采用官方 CUDA **FP8 V** 路线。LightX2V 原有 `sage_attn.py` 在 SM89 上覆盖为 `sageattn_qk_int8_pv_fp16_triton`，使用 per-block INT8 Q/K 和 **FP16 V**；本次 SF 缓存使用 per-thread INT8 Q/K 和 FP8 V，内核及量化方式均有变化。[SageAttention 官方 dispatcher](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py) 对 SM89 默认选择 `sageattn_qk_int8_pv_fp8_cuda`，其默认参数为 `qk_quant_gran="per_thread"`、`pv_accum_dtype="fp32+fp16"`、`smooth_v=False`，与本次选择一致。MG4 参考实现面向 SM120，支持分段量化和 LSE 合并的设计参考，不构成 RTX 40 系列实测依据。

此前 Sage2 专项只在 H200 验证了 SM89/SM120 native family 的接口、量化布局和数值参考；RTX 40/50 的精度、速度及 CUDA Graph 仍需在对应显卡运行验证。40 系列应以 LightX2V 原有 FP16 V 路径作优化前基线，检查实际视频及耗时。

V 不减均值。native kernel 的 LSE 是 log2，合并时转换成自然 log，再补每段减去 K mean 所造成的 score 偏移。INT8/FP8 引入量化误差，分段量化也与整窗量化有不同的 scale；视频对比应使用误差指标和实际画面。当前机器只有 H200，40/50 系列的 native attention、速度和视频需要在对应硬件运行验证。

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

零 cat 与零历史拷贝的范围是完整可见 ring：FA2/FA3 直接读物理 view，Sage2 使用对应量化分段。若 attention 窗口窄于存储区间，`RingAttention` 把有效 spans gather 到可复用 K/V buffer；Sage2 此时对该 buffer 调用 dense 后端。这个窄窗口路径仍可 compile/Graph，但包含历史拷贝，不能称零 copy。

量化缓存是原始 ring 的伴随缓存，没有替换原始 BF16/FP16 K/V。常见大段配置下，INT8 K、FP8 V 及 scale/mean/padding 的持久伴随缓存额外占用约原始 KV 一半的显存；短段的 padding 比例更高。这不包含分段 attention 的临时输出、Q 量化缓冲和 Graph 内存池。初始未满段按真实 KV 长度计算，后续写入复用同一分段布局。请求清理失效量化 valid 标记和条件 KV 内容，同时保留可复用存储；兼容请求可复用 Graph。动态 slot 不触发重新捕获；更换 shape、外部存储或整模型卸载重载会使对应 Graph 失效。

原生 `flash_attn2` / `flash_attn3` 共用物理 view 与窄窗口 buffer 两种布局，不走分段 LSE，也不额外保存 Sage 量化缓存。两者均接通 compile、warmup、CUDA Graph；代码接通与特定 runtime 的数值验收需要分别看待。

| 验证范围 | 当前证据与限制 |
| --- | --- |
| FA3 多卡与 offload | 本轮 H200 真实模型覆盖 SP2/TP2 及权重 block/KV offload 的具体组合。不是所有模型、所有组合都逐一验收。 |
| SF Sage2 | 下述 SF/Realtime14B/LingbotFast 真实视频与收益来自此前 H200 常驻专项；分段缓存、slot、compile/Graph 的合成测试不等同于 Sage2 全模型多卡/offload 性能验收，不能套用 FA3 的实测结果。 |
| FA2 | 此前真实视频验证使用 Torch 2.8/cu128 和下述外部编译器精度修复。本轮安装的 FA2 runtime 在不使用 KV cache 的长序列基线也产生 NaN，四项长序列数值测试因此未计为通过；本轮 Flash 多卡真实结果使用 FA3。 |
| RTX 40/50 | 已实现相应内核适配与布局检查；没有对应显卡的本轮精度、速度或 Graph 实测。 |

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
pytest tests/test_sage_ring_attention.py -q
```

## EdenLive AR

EdenLive 使用与 SF 相同的 `sage_attn2` 后端名，缓存位置策略由 `RoPERingAttention` 处理。FA2、FA3、Sage2 共用 raw-K ring、slot writer 和有序 RoPE workspace；Sage2 接收完整有序 K/V，调用原版 dense Sage2。它不复用 SF 的分段量化/LSE 路径。

每个 chunk 首次 forward 将历史 raw K 按逻辑窗口位置旋转，并与 V 一起写入复用 workspace；后续 denoising 只更新当前段。clean 后推进 ring 游标。KV offload 在层之间共享一组 workspace，转入新层时重新准备历史；Graph 捕获计算，传输留在 Graph 外。

有序路线保持原始 main 的整窗 K mean、V 量化尺度和 attention 输出舍入顺序。此前 H200 Sage2 专项的 cryo-crossing 85 帧和 717 帧测试同时开启 compile、CUDA Graph、warmup，完整 latent 与解码像素均与原 main 直接 Sage2 逐字节一致；该次 717 帧常驻请求为 275 次 Graph replay，没有正式请求捕获或重新编译。

本轮 2026-10-21 的 717 帧多卡主结果使用 **FA3、SP2、权重 block offload + KV offload**：完整 latent 与浮点解码视频均对齐原 main 的相同模式，正式请求 **0 次 capture、9000 次 replay**。逐层 staging 与常驻整栈的 Graph 计数不同；275 与 9000 属于不同后端、执行模式和测试轮次。性能按各自同卡、同配置的记录评估。

参考源码：

- [Matrix-Game-4 quantized ring](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/matrix_game/inference/rtx5090_sage/sage_attention.py)
- [MG4 current-stream patch](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/matrix_game/inference/rtx5090_sage/sageattention.patch)
- [SageAttention2 dispatch and LSE units](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py)
