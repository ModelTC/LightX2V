# LightX2V Compile Casebook

遇到相同现象时按“确认 → 根因 → 处理”复现，不要把历史结论直接套到新模型。

## 目录

- [1. staging block 重复编译](#1-staging-block-重复编译)
- [2. 公共 dispatcher 污染层号状态](#2-公共-dispatcher-污染层号状态)
- [3. LTX2.3 guider 触发重编译上限](#3-ltx23-guider-触发重编译上限)
- [4. Qwen 配置相同但 RMSNorm kernel 不同](#4-qwen-配置相同但-rmsnorm-kernel-不同)
- [5. custom op 没有减少 kernel launch](#5-custom-op-没有减少-kernel-launch)
- [6. Qwen one-pass RMSNorm 编译不稳定](#6-qwen-one-pass-rmsnorm-编译不稳定)
- [7. warmup 后 Step 1 仍慢](#7-warmup-后-step-1-仍慢)
- [8. cache cleanup 让 Step 1 重新变慢](#8-cache-cleanup-让-step-1-重新变慢)
- [9. offload 图外成本掩盖收益](#9-offload-图外成本掩盖收益)
- [10. scaled_mm 的 scale rank 错误](#10-scaled_mm-的-scale-rank-错误)
- [11. 把 torch.compile 当成 CUDA Graph](#11-把-torchcompile-当成-cuda-graph)
- [12. OOM 或其他进程污染实验](#12-oom-或其他进程污染实验)
- [13. Step 1 变快但冷启动总耗时更高](#13-step-1-变快但冷启动总耗时更高)
- [14. 最终输出相近但数值不一致](#14-最终输出相近但数值不一致)

## 1. staging block 重复编译

- **现象**：Qwen-Image/LTX2 block offload 只有少量 CUDA staging block，`compiled_blocks` 却接近逻辑层数。
- **确认**：比较 `cuda_buffers` 的对象身份、cache keys，并搜索图内是否读取 `self.block_idx`。
- **根因**：无层号依赖的 staging buffer 仍使用默认 `block_idx`。
- **处理**：offload infer 覆写 `get_compile_block_key()` 返回 `id(block)`。Wan 读取层号，不能照搬。

## 2. 公共 dispatcher 污染层号状态

- **现象**：`use_compile=false` 调用公共 `run_block()` 后也出现 `self.block_idx`。
- **确认**：检查 block 下游是否真正读取层号，以及需求是否只来自 Wan/AR 子类。
- **根因**：把专用路径的层号要求放进了公共 dispatcher。
- **处理**：公共层只做 eager/compile 分派；Wan 在自己的循环设置层号，LTX2 AR 只设置 `_ar_block_idx`。用单测保证无层号模型的 eager 对象不新增该字段。

## 3. LTX2.3 guider 触发重编译上限

- **现象**：单一 `self.compiled_block` 在多路 guider 下持续重编译并超过 Dynamo cache limit。
- **确认**：运行 `TORCH_LOGS="recompiles,graph_breaks" <command>`，查看 guards 是否来自 `_mm_skip_*`、block 身份或 guider 分支。
- **根因**：所有层共享一个入口，并读取变化的 Python set/`self` 状态。
- **处理**：使用公共 per-key cache；在循环中把 guider 状态计算成 `skip_*` 布尔参数传给 block。不要先提高 cache limit。

## 4. Qwen 配置相同但 RMSNorm kernel 不同

- **现象**：eager/compile 都配置 `rms_norm_type: "sgl-kernel"`，性能却不是同 kernel 的行为。
- **确认**：搜索仓库和依赖中的 `is_compiling|is_dynamo_compiling`，再用 trace 核对 kernel 名称和次数。当时的 `sglang-kernel 0.4.4` 会在 Dynamo 下分流。
- **根因**：backend 标签相同，第三方 wrapper 的运行分支不同。
- **处理**：先确定是比较生产最优栈还是严格同 kernel。前者注明实际实现；后者让 custom op 调用目标 SGL kernel，并用 trace 证明两边一致。

## 5. custom op 没有减少 kernel launch

- **现象**：RMSNorm/RoPE 设为叶子后 compile 可运行，但仍有大量调度，稳态收益有限。
- **确认**：profile custom-op、CPU dispatch 和 CUDA kernel 数量。
- **根因**：custom op 只阻止 Dynamo 进入内部，不会跨边界融合或合并多次调用。
- **处理**：兼容性目标保留小叶子；减少 launch 需要更大粒度的融合算子，不能靠“禁止编译叶子”实现。

## 6. Qwen one-pass RMSNorm 编译不稳定

- **现象**：one-pass eager 较快，纳入 block compile 后出现不兼容、首次 Triton 编译或路径不稳定。
- **确认**：最小化测试 RMSNorm 数值和实际 kernel，检查动态图对象及 warmup specialization。
- **根因**：当前 one-pass 实现不满足 Qwen block compile 的稳定要求。
- **处理**：当前 Qwen weights 对 `use_compile + one-pass` 直接报错，避免静默换 kernel。新模型先复现，不要机械继承禁令。

## 7. warmup 后 Step 1 仍慢

- **现象**：warmup 完成，但正式 Step 1 慢于后续；compile 的相对差距可能比 eager 更明显。
- **确认**：
  1. 分别计算 `Step1 - steady` 和 `Step1 / steady - 1`，保留至少三轮逐轮值；
  2. 用 `TORCH_LOGS="recompiles,graph_breaks"` 检查 guards；普通日志没有编译字样不能证明没有重编译；
  3. 对比 warmup/正式的 shape、token/frame、CFG/guider、dtype、layout、stage、对象身份和 leaf kernel；
  4. 核对计时边界，再检查 allocator reserved、attention/grouped-MM workspace、请求 cache 和数据依赖工作量。
- **根因**：可能是未覆盖的 graph signature，也可能只是正式尺寸首次分配/初始化、MoE 路由差异或测量噪声。compile 把 steady 加速后，固定首步成本会因分母变小而显得更大。
- **处理**：
  - 有 recompile/新 kernel 证据：修复 warmup 覆盖、guards 或 leaf dispatch；
  - 无编译证据且 Step 2 起稳定：用一次“正式 signature 的受控 warmup”验证首次资源开销，再决定是否值得扩大生产 warmup；
  - 不要默认做 request-aware warmup，也不要跳过 scheduler/request cache 的正确清理。具体修改遵循 `../../support_model_warmup/SKILL.md`。

`dynamic=None` 能接受新尺寸，不代表该尺寸的 allocator/workspace 已经预热。计时区间即使不包含 text encoder，文本长度仍会改变进入 transformer 的序列规模。

**LingBot-Video 实例**：warmup 使用两种固定分辨率、短 prompt 和空 negative prompt，正式 T2I/T2V 使用未精确覆盖的分辨率及长 CFG 文本。T2I compile 的 Step 1 相对额外耗时为 13.6%，高于 eager 的 9.7%，但绝对值反而更小（23.7 ms 对 32.6 ms）；I2V 某轮没有首步惩罚。这符合 signature 覆盖不足和首次资源使用的特征，不足以证明 compile 实现错误。

## 8. cache cleanup 让 Step 1 重新变慢

- **现象**：warmup 命中正式图，Step 1 仍慢；两者之间存在 RoPE/position cache 清理、`empty_cache()` 或 GC。
- **确认**：搜索 `rope|position.*cache|empty_cache|maybe_empty_cache|gc.collect`，分别对比 live allocation、allocator reserved cache 和 request-specific cache。
- **根因**：无条件 `empty_cache()` 释放 workspace；request-specific cache 的正确清理又被误认为性能缺陷。
- **处理**：allocator 使用已有 pressure-aware 路径；依赖请求 shape/内容的 cache 必须清除，只复用经验证的 shape-independent 部分。

## 9. offload 图外成本掩盖收益

- **现象**：model offload 首尾 step 偏慢；block offload compile 收益小或为负。
- **确认**：分别 profile compute、H2D/D2H、prefetch/swap 和 stream synchronize。
- **根因**：传输与同步在编译图外；首步可能包含 model H2D，末步可能包含 D2H。
- **处理**：保留完整 `infer_main` 指标并解释首尾成本，同时比较纯 compute。传输重叠属于 offload 优化，不属于 compile 适配。

## 10. scaled_mm 的 scale rank 错误

- **现象**：量化 MM 只在 Inductor 下因 `scale_a/scale_b` rank 或 shape 不匹配报错。
- **确认**：最小化复现 weight `apply()`，记录 input/weight scale shape，排除 checkpoint 加载问题。
- **根因**：标量、`(1,)`、`(1,1)` 和 `(1,N)` 混用。
- **处理**：仅在精确命中时统一 rank：per-tensor 两侧用 `(1,)`；weight 为 `(1,N)` 时 input 为 `(1,1)`。修改公共 `mm_weight.py` 后覆盖所有相关量化路径。

## 11. 把 torch.compile 当成 CUDA Graph

- **现象**：看到 compile 便认为已消除 launch，或直接加入 `mode="reduce-overhead"`。
- **确认**：公共调用只有 `torch.compile(callable, dynamic=None)`，不能据此宣称使用 CUDA Graph。
- **根因**：混淆 Inductor、算子融合、custom-op boundary 和 CUDA Graph capture。
- **处理**：保留默认 baseline。CUDA Graph 另行验证动态 shape、mutation、staging buffer、offload stream、显存和稳定收益。

## 12. OOM 或其他进程污染实验

- **现象**：正式 shape 在 eager/compile 的加载、warmup 或请求阶段 OOM，或重测期间目标 GPU 出现其他 PID。
- **确认**：先用 no-warmup eager 验证正式 shape；保存 traceback 和 allocated/reserved/peak；区分 compile 独有峰值、图池与基础 live tensor；查看 `nvidia-smi` PID。
- **根因**：基础模式超过单卡容量、compile 增加 tracing/Inductor/graph pool 峰值，或并发负载污染。
- **处理**：eager 也 OOM 时标记基础路径不可比较；仅 compile OOM 时报告 compile 显存代价。较小 shape/替代配置只能做 smoke test。发现明确外部 PID 后终止自己的轮次，空闲后从 warmup 重跑；不干预未知进程。

## 13. Step 1 变快但冷启动总耗时更高

- **现象**：compile + warmup 的正式 Step 1 接近稳态，但一次性任务反而更慢。
- **确认**：分别记录 compile/warmup 准备、正式 pipeline、两者之和；不要只截取 `infer_main cost`。
- **根因**：编译成本被移到服务就绪前，没有消失；短任务的稳态节省不足以摊销。
- **处理**：分别报告首次请求、稳态吞吐和一次性冷启动；按部署生命周期选择 eager 或 compile，不把成本迁移称为总加速。

## 14. 最终输出相近但数值不一致

- **现象**：同 seed 的 eager/compile 构图和语义相近，但像素、latent 或音频数值差异明显。
- **确认**：先证明 eager 重复运行可复现，再依次比较单 block、首个 denoise step 和最终输出，并核对 dtype、kernel dispatch 与容差。
- **根因**：BF16/Inductor 运算顺序的微小差异可能被多步扩散放大，也可能是真实分支、kernel 或状态不一致。
- **处理**：不要只凭最终视觉结果判断正确；先定位首个超出容差的位置。若单步已明显偏离，继续查实现；若仅长期累积，按项目数值契约决定是否接受并明确报告。
