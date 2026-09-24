# Operator Benchmark Agent 指南

本文档是 `tools/benchmarks/` 的目录级 Agent 指南，随工具提交和迁移。它规定如何把用户目标转成一次可复现的 benchmark 调研；CLI、schema 和指标定义以 `README.md` 与代码为准，不在这里重复维护。

机器专属的容器名、GPU 编号、checkpoint 绝对路径和实验产物不得写入本文档，应保存在本地工作记录中。

## 任务范围

Agent 应先把请求归入一种或多种工作流：

1. **模型 shape 实测**：消费模型导出的核心算子 shape，在指定硬件上寻找最佳 backend。
2. **理论 shape sweep**：按用户关心的维度生成网格，观察 backend 的适用区间和性能拐点。
3. **Profiler 配置审计**：用相同 shape、precision 和调用权重比较 observed backend 与候选 winner。
4. **新硬件 bring-up**：重新探测 backend、建立硬件 peak profile，再运行目标 suite；不复用其他硬件的候选和 peak。
5. **SP attention 配置搜索**：使用独立 SP suite 比较 Ulysses、Ring 及其合法选项；诊断通信时再执行 L2/L3。

模型级调度、TP/SP 混合并行、多机执行和跨硬件性能外推不属于当前工具范围。只对实际测试的指定硬件发布性能结论。

## 标准执行流程

1. 明确目标硬件、算子 family、precision、shape 来源和所需结论。缺少可选参数时采用 production 默认值或代表值，记录选择依据并继续。
2. 将模型 dump 或理论网格转换为 canonical suite；模型适配逻辑留在模型侧，benchmark 不维护 model catalog。
3. 先执行 `inspect`，修复合同错误。缺少 `call_count` 或 `observed_backend` 不阻塞逐 shape 排名，但会限制模型加权结论或 profiler 审计。
4. 在目标环境重新执行 backend/candidate probe。只运行 `eligible` 候选，分别保留依赖缺失、架构不支持、合同不适用和执行失败状态。
5. 使用新输出目录运行；正式结论默认至少 3 repeats、10 warmup、30 measurement iterations。续跑必须保持 suite、候选、环境和测量参数身份一致。
6. 检查稳定性、候选覆盖和错误状态。`measured_winner` 只代表已测集合；只有完整覆盖合同满足时才发布正式 `winner`。
7. 报告必须限定到 LightX2V commit、硬件、依赖环境、shape、precision、backend 配置和输入身份。归档 suite、命令、raw 及 report 即可，不建立额外任务控制面。

对 SP attention，持久化报告必须完整记录 winner 的 algorithm、attention backend、计算精度、sparse 参数、通信精度、fusion、pre/post、A2A、head pipeline、逻辑 Q/K/V shape、每 rank 计算 shape 和实际 payload/scale shape 及 dtype。“最优 sparse leaf”不能代替完整推荐配置。

交互窗口默认只给高密度结果：如果同时比较多个 SP 度数，表格只列 `SP` 和最优延迟，并链接到完整输出文档。只有用户明确追问时，才在对话中展开配置或通信 shape。Aux/main 切分、显存和吞吐属于文档中的辅助解释信息。

只有真实输入、checkpoint 或必要场景定义缺失且无法从仓库推导时，才向用户提出阻塞问题。非阻塞缺口应记录后继续可执行部分。

## Shape 与输入

- GEMM 使用 `m, n, k, bias`；MoE 使用 token、hidden/intermediate、expert、top-k 和 activation 合同。
- Dense attention 使用 batch、Q/KV sequence、Q/KV heads、head dim 和 causal 合同。
- SP attention 使用全局 sequence、head 信息、SP degree 和可选 auxiliary token 合同。
- 模型 shape 应保留来源、调用位置和必要时的 `call_count`；理论 sweep 必须明确轴范围，不能伪装成模型真实 shape。
- 随机生成的 GEMM、dense attention 和 MoE 输入可用于 shape 性能比较；对输入分布敏感的算子必须使用真实 artifact 或明确降低证据等级。

## Backend 与硬件 Peak

1. 候选目录跟随当前 LightX2V registry 和 benchmark adapter 演进，不能固化为某张卡上的静态列表。
2. `dependency_missing` 不是性能落选；发布最佳 backend 前，应覆盖目标 family 的全部 `eligible` 候选，并说明未覆盖项。
3. 新硬件必须建立与 GPU 名称、CUDA capability 及计算合同匹配的 peak profile。缺少 peak 不阻塞 latency、吞吐和 backend 排名，但不能计算对应效率。
4. FP8 等精度若随 accumulator 不同而峰值不同，profile 必须提供 variant，raw precision 必须声明 `accum_dtype`；不得回退到不匹配的通用 peak。
5. 名义 peak 用于定位实现效率 gap，不用于替代实测，也不能从一种硬件外推另一种硬件的性能。

## 算子专项规则

### GEMM 与 MoE

- GEMM backend 必须按实际 precision、bias 合同和 shape 比较；量化 wrapper 的动态量化等开销属于端到端测量。
- MoE 优先使用真实 routing 直方图；缺少 routing 时仍可做合成 shape 比较，但报告不得宣称代表模型真实负载。
- Profiler 审计应同时给出逐 shape 差距和按 `call_count` 加权影响，避免由低频 shape 主导模型结论。

### Dense 与 Sparse Attention

- Dense attention 候选按 causal、GQA/MQA、head dim 和架构能力做 eligibility 判断。
- Sparse attention 的正式推荐禁止使用随机 Q/K/V。应在 RoPE 等变换完成后、进入 attention backend 前捕获真实 Q/K/V，并记录 dtype、layout、shape、分支、step、block/module 和调用序号。
- Replay 使用 `operator_benchmark_qkv_replay_v1` manifest 和 SHA256 固定输入；sequence shards 必须连续、无重叠并覆盖完整序列。Artifact 写盘、哈希和初始化不进入 kernel latency。
- 用户未指定捕获点时，先选择主 denoise 分支、中间 block 和代表 step；routing 明显随调用变化时，再扩展 early/middle/late，且限定结论覆盖范围。
- 用户未指定 keep ratio 时，从 `0.10、0.15、0.20` 及 production 默认值起步；使用同一 replay 加入可用的 dense baseline。
- 同时报告 latency、实际 block density、`actual_tflops` 和 `dense_equivalent_tflops`。后者可能超过 dense peak，不得解释为实际 Tensor Core 利用率。

### Sequence-Parallel Attention

- SP 使用独立 `sp_bench.py`、SP suite 和 `torchrun`；进程数必须等于 `sp_size`。
- L1 production 端到端延迟是候选排序依据；L2 layout/communication 与 L3 collective 只用于解释 L1，不参与 winner 排名。
- L2/L3 日常入口应优先读取 L1 `report.json` 的逐 case 正式 winner，不要求用户重新描述长 candidate ID；结果同时保存机器可读 JSON 和面向人的 `diagnostic_report.md`。
- Aux 诊断必须遵循 production 通信语义：Ulysses 的 replicated aux Q/K/V 绕过 main QKV A2A，aux output 执行 all-gather；Ring 的 replicated aux Q/K/V 绕过 main K/V rotation。L3 必须拆分 main/aux 实际 payload，并把未通信的 aux 明确标为 bypass，不能用全量 dense tensor 估算替代。
- L3 分量必须独立计时并分别报告 latency、Bus bandwidth、spread 和 peak efficiency，禁止按整体延迟或总字节比例反推。混合路径的整体效率与分量效率同时保留；`all_to_all`、`all_gather`、`ring_p2p` 分别匹配自己的 peak profile。
- Ulysses 与 Ring 只在共同支持且已验证的 attention 语义上公平比较。Ring 的 GQA 必须排除；H100 当前 Ulysses/Ring aux 合法矩阵已验证 BF16/FP16、无量化/FP8 communication 和 fusion 开关，FP4 与新增 dense leaf 在专项验证完成前记录为 `not validated`，不能据此宣称 production 不支持。
- `sp_bench` 已支持 dense leaf 与真实 QKV replay 驱动的 sparse adapter。Sparse 候选必须使用独立 `--sparse-backend` 合同，不能把 sparse backend 冒充 `--dense-backend`。
- Sparse grouped Ulysses 必须使用同一 replay、同一 sparse leaf 和通信配置的 bulk Ulysses 做 reference 验证；bulk 只作为 reference，不能作为被测候选与自身比较。报告必须保留 `reference_candidate`，且不得把该结果解释为 sparse leaf 对 dense attention 的精度证明。
- Sparse Ring 只有在 backend 满足分块输出、LSE 和跨 block 合并合同时才进入候选；否则记录 `not_supported` 或 `not_applicable`，不阻塞 Ulysses。
- Sparse SP 候选必须从当前 backend catalog 派生。已知兼容 leaf 应进入 `SPARSE_SP_BACKENDS`；新出现的 eligible sparse adapter 若未完成 SP 合同分类，必须进入 `unmapped_sparse_sp_backends` 并阻断完整推荐，不能静默过滤。
- 互联 peak 必须匹配 GPU、world size 和 topology fingerprint；缺失时仍报告延迟和带宽，但不发布通信效率。
- NCCL `ALGO`、`PROTO` 和 channel 数只作为运行环境身份记录，不进入自动 sweep；需要比较时由用户显式设置环境并使用独立输出目录。
- 当前算法范围只有 Ulysses 与 Ring，不加入 `kv_all_gather` 第三路径。
- 每次新模型、新硬件或新需求开始时，必须主动审计相关候选的 exclusions，不能把 `not validated` 当作普通跳过项。这里的含义是“实现可能存在，但尚无足够证据进入推荐矩阵”。
- 只要 `not validated` 组合会影响本次候选完整性，且预计 3–5 轮工作内可验证、没有外部阻塞，Agent 必须立即登记显式 correctness tolerance，使用 `sp_bench validate-reference` 补 dense reference，再同步更新准入代码、测试和文档；不得等待用户再次提醒。
- “本次矩阵完整”指当前请求涉及的模型 shape、precision、硬件依赖和 backend 选项均有明确结论，不要求提前穷举未来所有交叉组合。结束任务时，不得静默遗留与本次推荐相关的 `not validated` exclusion。
- 只有缺少目标硬件、必要依赖、真实输入或模型语义等外部条件时才允许延期。延期项必须记录未完成内容、阻塞原因、触发补齐的条件和下一步动作；条件在后续任务中满足时，Agent 应主动恢复验证。
- Sparse SP 必须使用真实 replay，但采集拓扑不得与目标 SP 度数绑定。Artifact 必须描述原始 token 顺序下的全局完整、RoPE 后 Q/K/V；采集时可以使用单卡、模型并行或 sequence shards，benchmark 再按 production 模型语义切成用户指定的 SP 度数。
- Sparse L2/L3 必须复用与 L1 相同的 replay 输入；禁止因诊断方便而退化为随机 Q/K/V。
- 对“复制最大 conditioner 前缀、剩余 main 序列不 padding 切分”的模型，使用 `sp_bench derive-replay-suites`从同一完整 replay 派生每个目标 SP 的 suite；`conditioner_tokens` 必须来自模型语义，不能根据整除性猜测。
- 用户未指定 SP 范围时默认搜索 `SP=2/4/8`，也接受指定卡数。每个 SP 度数必须独立派生 replicated aux 与 sharded main 边界、验证 head/sequence 合同并发布各自 winner；不得用一个 SP 的局部分片冒充完整输入，也不得沿用另一个 SP 的 aux 边界。
- 整模采集显存不足时，先审计 transformer block 的结构、attention 合同和目标捕获点。可以删除目标层之后的层并在捕获后立即停止；若选择 block 0 等早期同构代表层，也可以构造缩层模型。不得删除目标层之前会改变其输入分布的计算后仍宣称是该层真实输入。缩层依据、保留层范围和代表性边界必须写入 provenance。
- 当前首选 MiniMax-H3，因为它具备 production aux 切分和 dynamic sparse 路径。用户只需说“帮我 dump MiniMax-H3 真实完整 sparse attention 输入，并分别搜索 SP=2/4/8 最优配置”，Agent 负责选择代表 block/step、生成 replay/suite 并执行。Sparse Ring leaf 不满足分块 LSE 合同时记录不支持。

## 代码与本地记录边界

以下内容必须进入受 Git 管理的 `tools/benchmarks/`：

- suite/replay/result schema 及其校验；
- 与模型无关的 shape/replay 转换器、捕获辅助能力和 backend adapter；
- backend 发现、eligibility、计时、指标和推荐规则；
- 用户入口和本目录级 Agent 指南。

以下内容可以只保留在本地工作目录：

- 容器名、GPU 使用顺序和 checkpoint 绝对路径；
- QKV shards、raw result、临时 suite 和 profiler 大文件；
- 一次性诊断 hook、失败日志和探索记录。

一次性脚本形成复用价值时，应按职责迁入受 Git 管理的位置：模型专属适配放在对应模型侧，通用能力放在 `tools/benchmarks/`。阶段反馈必须区分“已有受控工具支持”和“本次临时生成”，不能把只存在于本地目录的脚本描述为可迁移能力。
