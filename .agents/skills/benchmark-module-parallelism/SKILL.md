---
name: benchmark-module-parallelism
description: 比较 LightX2V 同一真实模型模块在固定全局输入与资源下的 TP/SP 执行方案，测量预热后的完成延迟、显存及数值状态；按需结合 SP 算子预筛和模块 profile 解释差异。仅有 QKV 的独立 SP 调优使用 benchmark-core-ops。
---

# 模块并行方案比较

先读[工具导航](../../../tools/AGENTS.md)与[Benchmark Agent 指南](../../../tools/benchmarks/AGENTS.md)的完整层 TP/SP 章节；API 和输入合同查[README](../../../tools/benchmarks/README.md#完整模型层tpsp-比较)。比较对象是同一完整生产模块，不将局部 attention 最快配置直接当成整层最优。

## 准备可比较的模块

1. 固定全局输入、权重/源码、精度、attention 和获准资源，明确层边界及输入是真实还是合成。用户未指定 attention 时，按指南以 FA2 作任务基线；依赖或路径不可执行时保留缺口，不静默换后端。
2. 沿生产加载与并行 mesh 准备一份 `PreparedModule`：真实 `call`、输入/状态 `reset`、完成所有目标 stream/通信的 `synchronize`、输出还原与参考 `check`，以及共同 identity 和实际 process/TP/SP 组。加载、最终输出还原等是否属于每层成本须沿生产调用判断。
3. 相同资源下列出拟比较的 TP/SP 与完整 SP 配置。候选按[配置遍历约定](../../../tools/benchmarks/AGENTS.md#完整层tpsp与sp配置遍历)优先复用现有 `sp_bench` probe/枚举；按实际 TP 分片确定局部 heads 与输入、按真实 SP 度数生成候选，SP=1 保留关闭方案。记录拒绝、失败和未测，不能把少量样例称完整遍历。

## 组合工具

| 当前目标 | 选择 |
|---|---|
| 比较完整模块完成延迟、显存与数值状态 | `tools.benchmarks.module_bench.benchmark`，结果用 `compare` 比较 |
| 需要先缩小 SP 配置范围 | 复用 [benchmark-core-ops](../benchmark-core-ops/SKILL.md) 的 SP 流程，用实际 TP 局部输入 |
| 解释已选方案的计算、通信或空隙 | 复用 [profile-model-module](../profile-model-module/SKILL.md)，独立采同一真实模块 |

已有明确模块候选时可直接比较；预筛和 profile 按疑点采用，不要求每次完整执行所有阶段。保留每个方案实际 attention、TP/SP 组、完整 SP candidate 配置及输入身份，核对配置意图与实际生产调用。

主结果使用无 profiler 的同进程预热与完整模块 benchmark：每次取各 rank host 完成延迟的最大值，含目标的异步工作完成。SP 算子 event-list、profile 时间线和整层完成延迟各有范围，不能替换或相加。执行和恢复按[资源指南](../../../tools/runtime/AGENTS.md)落实监测、期限、本人收尾、受扰作废及有界重测。

## 交付

保存任务 factory/配置、JSON/逐次 raw、命令、身份及退出状态，按[完整层报告模板](../../../tools/benchmarks/MODULE_REPORT_TEMPLATE.md)生成中文结果；涉及筛选时按[遍历范围说明](../../../tools/benchmarks/SEARCH_SCOPE_TEMPLATE.md)展开维度、拒绝原因和各阶段实测数量。

同卡数、同全局目标与输入的方案独立排名，报告预热后延迟、显存增量及数值范围。非零误差保留开发者质量决定；局部数值比较不代替最终图像/视频审核。结论限定本次模块、输入和资源，不自动外推整网或 serving。
