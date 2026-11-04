---
name: profile-model-module
description: 对 LightX2V 模型中的真实函数或模块采集并解释性能，按需截取输入、建立重放、关联计算量并分析多 rank 通信；适用于 transformer 层、attention、encoder、VAE 等范围的成本定位。后端竞速和完整模块并行方案排名使用 benchmark 工具。
---

# 模型模块性能分析

先读[工具导航](../../../tools/AGENTS.md)与[Profiler Agent 指南](../../../tools/profile/AGENTS.md)，按[README](../../../tools/profile/README.md)选择 CLI 或 API。范围可以是单个同步函数或模块，也可以是多 rank 的同一生产调用；按当前疑问选择最小充分范围。

只需 host 观测或单独算量时，按 README 选择 `--observe-only` 或 `--stats-only`，二者均不启动 profiler；需要时间线时再选择下述干净或联合采集。

## 组织真实调用

1. 从模型脚本追到实际加载的权重、配置、输入和目标调用。先核已有请求成本或正常测量，避免把标准合成层当真实模型；用户只要求局部分析时保持该范围。
2. 若需重放，保存真实 tensor、权重/量化表示、必要状态和捕获位置；在 buffer 被覆盖前取值，重建实际生产调用及并行组。任务侧提供加载与 reset，先比较原调用和重放输出；shape、路由摘要或相同 seed 不能替代真实输入。
3. 核入口能在同一进程提供足够的真实调用与状态恢复，再选择 `Capture` CLI/API。scheduler warmup 不提供跨进程预热；CLI 包装能力及不安全导入限制以 README 为准，有状态调用不能暗中重复执行。

## 采集与解释

| 当前需要 | 选择 |
|---|---|
| 正常时间线、空隙与同步疑点 | 干净 `Capture`，关闭计算量统计与 `record_shapes` |
| 同次 shape、计算量、实际稀疏区域和 kernel 归属 | 独立联合采集，再用 `trace_report` 分析 |
| 多 rank 的同一模块与通信 | 各 rank 独立目录，记录共同身份/批次和实际 TP/SP 组，用 `parallel_report` |

普通单卡 trace 也使用 `trace_report`。加载和 reset 在目标外时，按 active 调用的实际 scope 分析；GPU 异步尾部不能按 host 返回截断。多 rank 不从 WORLD_SIZE 猜 TP/SP，也不拼不同配置或批次。

计算量来自 owning core/backend 的同次事实，不开启 `record_shapes` 猜算量。编译图、未接入入口或缺最终区域时保留缺口；另跑 eager 统计可辅助理解，但不能冒充与原 trace 逐次绑定。高空隙先核同进程预热和采集扰动，再沿 CPU 等待与 GPU 活动追实际依赖。

正式性能比较仍使用无 profiler 的 benchmark，干净时间线与联合采样各自保留来源。

执行、进程收尾和中断恢复遵循 Profiler 指南及[资源指南](../../../tools/runtime/AGENTS.md)，先核已有进程、产物和身份再重试，受扰计时保留并作废。

## 交付

采到 trace 时，单卡和多卡均按[模块报告共同结构](../../../tools/profile/MODULE_PROFILE_REPORT_TEMPLATE.md)及[主体计算表约定](../../../tools/profile/CORE_COMPUTE_TABLE.md)呈现；多卡再增加[通信分析](../../../tools/profile/PARALLEL_REPORT_TEMPLATE.md)，保留原有计算、搬运、空隙和同步分析。提供一个已核可访问的 TensorBoard 原始入口，详细证据放 facts。

仅 host 观测或单独算量时，交付实际产生的 session/calls 或统计产物及其范围；不生成缺少时间线依据的 GPU 分析或 TensorBoard 入口，也不为填满模板追加采集。

区分工具事实、Agent 归因、任务补写接线与实际验证范围。新模型需要的捕获/重建脚本由任务补齐；稳定可复用能力随所属代码交付，不承诺任意模型零接线重放。
