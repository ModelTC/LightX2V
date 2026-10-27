---
name: benchmark-core-ops
description: 比较 LightX2V 核心算子的生产后端和执行配置，包括 GEMM、MoE、卷积、稠密/稀疏 attention 与独立 SP attention；用于固定输入的算子性能调优和候选覆盖核查。完整模块的 TP/SP 方案比较使用 benchmark-module-parallelism。
---

# 核心算子性能比较

比较对象是一次完整生产算子调用。SP attention 包含实际 attention、布局变换和通信，作为独立算子调优；完整模块的 TP/SP 比较按需复用这里的 SP 流程。

先读[工具导航](../../../tools/AGENTS.md)与[Benchmark Agent 指南](../../../tools/benchmarks/AGENTS.md)。命令、输入合同和指标查[README](../../../tools/benchmarks/README.md)，不另维护后端名单。

## 选择入口

| 目标 | 工具 |
|---|---|
| 单卡 GEMM、MoE、Conv2d/3d、dense/sparse attention | `tools.benchmarks.operator_bench` |
| 独立 SP attention 的后端与执行配置 | `tools.benchmarks.sp_bench`，使用独立 suite 和 `torchrun` |

已知 shape、只需离线检查或算量时，使用 `inspect` 和[共用计算量接口](../../../tools/benchmarks/README.md#计算量统计与真实moe回放)；目标设备性能比较再进入 probe 与正式计时。

现成 SP CLI 的真实 QKV 回放只覆盖稀疏路径；稠密 SP 默认生成输入。稠密 SP 使用固定真实 QKV 时需补任务侧输入接线并验证，不能仅改输入标签声称已经回放。

## 推进任务

1. 沿生产调用确定输入、精度和计时范围，选择代表性固定输入并生成 suite。Sparse 使用进入 backend 前、已完成模型原有 RoPE 等变换的真实 QKV；SP 还须核完整全局 pre-SP 边界及 replicated auxiliary 语义。MoE 的真实输入捕获和模型特有布局按指南在任务侧准备，不能用直方图冒充真实路由。
2. `inspect` 后在目标环境 probe 当前生产 registry、依赖和输入支持。按请求与预算选候选，保留纳入、拒绝、失败、未测及待补接入；新生产入口缺 adapter 时沿真实调用补齐或给出具体缺口，不把未接入当性能落选。
3. 固定输入和其余条件，同进程预热并完成必要库 autotune，再以工具的 event-list 计时。计算量取共用 core/backend 事实，正式计时关闭统计；路由、量化与布局是否计入以真实生产调用为准。
4. 先比较已测有效稳定项，再按疑点补最小诊断。SP 排名使用完整路径 L1；head 流水需要解释时采真实时间线，不用只支持 bulk 的分量模拟。数值验证独立组织，固定负载或输出有限不证明数值正确。

执行和恢复遵循[资源指南](../../../tools/runtime/AGENTS.md)及 Benchmark 指南：核授权与资源，落实同进程预热、期限、动态监测、受扰作废和有界重测。中断先核进程与已有产物；`--append` 跳过完整错误行，失败项另开恢复目录，不删除原错误。

## 交付

按[算子报告模板](../../../tools/benchmarks/REPORT_TEMPLATE.md)生成中文报告，保留英文 operator/backend 名称、输入与计算量、最快完整配置、覆盖表和原始 JSON 入口。SP 结果按度数分组；dense、sparse、SP 独立排名。

报告已有数值证据及其范围，最快项不等于质量批准。命令、suite、输入/源码身份、日志、退出与资源核对保存在当前任务记录；已有证据仍适用时复用，不因报告整理重跑矩阵。
