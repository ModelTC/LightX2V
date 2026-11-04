# 完整层报告模板

由Agent读取 `module_benchmark_v1` 或 `module_comparison_v1` 的JSON，替换下方占位符并展开行；不再实现一套渲染器。保留主表顺序，缺字段用“—”并说明原因。量级、单位和结论必须可回查JSON。

## 单方案

此页是无 profiler 的完成延迟、显存与数值摘要，不解释层内时间分布。开发者需要“单层性能分析”且已有 profile 时，将摘要放在模块分析页开头，紧接干净时间线和独立联合计算量；保留各次测量来源，不把摘要另列为内容相同的分析入口。

```markdown
# {target.name} · 完整层 bench

硬件：{hardware.name}；输入：{真实/合成及来源}；{rank_count} 个rank。
名义dense峰值：{hardware.nominal_dense_peak；缺可信值则仅分析绝对耗时}。

**预热后完成延迟 {mean:.3f} ms**；中位 {median:.3f} ms；标准差 {stdev:.3f} ms。

每次迭代取各rank同步完成host wall最大值，含目标的通信/全部stream完成等待；barrier、输入恢复和数值检查在计时外。

| rank | TP / SP | 峰值 / 结束显存 MiB | 峰值增量 MiB | 声明范围输出比较 |
|---:|---|---:|---:|---|
| {rank} | {实际组度数} | {measurement_peak / end} | {peak_increment} | {数值状态、NRMSE} |

显存为各rank CUDA allocator的allocated值；增量相对同进程预热/reset后的基线，不相加为整组峰值。
数值参考和输出还原由模型侧负责；非零差异不是产品质量批准。辅助profile另跑，不进入上述计时。

<details><summary>完整身份、实际分组、误差与显存</summary>

{identity、parameters及ranks的原始字段；包括比较scope、reference身份、max_abs和reserved}

</details>

原始结果：[report.json](report.json)，逐次数据在同目录raw.jsonl。
```

数值状态映射：`strict_equal`→严格一致，`pending_review`→数值差异已记录，`unavailable`→参考缺失；有NRMSE时追加百分数，仅 `screen_exceeded=true` 时粗体提示“超出推荐筛查值”。没有误差值不能写零。显存bytes转MiB除以2²⁰。

## 多方案

保持相同硬件/来源说明，标题注明实际卡数与范围，例如“固定配置下的四卡 TP/SP 比较”；明确同一全局目标、输入、权重/源码与硬件，无profiler，仅限当前层和输入。使用 `compare(reports)` 的 cases：

SP配置发生变化时，主表追加`SP配置`列，使用实际candidate_id字符串（从factory.config或capture恢复），说明其字段；不能只列TP/SP度数。比较任务未指定attention时基线为FA2，允许明确指定其他生产后端，已有数据保留真实后端。

主表前写清共同固定项；不同的项放入逐方案配置表：attention 算法/backend及稀疏率、SP 方法、head_parallel/group_size、通信 dtype/FP8 开关、tensor fusion、通信与 pre/post backend，以及 TP 实现。来源是实际配置、任务 factory 和 capture；未知留空说明，不能从 TP/SP 度数推断。只测度数组合就明确未搜索这些配置，不称全局最优。

同卡数方案组成同一排名；单卡参考单独列出，不能混入“四卡”主表。仅能给出该固定配置与输入下的本轮最快方案。

| TP | SP | 卡数 | 平均 / 中位 ms | 标准差 ms | 单卡峰值 / 结束 / 增量 MiB | 最大 NRMSE |
|---:|---:|---:|---:|---:|---:|---:|
| {tp} | {sp} | {rank_count} | {mean / median} | {stdev} | {各rank allocated最大值} | {max_normalized_rmse} |

延迟列粗体标本轮均值最短的全部并列项；差异列只在screen_exceeded时粗体警示。任一rank unavailable则明确“参考缺失”；误差缺失留空，不拿其他rank替它背书。不同卡数的资源消耗不同，不从整层延迟推通信带宽。

涉及配置遍历时，在主表后按[遍历范围说明](SEARCH_SCOPE_TEMPLATE.md)展开固定条件、组合维度、支持筛选及阶段入选过程；明确SP算子首筛、稳定性/数值补证与实际完整层测量各自的数量和计时范围。

折叠“各方案实际分组、数值范围和测量参数”：逐方案列parameters；逐rank列真实TP/SP成员、check.scope、max_abs、数值状态及参考身份。正文仅保留一个可访问的report.json入口；源报告路径/哈希由任务保存，不杜撰链接。报告不自动批准质量或改变默认。

折叠明细的表头保持：

| rank | 实际TP组 / SP组 | 比较范围 | 最大绝对差异 |
|---:|---|---|---:|
| {rank} | {tp_ranks / sp_ranks} | {check.scope} | {check.max_abs或—} |
