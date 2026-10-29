# 最终算子性能报告模板

由Agent读取本次`report.json`生成最终报告；工具的`report.md`只是简表。模板没有运行时解析器，不向其中嵌入报告程序。使用下面版式，替换`【占位项】`；按输入分组重复，未记录的事实留“—/未记录”。保留工具JSON，最终报告保存到本次产物目录的`REPORT.md`。

```markdown
# 算子bench报告

硬件：**【GPU型号】**，SM【capability】。

名义dense峰值：【profile已记录的各精度、算力与单位；有累加variants时注明】。来源：【厂商来源】。

【组数】组固定输入；候选覆盖【完整/尚未完整】。
延迟为各轮平均耗时的中位数，括号为轮间极差，单位ms；准备和预热不计入。
粗体表示本轮有效稳定项中计算效率最高者，不表示质量批准或建议采用。

## 【类型；仅convolution追加 · conv2d/conv3d，SP追加 · SP=度数】

输入：【shape；保留英文字段名】。
计算量（每次调用）：**【实际计算量范围与GFLOPs/TFLOPs，整数用GOPs/TOPs】**。
【稀疏组改列dense名义计算量、配置稀疏率/保留率、理论稀疏量；固定回放注明来源】

| 后端说明 | 延迟/ms<br>中位数（极差） | 计算效率 | 名义peak算力 | 峰值比例 | 数值差异 | 异常 |
|---|---|---|---|---|---|---|
| 【后端；必要算量/密度用<br>换行】 | 【延迟（极差）】 | 【效率/单位】 | 【peak/单位】 | 【比例】 | 【已有数值证据或—】 | 【失败/实际0%稀疏等异常或—】 |

<details>
<summary>精度与失败明细</summary>

| 后端 / 完整配置 | 输入 / 名义峰值基准 / 累加 | 数值验证 | 失败阶段与原因 |
|---|---|---|---|
| 【canonical名称】 | 【三个精度】 | 【未执行则独立验证】 | 【具体阶段、错误类型和原因】 |

【需要时补输入来源/捕获点/身份、历史参考范围与数值状态、缺失事实和误差解释。】

</details>

<details>
<summary>候选覆盖（完整选项表）</summary>

| 选项 | 本轮状态 | 具体原因 |
|---|---|---|
| 【已知选项】 | 【实测/待测/硬件或依赖或输入拒绝/待补接入】 | 【原始事实】 |

</details>

【有影响判断的发现时，表后用一两句说明；长归因/恢复动作放折叠，不逐行复制套话。】

计算量按每次调用，乘加计2次；计算效率=计算量÷实测延迟，名义dense峰值比例不代表物理利用率。
稀疏效率前项为实际，后项*为dense等价，可超过100%；QK/PV不含路由、softmax、量化。

<details>
<summary>原始数据</summary>

完整事实：[report.json](report.json)。

</details>
```

## 字段和显示约定

- 主视图先性能：开篇只保留硬件/峰值/来源、组数/覆盖、计时/高亮说明；每组的输入、计算量、至多一句输入来源后紧接主表。hash、capture字段、重复的验证规则、审查过程留折叠，不在表前展开。
- 耗时、极差、计算量和计算效率默认4位有效数字（如108.5 ms、68.74 TFLOPs）；百分比2位小数，主表peak沿用3位小数。接近结果需要额外精度时针对差值/波动解释，不全表加长；完整精度留JSON。
- 后端名、实际计算量和实际块稀疏率用`<br>`分行，不用长分号串；`specialized/library`显示为“专用入口/外部库入口”，canonical名称在明细保留。异常列只放失败、缺最终区域、实际0%稀疏等影响判断的事实，不放内部状态码或“为什么保留某字段”的说明。
- 类型由`operator_family`决定，不用难懂的case名称当章节标题；输入、`source/replay`与模型代表性分别说明。operator/backend名称保留英文。`sla.sage2.general`显示为`SLA + Sage2（General）`；其余稀疏名称同样按“算法 + 实现（入口）”。常用显示名：`flash_attn2/3/4`为FA2/3/4，`sage_attn2/3`为Sage2/3，`torch_linear`为Torch（直接MM），`Default`为Torch（生产MM Default）。
- 硬件与各已知精度的dense峰值读取`hardware`，不能只列主表使用的精度；不从文件名猜设备，不联网值替换本次已经匹配的peak。缺失则说明仅报告绝对耗时和效率。
- 单卡各行来自`workloads[].backends`：`latency_ms/spread_ms`、`rate/rate_unit`、`nominal_peak_rate/nominal_efficiency`；实际计算量用`work_range`，按输入下方一行展示；各后端不同时追加在后端说明中。缺失/不匹配的`peak_status`在异常列说明；精度读取`precision/peak_family`，`unknown/backend_default`等标未知。
- 稀疏用`dense_equivalent_work_range`和`sparse.keep_ratio`列名义/理论量。`block_density_actual_range`给实际块稀疏率，不能据此反推有效token对。效率与峰值比例分别并列`rate / dense_equivalent_rate*`和`nominal_efficiency / dense_equivalent_peak_efficiency*`；实际量缺失时两项均留空，说明未观测最终区域。
- 同一输入按延迟排序。至少两个`accepted`且有计算效率的项才对最大效率项加粗（后端、耗时、效率三格）；并列最大均高亮。不把不可执行、未知计算量或不稳定项标为最快。
- 覆盖必须包含已知catalog、未测、硬件/依赖/输入拒绝、失败、`excluded_production_backends`、`coverage_blockers.unmapped_production_backends`及`integration_gaps`，保留具体原因；`production_mappings`注明共用生产路径。没有catalog不声称完整。support仅为错误阶段，不据此认定输入被拒绝；CUDA不可用先视为设备问题。失败行简述阶段/错误，不再追加“参考未覆盖”等无关噪音。
- 有候选遍历或分阶段筛选时，在性能主表后按[遍历范围说明](SEARCH_SCOPE_TEMPLATE.md)解释固定条件、组合维度、支持筛选与实测入选过程，文字说明规则和原因，表格对账选项与数量。
- 单卡没有自动数值比较，数值列留“—”；历史JSON若已有correctness比较，在折叠明细保留原参考范围、误差与状态，注明是历史证据，不重算参考。不将finite/shape当数值正确。SP按AGENTS核对独立reference的输入、配置、实际seed与环境后，单独注明已验证范围并引用其文件；无匹配记录留空，不把run的结构检查或旧容差写成数值通过。推荐筛查和恢复判断遵循AGENTS。
- SP章节先用`measured_winner`及其完整`candidate`总结本轮稳定项最快配置（算法、attention backend、通信精度、fusion、head组/尾组/prepost/A2A、耗时）；不把部分覆盖写成正式采用。SP计算量/块密度读取candidate的`work_range/dense_equivalent_work_range/scheduled_tile_work_range/block_density_actual_range`，采样范围及缺口读取`compute_samples`；不能从速率和延迟中位数反算。SP主表用“并行方案 / attention实现”替换“后端说明”（共八列），只列有计时项，其余进完整覆盖表。Dense效率取`aggregate_effective_tflops`；sparse仅用已观测的`actual_tflops`，缺实际路由时实际/等价效率均留空，不用配置估计代替。峰值分母匹配设备数及attention输入/计算精度，不能因通信为FP8就换FP8分母；缺少对应事实留空。leaf、SP exclusions与blockers同样列入覆盖。详细JSON语义见README的SP章节。

交付前逐行核对输入、后端、数值、单位、缺口与JSON，同时确认首张表前没有审计长文、数字简洁、单元格分层。报告只保留一个原始JSON入口；验收USAGE/过程留本地工作包。只改排版和解释，不改事实。独立复核有新疑点时补最小验证，不因重排报告重跑GPU。
