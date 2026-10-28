# 层级并行诊断报告模板

由Agent读取 `parallel_profile_v1` 的facts.json，按以下顺序生成中文Markdown。数值来自实际capture，不重建时间线、计算量或通信关联；这些由工具计算。保持正常完整层bench与辅助profile分开。整体结构遵循[模块报告共同结构](MODULE_PROFILE_REPORT_TEMPLATE.md)，多卡不得省略辅助计算/数据搬运、空隙/同步分析。

```markdown
# 层级并行 · 辅助 profile

范围：{identity.target.name}；{真实/合成}全局输入；只解释已采集的{rank_count}个rank。
{共同run_id；全缺时提示需外部收据核同次，不能推跨rank关键路径}

配置：{实际卡数及TP/SP组}；{attention算法/backend与稀疏率}；{SP方法、head_parallel/group_size、通信dtype/FP8开关、tensor fusion与backend}。从实际配置/capture恢复，缺失则说明，不据度数猜测。

硬件：{hardware.name}；名义dense峰值：{hardware.nominal_dense_peak或缺可信值}。
这份报告解释一次采集，正式完整层延迟另看bench。联合统计会扰动CPU/GPU空隙，不拿它排名。record_shapes默认关闭，计算shape来自同次core事实。

| rank / 设备 | TP / SP | GPU跨度 ms | 主体计算 ms（占比） | 辅助计算 ms（占比） | 拷贝 ms（占比） | 填充 ms（占比） | 通信 ms（占比） | 计算/通信重叠 ms | 可见空隙 ms（占比） |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| {rank/device} | {实际组度数} | {span_ms} | {core_ms / 占比} | {auxiliary_compute_ms / 占比} | {copy_ms / 占比} | {fill_ms / 占比} | {communication_ms / 占比} | {core_communication_overlap_ms} | {observed_gap_ms / 占比} |

{仅当存在时列统计会话failed/采样上限等coverage_gaps，附scope和scope_call；勿隐去不完整统计}

各占比均为本类GPU覆盖时间除以当前rank的GPU跨度，格式为毫秒（百分比）；计算/通信重叠只列毫秒。各类取各设备活动并集，类别可能重叠，占比不要求相加为100%；可见空隙不是整卡物理空闲。辅助计算是排除核心和通信后的GPU kernel，含量化、路由、布局及未识别kernel；拷贝/填充分别为trace的gpu_memcpy/gpu_memset，kernel实现的数据移动仍属辅助计算。三类分别取并集，不用总时间相减估算；兼容字段other_ms保留三类合并后的活动并集。

## 实际本地核心计算

{按[主体计算表共同约定](CORE_COMPUTE_TABLE.md)生成；只在七列共同表头左侧追加rank列。保留kernel、名义峰值比例、计算/累加精度。}

## 辅助计算与数据搬运

{按共同结构展示auxiliary_activities，排除通信；保留名称/次数/覆盖时间及rank。}

## 空隙与同步分析

{消费各rank干净采样的timeline.gap_diagnostics；列主要同步等待、空隙交集和上层入口，再解释当前证据。}

## 通信分析

{从实际metadata解释TP/SP组，说明每rank调用数及用途；组不是两两GPU连接列表。选代表rank展示须明确rank，并核对其余rank覆盖。业务用途由实际owner和源码归因，不伪称工具自动阶段识别。}

<details><summary>逐次通信payload及包装自身活动</summary>

通信来自同次显式关联的record_param_comms；时间为关联kernel活动并集，一调用只记一次payload。逻辑输入GB/s含等待，不是NVLink/网络线速。

| rank / 调用序号 | 业务用途 | collective / 实际组 | dtype | 输入 / 输出 MiB | GPU ms | 逻辑输入 GB/s |
|---:|---|---|---|---:|---:|---:|
| {rank / host顺序} | {实际owner/源码归因；缺失留未知} | {collective / group_ranks} | {dtype} | {input_bytes / output_bytes} | {gpu_ms} | {logical_input_gbps} |

嵌套包装自身活动（仅有wrapper_own_activities时展示；已扣子owner及通信，不分摊父计算量，不声称全是通信辅助）：

| rank | 包装入口 | 自身GPU活动 ms |
|---:|---|---:|
| {rank} | {backend} | {own_activity_ms} |

</details>

{有communication_binding_gaps时说明项数和原因位置}
[TensorBoard原始入口]({一个实际可访问trace路径；其余rank路径见facts.json})
```

主体计算的列、单位、小数位、缺失值、排序和跨rank合并条件只维护在[共同约定](CORE_COMPUTE_TABLE.md)，不在并行模板另定一套。bytes转MiB除以2²⁰；跨rank不相加，不能按核名猜计算量。

time_basis须标明：`compute_kernel_group`→主体，`whole_operator_including_auxiliary`→含辅助，`whole_operator_including_lora`→含LoRA且效率留空，`compute_range_including_auxiliary`→专家阶段含辅助、不含阶段外路由合并。不同口径不能合行。

TensorBoard链接先确认文件存在且开发环境可访问；path用URL编码保留 `/` 和 `:`（例如Python `urllib.parse.quote(path, safe="/:")`），空格/括号不能原样拼接。只给一个原始入口，不造失效服务URL。完整逐次事实、缺口、源码/输入身份保存在facts.json，正文不堆路径列表。
