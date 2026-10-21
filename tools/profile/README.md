# 模型范围 Profiler

回答一次真实函数/模块的时间花在哪里。算子多后端竞速使用 `tools/benchmarks`；本工具产生诊断时间线、同次计算量和中文报告，不提供正式请求吞吐或后端排名。

## 最短路径

入口必须在**同一进程**实际调用目标至少 `skip + warmup + count` 次。工具不重放目标，也不替入口编造预热。

```bash
# 先看正常时间线；默认scheduler warmup=1，active=1，record_shapes关闭。
python -m tools.profile.capture --target my_model:Block.forward \
  --output /tmp/profile-clean --warmup 3 my_entry.py --entry-option value

# 需要同次shape、计算量、实际路由和kernel对应时，单独做联合诊断。
python -m tools.profile.capture --target my_model:Block.forward \
  --output /tmp/profile-facts --warmup 3 --workloads my_entry.py --entry-option value

python -m tools.profile.trace_report /tmp/profile-facts/*.pt.trace.json.gz \
  --output /tmp/profile-report --peaks /path/to/peaks.json --platform my_gpu
tensorboard --logdir /tmp/profile-facts
```

TensorBoard 使用 PyTorch Profiler 插件（`torch-tb-profiler`）。也可在 VS Code 的 TensorBoard 扩展中选择同一个 logdir。文件存在、插件解析成功、IDE 实际打开是三个不同证据。主要原始文件是 `.pt.trace.json.gz`，其内容为 PyTorch/Kineto trace；兼容旧插件的 runtime 分类转换保留原分类。无需搬运到本地浏览器。

目标格式为 `module:function` 或 `module:Class.method`，支持普通同步 Python 函数/实例方法。临时包装在退出/异常后恢复，不修改模型源码；目标定义与 `__main__` 入口须分开，同文件会在执行入口前拒绝，可用导入目标的独立 driver 或 API。异步、生成器、静态/类方法描述符和已缓存的函数别名不由此 CLI 接管。`--module` 表示入口是模块名；入口的参数放最后。需要更精确的调用位置时使用 API：

```python
from tools.profile.capture import Capture

with Capture(output, enabled=True, warmup=3, count=1, workloads=True,
             metadata={"description": "模型、权重、固定输入及精度来源"}) as capture:
    for request in real_requests:
        result = capture.call(model.forward, request)
```

`Capture` 默认 `enabled=False`，直接执行原函数。模型内置 warmup 准备模型，profiler scheduler warmup 准备本进程的采集器，两者用途不同；没有跨进程 cache 预热保证。进入 active 前完成一次当前 CUDA 设备的预热队列。`--synchronize-cuda` 才在每次**整个目标**前后同步当前设备；它不逐 kernel 同步，但可能改变与范围外工作的重叠。多设备队列需要调用方明确边界。

`--observe-only` 只记录 host 调用，不启动 profiler 或统计；异步 CUDA 的 host 返回不等于 GPU 完成。`--stats-only` 只采计算量，无 scheduler warmup；入口可自行先预热。`--record-shapes` 只供明确的 ATen shape 调试，默认关闭，报告不从该字段推算计算量。`--stats-max-calls` 默认每类 backend 4096 条，超过上限有 coverage gap。`--moe-distribution`只额外保存生产专家分布；稀疏实际区域随计算量统计自动取得，导出需显式callback；总量仍复用共用算子数学。

## 产物与范围

- `session.json`：状态、scheduler、调用/失败计数、加载的 Python 源身份和 metadata；`calls.jsonl` 是逐次 host 边界。
- 一份 TensorBoard trace：只包含 active 周期。调用不够为 `capture_incomplete`，CLI 非零退出，已有失败事实保留。
- 联合或 stats-only 模式下的 `call_*.stats/`：共用 core/backend 计算量、实际区域及 coverage。联合采样给真实 apply 唯一标记；形状、实际 route 与该次调用的 kernel 沿标记/CPU 嵌套/correlation 关联。core_ops 写盘移出核心 apply 标记；apply 内的 backend 区域统计、workloads 写盘及拷贝/同步仍可能影响采集，所以必须与干净时间线区分。
- 分析目录的 `report.md` 和 `facts.json`：前者首屏看总体/热点，后者保留完整事件下标、关联、精度证据与缺口。`--scope` 精确选 annotation，`--occurrence` 选其第几次；GPU 异步尾部按关联保留，不裁到 host 结束；其他scope中存在的唯一标记不计作本范围缺口。

统计会话的`status/coverage_gaps`会保留到普通及并行分析的facts，并在存在缺口时提示采样覆盖。达到采样上限、未接入等均不表示算子没有执行；有效样本只解释已观测调用，不外推完整调用分布。

统计覆盖实际 core-op 和已接入的 backend 统计入口，不覆盖任意 raw ATen 或编译图内部。eager 联合统计不改变生产 compile 默认；需要编译性能时采干净编译时间线，另外 eager 查计算量，不能将两次运行冒充逐次绑定。CUDA graph 内部缺少事实时也不能由名称补造。

同一调用多个计算 kernel 合并覆盖时间，计算量只列一次；无法识别主体 kernel 时仍展示实际 shape/计算量，以“含辅助”注明包含拷贝/填充的完整算子活动并集分母。LoRA激活时主线性计算量不含其两次GEMM，尚未单独定位主GEMM时按完整owner展示耗时、效率留空。完整owner计时不改变kernel身份：布局转换、逐元素和量化仍列辅助；卷积的implicit-GEMM属于卷积主体。组合父子只发布实际叶子计算量；父完整逻辑量仍见原始统计，其自身辅助保留在成本表。无法拆分的父额外计算只报时间，不从完整父量推效率。稀疏只按真实最终 mask/ranges 算 selected logical，scheduled tile 包含可证的 padding；dense-equivalent 只算 QK/PV。均不含路由、softmax、量化，也不声称是物理指令数。外部 Sparge 等未暴露最终区域时，实际效率留空；第二项 `*` 是稠密等价效率，不是硬件利用率。

已核实的 `lightx2v::dynamic_sparse_sage2` 内，GEMM 用于生成 SLA 路由图，归入辅助成本并保留 GEMM 家族；其计算量未采集，不计入 QK/PV 主体效率。不能将此规则推广到任意 attention 内的 GEMM。

MoE 专家量不含路由合并。已接入的 owning backend 用同次 `compute_trace_marker` 标记真实专家阶段，阶段外 combine 不再按 GEMM 名字混入专家主体；完整 owner 成本仍保留。缺少或无效专家范围时只给完整 owner，专家精度/效率留未知。阶段内主体不全可识别时按整阶段计时并标“专家阶段含辅助”。`gpu_groups.work_role` 区分核心、辅助、归属待核，保留原 native/kernel 家族；普通与并行消费者使用同一角色事实。

单卡和多卡报告的章节遵循[共同结构](MODULE_PROFILE_REPORT_TEMPLATE.md)，多卡增加通信，同时保留辅助计算/数据搬运、空隙/同步分析。

单卡和多卡报告的主体计算表共用[呈现约定](CORE_COMPUTE_TABLE.md)，固定列、单位、数值格式和合并条件；多卡只追加rank及允许的范围展示。

## 如何读时间和精度

GPU活动并集：合并重叠区间，只计一次，包含 kernel/拷贝/填充。

GPU首末跨度：首个到最后一个关联活动的时间，包含中间间隙。

可见空隙：上述跨度内，本 trace 没有显示本设备任何活动的区间。范围外但同期可见的活动不会误算为空隙；trace 看不见的其他进程仍需外部动态监测。各分类覆盖可能重叠，不相加。CPU 自身时间扣除同线程子区间并集，不扣异步 GPU，也不把等待当计算。分类结合真实native关联与kernel名称，不依赖计算量样本；无法识别的活动注明类型待识别，保留原因并汇入其他，不因此发布FLOPs。

同步 API 的等待、上层调用和空隙交集用于定位；不能仅凭 `cudaStreamSynchronize` 很长判它多余。高空隙先比较干净/联合采集，检查同进程预热、CPU准备和异步队列，再查生产依赖。profile 开销不进入正式 benchmark 的 event-list 结果。

精度分开保存实际输入/输出、名义计算/累加、部分归约许可。custom GEMM/FP4 类型和 PyTorch 原生 GEMM 合同在 `operator_facts.py` 集中，消费实际关联的CPU native名称及kernel类型；外层BF16可以经过量化，不能替代真实GEMM精度。联合采集在每次core入口前后读取设置，不拿预热前快照冒充active事实；缺失或变化时不应用这份设置推断名义累加。原始设置只存于该 core 的事实，不把配置的归约许可再发布成实际库行为；未核实的 cuBLAS 私有短名不作为精度合同。边界设置也不证明opaque内部所有部分归约，未知保留。只有已核实的版本/API/类型才发布，输入 BF16 不自动证明全程严格 FP32。`--library-log-dir` 在尚未 import torch 的新 CLI 进程打开 cuBLAS/cuBLASLt API 日志，分析时同名参数附上退出后的证据。日志包含预热，是进程级补证，不按时间/shape猜分配到某次 GEMM；API computeType 不证明每条物理指令的中间精度。

峰值复用 benchmark 的显式文件与 SKU/SM/dtype/accumulator 校验；没有匹配文件只报告绝对时间/计算效率。Agent 更换硬件时主动查厂商规格并建档，代码不联网或硬填 H100 常数。GEMM 理想字节量只计一次读 A/B、写输出，未测量 DRAM 流量；理想算术强度=FLOPs/上述字节量，roofline=min(名义计算峰值, 名义显存带宽×算术强度)。它只是理想参照，不能单凭它断定内存瓶颈。

## 源码阅读顺序

`capture.py`（真实调用与scheduler）→ `workload_stats.py` 的显式 trace_ranges（核心标记）→ `trace_report.py`（关联/并集/中文界面）。精度规则单看 `operator_facts.py`，库日志单看 `library_precision.py`；原 `TorchTraceProfileContext` 不变，原 `analyze_torch_trace_gap.py` 仍可单独使用。

## SP 算子的真实流水

已有 SP benchmark 选项可以直接组合采集 API，无需复制 Ulysses/Ring 流水。由 `diagnostic_configuration_from_report` 核对 suite、实测候选和基础 seed，再从当前候选目录取得完整配置。它不代表产品采用决定。

```python
from tools.benchmarks.sp_bench import _prepare_operation
from tools.profile.capture import Capture

actual_seed = base_seed + repeat * 1000
fn, dense_equivalent_flops = _prepare_operation(case, candidate, device, actual_seed, rank, inputs=inputs)
with Capture(rank_output, enabled=True, warmup=warmup, count=1,
             workloads=joint_statistics, metadata=metadata) as capture:
    for _ in range(warmup + 1):
        result = capture.call(fn)
```

`case/candidate/inputs/device/rank` 均沿用生产准备，保留实际 `head_parallel/head_parallel_group_size`，包括非整除尾组；不要改用只支持 bulk 的组件模拟诊断。每个 rank 采集自己的目录；metadata 的 identity.target 包含完整候选配置，另记录共同批次、suite/case、输入文件与源码身份、基础及实际 seed、repeat 和真实组。完整 SP 入口是组合边界，计算量来自实际本地叶子调用，不能把返回的稠密等价量分给每个 kernel。时间线采集先关闭统计，必要时另跑同配置的联合统计。

## 层级并行的辅助报告

正式TP/SP完整层延迟由`tools.benchmarks.module_bench`独立计时。解释某个方案时，沿同一PreparedModule真实call分别采各rank，Capture.metadata保存`identity`（共同全局目标/输入/源码）、`rank`、实际`tp_ranks/sp_ranks`，共同采集批次`run_id`及`measurement_ranks`；None的并行维度记录当前rank的单元素列表。每个目录只采一个active调用，先干净时间线，再按需另跑联合统计。

```bash
python -m tools.profile.parallel_report \
  /path/to/capture/rank0 /path/to/capture/rank1 \
  --output /path/to/new-parallel-profile \
  --peaks /path/to/peaks.json --platform actual_platform
```

输入须为同一次同配置的各rank目录，session与trace内嵌元数据一致，不重复rank或混合不同全局身份/TP/SP度数。各rank的run_id须相同；旧采集全部未记录run_id时保留缺口，是否同次由外部运行收据核对，不据此推跨rank关键路径。只传部分rank时只能解释已采部分，不能当跨rank完整覆盖。CLI仅输出并打印`facts.json`路径；中文展示由Agent按[并行诊断模板](PARALLEL_REPORT_TEMPLATE.md)生成，不再自动写Markdown。事实包含每设备活动并集/空隙、核心/通信/辅助计算/拷贝/填充活动及核心通信重叠、实际局部shape/计算量/计时口径/计算效率（未识别主体时标含辅助，LoRA效率留空）、逐次通信和包装自身活动。`collectives`按`host_start_us`排序；`auxiliary_activities`保留排除核心/通信后的逐kernel/拷贝/填充明细。范围耗时与诊断扰动沿用上文定义，不与无profilerbench混用。`auxiliary_compute_ms` 为排除核心和通信后的kernel并集；`copy_ms`/`fill_ms` 分别为gpu_memcpy/gpu_memset并集，kernel执行的数据移动仍计辅助计算。`other_ms` 兼容保留三类合并并集，各类重叠时不能相加。

通信事实只消费trace中经显式External id/correlation关联的`record_param_comms`，不按时间邻近配对，不新增生产hook。输入/输出字节由本次nelems与dtype计算；一调用多kernel只记一次payload，GPU耗时用其活动并集。缺明确关联、未知dtype或元素数保持缺口。逻辑输入GB/s=本次输入字节/通信GPU活动时间，含等待，**不是物理NVLink/网络链路带宽**；不从整层时延、跨rank时间和推算线速或物理峰值比例。原始分组、split和通信参数保存在facts，复杂算法物理发送量不猜。

识别优先查显式关联的CPU通信记录，不要求GPU kernel带有特定backend名称。NCCL名称或GPU事件的`Collective name`仅作为缺少关联时的通信线索；有线索但无法唯一配对时保留活动与缺口，不猜payload或带宽。没有通信证据的陌生活动仍保留在总活动中，不将其猜成通信。这里依赖当前PyTorch trace字段；更换采集格式时须核对实际协议。

嵌套包装按实际同线程core标记将GPU事件归最内层，父包装自身活动排除子owner和通信事件；不将父inclusive计算量除以剩余辅助时间。这些自身活动可能与子调用并行，也可能有未单独标记的计算，因此不声称全是通信辅助或已定位冗余。需要深入时沿原始trace和模型源码追依赖。
