# 核心算子 Benchmark

这是一套由 shape 驱动的单卡 benchmark，用于在**指定硬件**上回答四个问题：

1. 新模型或新硬件上的核心 shape 大致能达到什么性能；
2. profiler 中记录的当前 backend 是否是同 shape 下的最佳候选；
3. 一个 backend 在理论 shape 网格上的表现，以及与其他 backend 的差距；
4. 最佳实测结果距离硬件名义峰值还有多大 gap。

`operator_bench.py` 处理 GEMM、dense/sparse attention、Conv2d/3d 和单卡 MoE。Sequence-parallel dense/sparse attention 使用独立的
`sp_bench.py` 和 suite。本工具比较算子；完整模型层、TP/SP组合方案和时间线诊断分别使用其对应工具，不混入算子报告。

## 测试类型与候选命名

每种类型独立定义输入、筛选候选和排名。单卡`sparse_attention`不包含dense对照或SP入口；`dense_attention`独立比较稠密后端，`sp_attention`由`sp_bench.py`比较完整分布式attention路径。SP使用的dense/sparse叶子是其配置，不把SP结果并入单卡排名。

| 测试类型 | 候选名称表达的维度 | 命名与显示规则 |
|---|---|---|
| `gemm` | 量化/计算格式、实现库、生产封装或直接调用 | 保留生产registry的`fp8-sgl`等精度-实现名；`Default`显示为Torch生产MM，`torch_linear`为直接MM，二者不冒充两种数学算法。 |
| `dense_attention` | 实现库及代际 | `flash_attn2/3/4`、`sage_attn2/3`、`torch_sdpa`；显示FA2/3/4、Sage2/3、Torch SDPA。Causal、GQA和shape属于输入合同。 |
| `sparse_attention` | 稀疏算法、计算实现、生产调用入口 | ID统一为`算法.计算实现.入口`，如`sla.sage2.general`；显示SLA + Sage2（General）。 |
| `moe` | 实现库、专家矩阵乘组织方式 | `torch_expert_loop`和`torch_grouped_mm`分别表示专家循环与分组MM；top-k、expert配置与实际路由属于输入。 |
| `convolution` | 卷积维数、实现库 | 类型标题为convolution · conv2d/conv3d；共享后端`torch_convolution`，不因维数另造两个库后端。 |
| `sp_attention` | 并行算法、attention叶子、通信精度及执行配置 | Ulysses/Ring与leaf分别展示；候选ID保留通信、fusion、pre/post、A2A及head流水配置以便精确重跑，完整ID折叠。 |

稀疏ID的算法为`sla/sparge`；入口为`dynamic/general/specialized/library`。`specialized`对应SparseSage/SparseFlash专用生产类，`library`对应外部库完整稀疏接口；两者可能调用相同计算库，但路由、布局与调用开销可不同。Dynamic/General/专用类都会在调用内生成mask，不能据名称称其为静态/预计算路由。`triton_ar`是不同计算实现变体，保留该信息。

不在候选名追加`_replay`。回放能力仍由输入manifest和哈希表达，原生产registry key单独保存用于源码追溯。重复入口须核对实际数据流后再合并，不能仅凭同一算法或底层kernel名合并；未接入的适用实现继续作为缺口保留。

## 最小数据流

```text
shape dump / 理论 sweep / 真实 attention或MoE replay
                      |
                      v
                canonical suite
                      |
                      v
                   run
                      |
                      v
         raw.jsonl + report.json/report.md
```

工具没有 model catalog、任务队列、execution plan、bundle manifest 或跨 run 控制面。需要长期保留结果时，直接归档输入 suite、命令、`raw.jsonl` 和报告即可。

## 性能测量与实际输入

默认`repeat_runs=3`、每轮`warmup=10`次、正式`iterations=30`次，`prewarm_seconds=0`；CLI可显式覆盖。各轮在同一进程预热后计时，event-list机制与指标见下文。

单卡bench只测性能，不内置FP32数学参考、误差阈值或比较输出归档。计时前和计时后检查输出类型、有限性与shape；`correctness.checked=false`表示未做数值验证，不表示零误差。独立验证的时机与责任见[Agent指南](AGENTS.md#输入计算量和数值)。

稀疏attention在计时外执行一次生产调用，用共用统计取得其最终区域的实际计算量；正式计时关闭统计，不另跑router，不用块密度估算有效token对。缺少或观察到多个区域时，raw的`work`/`actual_tflops`为null，保留dense等价量和缺口。采样结果只代表该次调用，不保证内容相关路由恒定。

真实QKV和MoE回放用于固定性能负载，加载时核对输入身份。MoE保存activation、专家ID/scale及权重，不要求捕获输出或进行完整数值重建。专家直方图仍是合成路由，不能冒充真实模型输入；普通shape输入明确标为synthetic。不同后端使用相同operands与配置，不仅核对shape/seed。未测、硬件/依赖/输入拒绝及未接入仍分别显示。

## 计算量统计与真实MoE回放

计算量由生产算子基类提供，bench使用同一数学函数；不解析kernel名称，不依赖profiler或`record_shapes`。

`tools/compute_stats.py` 是离线数学转接模块，不是 CLI：例如 `from tools.compute_stats import gemm_flops; gemm_flops(3, 5, 7)` 得到 210。生产公式在 `lightx2v/common/ops/compute_stats.py`；转接用文件加载避免仅算几个乘法却触发 `lightx2v` 的平台/模型初始化。生产统计不反向依赖 tools。attention 的三种区域计算函数在 `lightx2v/common/ops/attn/compute_stats.py`，由 attention 基类公开，内部统一几何和面积→FLOPs换算。

Bench使用源码checkout；`tools`不随LightX2V wheel打包。默认推理与生产统计不依赖`tools`。`WorkloadStats` 是唯一显式入口，`families` 只选择算子类型，不按算法或实现库另开通道；这里的`attention`包含bench中分别比较的dense/sparse类型：

| family | 统计范围 |
|---|---|
| `gemm` | common MM基类，包括其FP8/INT8/FP4等具体实现 |
| `attention` | common dense/sparse attention；最终consumer自动补实际mask/ranges |
| `convolution` | common Conv2d/3d |
| `moe` | 同构专家FC1/FC2及gate的主矩阵计算 |

稀疏区域是准确算量的必要输入，选择attention后自动读取；采样超限时不再归约/导出。Dynamic由实际分支函数记录，General由选定consumer记录；不因算法叫SLA或Sparge而复制算术。MoE总量只需shape/专家配置；`moe_distribution=True`才额外读取生产专家分布，原始输入导出由callback单独开启。

边界dtype是Tensor存储类型，未列的名字原样保存；NVFP4/MXFP4等量化方案、计算/累加精度需backend合同，不能从BF16输入或uint8存储猜出。common之外的独立平台模板、paged decode、任意ATen和编译图内部仍有缺口。HF量化Linear的`nn.Module.forward`尚未共享接入；直接SGL调用在执行时明确报告绕过共同MM基类，不猜计算量或增加专用通道。普通SGL MM仍由基类统计。其他不可见旁路仅声明覆盖边界，不声称自动发现。

稀疏候选从生产路径接入：Dynamic支持SLA的Triton/Triton AR、Sage2/3、FA4、Magi；General支持SLA/SPARGE与Triton、Sage2/3、FA4、Magi、FlexBlock、FlashInfer的组合，另保留已有SparseSage/SparseFlash及外部Sparge入口。Dense后端在独立dense attention测试中比较。实际可尝试集合由架构、依赖及输入筛选，不表示这些组合已在所有设备验证。

默认尝试全部eligible候选，逐轮错开起点；显式`--backend`缩小范围时仍保留未测项。路由和布局变换在生产调用计时内，文件加载及独立统计在外。固定QKV属于输入回放合同，未固定或跳过路由；候选名不追加回放后缀。General与Dynamic分别保留生产入口差异，不因都叫SLA就合并成绩。

覆盖状态区分纳入比较、当前条件拒绝、待补接入或证据。SVG2、Sol-Attn等缺少可复用输入合同的路径保留unmapped；Nbhd、Radial、SVG、RainFusion、Draft等缺少QKV-only suite之外的布局或状态。新mask/operator registry项也会暴露为缺口并阻止完整覆盖结论。具体补输入与接线方向见[Agent指南](AGENTS.md#候选覆盖与补齐)。Magi ranges与变量块可提供实际区域计算量，不因此证明输出数值正确。

| 计算 | 逻辑计算量 | 范围 |
|---|---|---|
| MM | `2 * batch * m * n * k` | 仅矩阵乘；SGL沿用MM公式 |
| Attention | `2 * token_pairs * (qk_dim + value_dim)` | QK/PV，不含路由、softmax或量化 |
| Sparse attention | 上式使用实际选中的有效token对 | sparsity给出固定区域数；mask/ranges给出实际区域；尾块按有效长度计 |
| Conv2d/3d | `2 * output.numel() * product(weight.shape[1:])` | 含卷积padding计算，weight已反映groups |
| MoE | `routes * sum(2 * out_features * in_features)` | FC1/FC2及split gate；含scale为0的实际路由，不含激活、重排与combine |

MoE同构专家无drop时，总量由route数和矩阵shape决定；ID/scale改变expert负载与输出。相同块稀疏率也可能因尾块改变有效计算量。Scheduled tile FLOPs另列，不代表物理指令计数；未知量化布局或未暴露最终区域明确报告缺口。

Dynamic稀疏观测的observed_backend映射到canonical候选ID，production_operator另存实际hook身份；其他尚无映射的入口保留原生产名，不猜候选。输入回放只由replay/来源字段表达。

统计默认关闭。只包住现有调用，不需要改模型各层的缩进：

```python
from lightx2v.utils.workload_stats import WorkloadStats

with WorkloadStats("/path/to/new-stats", families=["attention"], max_calls=1):
    run_target()  # 原生产调用
```

`max_calls` 是每个backend采样桶的保存上限（core按具体类，consumer按operator名），不是推理/warmup次数。100次同类MM、上限1，原计算仍执行100次，只保存1条详细事实；不同权重/shape可能被截断，不能据此推整网总量。同类未返回的父调用也预占名额。整层诊断应按覆盖需要提高上限，单次实际区域统计通常1–2。

产物为`core_ops.jsonl`、中文`core_ops.md`、consumer侧`workloads.jsonl`和`session.json`。后者schema v2记录`families/moe_distribution`及覆盖缺口。parent_call与actual_backend_routes说明归属，父子/consumer可能描述同一工作，不能相加。统计会同步读区域和写文件，只用于独立eager诊断；正式event计时关闭它，compile/CUDA graph capture拒绝，graph replay不可恢复内部调用。

`WorkloadStats(on_moe_inputs=...)`只在显式提供回调时暴露实际MoE输入、路由和权重，保存方式由回调决定。

真实MoE输入准备通过现有`--plugin`与`register_backends(registry)`扩展，不规定公共保存格式。内置MoE adapter在prepare阶段接受`options["moe_inputs"]`，字段为`activation`、`fc1`、`fc2`、`expert_ids`、`scales`，以及可选`fc1_gate`、`fc1_bias`、`fc2_bias`；它们是符合实际production factory及case shape/精度的内存operands，不是序列化协议。继承适用adapter时调用父`prepare(case, device, {**options, "moe_inputs": inputs})`；布局无法复用时返回现有`PreparedOperation`准备真实production调用。

Suite用`replay.manifest/sha256`固定输入身份；文件内容、引用权重及实际tensor与suite的一致性由plugin校验，工具不自动解析MoE manifest。声明replay却未传入operands会报错；无replay的内置测例是显式合成负载。任务侧捕获、身份保存与稳定adapter归属见[Agent指南](AGENTS.md#moe真实输入准备)。

内置准备逻辑继续负责production factory、固定输入调用和FlashInfer计时外autotune。其他量化、分片或专家结构用任务adapter调用其真实backend，不将其强行转换为内置布局。

显式MoE捕获在生产调用前取得operands；与调用者的output缓冲共用storage者先克隆，回调在生产返回后保存实际输入，不保存输出。默认调用不克隆。Torch两候选及FlashInfer候选调用生产MoE。FlashInfer在计时外将split gate合并成value在前、SiLU gate在后的packed FC1，再消费同一真实路由和权重；不改变输入精度。其他量化或多shard布局不猜转换。FlashInfer在prepare阶段显式调用`flashinfer.autotune(tuning_buckets=(tokens,))`，只搜索当前token点，`tune_max_num_tokens=tokens`覆盖当前点。普通warmup不会启动该搜索；正式调用使用`autotune(False, tuning_buckets=(tokens,))`固定相同映射，只查缓存、不做搜索。这样同进程切换token数后也不依赖库共享配置的残留状态；False上下文开销属于本adapter的event-list计时。调优耗时属于setup，raw及报告保存`autotune`合同；未调优/调优记录不可混合聚合。需要安装的FlashInfer支持该autotune接口，不静默退回默认tactic。机制参考[FlashInfer autotuning](https://docs.flashinfer.ai/autotuning.html)。采集、加载和调优在计时外；纯计时仍采用event-list。`source`/`replay`保存在报告JSON；模型代表性取决于实际捕获范围。本工具不保存捕获输出，不据此提供数值正确性结论。

## Shape 合同

顶层最小结构：

```json
{
  "schema_version": 1,
  "kind": "operator_benchmark_shape_suite_v1",
  "suite_id": "my_model",
  "cases": [
    {
      "case_id": "ffn.up",
      "operator_family": "gemm",
      "shape": {"m": 512, "n": 13824, "k": 5120, "bias": true},
      "precision": {"input_dtype": "bf16"},
      "call_count": 40,
      "observed_backend": "Default"
    }
  ]
}
```

每个 case 需要 `case_id`、`operator_family`、`shape` 和 `precision.input_dtype`。可选字段：

- `call_count`：用于估算模型调用加权耗时；
- `observed_backend`：profiler 中实际使用的 backend 精确名称；
- `source`、`tags`：来源信息；
- `routing.expert_counts`：MoE 真实路由直方图。

Family 对应 shape：

- `gemm`：`m, n, k, bias`；
- `dense_attention`：`batch, seq_q, seq_kv, heads, kv_heads, head_dim, causal`；
- `sparse_attention`：与 dense attention 相同，另外要求 `sparse` 和 `replay`；
- `moe`：`tokens, hidden_size, intermediate_size, num_experts, top_k, activation, expert_bias`；
- `convolution`：`input, weight, stride, padding, dilation, groups, bias`；输入为NCHW/NCDHW，stride/padding/dilation为对应长度的整数列表。

当前 attention 合同只接受 fixed-length；`causal=true` 时要求 `seq_q == seq_kv`。Sparse attention 只接受真实 QKV 回放，不提供随机输入或理论 sweep。

`dense_attention`默认使用合成输入；也可添加`replay: {"manifest": "/path/to/qkv.json", "sha256": "<manifest SHA256>"}`，使用下述同一QKV manifest做真实输入计时。所有内置dense后端共用加载器，保留BSHD与batch维度，校验manifest、分片SHA、shape/dtype及完整序列覆盖；文件不存在或不匹配时失败，不退回随机QKV。原backend名不变，dense结果不进入sparse排名；GEMM/Conv等尚未实现回放的类型明确拒绝`replay`字段。来源仍受实际捕获边界约束，leaf输入不因此成为全局pre-SP输入。

单卡`Sage2`使用生产模块已解析的`sageattn`，包括SM89的显式Triton分发；第三方库默认dispatcher不能代替该生产选择。

## 检查

模型侧只需按上述合同导出 JSON。Benchmark 不内置逐模型 trace 解析器；不同 profiler 的转换逻辑应留在模型侧。运行前检查 suite：

```bash
python3 -m tools.benchmarks.operator_bench inspect \
  --suite /tmp/my_model.json
```

检查结果会列出 family/dtype 数量，以及缺少 `call_count` 或 `observed_backend` 的 case。缺失这些可选字段不影响逐 shape backend 排名。

## 理论 Shape Sweep

```bash
python3 -m tools.benchmarks.operator_bench sweep \
  --family gemm \
  --axis m=1,16,256,4096 \
  --axis n=1024,4096 \
  --axis k=1024,4096 \
  --axis bias=false,true \
  --dtype bf16 \
  --suite-id gemm_grid \
  --output /tmp/gemm_grid.json
```

Dense attention 和 MoE 使用各自的全部必填 shape 字段；不提交预生成网格，避免重复维护由这个命令即可生成的数据文件。CLI sweep只接受标量轴，不支持卷积的列表shape；Conv2d/Conv3d使用上述canonical suite及`run`入口。

## Sparse Attention 回放

已核实但由模型延迟导入的生产入口仍出现在候选清单中；报告保留其输入限制，不强制加载无关依赖。GeneralSparse的Nbhd/SVG mask也在覆盖表列出所缺frame/context语义，不将独立attention入口的排除当成它们的替代。

稀疏性取决于Q/K内容，shape加随机输入不能代表真实稀疏负载。回放合同的Q/K/V是进入attention backend前、已完成RoPE等变换的实际输入。捕获点与adapter归属见[Agent指南](AGENTS.md#输入计算量和数值)。

Suite case 需要固定 replay manifest 的哈希，并声明目标保留比例：

```json
{
  "case_id": "model.step20.block20.self_attn",
  "operator_family": "sparse_attention",
  "shape": {
    "batch": 1, "seq_q": 32130, "seq_kv": 32130,
    "heads": 40, "kv_heads": 40, "head_dim": 128, "causal": false
  },
  "precision": {"input_dtype": "bf16"},
  "sparse": {"keep_ratio": 0.2},
  "replay": {"manifest": "/path/to/qkv_replay.json", "sha256": "...64 hex..."}
}
```

Replay manifest 使用 `operator_benchmark_qkv_replay_v1`，layout 为 `BSHD`，并为 `q/k/v` 分别记录 dtype、完整 shape 和连续 sequence shards。每个 shard 必须包含相对 manifest 的路径、SHA256、tensor key、`sequence_start` 和 `sequence_end`。加载时会校验 manifest、所有 shard、dtype、shape 和序列覆盖，随后才搬到目标 GPU。

内置稀疏候选按以下规则生成，不枚举不存在的笛卡尔积：

| 入口 | 算法 | 计算实现 | 数量 |
|---|---|---|---:|
| Dynamic | SLA | Triton、Triton AR、Sage2、Sage3、FA4、Magi | 6 |
| General | SLA、SPARGE | Triton、Sage2、Sage3、FA4、Magi、FlexBlock、FlashInfer | 14 |
| 专用类 | SLA、SPARGE | Sage2、Sage3、FA4 | 6 |
| 外部库完整接口 | SPARGE | Sage2 | 1 |

这是27条已接入的生产调用路径，不代表27种算法、独立kernel或当前设备全部可用。实际按硬件、依赖和输入筛选；稀疏候选目录不包含FA3等dense选项。
`setup_ms`覆盖adapter.prepare及其后的CUDA同步，包含其中执行的artifact加载、校验和显式autotune。prepare之后的独立预检/计算量采样、warmup和首次惰性初始化不计入setup；正式kernel latency只含event-list测量段。稀疏backend的block-map/routing包含在每次计时中。Raw case固定replay SHA与稀疏配置，append与report拒绝身份漂移。

延迟取各轮平均耗时的中位数；spread是轮间极差（最大值减最小值），不是标准差或置信区间。点估计领先不保证显著或可重复领先，也不表示质量已接受。数值未测不等于数值严格一致。

稀疏理论计算量为对应dense逻辑量×`keep_ratio`，不含选块取整、保护区、校准和padding。实际量来自raw `work`及其范围，不由速率或mask块密度反推；缺最终区域时不发布实际计算效率。展示位置、双效率和高亮统一见[固定报告模板](REPORT_TEMPLATE.md)。

单算子默认稀疏率0.8（`keep_ratio=0.2`），suite显式保存，不改生产或SP默认。某family未指定`--backend`时，run探测并尝试全部eligible候选；`--backend FAMILY=...`显式限制后端，不过滤suite中的其他family，也不支持前缀通配符。目标case由suite决定。eligible仅是依赖/架构预检，实际失败单独保存；报告对账保留未测项。SP的`--dense-backend sage_attn2`可替换`flash_attn2`，再派生对应合法配置。候选选择与完整覆盖判断见[Agent指南](AGENTS.md#候选覆盖与补齐)。

LightX2V的`SparseSageAttn2/3Weight`和`SparseFlashAttn4Weight`已有SLA/Sparge两条生产路由。`sla.*.specialized`保留SLA合同，`sparge.*.specialized`选择同一类的`sparge_mode`，共享加载与实际mask统计，不复制路由实现。新增组合尚需GPU验收，SP暂明确排除；外部`sparge.sage2.library`调用生产`SpargeAttnWeight`，再进入`spas_sage_attn.core.spas_sage2_attn_meansim_topk_cuda`，在库内完成选区与计算。无法提供最终mask时仍报告成功耗时；当前该路径无法提供最终区域，实际计算量留缺口，不把缺证据写成运行失败。

Sparse attention 同时输出两套口径：

- `actual_tflops`：以本次实际选中有效token对的QK/PV计算量为分子，扣除尾块padding；不计routing FLOPs，但延迟包含每次调用的原生block-map/routing。缺少实际最终区域时为null，不用块密度估算替代；
- `dense_equivalent_tflops`：以完整 dense work `4*B*H*Sq*Sk*D` 为分子，用于观察同 shape 的等效吞吐和加速收益。

提供硬件 peak profile 后，报告分别输出 `actual_peak_efficiency` 和 `dense_equivalent_peak_efficiency`。两者都使用 suite 输入精度的普通 dense peak，不使用 2:4 structured-sparsity peak。前者使用该候选本次最终mask得到的有效QK/PV计算量；缺少最终区域时，实际计算量及对应速率为null，并记录缺口。它是逻辑计算效率；后者是等效效率，可能超过 100%，不能解释为实际 Tensor Core 利用率。若 backend 内部混合了量化 QK 与浮点 PV 等不同算术，该分母只是统一诊断基准；精确的硬件指令效率还需要 backend 声明计算混合比例。

### Agent 协作入口

用户只需说明模型/场景、目标硬件和希望搜索的 SP 范围，例如：

```text
帮我 dump MiniMax-H3 真实完整 sparse attention 输入，并分别搜索 SP=2/4/8 最优配置。
```

输入选择、QKV dump、候选探测和降级规则由 Agent 按 [`AGENTS.md`](AGENTS.md) 处理；用户仍可选填 step/block、keep ratio 或 backend。Dump 目标始终是原始 token 顺序下的完整全局 Q/K/V，不与采集用了几张卡或最终 `SP` 度数绑定。

## 运行

内置 Torch 基线、LightX2V production MM registry、Flash/Sage attention 和 Torch grouped MoE adapter。常规 per-channel MM adapter 覆盖 vLLM、SGL、TorchAO、Q8F 和 Triton；需要 checkpoint 专用 packing、scale 或校准元数据的 registry key 会在 probe 中带原因排除。以下示例比较 production GEMM backend：

```bash
python3 -m tools.benchmarks.operator_bench run \
  --suite /tmp/my_model.json \
  --output-dir /tmp/my_model_h100 \
  --backend gemm=Default,fp8-vllm \
  --repeat-runs 3 \
  --warmup 10 \
  --iterations 30 \
  --device cuda:4 \
  --peaks /path/to/hardware_peaks.json \
  --platform h100_sxm_80gb
```

`run` 直接生成：

- `raw.jsonl`：每个 case/backend/repeat 的原始记录；
- `backend_catalog.json`：本次硬件与依赖环境下冻结的候选目录；
- `run_summary.json`：本次写入和状态计数；
- `report.json`：完整性能、精度、覆盖与来源事实；`report.md`：CLI简短结果表。Agent按[固定模板](REPORT_TEMPLATE.md)生成面向开发者的最终`REPORT.md`。

单卡报告只给事实：`measured_winner`是已测稳定候选中的最快项，`correctness.status=not_checked`明确未执行数值验证，`coverage_blockers`列未测、缺依赖与未接入；不生成`winner`或`quality_approved`。采用、质量取舍与补测由Agent/开发者据事实决定。缩小`--backend`仍可探索，但不能把subset称为全部后端最优。离线`report`传入本次`--backend-catalog`才能重建完整候选对账。

中断后以相同参数增加 `--append`。工具按 `(repeat, case, backend)` 跳过已有记录，并拒绝算子类型、shape、输入/后端精度、实际路由配置、replay/sparse、测量参数、seed、LightX2V commit、GPU 或 CUDA/PyTorch 环境漂移；标签、来源注释和call_count可更新；这些身份变化时应使用新目录。raw的`requested_input_dtype`保存suite请求，`precision`保存adapter声明；即使后端在support阶段拒绝，原请求身份也不会被声明精度覆盖。

单卡`run`通过旧raw身份校验后、首个case前保存`backend_catalog.json`。已识别的fatal CUDA上下文错误先写入并flush失败行，再非零退出；不生成成功汇总，已有raw可用离线`report`重建，fatal行归`measurement_error`并阻止完整覆盖。普通后端异常仍单独保存并继续。JSONL按物理换行读取，完整末行无换行时append补分隔；不完整行严格拒绝，不自动删除或修补。

`--device`使用当前进程可见的CUDA ordinal；资源核对与受扰测量处置见[Agent指南](AGENTS.md#一次任务怎样推进)。

## Backend

查看内置及指定插件注册的 backend：

```bash
python3 -m tools.benchmarks.operator_bench backends
```

`operator_backends.py` 中的可选依赖只在实际准备 case 时导入；列出 backend 不要求当前环境安装所有 kernel。外部扩展仍可通过 `--plugin module.name` 注册。

更换硬件或更新 LightX2V 后，先在目标设备上执行候选探测：

```bash
python3 -m tools.benchmarks.operator_bench backends \
  --probe --device cuda:0 \
  > /tmp/backend_catalog.json
```

Probe 将 LightX2V 当前 `MM_WEIGHT_REGISTER`、`ATTN_WEIGHT_REGISTER` 与 benchmark adapter 对照，并冻结 `catalog_fingerprint`。每个候选具有以下状态：

- `eligible`：声明的架构与必要 Python symbol 均满足，可以进入真实 shape benchmark；这不是 kernel smoke test，实际编译或执行失败仍由 `run` 记录；
- `unsupported_arch`：该代际不支持当前 CUDA capability，可以合法排除；
- `dependency_missing`：当前架构适用但环境缺少依赖，不能静默排除；
- `cuda_unavailable`：无法探测目标 CUDA 设备。

Production registry 中的新 backend 如果既没有 adapter，也没有带原因的排除项，会进入 `unmapped_mm_backends` 或 `unmapped_attention_backends`，汇总到 `unmapped_production_backends`，并令 `catalog_complete=false`。当前架构适用的候选缺少依赖时，`environment_complete=false`。只有两者均成立时 `coverage_complete=true`。

候选目录随LightX2V registry演进，每次probe的fingerprint标识本次已发现集合，不是代码/依赖身份校验。尚未进入registry或环境的外部kernel无法自动发现，可通过`--plugin`注册adapter。

GeneralSparse组合的descriptor使用`production_backend="general_sparse_attn"`和`production_variant="mask=算法;operator=注册名"`关联生产子项；descriptor只描述接线，不能证明任意算法×实现组合可执行。

单卡入口`operator_bench.py`负责shape、执行、报告和CLI；`operator_backends.py`负责backend合同与实现，`reference.py`仅保留SP显式验证使用的小型输出比较函数，单卡执行不调用它；真实MoE输入准备留任务侧plugin。分布式 SP attention 的进程组、候选约束和计时集中在独立的 `sp_bench.py`，避免让单卡工具依赖 `torchrun`。

卷积覆盖按conv2d/conv3d分别读取生产registry：Default映射到同一路径的torch_convolution，ForceFp32明确要求独立精度/转换合同；新增registry项阻止完整推荐。已知VAE FP8分发通过共享lightx2v_kernel.fp8_conv3d执行（Conv2d转单深度Conv3d），当前缺量化权重、scale、累加精度及硬件接线，作为待补项保留，不猜测硬件拒绝或视为已测。历史catalog没有卷积清单时只给已测排序；未登记的原生分发仍需Agent沿生产调用审计，工具不声称自动扫描任意Python分支。

## Sequence-Parallel Attention

SP候选名称与单卡registry同步；catalog保留leaf的硬件/依赖/接入事实，待验证的SP语义单独记录并使覆盖保持未完整。以下L1报告与诊断仍使用本节既有JSON和`--recommendation-report`入口。

当前 `sp_bench` 支持 Ulysses/Ring dense attention，以及使用真实 QKV replay 的 Ulysses sparse attention；不包含 `kv_all_gather`。Sparse leaf 若不提供 Ring 需要的分块输出和 LSE 合并合同，Ring 会明确记为不支持，不会被当作性能失败。Sparse SP 候选从通用 backend catalog 中筛选；当前已适配 H100 与 Blackwell 的 production sparse leaf，未来出现 eligible 但尚无 SP 合同的 adapter 时会记录为 `unmapped_sparse_sp_backends` 并阻止发布完整候选最优。`sage_attn2` 是低精度 dense attention，不代表稀疏计算。

SP benchmark 使用独立的 `sp_attention_benchmark_shape_suite_v1`。模型侧只需导出全局主序列长度、Q/KV head、head dim、SP degree 和可选 replicated auxiliary token：

```json
{
  "schema_version": 1,
  "kind": "sp_attention_benchmark_shape_suite_v1",
  "suite_id": "my_model_sp",
  "cases": [{
    "case_id": "block.self_attn",
    "shape": {
      "sequence": 32768,
      "heads": 32,
      "kv_heads": 32,
      "head_dim": 128,
      "sp_size": 4,
      "aux_tokens": 0,
      "aux_q": false,
      "aux_first": false,
      "causal": false
    },
    "precision": {"input_dtype": "bf16"}
  }]
}
```

Sparse SP 的 suite 不直接保存某个 rank 的局部输入，而是引用同一份完整 `operator_benchmark_qkv_replay_v1` manifest。对于“复制最大 conditioner 前缀、剩余 main 序列不 padding 切分”的模型，可以从一份 replay 派生默认 `SP=2/4/8` 或指定卡数的 suite：

```bash
python3 -m tools.benchmarks.sp_bench derive-replay-suites \
  --manifest /path/to/full_qkv_replay.json \
  --output-dir /tmp/model_sparse_sp \
  --suite-id model_real_sparse_sp \
  --conditioner-tokens 89 \
  --keep-ratio 0.15
```

`--conditioner-tokens` 必须来自模型语义；工具会为每个 SP 度数重新计算 aux/main 边界，并在每个 rank 上只加载 replicated aux 与当前 local main 范围。需要非默认卡数时，可以重复传入 `--sp-size`。

先检查合同，再在目标硬件上探测 production dense leaf backend、平台 A2A backend 和所有合法组合：

```bash
python3 -m tools.benchmarks.sp_bench inspect --suite /tmp/model_sp.json

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m tools.benchmarks.sp_bench candidates \
  --suite /tmp/model_sp.json --device cuda:0
```

候选包含 Ulysses 的 pre/post、A2A、通信精度、tensor fusion、head pipeline，以及 Ring 的通信精度和 K/V fusion。Ulysses 启用 head pipeline 时会自动枚举 `1..local_heads` 的全部 `head_parallel_group_size`；group size 不要求整除 local head 数，尾组按实际剩余 head 计入候选 shape 和通信 payload。工具会按当前 production 约束排除无效组合，并记录原因；Ring 只接受原生提供 `apply_with_lse()` 的 dense backend。

Aux Q/K/V 按 replicated 语义在各 rank 使用相同输入；raw 同时检查 main/aux output shape、finite 和 aux output 跨 rank 一致性。H100 上已用 dense reference 验证当前全部 Ulysses/Ring leaf 与合法选项，覆盖 BF16/FP16、无量化/FP8 communication、fusion、aux K/V only/完整 aux Q/K/V，以及 `aux_first` 两种顺序。精确 backend、近似 backend 和量化通信的门限由 `SP_REFERENCE_TOLERANCES` 显式声明，并随验证结果写入 correctness。

`not validated` 的含义是“代码可能支持，但本工具还没有足够证据把它放入候选”，不是性能落选，也不是可以长期忽略的状态。新模型、新硬件或新需求命中这类 exclusion 时，Agent 必须先补验证并更新准入矩阵；只有缺少目标硬件、依赖或真实输入等外部阻塞时才允许延期。

新增 backend、精度或通信路径时，先运行 reference 验证。Synthetic dense suite 对比 Torch SDPA；真实 sparse replay 的 grouped Ulysses 对比同一输入、同一 sparse leaf 和通信配置的 bulk Ulysses，从而隔离 head grouping 引入的差异。该命令允许枚举普通推荐路径尚未放行的 dense 候选，但不会自动修改准入合同：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc-per-node=4 \
  -m tools.benchmarks.sp_bench validate-reference \
  --suite /tmp/model_sp.json \
  --output /tmp/model_sp_reference.json
```

结果逐 case/candidate 记录 main/aux 最大绝对误差、显式 tolerance policy、各 rank shape/finite/allclose 状态和环境身份。尚未登记的新 backend 在探索验证中临时使用 `atol=rtol=0.10`，该值不是正式正确性合同；验证通过后仍需按实测误差更新 `SP_REFERENCE_TOLERANCES` 或对应候选门禁并补单测，才算正式解锁。FP4 communication 当前因依赖缺失未验证；新增 dense leaf 默认以 `not validated` 排除。

Sparse replay 可用 `--sparse-backend` 限定 leaf。被测集合只包含 grouped Ulysses，bulk 仅作为 reference，不生成与自身比较的记录。输出中的 `reference_candidate` 明确记录用于对照的 bulk 配置；该验证只证明 grouped 与 bulk 路径在同一 sparse leaf 下的一致性，不把 sparse 近似本身声明为等价于 dense attention。

运行必须由 `torchrun` 启动，进程数必须等于 suite 的 `sp_size`：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc-per-node=4 \
  -m tools.benchmarks.sp_bench run \
  --suite /tmp/model_sp.json \
  --output-dir /tmp/model_sp_h100 \
  --dense-backend flash_attn3 \
  --algorithm ulysses --algorithm ring
```

Sparse replay suite 使用同一 `run` 入口；不指定 `--sparse-backend` 时遍历目标环境中全部 eligible sparse SP adapter：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc-per-node=4 \
  -m tools.benchmarks.sp_bench run \
  --suite /tmp/model_sparse_sp/model_real_sparse_sp_sp4.json \
  --output-dir /tmp/model_sparse_sp_h100 \
  --algorithm ulysses \
  --repeat-runs 3 --warmup 10 --iterations 30
```

Sparse raw/report 同时输出 `actual_tflops`、`dense_equivalent_tflops` 和实际 block density。两种 TFLOPS 都是参与 GPU 的聚合值；前者以实际选中 block 工作量为分子，后者以完整 dense attention 工作量为分子。

不指定 `--candidate` 时会运行所选算法的全部合法候选；可先从 `candidates` 输出中选择精确 candidate ID，以 `--candidate ID` 缩小实验。每次迭代取所有 rank 的最大 CUDA event 延迟，measured loop 内不插入 barrier 或同步；输出为 `candidate_catalog.json`、`raw.jsonl`、`run_summary.json` 和 `report.json`。

中断后使用完全相同的测量参数增加 repeat，并传入 `--append`。工具按 `(repeat, case, candidate)` 跳过已有记录，同时拒绝 suite、候选 fingerprint、测量参数、LightX2V commit、GPU UUID/PCI 映射、CUDA/PyTorch/NCCL 或 NCCL 环境漂移。NCCL `ALGO`、`PROTO` 和 channel 数只记录、不自动 sweep；显式更换后应使用独立输出目录。每条 raw 在候选完成后立即 flush，已经完成的工作不会因后续候选失败而丢失。

报告区分 `measured_winner` 和 `winner`：前者只是已测集合中最快的稳定候选；后者要求底层 attention registry 没有适用但缺依赖或未映射的 backend，且候选目录中的每个合法组合都得到稳定结果或明确的不可用/正确性结论。少于三个 repeat、波动超门限、测量错误或候选未覆盖时，`coverage_complete=false` 且不发布 `winner`。

正式 winner 发布时，workload 的 `recommendation` 直接携带完整候选配置、原始 workload case（包含 precision、sparse/replay 与 shape 身份）、延迟、逻辑 Q/K/V shape、每 rank attention shape，以及按通信量化/fusion/head-pipeline group 配置展开的逐调用 payload 和 scale shape。这是机器可读的完整推荐；aux/main 边界和吞吐指标用于后续解释。

完整 production SP attention 是搜索配置的 L1 指标，也是推荐配置的唯一排序依据。对已选候选可进一步运行诊断层：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc-per-node=4 \
  -m tools.benchmarks.sp_bench diagnose \
  --suite /tmp/model_sp.json \
  --output-dir /tmp/model_sp_h100_diagnostic \
  --recommendation-report /tmp/model_sp_h100/report.json
```

`--recommendation-report` 会读取 L1 正式报告中的 winner，因此日常诊断不需要手写长 candidate ID；调试单个候选时仍可使用一个或多个 `--candidate`。CLI 摘要会返回 `diagnostic_report.md` 路径，它是默认的人类可读结果页；同目录的 JSON 保留完整机器可读结果。

`diagnose` 输出以下互相独立的测量层：

- L2：跳过 attention 计算后的完整 layout + communication 路径，以及 main/aux 的 pack、量化、exchange、unpack 分段；
- L3：只执行候选真实 payload 的 A2A、all-gather 或 Ring P2P，按通信分量报告每 rank 逻辑 payload、跨 GPU link 字节、整组聚合字节、algorithmic bandwidth 和 bus bandwidth；
- overlap 诊断：另外测量 production L1 和 compute-only 路径。只有三个独立测量在容差内可加时才发布估算比例；存在未解释的组合开销或跨测量倒置时，`estimate_status` 给出原因，比例保持 `null`。

诊断同样要求至少三个 repeat，并对 L2、L3 整体及各通信分量分别应用 spread 门禁。L2/L3 用于解释 L1，不参与候选排名。分量不稳定时保留带宽和带 `~` 的 observed efficiency，但正式 `bus_peak_efficiency` 保持 `null`。当前诊断要求 `head_parallel=false`，支持 dense suite 以及真实 replay 驱动的 sparse suite。Sparse 诊断会复用 L1 的真实 Q/K/V，不会重新生成随机输入。

Aux 按 production 语义拆分：Ulysses 的 replicated aux Q/K/V 不进入 main QKV A2A，aux output 使用 all-gather；Ring 的 replicated aux Q/K/V 不进入 main K/V rotation，只在本地 attention layout 中拼接。报告会同时列出 main 与 aux 通信分量各自独立实测的 latency、Bus bandwidth、稳定性和峰值效率。分量使用独立 Event-list 测量，不是按整体耗时比例摊分；因此各分量延迟之和不要求等于整体 L3 路径延迟。

混合 Ulysses 整体路径仍以 `all_to_all` profile 作为主链路峰值分母，并显式计入 aux all-gather 字节；分量报告则分别使用 `all_to_all` 和 `all_gather` peak。缺少某个分量的 peak 时仍报告其延迟和带宽，但不发布该分量效率。

### SP 互联峰值

首次到达新硬件时，先不提供 peak 运行 `diagnose`。报告的 `hardware.topology_fingerprint` 由 rank 到 GPU PCI 映射和规范化后的 `nvidia-smi topo -m` 共同生成。调研当前硬件后再建立 profile：

```json
{
  "schema_version": 1,
  "platforms": {
    "my_4gpu_topology": {
      "identification": {
        "gpu_name_regex": "^NVIDIA H100 80GB HBM3$",
        "cuda_capability": "9.0",
        "world_size": 4,
        "topology_fingerprint": "..."
      },
      "interconnect_peaks": {
        "all_to_all": {
          "bus_bandwidth": 450.0,
          "unit": "GB/s",
          "kind": "theoretical",
          "source": "https://docs.nvidia.com/cuda/hopper-tuning-guide/#fourth-generation-nvlink"
        },
        "all_gather": {
          "bus_bandwidth": 450.0,
          "unit": "GB/s",
          "kind": "theoretical",
          "source": "https://docs.nvidia.com/cuda/hopper-tuning-guide/#fourth-generation-nvlink"
        },
        "ring_p2p": {
          "bus_bandwidth": 450.0,
          "unit": "GB/s",
          "kind": "theoretical",
          "source": "https://docs.nvidia.com/cuda/hopper-tuning-guide/#fourth-generation-nvlink"
        }
      }
    }
  }
}
```

`bus_bandwidth` 必须与报告的 `network_bytes_per_rank / latency` 使用同一口径：只计每 rank 发出的跨 GPU link 字节。上例的 H100 SXM 官方规格为每 GPU 900 GB/s 双向 NVLink 带宽，因此单向分母为 450 GB/s；该数值只适用于 profile 明确匹配的 H100 拓扑，不是其他硬件的默认值。理论物理峰值可以在多个通信模式下相同；实测饱和包络必须按 `all_to_all`、`all_gather`、`ring_p2p` 分别调研。`kind` 可取理论峰值 `theoretical` 或实测饱和包络 `empirical_envelope`，两者都必须保留来源。

补完 profile 后无需重跑 benchmark，可从 raw 离线重建报告：

```bash
python3 -m tools.benchmarks.sp_bench report-diagnostics \
  --suite /tmp/model_sp.json \
  --raw /tmp/model_sp_h100_diagnostic/diagnostic_raw.jsonl \
  --output-dir /tmp/model_sp_h100_diagnostic \
  --interconnect-peaks /path/to/interconnect_peaks.json \
  --platform my_4gpu_topology
```

GPU 名称、capability、world size 或拓扑指纹任一不匹配时，工具拒绝使用该 profile。缺少 profile 或通信模式 peak 时仍保留延迟和带宽；测量未通过 repeat/spread 门禁时保留 nominal peak，但不计算效率。

## 接纳与效率

硬件峰值由运行环境提供，不在工具中固化。最小格式：

```json
{
  "platforms": {
    "h100_sxm_80gb": {
      "identification": {
        "gpu_name_regex": "^NVIDIA H100 80GB HBM3$",
        "cuda_capability": "9.0"
      },
      "peaks": {
        "bf16": {"dense_rate": 989.5, "unit": "TFLOPS"},
        "fp16": {"dense_rate": 989.5, "unit": "TFLOPS"},
        "fp8": {"dense_rate": 1979.0, "unit": "TFLOPS"},
        "int8": {"dense_rate": 1979.0, "unit": "TOPS"}
      }
    }
  }
}
```

上例数值对应 H100 SXM 80GB 的 dense Tensor Core 名义峰值。NVIDIA 规格页列出的是带结构化稀疏的数值，
这里按其一半记录非稀疏峰值：BF16/FP16 为 989.5 TFLOPS，FP8 为 1979 TFLOPS，INT8 为
1979 TOPS。来源：<https://www.nvidia.com/en-us/data-center/h100/>。

更换硬件后必须为目标型号重新调研并提供 profile，不能沿用 H100 数值；新增 profile 不需要修改
benchmark 代码。缺少当前精度的 peak 时，延迟、吞吐和 backend 排名仍然有效，但报告会输出
`peak_status=peak_missing`，并停止计算该项效率。浮点 rate 只匹配 TFLOPS，INT8 rate 只匹配 TOPS。

如果同一种计算精度随累加类型具有不同峰值，使用 accumulator variant。以下是 RTX 5090 FP8 的
dense 示例；斜杠后的 structured-sparsity 数值不用于本工具：

```json
{
  "fp8": {
    "variants": {
      "fp32": {"dense_rate": 419.0, "unit": "TFLOPS"},
      "fp16": {"dense_rate": 838.0, "unit": "TFLOPS"}
    }
  }
}
```

单值形式只适用于该平台不需要区分 accumulator peak 的 precision family；只要不同累加类型的峰值
不同，就必须使用 variants。

Backend 必须在 raw precision 中明确记录 `accum_dtype=fp32` 或 `fp16`，报告才会选择对应分母。
存在 variants 时不会回退到通用 FP8 peak：累加类型未知会标记 `precision_variant_unknown`，已知但
profile 未收录会标记 `peak_variant_missing`。来源：
<https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>。

- 默认至少 3 个 repeat；
- 每轮先完成 warmup，再为各 iteration 记录 event pair，所有 kernel 入队后只同步最后一个 event；这避免逐 iteration 的 CPU/GPU 同步间隙；
- 单算子返回Tensor。计时结束后检查末次输出的类型、有限性及声明shape；未声明shape时核对已验证的预检原始输出shape。失败记录阶段与原因，整次结果不进入有效计时排序。不逐iteration校验，不做数学参考；
- 对数微秒级 kernel，event pair 本身仍可能形成可观扰动；报告的是逐调用 event latency，不等同于整段循环摊销吞吐；
- 跨轮 spread 同时超过 `5%` 和 `0.005 ms` 才判为不稳定；
- capability/measurement error、显式 correctness failure、repeat 不足和不稳定结果不进入 `measured_winner`；
- 性能排序与数值验证分开；数值未测不阻塞性能计时，任何性能结果都不自动批准质量；
- “计算效率”表示逻辑计算量除以实测延迟（TFLOPS/TOPS）；名义peak算力给出峰值比例的分母。没有本次最终mask的稀疏估计量不进入开发者主表，保留缺口原因及已有区域统计。独立benchmark不默认采profiler；若报告链接其他诊断trace，必须注明它不是正式计时那次运行；
- 稀疏的稠密等价计算效率为完整dense量除以同一耗时，可超过dense峰值，不代表物理利用率；实际区域未知时两项效率均留空。
- GEMM 使用 `2*M*N*K`；dense attention/MoE 使用 effective work；sparse attention 同时报告实际 block work 和 dense-equivalent work；
- 量化 GEMM 的 effective rate 包含动态 activation quantization 等 wrapper 开销，相对纯 Tensor Core 峰值的效率是端到端诊断指标；
- 峰值 profile 必须与 raw 中的 GPU 名称和 CUDA capability 匹配，否则拒绝发布效率。

CLI只生成结构化事实与简表；最终中文报告由Agent按[固定模板](REPORT_TEMPLATE.md)生成，包含硬件峰值、输入/计算量、性能主表、精度/失败及完整候选覆盖。裸CLI不再自动生成该完整页面。


已有 raw 可单独重建报告：

```bash
python3 -m tools.benchmarks.operator_bench report \
  --suite /tmp/my_model.json \
  --raw /tmp/my_model_h100/raw.jsonl \
  --backend-catalog /tmp/my_model_h100/backend_catalog.json \
  --output-dir /tmp/my_model_h100 \
  --peaks /path/to/hardware_peaks.json \
  --platform h100_sxm_80gb
```
