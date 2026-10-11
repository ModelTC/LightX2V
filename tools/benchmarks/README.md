# 核心算子 Benchmark

这是一套由 shape 驱动的单卡 benchmark，用于在**指定硬件**上回答四个问题：

1. 新模型或新硬件上的核心 shape 大致能达到什么性能；
2. profiler 中记录的当前 backend 是否是同 shape 下的最佳候选；
3. 一个 backend 在理论 shape 网格上的表现，以及与其他 backend 的差距；
4. 最佳实测结果距离硬件名义峰值还有多大 gap。

`operator_bench.py` 只处理 GEMM、dense/sparse attention 和单卡 MoE。Sequence-parallel dense/sparse attention 使用独立的
`sp_bench.py` 和 suite；模型级调度、TP/SP 混合并行、多机执行和跨硬件性能外推不属于当前版本。

## 最小数据流

```text
shape dump / 理论 sweep / 真实 QKV replay
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
- `moe`：`tokens, hidden_size, intermediate_size, num_experts, top_k, activation, expert_bias`。

当前 attention 合同只接受 fixed-length；`causal=true` 时要求 `seq_q == seq_kv`。Sparse attention 只接受真实 QKV 回放，不提供随机输入或理论 sweep。

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

Dense attention 和 MoE 使用各自的全部必填 shape 字段；不提交预生成网格，避免重复维护由这个命令即可生成的数据文件。

## Sparse Attention 回放

稀疏性取决于 Q/K 内容，只保存 shape 并随机生成输入会让稀疏率和耗时失真。模型侧应在进入 attention backend 前捕获已经完成 RoPE 等变换的 Q/K/V，工具只消费稳定的回放合同。模型专属 hook 留在对应模型侧或本地工作区；与模型无关的可复用捕获能力才进入 benchmark。

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
  "sparse": {"keep_ratio": 0.15},
  "replay": {"manifest": "/path/to/qkv_replay.json", "sha256": "...64 hex..."}
}
```

Replay manifest 使用 `operator_benchmark_qkv_replay_v1`，layout 为 `BSHD`，并为 `q/k/v` 分别记录 dtype、完整 shape 和连续 sequence shards。每个 shard 必须包含相对 manifest 的路径、SHA256、tensor key、`sequence_start` 和 `sequence_end`。加载时会校验 manifest、所有 shard、dtype、shape 和序列覆盖，随后才搬到目标 GPU。

内置回放 backend：

- `dynamic_sparse_triton_replay`：LightX2V production Triton sparse attention；
- `dynamic_sparse_sage2_replay`：LightX2V production dynamic sparse attention；
- `dynamic_sparse_sage3_replay`、`dynamic_sparse_fa4_replay`：Blackwell 代际候选；
- `sparge_sage2_replay`：production Sparge attention；
- `spas_sage2_replay`、`spas_sage3_replay`、`spas_fa4_replay`：LightX2V production static sparse 路径；
- `flash_attn3_replay`、`flash_attn4_replay`、`sage_attn3_replay`：同一份真实 QKV 上的代际 dense 基线。

Artifact 加载、校验和首次初始化计入 `setup_ms`，不进入 kernel latency；稀疏 backend 的原生 block-map/routing 则包含在每次计时中。Raw case 会固化 replay SHA 和稀疏配置，append 与 report 都拒绝身份漂移。

Sparse attention 同时输出两套口径：

- `actual_tflops`：以 `dense work * 实际 block density` 估算选中 QK/PV 的 FLOPs；分子不计 routing FLOPs，但延迟包含每次调用的原生 block-map/routing，用于观察端到端实际算力利用；
- `dense_equivalent_tflops`：以完整 dense work `4*B*H*Sq*Sk*D` 为分子，用于观察同 shape 的等效吞吐和加速收益。

提供硬件 peak profile 后，报告分别输出 `actual_peak_efficiency` 和 `dense_equivalent_peak_efficiency`。两者都使用 suite 输入精度的普通 dense peak，不使用 2:4 structured-sparsity peak。前者是实际有效计算效率的近似值；后者是等效效率，可能超过 100%，不能解释为实际 Tensor Core 利用率。若 backend 内部混合了量化 QK 与浮点 PV 等不同算术，该分母只是统一诊断基准；精确的硬件指令效率还需要 backend 声明计算混合比例。

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
- `report.json`、`report.md`：稳定 backend 排名、observed backend 差距和硬件效率。

报告区分 `measured_winner` 与正式 `winner`。前者是已测稳定候选中的最快项；后者还要求目标 family 的 production registry 无未映射项、适用依赖无缺失，并且 catalog 中每个 `eligible` 候选都有稳定结果或明确的不可用/正确性结论。缩小 `--backend` 范围仍可用于快速探索，但只生成 measured subset，不发布正式最佳配置。离线 `report` 要恢复正式推荐时需同时传入本次 `--backend-catalog`。

中断后以相同参数增加 `--append`。工具按 `(repeat, case, backend)` 跳过已有记录，并拒绝 suite、backend、测量参数、seed、LightX2V commit、GPU 或 CUDA/PyTorch 环境漂移；这些身份变化时应使用新目录。

共享机器上的卡是否空闲应在启动前用 `nvidia-smi` 确认。`--device` 使用当前进程可见的 CUDA ordinal。

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

候选目录随当前 LightX2V registry 演进，但每次 probe 的 fingerprint 固定本次比较集合。发布“最佳 backend”结论前，应对目标 family 的全部 `eligible` 候选运行目标 shape；不能把 `dependency_missing` 当作性能落选。尚未进入 LightX2V、也未安装在环境中的外部 kernel 无法自动发现，需要先安装并通过 registry 或 `--plugin` 注册 adapter。

单卡 benchmark 实质代码只有两个文件：`operator_bench.py` 负责 shape、执行、报告和 CLI，`operator_backends.py` 负责 backend 合同与实现。分布式 SP attention 的进程组、候选约束和计时集中在独立的 `sp_bench.py`，避免让单卡工具依赖 `torchrun`。

## Sequence-Parallel Attention

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
- 对数微秒级 kernel，event pair 本身仍可能形成可观扰动；报告的是逐调用 event latency，不等同于整段循环摊销吞吐；
- 跨轮 spread 同时超过 `5%` 和 `0.005 ms` 才判为不稳定；
- capability/measurement error、显式 correctness failure、repeat 不足和不稳定结果不进入 winner；
- correctness 未检查不会阻塞当前版本；
- GEMM 使用 `2*M*N*K`；dense attention/MoE 使用 effective work；sparse attention 同时报告实际 block work 和 dense-equivalent work；
- 量化 GEMM 的 effective rate 包含动态 activation quantization 等 wrapper 开销，相对纯 Tensor Core 峰值的效率是端到端诊断指标；
- 峰值 profile 必须与 raw 中的 GPU 名称和 CUDA capability 匹配，否则拒绝发布效率。

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
