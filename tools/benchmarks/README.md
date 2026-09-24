# 核心算子 Benchmark

这是一套由 shape 驱动的单卡 benchmark，用于在**指定硬件**上回答四个问题：

1. 新模型或新硬件上的核心 shape 大致能达到什么性能；
2. profiler 中记录的当前 backend 是否是同 shape 下的最佳候选；
3. 一个 backend 在理论 shape 网格上的表现，以及与其他 backend 的差距；
4. 最佳实测结果距离硬件名义峰值还有多大 gap。

它只处理 GEMM、dense/sparse attention 和单卡 MoE。Distributed、模型级调度和跨硬件性能外推不属于当前版本。

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

稀疏性取决于 Q/K 内容，只保存 shape 并随机生成输入会让稀疏率和耗时失真。模型侧或 agent 应在进入 attention backend 前捕获已经完成 RoPE 等变换的 Q/K/V，工具只消费稳定的回放合同。新模型不需要把捕获逻辑加入 benchmark；应在 `wyr_agent` 中保存该模型的一次性 hook、转换命令和来源说明。

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

候选目录随当前 LightX2V registry 演进，但每次 probe 的 fingerprint 固定本次比较集合。发布“最佳 backend”结论前，应对目标 family 的全部 `eligible` 候选运行目标 shape；不能把 `dependency_missing` 当作性能落选。尚未进入 LightX2V、也未安装在环境中的外部 kernel 无法由工具自动发现，需要 agent 在新硬件调研阶段补充。

Benchmark 实质代码只有两个文件：`operator_bench.py` 负责 shape、执行、报告和 CLI，`operator_backends.py` 负责 backend 合同与实现。

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
