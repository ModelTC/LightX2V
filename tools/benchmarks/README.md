# 核心算子 Benchmark

这是一套由 shape 驱动的单卡 benchmark，用于在**指定硬件**上回答四个问题：

1. 新模型或新硬件上的核心 shape 大致能达到什么性能；
2. profiler 中记录的当前 backend 是否是同 shape 下的最佳候选；
3. 一个 backend 在理论 shape 网格上的表现，以及与其他 backend 的差距；
4. 最佳实测结果距离硬件名义峰值还有多大 gap。

它只处理 GEMM、dense attention 和单卡 MoE。Distributed、模型级调度、跨硬件性能外推和 sparse attention 真实输入均不属于当前版本。

## 最小数据流

```text
canonical shape dump / 理论 sweep
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
- `moe`：`tokens, hidden_size, intermediate_size, num_experts, top_k, activation, expert_bias`。

当前 attention 合同只接受 fixed-length；`causal=true` 时要求 `seq_q == seq_kv`。Sparse attention 延后，因此不会用随机 dense 输入生成误导性的稀疏性能。

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

Attention 和 MoE 使用各自的全部必填 shape 字段；不提交预生成网格，避免重复维护由这个命令即可生成的数据文件。

## 运行

内置 Torch 基线、LightX2V production MM registry、Flash/Sage attention 和 Torch grouped MoE adapter。以下示例比较 production GEMM backend：

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
- `run_summary.json`：本次写入和状态计数；
- `report.json`、`report.md`：稳定 backend 排名、observed backend 差距和硬件效率。

中断后以相同参数增加 `--append`。工具按 `(repeat, case, backend)` 跳过已有记录；更换 suite、backend 或测量参数时应使用新目录。

共享机器上的卡是否空闲应在启动前用 `nvidia-smi` 确认。`--device` 使用当前进程可见的 CUDA ordinal。

## Backend

查看内置及指定插件注册的 backend：

```bash
python3 -m tools.benchmarks.operator_bench backends
```

`operator_backends.py` 中的可选依赖只在实际准备 case 时导入；列出 backend 不要求当前环境安装所有 kernel。外部扩展仍可通过 `--plugin module.name` 注册。

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
- GEMM 使用 `2*M*N*K`；attention/MoE 使用 effective work，因此后二者的峰值效率只用于定位短板；
- 量化 GEMM 的 effective rate 包含动态 activation quantization 等 wrapper 开销，相对纯 Tensor Core 峰值的效率是端到端诊断指标；
- 峰值 profile 必须与 raw 中的 GPU 名称和 CUDA capability 匹配，否则拒绝发布效率。

已有 raw 可单独重建报告：

```bash
python3 -m tools.benchmarks.operator_bench report \
  --suite /tmp/my_model.json \
  --raw /tmp/my_model_h100/raw.jsonl \
  --output-dir /tmp/my_model_h100 \
  --peaks /path/to/hardware_peaks.json \
  --platform h100_sxm_80gb
```
