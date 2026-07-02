# Transformer Profiling

## 概述

LightX2V 当前提供两类 transformer profile：

- `full`：采集某个 infer step 的完整 transformer 执行过程，输出 PyTorch Profiler / TensorBoard trace。
- `block`：采集某个 infer step 的某个 transformer block，输出 TensorBoard trace、逻辑 op trace，以及按 region 聚合的人类可读报告。

这两类 profile 都由环境变量 `LIGHTX2V_PROFILE_MODE` 开启：

```bash
export LIGHTX2V_PROFILE_MODE=full   # 或 block
```

目前支持的主要路径：

- Hunyuan3D shape transformer
- SekoTalk / Wan audio transformer

不开启 `LIGHTX2V_PROFILE_MODE` 时，推理走普通路径。

---

## Full Profile

`full` 模式只包住目标 infer step 的 transformer 执行，不生成 layer report。它适合查看整体 timeline、kernel 分布、CPU/GPU gap、异步调度等问题。

### 运行方式

以 Hunyuan3D 为例，在常规脚本里设置：

```bash
export LIGHTX2V_PROFILE_MODE=full
bash scripts/hunyuan3d/run_hunyuan3d.sh
```

输出目录形如：

```text
prof_results/hunyuan3d_transformer_profile/full_step_10_YYYYmmdd_HHMMSS/
└── *.pt.trace.json
```

可用 TensorBoard 打开该目录：

```bash
tensorboard --logdir prof_results/hunyuan3d_transformer_profile/full_step_10_YYYYmmdd_HHMMSS --port 16006 --bind_all
```

浏览器访问：

```text
http://127.0.0.1:16006/#pytorch_profiler
```

注意要打开 **PYTORCH PROFILER** 页，不是 SCALARS 页。

### ProfilerStep gap 快速统计

如果只想量化 TensorBoard GPU timeline 里 `ProfilerStep#N` 的 wall time、GPU active time 和空白 gap，可以直接解析 trace：

```bash
python tools/profile/profiler_step_gap.py --brief \
  prof_results/hunyuan3d_transformer_profile/full_step_10_YYYYmmdd_HHMMSS/*.pt.trace.json
```

输出示例：

```text
ProfilerStep#0: wall=56.154 ms, gpu_active=43.612 ms, self_gap=12.542 ms, raw_gpu_sum=43.613 ms, gpu_memcpy=24, gpu_memset=131, kernel=1733
```

该工具使用 trace 中的 `gpu_user_annotation` 作为 `ProfilerStep#N` 窗口，`gpu_active` 是窗口内 GPU activity 区间 merge 后的耗时，`self_gap = wall - gpu_active`。`raw_gpu_sum` 是未 merge 的 GPU activity 耗时求和，后面的 `gpu_memcpy/gpu_memset/kernel` 是窗口内事件数量，适合对比两次 trace 的 gap 和 memcpy 数量变化。

---

## Block Profile

`block` 模式用于单层深入分析。它会在目标 block 上同时采集：

- PyTorch profiler trace：`*.pt.trace.json`
- 逻辑 op shape：`block_{layer}_op_trace.jsonl`
- 汇总报告：`block_{layer}_layer_trace.txt`

汇总报告会把 CUDA kernels 归属到粗粒度 region，并结合逻辑 op shape 估算 FLOPs / TFLOPS。

### 运行方式

```bash
export LIGHTX2V_PROFILE_MODE=block
bash scripts/hunyuan3d/run_hunyuan3d.sh
```

输出目录形如：

```text
prof_results/hunyuan3d_transformer_profile/block_step_10_layer_0_YYYYmmdd_HHMMSS/
├── *.pt.trace.json
├── block_0_op_trace.jsonl
└── block_0_layer_trace.txt
```

`block_0_layer_trace.txt` 示例：

```text
Layer index: 0   ProfilerStep#0   GPU events: 64
Region assigned: 64/64
ProfilerStep  compute=1.673 ms  wall=3.878 ms

── self_attn ── kernel_sum=0.788 ms
── cross_attn ── kernel_sum=0.456 ms
── dense_ffn ── kernel_sum=0.430 ms
```

---

## 修改采集目标

为了减少配置和 review 负担，当前只保留一个 env 开关。采集目标在代码默认值里直接写明：

```python
# lightx2v/utils/transformer_profile.py
class TransformerProfile:
    profile_infer_step = 10
    profile_block_idx = 2
```

开发者临时分析其他 step 或 block 时，直接修改这两个默认值即可。例如分析 block 15：

```python
profile_block_idx = 15
```

调试完成后再改回默认值。

---

## Region 与 Op Trace

`block` 模式里有两层信息：

1. `region_profile`：在 PyTorch trace 中标出粗粒度执行区间，例如 `self_attn`、`cross_attn`、`dense_ffn`、`moe`、`audio_adapter`。
2. `op_shape_trace`：写出逻辑 op 的 shape / FLOPs，例如 GEMM、ATTN、MOE。

二者结合后，`block_profile_report.py` 会把 kernel 时间和理论计算量对应起来，生成 `block_{layer}_layer_trace.txt`。

### Kernel 到 Region 的归属口径

`block` report 需要把 GPU timeline 上的 kernel / memcpy / memset 归属到粗粒度 region。这里不能只看 GPU annotation 的起止重叠：GPU annotation 的尾部和实际 kernel 结束时间可能存在 overlap 或截断，容易把 region 末尾 launch 的 kernel 漏到下一个区域或漏成未归属。

因此 `tools/profile/region_event_trace.py` 会通过 trace correlation 找到 kernel 对应的 CPU `cudaLaunchKernel` timestamp，并优先用这个 launch timestamp 落入的 CPU `user_annotation` region 作为可靠的归属依据。这样更符合“哪个 region 发起了这个 GPU work”的语义，也能覆盖 kernel 在 GPU 上延迟执行、跨过 annotation 尾部的情况。只有在 CPU launch correlation 找不到时，才退回 GPU annotation overlap 做兜底。

注意这和 `profiler_step_gap.py --brief` 的口径不同：gap 统计刻意使用 GPU timeline 里的 `gpu_user_annotation` `ProfilerStep#N` 窗口来量化 wall/gpu_active/self_gap；region report 则关心 kernel 属于哪个 CPU 发起的 region。

---

## 文件职责

### Full 与 Block 共用

| 文件 | 说明 |
|------|------|
| `lightx2v/utils/transformer_profile.py` | 读取 `LIGHTX2V_PROFILE_MODE`，选择 full/block，创建输出目录 |
| `lightx2v/utils/torch_trace_profiler.py` | PyTorch profiler trace 导出 |
| `tools/profile/profiler_step_gap.py` | 解析 full/block trace 中的 `gpu_user_annotation` ProfilerStep，快速统计 wall/gpu_active/self_gap/raw_gpu_sum 和 GPU event 数量 |
| 模型 `transformer_infer.py` | 在目标 infer step 调用 `record_full()` 或 `record_block()` |

### Block 专用

| 文件 | 说明 |
|------|------|
| `lightx2v/utils/region_profile.py` | region decorator 与当前 active profile 管理 |
| `lightx2v/utils/op_shape_trace.py` | 写 `block_{layer}_op_trace.jsonl` |
| `tools/profile/region_event_trace.py` | 解析 torch trace + op trace，生成 region report 和 ProfilerStep summary |
| `tools/profile/trace_correlation.py` | kernel 与 CPU launch / region 关联辅助 |
| `lightx2v/models/networks/*/infer/block_profile.py` | 模型侧 op shape hook |
| `lightx2v/models/networks/*/infer/block_profile_report.py` | 模型侧 report 配置 |

---

## 底层 TorchProfileContext

`lightx2v.utils.torch_trace_profiler.TorchProfileContext` 仍然可以用于临时 profile 任意调用点，但它不是 Hunyuan3D/Seko transformer profile 的主入口。

示例：

```python
from lightx2v.utils.torch_trace_profiler import TorchProfileContext

with TorchProfileContext(
    "my_forward",
    tb_dir="save_results/torch_profile",
    with_stack=True,
) as profile:
    profile.run(my_forward, arg1, arg2)
```

默认 schedule 为 `wait=1 / warmup=3 / active=1`，会重复调用目标函数共 5 次。这个行为适合独立函数分析，但不适合直接包住带状态推进的扩散循环。因此常规模型 profile 优先使用 `LIGHTX2V_PROFILE_MODE=full/block`。

---

## 常见问题

### full 模式为什么没有 `block_x_layer_trace.txt`？

`full` 只产出 TensorBoard trace，用于整体 timeline 分析。单层 kernel_sum / TFLOPS 报告属于 `block` 模式。

### block 模式为什么需要 `block_x_op_trace.jsonl`？

PyTorch trace 只有 kernel 名称和耗时，不知道某个 kernel 对应的逻辑 GEMM / ATTN shape。`op_trace.jsonl` 提供 shape 和 FLOPs，报告才能计算 TFLOPS 和效率。

### 为什么默认只用一个 env？

这个功能主要面向能直接改代码的开发者。保留大量 env 配置会增加实现和 review 复杂度；当前选择是 env 只控制模式，step/block 默认值直接在 `TransformerProfile` 中修改。

### 非 H100 机器报 peak TFLOPS 错误怎么办？

block report 当前会自动识别 H100 的 BF16/FP8 peak TFLOPS。其他硬件需要在 `tools/profile/region_event_trace.py` 的 `infer_peak_tflops_from_device()` 中补充硬件峰值。
