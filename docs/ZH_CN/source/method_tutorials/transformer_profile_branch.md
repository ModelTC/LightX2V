# reserved_dev_tool/transformer-profile 分支说明

## 分支定位

`reserved_dev_tool/transformer-profile` 是一个保留开发工具分支，用来沉淀 transformer full/block profile 的通用做法。它更像可参考、可迁移的性能分析模板，不是必须合入 `main` 的产品功能。

这个分支主要解决两个问题：

- 用常规运行脚本，通过 `LIGHTX2V_PROFILE_MODE=full|block` 临时打开 profile。
- 在保持模型推理逻辑基本不变的前提下，把单个 transformer block 的粗粒度 region、核心 op shape、kernel 时间对应起来。

当前已经验证过的路径：

- Hunyuan3D shape transformer
- SekoTalk / Wan audio transformer

## 功能边界

`full` 模式只 profile 目标 infer step 的 transformer 执行，输出 PyTorch Profiler trace，适合用 TensorBoard 看整体 timeline。

`block` 模式只 profile 目标 infer step 的目标 block，输出：

- `*.pt.trace.json`
- `block_{layer}_op_trace.jsonl`
- `block_{layer}_layer_trace.txt`

为了降低 review 负担，当前只保留一个环境变量开关。默认 infer step 和 block index 直接写在 `lightx2v/utils/transformer_profile.py` 的 `TransformerProfile` 类里，开发者临时分析时直接修改对应默认值。

## 文件组织

通用文件：

| 文件 | 角色 |
|------|------|
| `lightx2v/utils/transformer_profile.py` | 读取 profile mode，管理 full/block 输出目录和一次性 profiler context |
| `lightx2v/utils/torch_trace_profiler.py` | 封装 PyTorch Profiler trace 导出 |
| `lightx2v/utils/region_profile.py` | 提供 region decorator 和当前 active profile |
| `lightx2v/utils/op_shape_trace.py` | 写出逻辑 op shape JSONL |
| `tools/profile/region_event_trace.py` | 解析 torch trace + op trace，生成 block report |
| `tools/profile/trace_correlation.py` | kernel 与 CPU launch / region 的关联辅助 |

模型侧文件：

| 文件 | 角色 |
|------|------|
| `lightx2v/models/networks/<model>/infer/block_profile.py` | 模型侧 op shape hook，绑定权重 shape 和运行时 shape |
| `lightx2v/models/networks/<model>/infer/block_profile_report.py` | 模型侧 report 配置，例如 region 顺序、特殊 op 展开 |
| `lightx2v/models/networks/<model>/infer/transformer_infer.py` | 在 transformer infer / block loop 附近接入 full/block profile |
| 其他 attention / moe / ffn 文件 | 只在已有自然边界上加 `@region_profile(...)` 或少量 op logging hook |

## 新模型迁移步骤

假设要在另一个分支上给模型 C 增加同类 profile，建议按下面顺序迁移。

1. 从模型 C 的开发分支拉出新分支，再从 `reserved_dev_tool/transformer-profile` 参考或拷贝通用文件。

2. 先迁移通用文件：

```text
lightx2v/utils/transformer_profile.py
lightx2v/utils/torch_trace_profiler.py
lightx2v/utils/region_profile.py
lightx2v/utils/op_shape_trace.py
tools/profile/region_event_trace.py
tools/profile/trace_correlation.py
```

如果目标分支上这些文件已经有本地改动，优先手工合并，避免直接覆盖。

3. 找到模型 C 的 transformer infer 入口和 block loop。目标代码形态尽量保持直观：

```python
mode = self.transformer_profile.mode_for_step(infer_step)
if mode == "full":
    with self.transformer_profile.record_full():
        return self._infer(...)
if mode == "block":
    block_profile = self.transformer_profile.start_block()
    output = self._infer(...)
    block_profile.write_block_report()
    return output
return self._infer(...)
```

实际迁移时不要求完全照抄这段伪代码，原则是让 `full`、`block`、普通推理三个路径在入口处一眼可见。

4. 在 block loop 里只处理目标 block。普通 block 继续走原有调用；目标 block 在运行前绑定 shape，然后用 `record_block()` 包住原本的 block 调用。

```python
if mode == "block" and block_idx == self.transformer_profile.block_idx:
    self.block_profile.bind(block_weights, cond_len, hidden_states)
    with self.transformer_profile.record_block(block_idx):
        hidden_states = self.infer_block(...)
else:
    hidden_states = self.infer_block(...)
```

5. 新增模型侧 `block_profile.py`。这个文件负责两件事：绑定 shape，以及在 region hook 被调用时写 op trace。

通常需要包含：

- 模型专属 env，例如 `MODEL_C_BLOCK_PROFILE`。
- `region_profile = partial(_region_profile, annotate_env=...)`。
- `ModelCBlockProfile.profile_env`。
- `ModelCBlockProfile.block_profile_report_module`。
- `bind(...)`：记录运行时 `M/seq/context`，以及从权重读取 GEMM 的 `N/K`。
- `self_attn()`、`cross_attn()`、`dense_ffn()`、`moe()` 等 hook。

6. 在模型已有计算边界上添加 region decorator。region 名称保持粗粒度，op tag 可以更细：

```python
@region_profile("self_attn", emit="self_attn")
def infer_self_attn(...):
    ...
```

如果模型已有 `infer_self_attn`、`infer_cross_attn`、`infer_moe_block` 这类自然边界，优先只加 decorator。只有在原代码完全没有合适边界时，才考虑拆一个小函数。

7. 新增模型侧 `block_profile_report.py`。常规内容是：

- `RegionTraceConfig(region_order=...)`
- 必要的 `skip_regions` / `gpu_skip_prefixes`
- `analyze(...)` 包装 `analyze_region_trace(...)`

只有当模型存在特殊 op 统计方式时，才添加 `expand_op`。例如 MoE routed 部分可能需要根据 backend 展开成两组或三组 GEMM。

8. 用常规脚本验证。

```bash
python -m py_compile lightx2v/models/networks/<model>/infer/transformer_infer.py

export LIGHTX2V_PROFILE_MODE=full
bash scripts/<model>/run_<model>.sh

export LIGHTX2V_PROFILE_MODE=block
bash scripts/<model>/run_<model>.sh
```

`full` 至少应生成 `*.pt.trace.json`。`block` 至少应生成 `*.pt.trace.json`、`block_{layer}_op_trace.jsonl`、`block_{layer}_layer_trace.txt`。

## 迁移原则

迁移时优先保证 review 友好：

- 不为了 profile 重写原有推理逻辑。
- 不做格式化、空行、条件改写等无关变动。
- 不新增大量 env 配置；默认只保留 `LIGHTX2V_PROFILE_MODE`，具体 step / block 由开发者改默认值。
- region 保持粗粒度，例如 `self_attn`、`cross_attn`、`dense_ffn`、`moe`、`audio_adapter`。
- op logging 可以比 region 更细，但只服务于计算量和 kernel 对照。
- `full` 路径只负责 TensorBoard trace，不要求复用 block report 逻辑。
- `block` 路径只包目标 block，不扩大到整网或多个 step。
- 不提交 `prof_results/`、临时 mesh、个人 docker/run 脚本等生成物。

如果模型 C 的 block 结构差异很大，先迁移 `full` 模式也可以。等确认整体 timeline 有价值，再补 `block_profile.py` 和 `block_profile_report.py`。

适配新模型时，下面几类功能可能需要扩展：

- 硬件峰值算力：report 里的 TFLOPS / efficiency 需要硬件 BF16/FP8 峰值；非 H100 机器需要补 `infer_peak_tflops_from_device()`。
- op shape hook：不同模型的 QKV、FFN、MoE、adapter、fused linear 形态不同，`block_profile.py` 需要按模型实际权重和运行时 shape 写出逻辑 op。
- 特殊 op 展开：如果某个逻辑 op 会对应多组 kernel，例如不同 backend 的 MoE routed GEMM，需要在 `block_profile_report.py` 里扩展 `expand_op`。
- kernel 识别与关联：新 attention / GEMM backend 可能有新的 kernel 名称或 trace 关联字段，需要扩展 `tools/profile/region_event_trace.py` 或 `trace_correlation.py`。
- region 边界：如果模型没有自然的 self-attn / cross-attn / ffn 函数边界，可能需要极少量拆分，原因是 PyTorch trace 只能按实际 `record_function` 区间归属 kernel。

## 验证检查

迁移完成后，重点检查：

- 普通推理不开 `LIGHTX2V_PROFILE_MODE` 时结果不变。
- `full` 模式只输出 TensorBoard trace，不生成 block report。
- `block` 模式的 report 里有 `ProfilerStep`、`Region assigned` 和各 region 的 `kernel_sum`。
- 关键 GEMM / ATTN / MOE op 能在 report 里看到 shape、FLOPs、TFLOPS。
- 非 H100 硬件如需计算效率，需要在 `infer_peak_tflops_from_device()` 中补充对应硬件峰值。
