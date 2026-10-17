# XPU 故障排查

从 [XPU 主文](../SKILL.md) 按故障现象进入对应章节；保留原排障实例供对照。

## Case 2: 进程静默消失，无任何输出

先记录退出码、最后一条已 flush 的日志、CPU 提交内存和设备内存，再查操作系统的 OOM/驱动记录。文件大小本身不能证明 mmap 是原因，也没有通用的“5 GB 必崩”阈值。

用小 shard 或单 tensor 读取定位加载阶段；如果确认是峰值内存问题，再选择已支持的分片加载或在内存足够的机器转换。当前 `lightx2v/common/ops/utils.py` 没有 `_read_tensor_no_mmap()`，不要调用历史文档中的不存在接口。

---

## Case 3: 推理中途静默 kill，日志截断

**历史线索**：某些 Arc 140V 环境曾在 int8 转换/传输处退出。先用最小 tensor 复现并记录驱动、PyTorch、oneDNN 版本；日志截断不能直接证明是该原因。

**两种触发路径**：

| 操作 | 后果 |
|------|------|
| `int8_tensor.to(torch.float16)` | 对比直接 cast 与 FP32 中转，定位是否为当前环境问题 |
| `int8_xpu_tensor.to("cpu")` | 检查直接 D2H 与分块传输的结果及峰值 |

**定位**（建议用 stderr 并立即 flush）：
```python
import sys
print("[debug] before op", file=sys.stderr, flush=True)
```

**局部诊断示例**（仅适用于反量化线性层，不替换 W8A8 GEMM）：
```python
# BAD
w = self.weight.to(torch.float16)
# GOOD
w = self.weight.to(torch.float32) * self.weight_scale
return F.linear(x, w.to(x.dtype), ...)
```

**二维 int8 矩阵的分块传输示例**（确认复现后再采用）：
```python
def _xpu_int8_to_cpu_chunked(t, chunk_mb=50):
    torch.xpu.empty_cache()
    cpu_out = torch.empty_like(t, device="cpu")
    chunk_rows = max(1, (chunk_mb << 20) // (t.shape[1] * 4))
    for start in range(0, t.shape[0], chunk_rows):
        end = min(start + chunk_rows, t.shape[0])
        cpu_out[start:end] = t[start:end].to(torch.float32).cpu().to(torch.int8)
    return cpu_out
```

---

## Case 4: OOM（内存不足）

### 4a — 加载文本编码器时 OOM

先区分权重加载峰值、forward 激活和其他组件驻留，读取实际设备容量；不能固定按 16 GB 判断。

检查该编码器现有的量化与 offload 选项（见 [INT8 文本编码器](../SKILL.md#int8-文本编码器)）；没有实现时不能靠配置自动启用。

### 4b — 第二次 infer 时 OOM

比较两次请求结束后的 allocated/reserved、CPU 内存和存活引用，检查 encoder、KV/shape cache 与 offload cleanup。统一内存仍受总物理容量限制，但 `.to()` 是否创建副本或释放旧存储取决于引用与迁移方式。复用现有 runner 的组件生命周期，不添加与实际设备状态脱节的 `_model_on_device` 标志。

### 4c — int8 中间态 OOM

检查大 embedding 表转 float32 时的临时 tensor 峰值。确认是该路径后，可评估 `_xpu_int8_to_cpu_chunked` 分块处理（见 Case 3）；`empty_cache()` 不会释放仍被引用的 tensor。

---

## Case 5: KeyError / AttributeError 权重加载失败

### KeyError: 'transformer_blocks.0.xxx.weight'

**最可能原因**：lazy 模式下 `weight_dict` 只含 `non_block.safetensors`，block 权重在磁盘，但 block 未传 `lazy_load_path`。

```python
# BAD
MyModelBlock(i, ..., create_cuda_buffer=True)
# GOOD
MyModelBlock(i, ..., create_cuda_buffer=True,
             lazy_load=self.lazy_load, lazy_load_path=lazy_load_path)
```

**其他原因**：INT8/FP8 key 名与 `_apply_weights` 不一致 → `safe_open('block_0.safetensors')` 打印 key 逐一对照。

### AttributeError: 'MyAttnWeights' has no attribute 'load_state_dict_from_disk'

先检查算子被注册为 parameter 还是 module。当前 `WeightModule.load_state_dict_from_disk()` 对 parameter 有 `hasattr` 检查，对 module 则直接递归；平台 `AttnWeightTemplate` 尚无该方法。仅在目标 lazy 路径确实将无状态 attention 放入 `_modules` 时补齐 no-op，不要宣称公共模板已具备它：

```python
# lightx2v_platform/ops/attn/template.py
def load_state_dict_from_disk(self, *args, **kwargs):
    pass
```

---

## Case 6: NoneType / Stream / 注意力限制崩溃

### TypeError: 'NoneType' object is not callable

**最可能原因**：CUDA-only 库（flash_attn / flashinfer）在 XPU 初始化为 None，或 attn_type 配置错误。

按模型的配置字段选择已构建且契约匹配的 backend：Linux 可评估 `intel_xpu_cute_attn`；Windows 构建关闭 CUTE，可评估 `intel_xpu_flash_attn` 或 `torch_sdpa`。Wan 使用 `self_attn_1_type` 等字段，H3 使用 `attn_type`，不能只改一个通用 key。

### 因果掩码 / 单向注意力语义不匹配

两个 XPU adapter 都未传递 causal/mask，可能静默产生错误结果，而不是抛出 `AttributeError`。

**修复**：使用支持该语义的 backend；例如 H3 配置字段为：
```json
{"attn_type": "torch_sdpa"}
```

同时检查调用方是否向 `torch_sdpa.apply()` 传入 `causal=True` / `attn_mask`；仅切换 backend 不会自动添加因果语义。

### Stream 推理崩溃

对照当前 Wan/H3 offload infer：XPU 分支在调用方 stream 上执行 `run_block()`，由 `swap_blocks()` 做设备同步。公共 manager 的非 CUDA 分支仍创建 `compute_stream(priority=-1)`；不能把“所有 stream 均无 priority”写成现状，也不要为单设备问题全局改写调度。

---

## Case 7: TP 报错 `torch.addmm` 维度不匹配（列并行 bias 未切分）

**最可能原因**：TP 权重切分逻辑把"`tensor.ndim < 2` 就跳过切分"当成通用规则，导致列并行（column-parallel）linear 的 1-D bias/scale 也被一起跳过，权重 out_dim 切了但 bias 没切。例如 TP=2 时 AdaLN weight out_dim 被切成 48384，但 bias 仍是原始的 96768，`torch.addmm` 两者维度对不上。

**修复**：按张量的语义区分处理，不能只看维度数：
```python
def _tp_split(self, tensor, key):
    split_type = self._tp_split_type(key)
    if split_type is None:
        return tensor
    if tensor.ndim == 0:          # 标量元数据（如 ConvRot 的 groupsize），所有 rank 共享
        return tensor
    if split_type == "row":
        if tensor.ndim < 2:       # 行并行的 bias 属于已 reduce 的输出，保持复制
            return tensor
        ...                       # 行并行权重：按输入维度切分
    # 列并行：weight 和 1-D 的 bias/scale 都要跟着输出维度一起切
    ...
```

**判断标准**：新模型接入 TP 时，检查切分逻辑是否只按 `tensor.ndim < 2` 一刀切保留复制——真正应该复制的只有标量元数据和行并行 bias，列并行层的 1-D bias/scale 必须和它对应的权重一起切分。

---
