# /xpu — Intel XPU 新模型接入 & 崩溃排查

LightX2V on Intel Arc 140V：接入新模型 + 推理崩溃排查。

---

## Quick Start

**接入新模型**：
```
"Use xpu skill. 接入新模型 {ModelName}，DiT ~XX GB BF16，XX transformer blocks。
Model path: /path/to/{ModelName}"
```

**排查崩溃**：
```
"Use xpu skill. 进程静默消失，无任何输出。Stage: 权重加载。文件大小: ~XX GB"
"Use xpu skill. OOM at second infer. Error: Tried to allocate XX MiB."
"Use xpu skill. 推理中途日志截断，Stage: 文本编码器 forward"
```

---

# 常见场景

## Case 1: 接入新模型到 Intel XPU

**Ask Claude**：
```
"Use xpu skill. 接入新模型 {ModelName} 到 Intel XPU.
Model path: /path/to/{ModelName}
DiT: ~XX GB BF16, XX transformer blocks
Text encoder: {TextEncoder} ~XX GB
优先使用 INT8 量化方案"
```

**关键决策**：`cpu_offload`（必要时加 `lazy_load`）能跑任意大小的 BF16 DiT——`offload_granularity=block` 下 XPU 上一次只需要 ~2 个 block 的显存，量化不是跑起来的必要条件，而是"缩小体积 + 减少每步搬运量"的可选优化。真正决定要不要量化的是**内存 profile**，不是单看 DiT 体积：

| 硬件 profile | 内存特点 | 建议 |
|---|---|---|
| 独立显卡 + 大内存服务器（如 Arc Pro B60） | 显存和系统内存分离，系统内存通常远大于模型体积 | `cpu_offload` 即可，不量化也能跑——MiniMax H3 62 GB BF16 DiT 就是这样跑的（`configs/platforms/intel_xpu/minimax_h3_t2av.json`，无 `dit_quantized`） |
| Arc 140V 等统一内存设备（32 GB 全局共享） | DiT + 文本编码器 + VAE + OS 挤在同一 32 GB 池子里，CPU/XPU 抢同一块内存 | DiT 越大越吃紧，量化（体积、搬运量减半）+ `lazy_load`（不用整份常驻内存）更容易让 pipeline 撑住 |

粗略参考（**经验值，不是硬性门槛**，具体看整机可用内存）：

| DiT BF16 体积 | 独立显卡 + 大内存 | Arc 140V 等统一内存 |
|---|---|---|
| < 5 GB | 直接加载 | 直接加载 |
| 5–16 GB | `cpu_offload` 足够，量化可选 | 视整机可用内存，量化开始有用 |
| > 16 GB | `cpu_offload`（+ `lazy_load`）仍然可行，量化是加速/省盘的可选项 | 量化 + `lazy_load` 通常是让 pipeline 挤进 32 GB 的现实选择 |

**说明**：量化是可选的性能/内存优化，不是运行的先决条件。要量化时，INT8 性能优于 FP8，推荐优先使用。

→ 完整 8 步流程见[接入新模型：完整流程](#接入新模型完整流程)

---

## Case 2: 进程静默消失，无任何输出

**最可能原因**：safetensors 默认 mmap，单文件 > ~5 GB 时 Windows OS 在 C++ 层 kill 进程，Python 无法捕获。

**诊断**：
```bash
python -c "import os; print(os.path.getsize('MODEL_PATH/model.safetensors') / 1e9, 'GB')"
```

**修复**：改用 `_read_tensor_no_mmap(path, key, target_dtype=torch.bfloat16)`。INT8/FP8 转换脚本中使用 `struct+readinto` 方式（Step 2 模板已内置）。

**Ask Claude**：
```
"Use xpu skill. 加载权重时进程静默消失，无输出。
使用 from_pretrained 加载，文件大小 ~XX GB"
```

---

## Case 3: 推理中途静默 kill，日志截断

**最可能原因**：int8 tensor op 触发 Arc 140V driver SIGABRT。loguru 有缓冲，崩溃前来不及 flush。

**两种触发路径**：

| 操作 | 后果 |
|------|------|
| `int8_tensor.to(torch.float16)` | 触发 oneDNN SIGABRT |
| `int8_xpu_tensor.to("cpu")` | 触发 Level-Zero driver 崩溃 |

**定位**（必须用 stderr）：
```python
import sys
print("[debug] before op", file=sys.stderr, flush=True)
```

**修复 int8 → fp16**（两步 cast）：
```python
# BAD
w = self.weight.to(torch.float16)
# GOOD
w = self.weight.to(torch.float32) * self.weight_scale
return F.linear(x, w.to(x.dtype), ...)
```

**修复 int8 XPU → CPU**（分块转移）：
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

**Ask Claude**：
```
"Use xpu skill. 推理中途日志截断，SIGABRT。
Stage: DequantLinearInt8 forward，使用 int8 weight"
```

---

## Case 4: OOM（内存不足）

### 4a — 加载文本编码器时 OOM

**最可能原因**：文本编码器 fp16 体积接近或超过 XPU 可用内存（约 16 GB）。

**修复**：在线 INT8 量化加载（见 [XPU 平台约束 → INT8 文本编码器](#int8-文本编码器)）。

**Ask Claude**：
```
"Use xpu skill. 加载文本编码器 OOM。
模型: {TextEncoder}，fp16 约 XX GB"
```

### 4b — 第二次 infer 时 OOM

**最可能原因**：统一内存下 CPU/XPU 共享物理池，`.to(device)` 不释放物理内存，第二次调用重复分配。

**修复**：
```python
def infer(self, texts):
    if not getattr(self, "_model_on_device", False):
        self.model = self.model.to(AI_DEVICE)
        self._model_on_device = True
    # 不在每次 infer 后 .to("cpu")
    # 由 runner unload_modules=true 统一释放
```

### 4c — int8 中间态 OOM

**最可能原因**：大 embedding 表转 float32 临时 tensor 过大。先 `torch.xpu.empty_cache()` 再用 `_xpu_int8_to_cpu_chunked` 分块处理（见 Case 3）。

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

`WeightModule.load_state_dict_from_disk` 递归所有 `_modules`，AttnWeight 子类必须有 no-op：

```python
# lightx2v_platform/ops/attn/template.py
def load_state_dict_from_disk(self, *args, **kwargs):
    pass
```

**Ask Claude**：
```
"Use xpu skill. KeyError: transformer_blocks.0.attn.to_q.weight，
使用 lazy_load=True，offload_granularity=block"
```

---

## Case 6: NoneType / Stream / 注意力限制崩溃

### TypeError: 'NoneType' object is not callable

**最可能原因**：CUDA-only 库（flash_attn / flashinfer）在 XPU 初始化为 None，或 attn_type 配置错误。

**修复**（config）：
```json
// Linux
"attn_type": "intel_xpu_cute_attn",

// Windows（cute_attn 不支持）
"attn_type": "intel_xpu_flash_attn"
```

### AttributeError: CUTE attention 不支持因果掩码 / 单向注意力

**最可能原因**：模型使用单向或因果注意力，但 config 设置了 `intel_xpu_cute_attn` 或 `intel_xpu_flash_attn`（两者均仅支持双向）。

**修复**：改用标准 PyTorch attention：
```python
# config
"attn_type": "torch_sdpa"  # 支持单向和因果掩码
```

**判断标准**：检查 attn 实现中是否有 `causal_mask`、`is_causal` 或类似的单向掩码标记。

### Stream 推理崩溃

**最可能原因**：`priority=-1` 在 Arc 140V 不支持 compute kernel。

```python
# BAD
torch.xpu.Stream(priority=-1)
# GOOD
torch.xpu.Stream()  # 不设 priority，copy 和 compute 均可用
```

---

# XPU 平台约束

## 统一内存

Arc 140V 总内存 32 GB（LPDDR5X），XPU 侧约 16 GB（PyTorch 可见 16.46 GiB）。CPU 与 XPU **共享同一物理内存池**。

| ✅ DO | ❌ DON'T |
|------|---------|
| `_model_on_device` flag 避免重复 `.to(device)` | 每次 infer 后 `.to("cpu")` 再 `.to(xpu)` |
| `unload_modules=true` 统一释放不用的组件 | 以为 CPU↔XPU 移动会释放物理内存 |
| 文本编码器体积接近 16 GB → INT8 量化 | 直接 fp16 加载超大文本编码器 |

## Stream 与同步

| ✅ DO | ❌ DON'T |
|------|---------|
| 所有 stream 用 `torch.xpu.Stream()`（无 priority） | `Stream(priority=-1)` 用于 compute kernel |
| `swap_blocks()` 前调 `torch.xpu.synchronize()`（device-wide） | 仅依赖 per-stream sync |

`swap_blocks()` 必须 device-wide sync：XPU 跨 stream 无内存可见性保证。

## pin_memory + non_blocking

必须同时启用，否则 offload 流水退化为同步拷贝：
- `create_cpu_buffers()` 分配 page-locked 内存，H2D 带宽 ~6 → ~14 GB/s
- `non_blocking=True` 使 H2D copy 与 compute kernel 真正重叠

## INT8 DiT 权重

DiT INT8 量化相比 FP8 提供更高精度和性能。**不是**反量化成 float 后走普通 matmul（那样等于退化成 bf16 matmul，没有性能收益）——`MMWeightInt8IntelXpu`（`lightx2v_platform/ops/mm/intel_xpu/mm_weight.py`）走的是动态 per-token W8A8：激活先在线量化成 int8，再调用真正的 INT8×INT8 GEMM 内核：

```python
def apply(self, input_tensor):
    # 激活 + 权重均为 int8，走真正的 INT8 GEMM kernel，而非反量化后 matmul
    output = sycl_kernels.onednn_w8a8_int8(
        input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous(),
        self.weight.contiguous(),                        # int8, [out_dim, in_dim]
        self.weight_scale.reshape(-1).float().contiguous(),
        self.bias,
    )
    return output.reshape(*input_tensor.shape[:-1], self.weight.shape[0])
```

**关键点**：INT8 scale 为每行（per-channel）保存（shape `(out_dim, 1)`），与 FP8 一致；`int8-intel-xpu` 要求激活为 FP16/BF16，且需要 `lightx2v_kernel_xpu` 编译出 `sycl_kernels.onednn_w8a8_int8`（见下方"前置条件"）。

---

## 多卡分布式推理（Linux 仅）

⚠️ **Windows 不支持多卡**，仅 Linux 可用。XPU 多卡使用 `torchrun` + PyTorch XCCL/oneCCL。

**进程数和并行大小关系**：
```
进程数 = tensor_p_size × seq_p_size × cfg_p_size
```

**启动命令（Linux）**：
```bash
export ZE_AFFINITY_MASK=0,1,2,3          # 选择卡 0-3
export PLATFORM=intel_xpu
export PYTHONFAULTHANDLER=1

# TP（张量并行）4卡
torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_tp.json ...

# SP+TP（序列+张量并行）8卡
export CCL_SYCL_ALLTOALL_ARC_LL=1
export CCL_SYCL_ALLTOALL_TMP_BUF=1
export CCL_SYCL_CCL_BARRIER=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_sp_tp.json ...
```

**并行策略**：
- **TP**：模型不适合单卡或需要分割 linear layer
- **SP**：长视频或高分辨率，使用 Ulysses all-to-all 通信
- **CFG**：两组设备分别计算正负条件分支（通常 `cfg_p_size=2`）
- **SP+TP**：先单独验证 TP 和 SP，再组合

**要点**：
- `ZE_AFFINITY_MASK` 选择卡（不用 `CUDA_VISIBLE_DEVICES`）
- oneCCL 变量仅 SP/SP+TP 需要，TP 不需要
- 多卡不一定线性加速，小模型/低分辨率可能通信开销更大
- 文本编码器/VAE 可能仍在所有卡上复制，不能简单按卡数除以内存

---

## INT8 文本编码器

体积接近 16 GB 时必须在线量化加载。加载流程：

1. `init_empty_weights` 建模型骨架（0 RAM，全是 meta tensor）
2. `_replace_linear_int8` 递归替换所有 `nn.Linear`（仍 0 RAM）
3. 逐 shard `struct+readinto` 读取，fp16 → int8 即时量化赋值，peak RAM ≈ 单 shard

`DequantLinearInt8.forward` 关键（两步 cast，绕过 oneDNN SIGABRT）：
```python
def forward(self, x):
    w = self.weight.to(torch.float32) * self.weight_scale  # int8→fp32
    return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
```

**non-persistent buffer**：`init_empty_weights` 后 `register_buffer(..., persistent=False)`（如 RoPE `inv_freq`）仍是 meta tensor，加载权重后需手动重算。

---

# 接入新模型：完整流程

## 第一步：读取原始权重，确定架构参数

```bash
python -c "
from safetensors import safe_open
import glob, os
for s in sorted(glob.glob('MODEL_PATH/**/*.safetensors', recursive=True))[:2]:
    with safe_open(s, framework='pt') as f:
        keys = list(f.keys())
    print(os.path.basename(s), len(keys), 'keys')
    for k in keys[:20]: print(' ', k)
"
```

| 需要确认 | 用途 |
|---------|------|
| block key 正则（如 `transformer_blocks\.(\d+)\.`） | converter.py `model_type_keys_map[...]["key_idx"]`；流式脚本 fallback 用 `_BLOCK_RE` |
| block 总数 | 权重类 `num_layers` |
| 需量化的子模块名（attention + FFN 的 2D weight） | converter.py `model_type_keys_map[...]["target_keys"]`；流式脚本 fallback 用 `_TARGET_PARTS` |
| 文本编码器体积是否接近 16 GB | 是否需 INT8 量化 |
| 调度器类型（Flow Matching / DDIM） | Scheduler 实现 |

---

## 第二步：INT8/FP8 转换

### 优先扩展通用转换工具

`tools/convert/converter.py` 已支持 `wan_dit` / `h3` / `hunyuan_dit` / `wan_t5` / `wan_clip` / `wan_animate_dit` / `qwen_image_dit` / `qwen25vl_llm` / `z_image_dit` / `self_forcing` 等 `--model_type`。量化算法是 per-channel 对称 int8/fp8（`tools/convert/quant/quant.py` 的 `QuantWeightINT8`/`QuantWeightFP8`：`scales = max_val.amax(dim=1) / 127`），scale shape `(out_dim, 1)`，和运行时 `MMWeightInt8IntelXpu`/`MMWeightFp8IntelXpu` 的加载格式完全兼容。**新模型优先在这里加一个 `model_type`，而不是新写一份转换脚本。**

在 `tools/convert/converter.py` 里改 3 处：

1. **key 映射**（仅当原始 checkpoint 的 key 名和 LightX2V runtime 权重类注册的 key 名不一致时才需要）：在 `get_key_mapping_rules()` 里加一段 `forward`/`backward` 正则规则（参考其中 `wan_dit` 的写法）。如果原始 checkpoint 已经用 LightX2V runtime key（比如 MiniMax H3），像 `h3` 一样直接 `return []`。
2. **注册 model_type**：`main()` 里 `--model_type` 的 `choices=[...]` 加上新模型名。
3. **量化范围**：`main()` 里的 `model_type_keys_map` 字典加一条（用 Step 1 确认的 block key 正则位置和量化子模块名）：
   ```python
   "{model}": {
       "key_idx": 2,                       # key.split(".") 中子模块名的下标（block key 正则里那个位置）
       "target_keys": ["attn", "ffn"],      # 需要量化的 2D weight 所在子模块名
       "ignore_key": None,                  # 整条 key 都跳过加载的子串，没有就 None
       # "preserve_non_quant_dtype": True,  # 模型里混有需要保留精度的 FP32 tensor 时打开（参考 h3）
   },
   ```

**运行**（参考真实的 MiniMax H3 命令 `tools/convert/examples/convert_minimax_h3_int8_cpu.sh`）：
```bash
python tools/convert/converter.py \
    --source <原始目录或单个 safetensors> \
    --output <int8目录> \
    --output_name {model}_int8 \
    --model_type {model} \
    --quantized --linear_type int8 \
    --device cpu \
    --single_file
```
FP8 把 `--linear_type` 换成 `fp8` 即可，其余不变。需要 `lazy_load` 分块加载时，把 `--single_file` 换成 `--save_by_block`——转换器按 key 里的 `blocks\.(\d+)\.` 正则自动分组成 `block_N.safetensors` + `non_block.safetensors` + `index.json`，`_scale` 权重会和对应权重落在同一分片（这条路径在本仓库暂无现成 example 验证过，用前建议先跑通并抽查一个 block 的 key/dtype）。

### ⚠️ Windows / Arc 140V 上转换大文件的内存陷阱

`converter.py` 用 `safe_open(...).get_tensor()` 把**所有**源文件一次性读进内存再统一量化保存，没有流式读取。这意味着它和普通 `from_pretrained` 加载一样会撞上 [Case 2](#case-2-进程静默消失无任何输出) 的问题：单个 safetensors 文件 > ~5 GB 时，Windows 下 safetensors 默认的 mmap 会被 OS 在 C++ 层 kill 掉，Python 侧完全捕获不到。

- 原始 DiT 已经是多个 < 5 GB 的 shard，或者转换是在 Linux/内存充足的机器上做的（MiniMax H3 的 INT8 转换实际就是这样跑的，见上面的 example 脚本）→ 直接用 `converter.py`。
- 必须在 Arc 140V/Windows 本机转换、且单文件 > 5 GB → `converter.py` 目前不安全，改用下面的流式转换脚本，或者先在别的机器转换好再把量化产物拷过来。

### 内存受限：流式转换脚本（fallback）

仅在上面两条路都走不通时才新建 `tools/convert/{model}_int8_convert.py`。**只改顶部三处**（`# ← 修改`）：

```python
#!/usr/bin/env python3
import argparse, gc, json, os, re, struct
from collections import defaultdict
import torch
from loguru import logger
from safetensors.torch import save_file

# ← 修改 1：block key 正则
_BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")

# ← 修改 2：需量化的子模块名（只量化 2D weight）
_TARGET_PARTS = {"attn", "ff"}

# ← 修改 3：_TARGET_PARTS 在 key.split(".") 中的位置（0-indexed）
# transformer_blocks.0.attn.to_q.weight → split[2]="attn" → _KEY_IDX=2
_KEY_IDX = 2

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_NON_LIN_DTYPE = torch.bfloat16
_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
}

def _block_index(key):
    m = _BLOCK_RE.search(key); return int(m.group(1)) if m else None

def _should_quant(key, tensor):
    if tensor.dim() != 2: return False
    parts = key.split(".")
    return len(parts) > _KEY_IDX and parts[_KEY_IDX] in _TARGET_PARTS

def _int8_quant(w):
    # INT8：[-128, 127]，精度高于 FP8
    w_f32 = w.float()
    max_v = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = (max_v / 127).to(torch.float32)
    return (w_f32 / scales).round().clamp(-128, 127).to(torch.int8), scales

def _fp8_quant(w):
    # FP8：fallback 方案
    w_f32 = w.float()
    max_v = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = (max_v / _FP8_MAX).to(torch.float32)
    return (w_f32 / scales).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn), scales

# ← 修改 0：量化方案（"int8" 或 "fp8"）
_QUANT_SCHEME = "int8"

def _quant(w):
    return _int8_quant(w) if _QUANT_SCHEME == "int8" else _fp8_quant(w)

def _read_tensors(src_path, data_start, items):
    result = {}
    with open(src_path, "rb") as fh:
        file_pos = data_start; fh.seek(file_pos)
        for name, begin, end, dtype_str, shape in items:
            abs_begin = data_start + begin
            if file_pos != abs_begin: fh.seek(abs_begin); file_pos = abs_begin
            buf = bytearray(end - begin); mv = memoryview(buf); n_read = 0
            while n_read < len(buf):
                chunk = fh.readinto(mv[n_read:])
                if not chunk: raise EOFError(f"EOF reading '{name}'")
                n_read += chunk
            file_pos = abs_begin + len(buf)
            t = torch.frombuffer(buf, dtype=_ST_DTYPE.get(dtype_str, torch.bfloat16))
            result[name] = t.reshape(shape if shape else []).clone(); del buf, mv
    return result

def _write_block(tensors, block_id, output_dir):
    fname = f"block_{block_id}.safetensors"; d, wm = {}, {}
    for key, t in tensors.items():
        if _should_quant(key, t):
            w_q, scales = _quant(t)  # 使用统一的量化函数
            d[key] = w_q; d[key + "_scale"] = scales
            wm[key] = wm[key + "_scale"] = fname
        else:
            d[key] = t.to(_NON_LIN_DTYPE) if t.dtype.is_floating_point else t; wm[key] = fname
    save_file(d, os.path.join(output_dir, fname)); del d; gc.collect()
    return wm

def convert(source_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(source_path, "rb") as f:
        hdr_n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(hdr_n))
    data_start = 8 + hdr_n
    block_items, non_block_items = defaultdict(list), []
    for name, meta in header.items():
        if name == "__metadata__": continue
        begin, end = meta["data_offsets"]
        entry = (name, begin, end, meta["dtype"], meta["shape"])
        bid = _block_index(name)
        (block_items[bid] if bid is not None else non_block_items).append(entry)

    logger.info(f"{len(block_items)} blocks | {len(non_block_items)} non-block tensors")
    out_wm = {}
    for i, bid in enumerate(sorted(block_items)):
        tensors = _read_tensors(source_path, data_start, sorted(block_items[bid], key=lambda x: x[1]))
        out_wm.update(_write_block(tensors, bid, output_dir)); del tensors; gc.collect()
        if (i + 1) % 10 == 0: logger.info(f"  {i+1}/{len(block_items)} done")

    if non_block_items:
        tensors = _read_tensors(source_path, data_start, sorted(non_block_items, key=lambda x: x[1]))
        fname, d = "non_block.safetensors", {}
        for key, t in tensors.items():
            if _should_quant(key, t):
                w_q, scales = _quant(t); d[key] = w_q; d[key + "_scale"] = scales
                out_wm[key] = out_wm[key + "_scale"] = fname
            else:
                d[key] = t.to(_NON_LIN_DTYPE) if t.dtype.is_floating_point else t; out_wm[key] = fname
        save_file(d, os.path.join(output_dir, fname)); del d, tensors; gc.collect()

    total = sum(os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir) if f.endswith(".safetensors"))
    idx = os.path.join(output_dir, "diffusion_pytorch_model.safetensors.index.json")
    with open(idx, "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": out_wm}, f, indent=2)
    logger.info(f"Done. {total/1e9:.2f} GB → {idx}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True); p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    idx = os.path.join(args.output, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(idx) and not args.force:
        logger.info("Already converted. Use --force to re-run."); return
    convert(args.source, args.output)

if __name__ == "__main__":
    main()
```

多 shard：Phase 1 遍历所有 shard 合并 `block_items`/`non_block_items`；Phase 2/3 记录每个 tensor 来源 shard。

验证（流式脚本 / `converter.py --save_by_block` 输出的分块产物）：
```bash
python -c "
from safetensors import safe_open
with safe_open('OUTPUT_INT8_PATH/block_0.safetensors', framework='pt') as f:
    for k in f.keys():
        t = f.get_tensor(k)
        print(k, tuple(t.shape), t.dtype)
# 期望（INT8）：2D weight → int8，_scale → float32 (out_dim,1)，其余 → bfloat16
# 期望（FP8）：2D weight → float8_e4m3fn，_scale → float32 (out_dim,1)，其余 → bfloat16
"
```

验证（`converter.py --single_file` 输出的单文件产物）：
```bash
python -c "
from safetensors import safe_open
with safe_open('OUTPUT_INT8_PATH/{model}_int8.safetensors', framework='pt') as f:
    keys = list(f.keys())
    for k in keys[:20]:
        t = f.get_tensor(k)
        print(k, tuple(t.shape), t.dtype)
"
```

---

## 第三步：创建文件骨架

```bash
MODEL=your_model_name

mkdir -p lightx2v/models/networks/$MODEL/weights
mkdir -p lightx2v/models/networks/$MODEL/infer/offload
touch lightx2v/models/networks/$MODEL/__init__.py
touch lightx2v/models/networks/$MODEL/weights/__init__.py
touch lightx2v/models/networks/$MODEL/infer/__init__.py
touch lightx2v/models/networks/$MODEL/infer/offload/__init__.py
mkdir -p lightx2v/models/video_encoders/hf/$MODEL
mkdir -p lightx2v/models/input_encoders/hf/$MODEL
mkdir -p lightx2v/models/runners/$MODEL
mkdir -p lightx2v/models/schedulers/$MODEL
touch lightx2v/models/runners/$MODEL/__init__.py
touch lightx2v/models/schedulers/$MODEL/__init__.py
```

| 文件 | 内容 |
|------|------|
| `networks/{model}/weights/pre_weights.py` | patch_embed、time_embed 等全局权重 |
| `networks/{model}/weights/transformer_weights.py` | block 权重 + offload buffer（见第四步） |
| `networks/{model}/infer/transformer_infer.py` | 无 offload 推理（见第五步） |
| `networks/{model}/infer/offload/transformer_infer.py` | offload 推理（见第五步） |
| `networks/{model}/infer/pre_infer.py` | patchify + position embed + time embed |
| `networks/{model}/infer/post_infer.py` | unpatchify |
| `networks/{model}/model.py` | 主模型类（见第六步） |
| `video_encoders/hf/{model}/vae.py` | 包装 diffusers AutoencoderKL* |
| `input_encoders/hf/{model}/text_encoder.py` | 包装 T5 / CLIP / Qwen 等 |
| `runners/{model}/{model}_runner.py` | Runner 注册（见第七步） |
| `schedulers/{model}/{model}_scheduler.py` | Flow Matching / DDIM 等 |

---

## 第四步：实现 `weights/transformer_weights.py`

`MyModelTransformerWeights(WeightModule)` 中创建三组 WeightModuleList（**名称固定，框架依赖**）：

| 属性名 | 数量 | create_cuda_buffer | create_cpu_buffer | 条件 |
|--------|------|--------------------|-------------------|------|
| `self.blocks` | `num_layers` | False | False | 始终 |
| `self.offload_block_cuda_buffers` | 2 | True | False | `cpu_offload` |
| `self.offload_block_cpu_buffers` | 2 | False | True | `cpu_offload` + `lazy` |

每组创建后调 `self.add_module(name, list)`。每个 block **必须**传 `lazy_load=lazy, lazy_load_path=lazy_load_path`。无 offload 时两个 buffer 属性设为 `None`。

`MyModelBlock(WeightModule)` 接收 `(block_index, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_path)`，按 Step 1 的 key 名用 `MM_WEIGHT_REGISTER[mm_type](key)` 注册 2D weight，`RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](key)` 注册 RMSNorm，`ROPE_REGISTER[rope_type](key)` 注册旋转位置编码。

**mm_type 选项**：
- `"Default"` = BF16 无量化
- `"fp8-intel-xpu"` = FP8 GEMM（`onednn_w8a16_fp8`；kernel 不可用时退化为反量化+matmul，fallback）
- `"int8-intel-xpu"` = INT8 W8A8 GEMM（`onednn_w8a8_int8`，激活动态量化，推荐）

**rms_type 选项**：
- `"torch_native"` / `"torch"` = 标准 RMSNorm
- `"intel_xpu"` = XPU 优化的 RMSNorm

**rope_type 选项**：
- `"torch"` = 标准旋转位置编码
- `"minimax_h3_xpu_rope"` = MiniMax H3 XPU 优化版

---

## 前置条件：编译 sycl_kernels

使用新内核（CUTE attention、XPU RMSNorm、INT8 GEMM）需要先编译 `sycl_kernels`。

**Windows**：
```cmd
cd lightx2v_kernel_xpu
call build.bat
pip install dist\sycl_kernels-0.0.1-cp311-win_amd64.whl --force-reinstall --no-deps
```

**Linux**：
```bash
source /opt/intel/oneapi/setvars.sh
cd lightx2v_kernel_xpu
./build.sh
pip install dist/sycl_kernels-0.0.1-cp311-linux_x86_64.whl --force-reinstall --no-deps
```

版本对齐：oneAPI、oneDNN、PyTorch 版本必须匹配。

---

## 第五步：实现推理类

`infer/transformer_infer.py` 基础结构：`infer()` 调 `infer_func(weights.blocks, x, pre_infer_out)`，`infer_without_offload` 顺序遍历所有 block，`infer_block` 实现单 block 的 AdaLN + attn + FFN + 残差。

**新算子集成**：
```python
class MyModelTransformerInfer:
    def infer_block(self, block_weights, x, pre_infer_out):
        # INT8 W8A8 GEMM（自动在 attn/FFN 的 mm_weight.apply() 中调用，非反量化）
        # 使用新的 attention 实现
        x = self.attn(x, block_weights.attn)  # 内部使用 intel_xpu_cute_attn（Linux）或 intel_xpu_flash_attn（Windows）

        # 使用新的 RMSNorm（XPU 优化）
        x = self.norm(x, block_weights.norm)  # 内部使用 intel_xpu RMSNorm

        # 使用新的旋转位置编码
        # rope 在 pre_infer 或 attn 内调用 minimax_h3_xpu_rope
        return x
```

**注意**：
- `intel_xpu_cute_attn` 和 `intel_xpu_flash_attn` 均**仅支持双向注意力**（Bidirectional），不支持单向或因果掩码
- `intel_xpu_cute_attn` 在 **Windows 平台不可用**，此时必须用 `intel_xpu_flash_attn`
- 对于单向注意力（如 decoder-only）的模型，需改用 `"attn_type": "torch_sdpa"`

offload 版（`infer/offload/transformer_infer.py`），`__init__` 创建 `WeightAsyncStreamManager(offload_granularity=granularity)`，按粒度设 `infer_func`；phase+lazy 时调 `init_lazy_load(num_workers)`。

**Offload 粒度选择**：

| 粒度 | 缓冲数 | 何时用 | 特点 |
|------|--------|--------|------|
| **block** | 2 个完整 block | 标准情况 | 一次加载整个 block，双缓冲流水 |
| **phase** | 3 个 phase（attn/cross_attn/FFN） | 极限内存 | 细粒度 offload，内存峰值最低 |

**Block offload 核心循环**：
```python
def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
    for block_idx in range(len(blocks)):
        if self.offload_manager.need_init_first_buffer:
            self.offload_manager.init_first_buffer(blocks)
        self.offload_manager.prefetch_weights((block_idx + 1) % len(blocks), blocks)
        with torch_device_module.stream(self.offload_manager.compute_stream):
            x = self.infer_block(self.offload_manager.cuda_buffers[0], x, pre_infer_out)
        self.offload_manager.swap_blocks()  # device-wide sync + swap ping/pong
    return x
```

**Phase offload**：在上述基础上细分为 3 个 phase（self_attn/cross_attn/FFN），lazy 时额外调用 `start_prefetch_block` / `swap_cpu_buffers` / `prefetch_phase` / `swap_phases`。内存更紧张时使用。

---

## 第六步：实现 `model.py`

继承 `BaseTransformerModel`，设置 `pre_weight_class` / `transformer_weight_class` 类属性。

`__init__` 关键点：
- lazy 时 `self.remove_keys.extend(["transformer_blocks."])` —— 跳过 block 权重初始加载
- `_init_infer_class` 按 `self.cpu_offload` 选择 offload 或普通推理类
- `_init_infer` 末尾：若 `transformer_infer` 有 `offload_manager`，调 `self._init_offload_manager()`（连接 cuda/cpu buffers，由基类提供）

---

## 第七步：创建 Runner 并注册

**创建 `runners/{model}/{model}_runner.py`**：
```python
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER

@RUNNER_REGISTER("{model_cls}")
class MyModelRunner(DefaultRunner):
    def load_transformer(self):
        return MyModel(self.config["model_path"], self.config, self.init_device)
    def load_text_encoder(self):
        return [MyTextEncoder(self.config)]
    def load_vae_decoder(self):
        return MyVAE(self.config)
    def init_scheduler(self):
        self.scheduler = MyScheduler(self.config)
```

**修改 `lightx2v/infer.py`**：
```python
import lightx2v.models.runners.{model}.{model}_runner  # noqa（触发注册）
parser.add_argument("--model_cls", choices=[..., "{model_cls}"])
```

**修改 `configs/model_pipeline.json`**：参考已有条目格式加入 pipeline 定义。

---

## 第八步：创建 XPU Config

**创建 `configs/platforms/intel_xpu/{model}_t2v.json`**：

标准配置 — **Linux 平台**（INT8 > 5 GB，推荐）：
```json
{
    "attn_type": "intel_xpu_cute_attn",
    "rms_type": "intel_xpu",
    "rope_type": "minimax_h3_xpu_rope",
    "cpu_offload": true,
    "offload_granularity": "block",
    "lazy_load": true,
    "feature_caching": "NoCaching",
    "dit_quantized": true,
    "dit_quant_scheme": "int8-intel-xpu",
    "dit_quantized_ckpt": "/path/to/int8_output",
    "vae_cpu_offload": true,
    "unload_modules": true
}
```

标准配置 — **Windows 平台**（INT8 > 5 GB，推荐）：
```json
{
    "attn_type": "intel_xpu_flash_attn",
    "rms_type": "intel_xpu",
    "rope_type": "minimax_h3_xpu_rope",
    "cpu_offload": true,
    "offload_granularity": "block",
    "lazy_load": true,
    "feature_caching": "NoCaching",
    "dit_quantized": true,
    "dit_quant_scheme": "int8-intel-xpu",
    "dit_quantized_ckpt": "/path/to/int8_output",
    "vae_cpu_offload": true,
    "unload_modules": true
}
```

**关键限制**：
- `intel_xpu_cute_attn` 不支持 Windows，改用 `intel_xpu_flash_attn`
- 两者均**仅支持双向注意力**（非因果掩码）
- 单向注意力模型需改用 `"attn_type": "torch_sdpa"`

**Fallback 配置（无 INT8 脚本时，使用 FP8）**：
```json
{
    "attn_type": "intel_xpu_flash_attn",
    "rope_type": "torch",
    "cpu_offload": true,
    "offload_granularity": "block",
    "lazy_load": true,
    "feature_caching": "NoCaching",
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-intel-xpu",
    "dit_quantized_ckpt": "/path/to/fp8_output",
    "vae_cpu_offload": true,
    "unload_modules": true
}
```

内存极限配置（phase + 磁盘预取，INT8）：
```json
{
    "attn_type": "intel_xpu_cute_attn",
    "rms_type": "intel_xpu",
    "rope_type": "minimax_h3_xpu_rope",
    "cpu_offload": true,
    "offload_granularity": "phase",
    "lazy_load": true,
    "num_disk_workers": 4,
    "dit_quantized": true,
    "dit_quant_scheme": "int8-intel-xpu",
    "dit_quantized_ckpt": "/path/to/int8_output"
}
```

运行验证：
```bash
export PLATFORM=intel_xpu
python lightx2v/infer.py \
    --model_cls {model_cls} --task t2v \
    --model_path /path/to/model \
    --config_json configs/platforms/intel_xpu/{model}_t2v.json \
    --prompt "A red ball bouncing" \
    --save_result_path output/test.mp4
```

---

# 快速参考

## 多卡快速检查

```bash
# 1. 验证设备可见
python -c 'import torch; print(torch.xpu.is_available(), torch.xpu.device_count())'

# 2. 计算进程数 = TP_SIZE × SP_SIZE × CFG_SIZE
# 3. 设置 ZE_AFFINITY_MASK 和 oneCCL 变量（SP/SP+TP 时）
# 4. 检查配置文件中的 parallel 部分
# 5. 启动 torchrun --nproc_per_node=<进程数>
```

| 并行策略 | 进程数 | oneCCL | 何时用 |
|---------|--------|--------|--------|
| TP 仅 | tensor_p_size | 否 | 模型不适合单卡 |
| SP 仅 | seq_p_size | 是 | 长视频/高分辨率 |
| CFG 仅 | cfg_p_size (通常=2) | 否 | Classifier-Free Guidance |
| SP+TP | seq_p_size×tensor_p_size | 是 | 需要同时分割 |

---

## 移植检查清单

- [ ] 单文件 > ~5 GB：改用 `_read_tensor_no_mmap`
- [ ] **量化（可选）**：仅当 DiT > 5 GB 时考虑，优先给 `tools/convert/converter.py` 加 `model_type`（见第二步），内存/mmap 撞坑时才退回流式脚本
  - [ ] 不需要 `lazy_load` → `--single_file` 单个 safetensors
  - [ ] 需要 `lazy_load` → `--save_by_block` 输出 `block_N.safetensors` + `non_block.safetensors` + `index.json`
  - [ ] INT8：scale shape `(out_dim, 1)` per row，dtype float32；weight dtype：int8
  - [ ] 或 FP8（fallback）：scale shape 同上；weight dtype：float8_e4m3fn
- [ ] Offload buffer 传参：每个 block 传 `lazy_load=True` + `lazy_load_path`
- [ ] `remove_keys`：lazy_load 时跳过 block 权重初始加载
- [ ] `AttnWeightTemplate` 子类有 `load_state_dict_from_disk` no-op
- [ ] Stream：不设 priority，直接 `torch.xpu.Stream()`
- [ ] `_init_offload_manager()` 在 `_init_infer` 末尾调用
- [ ] **新算子配置**（推荐，仅支持双向注意力）：
  - [ ] Linux：`attn_type: intel_xpu_cute_attn`
  - [ ] Windows：`attn_type: intel_xpu_flash_attn`（cute_attn 不支持）
  - [ ] `rms_type: intel_xpu`
  - [ ] `rope_type: minimax_h3_xpu_rope`
  - [ ] `dit_quant_scheme: int8-intel-xpu`
  - [ ] ⚠️ 单向注意力模型改用 `attn_type: torch_sdpa`
- [ ] 或 Fallback（无新内核时）：
  - [ ] `attn_type: torch_sdpa`（支持双向和单向注意力）
  - [ ] `rope_type: torch`
  - [ ] `dit_quant_scheme: fp8-intel-xpu`
- [ ] 文本编码器体积接近 16 GB：启用 INT8 量化
- [ ] `_model_on_device` flag 防止多次 infer OOM

## 调试工具

```python
# 定位 SIGABRT 前的最后位置（必须 stderr）
import sys
print("[debug] before op", file=sys.stderr, flush=True)

# XPU 内存状态
print(f"alloc: {torch.xpu.memory_allocated()/1e9:.2f} GB  "
      f"reserved: {torch.xpu.memory_reserved()/1e9:.2f} GB")
torch.xpu.empty_cache()
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `lightx2v/common/offload/manager.py` | `WeightAsyncStreamManager`：双缓冲核心，Stream 配置 |
| `lightx2v/common/modules/weight_module.py` | `WeightModule` / `WeightModuleList` 基类 |
| `lightx2v/common/ops/utils.py` | `_read_tensor_no_mmap`；`create_cuda/cpu_buffers` |
| `tools/convert/converter.py` | 通用转换 CLI；新模型加 `model_type`：`get_key_mapping_rules()` + `model_type_keys_map` |
| `tools/convert/quant/quant.py` | `CONVERT_WEIGHT_REGISTER`；`QuantWeightINT8`/`QuantWeightFP8`（per-channel 对称量化） |
| `lightx2v_platform/ops/mm/intel_xpu/mm_weight.py` | `MM_WEIGHT_REGISTER`；`MMWeightInt8IntelXpu`；`MMWeightFp8IntelXpu` |
| `lightx2v_platform/ops/norm/intel_xpu/xpu_rms_norm.py` | `PLATFORM_RMS_WEIGHT_REGISTER`（合并进 `RMS_WEIGHT_REGISTER`）；`IntelXpuRMSWeight` |
| `lightx2v_platform/ops/rope/intel_xpu/minimax_h3_rope.py` | `PLATFORM_ROPE_REGISTER`（合并进 `ROPE_REGISTER`）；`MiniMaxH3XpuRope` |
| `lightx2v/models/networks/base_model.py` | `BaseTransformerModel`；`_init_offload_manager` |
| `lightx2v/utils/registry_factory.py` | 所有 Register 注册表 |
| `lightx2v_platform/ops/attn/template.py` | `AttnWeightTemplate`：`load_state_dict_from_disk` no-op |
| `lightx2v_platform/ops/attn/intel_xpu/xpu_cute_attn.py` | `IntelXpuCuteAttnWeight`（CUTE 优化注意力） |
