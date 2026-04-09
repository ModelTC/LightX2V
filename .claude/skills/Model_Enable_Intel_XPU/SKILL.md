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
Text encoder: {TextEncoder} ~XX GB"
```

**关键决策**（DiT FP8 体积）：

| DiT FP8 | 策略 |
|---------|------|
| ≤ 7 GB | lazy offload（推荐） |
| > 7 GB | **必须** lazy offload，peak = 2 block |

→ 完整 8 步流程见[接入新模型：完整流程](#接入新模型完整流程)

---

## Case 2: 进程静默消失，无任何输出

**最可能原因**：safetensors 默认 mmap，单文件 > ~5 GB 时 Windows OS 在 C++ 层 kill 进程，Python 无法捕获。

**诊断**：
```bash
python -c "import os; print(os.path.getsize('MODEL_PATH/model.safetensors') / 1e9, 'GB')"
```

**修复**：改用 `_read_tensor_no_mmap(path, key, target_dtype=torch.bfloat16)`。FP8 转换脚本中使用 `struct+readinto` 方式（Step 2 模板已内置）。

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

**其他原因**：FP8 key 名与 `_apply_weights` 不一致 → `safe_open('block_0.safetensors')` 打印 key 逐一对照。

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

## Case 6: NoneType / Stream 崩溃

### TypeError: 'NoneType' object is not callable

**最可能原因**：CUDA-only 库（flash_attn / flashinfer）在 XPU 初始化为 None。

**修复**（config）：`"attn_type": "intel_xpu_flash_attn"`，`"rope_type": "torch"`

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
| block key 正则（如 `transformer_blocks\.(\d+)\.`） | FP8 脚本 `_BLOCK_RE` |
| block 总数 | 权重类 `num_layers` |
| 需量化的子模块名（attention + FFN 的 2D weight） | FP8 脚本 `_TARGET_PARTS` |
| 文本编码器体积是否接近 16 GB | 是否需 INT8 量化 |
| 调度器类型（Flow Matching / DDIM） | Scheduler 实现 |

---

## 第二步：FP8 转换脚本

新建 `tools/convert/{model}_fp8_convert.py`。**只改顶部三处**（`# ← 修改`）：

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

def _fp8_quant(w):
    w_f32 = w.float()
    max_v = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = (max_v / _FP8_MAX).to(torch.float32)
    return (w_f32 / scales).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn), scales

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
            w_q, scales = _fp8_quant(t)
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
                w_q, scales = _fp8_quant(t); d[key] = w_q; d[key + "_scale"] = scales
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

验证：
```bash
python -c "
from safetensors import safe_open
with safe_open('OUTPUT_FP8_PATH/block_0.safetensors', framework='pt') as f:
    for k in f.keys():
        t = f.get_tensor(k)
        print(k, tuple(t.shape), t.dtype)
# 期望：2D weight → float8_e4m3fn，_scale → float32 (out_dim,1)，其余 → bfloat16
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

`MyModelBlock(WeightModule)` 接收 `(block_index, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_path)`，按 Step 1 的 key 名用 `MM_WEIGHT_REGISTER[mm_type](key)` 注册 2D weight，`LN_WEIGHT_REGISTER["torch"](key)` 注册 norm。`mm_type="Default"` = BF16，`"fp8-intel-xpu"` = FP8 dequant。

---

## 第五步：实现推理类

`infer/transformer_infer.py` 基础结构：`infer()` 调 `infer_func(weights.blocks, x, pre_infer_out)`，`infer_without_offload` 顺序遍历所有 block，`infer_block` 实现单 block 的 AdaLN + attn + FFN + 残差。

offload 版（`infer/offload/transformer_infer.py`），`__init__` 创建 `WeightAsyncStreamManager(offload_granularity=granularity)`，按粒度设 `infer_func`；phase+lazy 时调 `init_lazy_load(num_workers)`。核心循环：

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

**phase 粒度**在此基础上细分为 3 个 phase（self_attn/cross_attn/FFN），lazy 时额外调用 `start_prefetch_block` / `swap_cpu_buffers` / `prefetch_phase` / `swap_phases`。

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

标准配置（FP8 > 7 GB）：
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

内存极限（phase + 磁盘预取）：
```json
{
    "cpu_offload": true,
    "offload_granularity": "phase",
    "lazy_load": true,
    "num_disk_workers": 4,
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-intel-xpu",
    "dit_quantized_ckpt": "/path/to/fp8_output"
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

## 移植检查清单

- [ ] 单文件 > ~5 GB：改用 `_read_tensor_no_mmap`
- [ ] FP8 输出格式：`block_N.safetensors` + `non_block.safetensors` + `index.json`
- [ ] FP8 scale shape：`(out_dim, 1)` per row
- [ ] Offload buffer 传参：每个 block 传 `lazy_load=True` + `lazy_load_path`
- [ ] `remove_keys`：lazy_load 时跳过 block 权重初始加载
- [ ] `AttnWeightTemplate` 子类有 `load_state_dict_from_disk` no-op
- [ ] Stream：不设 priority，直接 `torch.xpu.Stream()`
- [ ] `_init_offload_manager()` 在 `_init_infer` 末尾调用
- [ ] `attn_type: intel_xpu_flash_attn`，`rope_type: torch`
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
| `lightx2v/common/ops/mm/mm_weight.py` | `MM_WEIGHT_REGISTER`；`MMWeightFp8IntelXpu` |
| `lightx2v/models/networks/base_model.py` | `BaseTransformerModel`；`_init_offload_manager` |
| `lightx2v/utils/registry_factory.py` | 所有 Register 注册表 |
| `lightx2v_platform/ops/attn/template.py` | `AttnWeightTemplate`：`load_state_dict_from_disk` no-op |
