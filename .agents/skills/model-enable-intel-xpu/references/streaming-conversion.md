# 内存受限时的流式转换参考

本页保留原 XPU 指南中的单文件、按 block 读取与量化模板。它与当前
`tools/convert/converter.py --save_by_block` 不等价：后者改变输出组织，
并未实现这里的按 block 读取。仅在已确认转换峰值内存问题、常规 converter
不可用且无法换到内存足够的环境时阅读；优先把所需能力接入现有转换工具。

下面代码按原样保留供比对，尚未按当前模型全面验证，不能只改三个常量就
用于任意模型。使用或迁移前须处理这些已知限制：

- 仅支持单个源文件；跨 shard 的同一 block 需先建立带来源文件的索引。
- `_ST_DTYPE.get(..., torch.bfloat16)` 会静默误读未知 dtype，必须改为明确
  的 dtype 映射和拒绝未知值，并检查 shape、offset、字节长度。
- `_NON_LIN_DTYPE` 强制转换浮点 non-linear 权重，会破坏需要保留 FP32 的
  模型；按模型的 `preserve_non_quant_dtype` 契约处理。
- 模板没有 converter 的 checkpoint key 映射、LoRA/adapter 等分支；必须
  核对 runtime 实际加载的 key、block 前缀和量化范围。
- 每个 block 与全部 non-block tensor 仍会各自聚合到内存，峰值并非单 tensor；
  原模板没有原子输出或中断恢复，已有 index 也不能证明转换完整。
- 原注释中的“INT8 精度高于 FP8”和示例中“其余均 BF16”不作为通用结论。
  对实际模型检查 scale/dtype、index 完整性，并比较量化后的数值与输出。

主流程及当前 converter 入口见 [XPU 指南](model-integration.md#第二步int8fp8-转换)。

## 原始模板与产物检查示例

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
