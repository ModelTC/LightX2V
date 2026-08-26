# /xpu-exec — Intel XPU 执行诊断

用于帮助用户在 Intel Arc XPU 上首次成功运行模型。
崩溃排查（OOM / SIGABRT / 驱动问题）→ `/xpu`。

---

## 步骤 1：确认环境

运行以下两条命令，确认输出符合预期：

```powershell
python -c "import torch; print(torch.__version__, '| XPU:', torch.xpu.is_available(), '| mem:', torch.xpu.get_device_properties(0).total_memory/1e9, 'GB')"
# 期望：版本含 "xpu"，XPU: True，mem ~16.46 GB
```

若 `XPU: False`：Arc 驱动未正确安装，重装驱动后重启。
若 import 报错：PyTorch XPU 版本未安装，按顺序执行：
```powershell
pip install --no-cache-dir -r requirements_win.txt
pip install --no-cache-dir torch==2.9.1+xpu torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install --no-cache-dir -e .
```

---

## 步骤 2：找 config 文件

所有 XPU 配置在 `configs/platforms/intel_xpu/`，文件名格式 `{模型}_{任务}.json`。

```powershell
dir configs\platforms\intel_xpu\
```

config 顶部 `model_cls` 字段 = 推理命令的 `--model_cls` 参数值。

**`--task` 合法值**：`t2v` 文生视频 / `i2v` 图生视频 / `t2i` 文生图 / `i23d` 图生3D

---

## 步骤 3：确认组件完整

一个可运行的模型需要：**DiT**（主模型）、**文本编码器**、**VAE**。

检查 config 里的 `text_encoder_path`，确认路径存在且含权重文件。路径不存在时推理报 `FileNotFoundError`，**不自动下载**。

| 模型 | 文本编码器 | 是否随模型打包 | 缺失时下载来源 |
|------|-----------|--------------|--------------|
| Wan2.1 / Wan2.2 | UMT5-XXL | ✓ | — |
| HunyuanVideo-1.5 | CLIP + Qwen2.5-VL-7B | ✗ Qwen 需单独下载 | HuggingFace: `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen-Image-2512 | Qwen2.5-VL-7B | ✓ | — |
| CogVideoX | T5-XXL | ✓ | — |

---

## 步骤 4：判断是否需要 INT8/FP8 转换

查看 DiT 权重总大小：

```powershell
python -c "
import os, glob
d = 'D:/path/to/your/ckpt'
fs = glob.glob(d+'/**/*.safetensors', recursive=True) + glob.glob(d+'/**/*.bin', recursive=True)
for f in sorted(fs): print(f'{os.path.getsize(f)/1e9:.1f}GB  {os.path.basename(f)}')
"
```

`cpu_offload`（+ `lazy_load`）本身能跑任意大小的 BF16 DiT，不量化也能跑；量化只是在 Arc 140V 这种 CPU/XPU 共享 32 GB 内存的机器上，把 DiT 体积和每步搬运量减半，让 pipeline（DiT + 文本编码器 + VAE + OS）更容易挤进这 32 GB。经验参考：

| DiT BF16 总大小 | 操作 | 量化方案 | Config 关键字段 |
|----------------|------|---------|-----------------|
| < 5 GB | 直接加载 BF16 | 无 | 无需 `dit_quantized` |
| 5–16 GB | 可选：直接加载或量化后加载 | 无 或 `int8-intel-xpu`/`fp8-intel-xpu` | 可选 `dit_quantized` |
| > 16 GB | `cpu_offload`（+ `lazy_load`）通常仍可行；量化能明显降低 OOM 风险和搬运耗时 | `int8-intel-xpu`（推荐）或 `fp8-intel-xpu` | `dit_quantized: true`, `cpu_offload: true`, `lazy_load: true` |

已接入 LightX2V 的模型统一走通用工具 `tools\convert\converter.py`（`--model_type` 里已经有该模型）。如果 `--model_type` 里没有这个模型 → 该模型尚未接入 LightX2V，去 `/xpu` 走接入新模型流程（给 `converter.py` 加 `model_type`，不要在这里新写转换脚本）。

**INT8 转换（推荐，只需运行一次）**：
```powershell
python tools\convert\converter.py --source <原始目录或单个safetensors> --output <int8目录> --output_name <model>_int8 --model_type <model> --quantized --linear_type int8 --device cpu --single_file
```

**FP8 转换（fallback）**：把 `--linear_type` 换成 `fp8`，其余不变。

需要 `lazy_load` 分块加载时，把 `--single_file` 换成 `--save_by_block`，输出目录会是 `block_0.safetensors … block_N.safetensors` + `non_block.safetensors`。

⚠️ `converter.py` 会把源文件整份读进内存转换，单个 safetensors 文件 > ~5 GB 时在 Windows 上可能撞上 mmap 静默 kill 的问题（`/xpu` Case 2）。遇到这种情况，改到别的机器转换好再把量化产物拷过来，不要在 Arc 140V/Windows 本机上硬跑。

转换后确认：`--single_file` 模式确认输出文件存在；`--save_by_block` 模式确认输出目录包含 `block_0.safetensors … block_N.safetensors` 和 `non_block.safetensors`。

---

## 步骤 5：推理命令

每次开新终端必须先设环境变量（关闭终端即丢失）：
```powershell
$env:PLATFORM = "intel_xpu"    # PowerShell
# set PLATFORM=intel_xpu       # CMD
```

**方式 A：CLI（推荐，参数来自 config 文件）**
```powershell
python lightx2v/infer.py `
    --model_cls   <见 config 顶部> `
    --task        <t2v/i2v/t2i/i23d> `
    --model_path  <模型根目录> `
    --config_json configs\platforms\intel_xpu\<config>.json `
    --prompt      "your prompt" `
    --save_result_path output\result.mp4
```
图生视频 / 图生3D 加：`--image_path <图片或视频路径>`

**方式 B：Python API**
```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path=r"D:\path\to\your\ckpt",
    model_cls="wan2.1",
    task="t2v",
)

# 方式 B1：使用 config 文件
pipe.create_generator(config_json="configs/platforms/intel_xpu/wan_t2v_1_3.json")

# 方式 B2：手动指定参数（attn_mode 替代 config 中的 attn_type）
pipe.create_generator(
    attn_mode="torch_sdpa",   # XPU 使用 torch_sdpa
    infer_steps=50,
    height=480,
    width=832,
    num_frames=33,
    guidance_scale=5.0,
)

pipe.generate(seed=42, prompt="a cat", save_result_path="output.mp4")
```

---

## Config 关键参数速查

**推荐配置（INT8 + 新算子）— Linux 平台**：
```jsonc
{
    // ── 注意力与位置编码（新优化） ────────────────────────
    "attn_type": "intel_xpu_cute_attn",   // CUTE 优化注意力（仅双向，Linux only）
    "rms_type": "intel_xpu",              // XPU 优化 RMSNorm
    "rope_type": "minimax_h3_xpu_rope",   // MiniMax H3 XPU 优化

    "cpu_offload": true,                  // DiT > 5 GB INT8 时必须
    "offload_granularity": "block",       // 推荐 "block"
    "lazy_load": true,                    // 大模型必须，block 按需从磁盘加载
    "num_disk_workers": 4,

    "dit_quantized": true,
    "dit_quant_scheme": "int8-intel-xpu", // INT8：性能优于 FP8
    "dit_quantized_ckpt": "D:/path/to/your/int8_ckpt",

    "vae_cpu_offload": true,              // VAE 在 CPU 运行（节省 XPU 内存，属正常）
    "unload_modules": true,               // 文本编码器推理后释放，为 DiT 腾空间
    "feature_caching": "NoCaching"        // 首次运行推荐
}
```

**推荐配置（INT8 + 新算子）— Windows 平台**：
```jsonc
{
    // ── 注意力与位置编码（新优化） ────────────────────────
    "attn_type": "intel_xpu_flash_attn",  // Windows 不支持 CUTE，用 flash_attn（仅双向）
    "rms_type": "intel_xpu",              // XPU 优化 RMSNorm
    "rope_type": "minimax_h3_xpu_rope",   // MiniMax H3 XPU 优化

    "cpu_offload": true,                  // DiT > 5 GB INT8 时必须
    "offload_granularity": "block",       // 推荐 "block"
    "lazy_load": true,                    // 大模型必须，block 按需从磁盘加载
    "num_disk_workers": 4,

    "dit_quantized": true,
    "dit_quant_scheme": "int8-intel-xpu", // INT8：性能优于 FP8
    "dit_quantized_ckpt": "D:/path/to/your/int8_ckpt",

    "vae_cpu_offload": true,
    "unload_modules": true,
    "feature_caching": "NoCaching"
}
```

**Fallback 配置（无 INT8 脚本时，使用 FP8）**：
```jsonc
{
    "attn_type": "intel_xpu_flash_attn",  // Fallback
    "rope_type": "torch",

    "cpu_offload": true,
    "offload_granularity": "block",
    "lazy_load": true,
    "num_disk_workers": 4,

    "dit_quantized": true,
    "dit_quant_scheme": "fp8-intel-xpu",
    "dit_quantized_ckpt": "D:/path/to/your/fp8_ckpt",

    "vae_cpu_offload": true,
    "unload_modules": true,
    "feature_caching": "NoCaching"
}
```

---

## 多卡分布式推理（Linux 仅）

⚠️ **Windows 不支持多卡**，仅 Linux 可用。

### 快速启动

**单卡**：跳过本部分，使用单卡推理脚本。

**多卡（4 卡 TP，Linux）**：
```bash
export ZE_AFFINITY_MASK=0,1,2,3
export PLATFORM=intel_xpu

torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_tp.json \
    --model_path ... --prompt "..." --save_result_path ...
```

**多卡（8 卡 SP+TP，Linux）**：
```bash
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
export PLATFORM=intel_xpu
export CCL_SYCL_ALLTOALL_ARC_LL=1
export CCL_SYCL_ALLTOALL_TMP_BUF=1
export CCL_SYCL_CCL_BARRIER=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_sp_tp.json ...
```

### 进程数和并行大小

**公式**：进程数 = `tensor_p_size` × `seq_p_size` × `cfg_p_size`

Config 中的 `parallel` 部分：
```json
{
    "parallel": {
        "tensor_p_size": 2,    // 张量并行
        "seq_p_size": 2,       // 序列并行
        "cfg_p_size": 1        // CFG 并行（通常 1 或 2）
    }
}
```

例：`tensor_p_size=2, seq_p_size=2, cfg_p_size=1` → 需要 4 个进程

### oneCCL 环境变量（仅 SP/SP+TP）

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `CCL_SYCL_ALLTOALL_ARC_LL` | 1 | Ulysses all-to-all 通信优化 |
| `CCL_SYCL_ALLTOALL_TMP_BUF` | 1 | 临时缓冲区 |
| `CCL_SYCL_CCL_BARRIER` | 1 | 屏障同步 |
| `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` | 4294967296 | allreduce 阈值 |
| `CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD` | 4294967296 | reduce-scatter 阈值 |
| `CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD` | 4294967296 | allgatherv 阈值 |

**TP 仅**：不需要这些变量。

---

## 各模块设备分工（正常行为参考）

| 模块 | 运行设备 | 备注 |
|------|---------|------|
| 图像/视频预处理、结果保存 | CPU | 正常 |
| 文本编码器（推理期间） | XPU | 超大模型自动 INT8 量化后上 XPU |
| 文本编码器（DiT 推理后） | 已释放 | `unload_modules: true` |
| DiT（无 offload） | XPU | GPU 使用率持续高位；新算子（CUTE attn / XPU RMSNorm）会提升 GPU 效率 |
| DiT（有 offload） | 计算在 XPU，权重从磁盘/CPU 流入 | GPU 使用率呈**间歇脉冲** + CPU 有搬运负荷，均正常 |
| DiT（INT8 推理） | XPU（INT8 计算单元） | 激活动态量化为 int8 后走真正的 INT8×INT8 GEMM（`onednn_w8a8_int8`），而非反量化+matmul，功耗低、性能高 |
| RoPE（旋转位置编码） | XPU | 新 `minimax_h3_xpu_rope` 在 XPU 优化实现 |
| VAE | CPU | `vae_cpu_offload: true` 时，正常 |

**判断标准**：DiT 推理阶段（占总时间 80%+）GPU 应有明显活动。若全程 GPU 为零 → 见下方异常诊断。

---

## 异常：GPU 全程闲置

按顺序排查：

**① `PLATFORM` 未设置**（最常见）
```powershell
echo $env:PLATFORM   # 若为空则未设置
$env:PLATFORM = "intel_xpu"
```

**② XPU 不可用**
```powershell
python -c "import torch; print(torch.xpu.is_available())"
# False → 重装 Arc 驱动后重启
```

**③ Config 字段错误**

确认 config 包含以下之一：

**推荐配置**（Linux，仅支持双向注意力）：
```json
"attn_type": "intel_xpu_cute_attn",
"rms_type": "intel_xpu",
"rope_type": "minimax_h3_xpu_rope"
```

**推荐配置**（Windows，仅支持双向注意力）：
```json
"attn_type": "intel_xpu_flash_attn",
"rms_type": "intel_xpu",
"rope_type": "minimax_h3_xpu_rope"
```

**Fallback 配置**（单向注意力或无新内核）：
```json
"attn_type": "torch_sdpa",
"rope_type": "torch"
```

缺失或使用 `"flash_attn"` → 注意力回落到 CPU。新字段缺失 → 降级到 fallback 方案。

---

## 常见报错速查

| 报错 / 现象 | 原因 | 处理 |
|------------|------|------|
| `FileNotFoundError` | 文本编码器路径不存在 | 下载后在 config `text_encoder_path` 填写 |
| `KeyError: block_0.safetensors` | INT8/FP8 未转换或路径错 | 检查 `dit_quantized_ckpt`，重跑 `tools\convert\converter.py`（`--linear_type int8\|fp8`） |
| `AttributeError: 'NoneType' object has no attribute ...` | Config 中 `attn_type` / `rms_type` / `rope_type` 错误或缺失 | 对照推荐配置逐字段检查 |
| `RuntimeError: cute_attn not available on Windows` | Windows 平台使用了 `intel_xpu_cute_attn` | 改用 `attn_type: intel_xpu_flash_attn` |
| `RuntimeError: Attention only supports bidirectional` | 模型使用单向/因果注意力，但 config 用了 CUTE 或 flash_attn | 改用 `attn_type: torch_sdpa` |
| 推理卡住 2–5 分钟无输出 | XPU JIT 编译 oneDNN/CUTE kernel | 正常，等待即可 |
| `ImportError: No module named cute_attn` | CUTE attention 内核未安装 | 参考项目安装文档，或降级到 `attn_type: intel_xpu_flash_attn` |
| `RuntimeError: tensor_p_size × seq_p_size × cfg_p_size != nproc` | 进程数与并行大小不匹配 | 调整 `--nproc_per_node` 或 config 中的 `parallel` 大小 |
| 多卡初始化卡住 | 进程数、设备数或并行大小不匹配，或 oneCCL 不兼容 | 确认设备数、检查 `ZE_AFFINITY_MASK`、验证进程数公式 |
| All-to-all/all-reduce 卡住（SP/SP+TP） | oneCCL 设置不正确或驱动不兼容 | 从 2 卡验证，检查 oneCCL 环境变量是否设置，先验证纯 TP 和纯 SP |
| 某个 rank 提前退出 | 不同进程状态不同步或配置不一致 | 设置 `PYTHONFAULTHANDLER=1` 查看所有 rank 的日志，确保配置和模型路径一致 |
| OOM / SIGABRT / 进程静默消失 | 内存或驱动问题 | → `/xpu` Case 2/3/4 |
