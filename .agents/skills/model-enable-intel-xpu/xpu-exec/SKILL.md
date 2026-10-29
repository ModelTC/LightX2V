---
name: xpu-exec
description: 核对 LightX2V Intel XPU 环境、模型入口、配置、Python API 和运行结果，定位首次推理失败。
---

# Intel XPU 执行诊断

用于帮助用户在 Intel Arc XPU 上首次成功运行模型。
崩溃排查与模型适配见 [XPU 主文](../SKILL.md)。仓库路径以根目录为基准。

---

## 步骤 1：确认环境

先记录实际 Python、PyTorch 和 XPU 设备；只在 XPU 可用后读取设备属性：

```powershell
python -c "import sys,torch; ok=torch.xpu.is_available(); print(sys.executable, torch.__version__, 'XPU:', ok); print(torch.xpu.get_device_properties(0) if ok else 'No XPU available')"
```

`XPU: False` 时分别检查当前解释器、PyTorch XPU 构建和驱动；不能直接判定必须重装驱动。依赖与构建入口见 [XPU README](../../../../scripts/platforms/intel_xpu/README_zh.md)、[内核工程](../../../../lightx2v_kernel/intel_xpu/README.md) 和 [主文构建说明](../references/model-integration.md#前置条件编译-sycl_kernels)。不要在本文件重复维护固定版本安装命令。

---

## 步骤 2：找 config 文件

XPU 示例配置在 `configs/platforms/intel_xpu/`，多卡配置在其 `dist_infer/` 子目录。由匹配的 `scripts/platforms/intel_xpu/` 脚本确定模型类、任务、模型 variant 与 JSON；文件名不保证代表当前请求任务。

```powershell
dir configs\platforms\intel_xpu\
```

当前示例 JSON 通常不含 `model_cls` / `task`。CLI 的 `--model_cls` 来自 `runner_factory.RUNNER_MODULES`，`--task` 还须通过 runner 的支持范围与请求字段校验。

常见任务包括 `t2v/i2v/t2i/i2i/t2av/i2av/fl2av/ref2av/i23d`；完整 CLI choices 见 `lightx2v/infer.py`，每个模型支持的子集以 runner 和匹配脚本为准。

---

## 步骤 3：确认组件完整

按 runner 的加载顺序检查 DiT、文本/图像/音频编码器、VAE 和模型专用资产。并非所有任务都包含相同组件。

路径字段按模型读取：例如 Wan 的 `t5_original_ckpt` / `clip_original_ckpt` / `vae_original_ckpt`，其他家族可能使用 `text_encoder_path` 或模型根目录下的子目录。以 `load_model_config()` 和组件 loader 为准，不假定都叫 `text_encoder_path`。

MiniMax H3 CPU offload 还需要与权重、variant、步数和 flow shift 匹配的 AdaLN cache。检查脚本中的实际路径；缺失资产先报告，不自动下载或改变模型语义。

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

block offload 降低设备权重驻留，但激活、workspace、非 block 权重和 CPU 常驻源仍占内存；它不能保证任意大小的模型都可运行。量化是否有益由峰值、精度和吞吐验证决定，不以固定 5/16 GB 为阈值。

检查 `tools/convert/converter.py` 的 `--model_type`、key 映射和量化范围。converter 尚无 model_type 不代表模型未接入推理；它可能直接读取原始 checkpoint。MiniMax H3 当前支持 model/block offload，明确拒绝 lazy 和 phase。

**INT8 转换模板（先确认 model_type 与权重格式）**：
```powershell
python tools\convert\converter.py --source <原始目录或单个safetensors> --output <int8目录> --output_name <model>_int8 --model_type <model> --quantized --linear_type int8 --device cpu --single_file
```

**FP8 转换**：把 `--linear_type` 换成 `fp8`，其余不变。

仅当目标模型已支持 `lazy_load` 分块加载时，把 `--single_file` 换成 `--save_by_block`，输出目录会是 `block_0.safetensors … block_N.safetensors` + `non_block.safetensors`。

`--save_by_block` 只改变输出组织，不代表转换阶段是流式加载。先估算源权重和转换中间 tensor 的峰值；单文件大小不能直接诊断 mmap 崩溃。转换目录的 index 为 `diffusion_pytorch_model.safetensors.index.json`。

转换后确认：`--single_file` 模式确认输出文件存在；`--save_by_block` 模式确认输出目录包含 `block_0.safetensors … block_N.safetensors` 和 `non_block.safetensors`。

---

## 步骤 5：推理命令

每次开新终端必须先设环境变量（关闭终端即丢失）：
```powershell
$env:PLATFORM = "intel_xpu"    # PowerShell
# set PLATFORM=intel_xpu       # CMD
```

**方式 A：CLI（模型/任务来自匹配脚本，部署设置来自 JSON）**
```powershell
python -m lightx2v.infer `
    --model_cls   <匹配脚本中的 model_cls> `
    --task        <runner 支持的 task> `
    --model_path  <模型根目录> `
    --config_json configs\platforms\intel_xpu\<config>.json `
    --prompt      "your prompt" `
    --save_result_path output\result.mp4
```
图像输入使用 `--image_path <图片路径>`；视频输入按 task 使用 `--video_path`。输出扩展名与任务一致。

**方式 B：Python API**
```python
import os
os.environ["PLATFORM"] = "intel_xpu"  # 在导入 LightX2V 前设置
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path=r"D:\path\to\your\ckpt",
    model_cls="wan2.1",
    task="t2v",
)

# 同一个 pipe 只能调用一次 create_generator()。
pipe.create_generator(config_json="configs/platforms/intel_xpu/wan_t2v_1_3.json")

pipe.generate(seed=42, prompt="a cat", save_result_path="output.mp4")
# size/num_frames 等请求覆盖仍受 runner 的 supported_request_fields_by_task 限制。
```

不使用 JSON 时，可在这一次 `create_generator()` 调用中传入 `attn_mode`、`infer_steps`、`size=(height, width)` 等实际参数。传入 `config_json` 后，除 `fps` 外的这些推理参数以文件内容为准；不要同时传两套设置并期待合并。

---

## Config 关键参数速查

不要将 MiniMax H3 的字段当作所有模型的通用模板：

| 设置 | 当前契约 |
|---|---|
| Wan attention / norm | `self_attn_1_type`、`cross_attn_1_type` 等；`rms_norm_type`、`layer_norm_type` |
| MiniMax H3 attention / norm | `attn_type`、`refiner_attn_type`、`rms_type` |
| RoPE | 通用注册名有 `torch_real_rope` / `torch_complex_rope`；`minimax_h3_xpu_rope` 仅适用于 H3 的部分旋转契约，不存在通用 `"torch"` RoPE 注册名 |
| `cpu_offload` / `offload_granularity` | 查模型的 model/block/phase 分派，不由配置值推断支持 |
| `lazy_load` | 需要完整磁盘 staging 链；H3 当前不支持 |
| `dit_quant_scheme` | `int8-intel-xpu` 需要 W8A8 内核；`fp8-intel-xpu` 的实际 backend/fallback 应从代码和日志确认 |
| `vae_cpu_offload` | 控制权重驻留/迁移；不表示 VAE 必然在 CPU 计算 |
| `use_compile` / `warmup` | 启动 JSON 设置，无同名 CLI 开关；warmup 与 `unload_modules` 互斥 |
| `size` / `fps` | `size=[height,width]`；`fps` 是启动设置。Python API 的尺寸参数也是 `size` |

H3 的 QKV 融合、SLA 和 VAE CUTE/compile 见 [主文](../SKILL.md#torchcompile-与融合算子minimax-h3-xpu-配置) 及实际配置。先确认形状、算子和资产，再逐项启用。

---

## 多卡分布式推理

平台当前使用 XCCL。非 Linux 且 PyTorch `< 2.10.0+xpu` 会被初始化检查拒绝；其他环境仍需设备实测。以下沿用 Linux 示例。

### 快速启动

**单卡**：跳过本部分，使用单卡推理脚本。

**多卡（4 卡 TP，Linux）**：
```bash
export ZE_AFFINITY_MASK=0,1,2,3
export PLATFORM=intel_xpu

torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
    --model_cls minimax_h3 --task t2av --model-variant fl2av \
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
    --model_cls minimax_h3 --task t2av --model-variant fl2av \
    --config_json configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_sp_tp.json \
    --model_path /path/to/MiniMax-H3 --prompt "..." --save_result_path output/result.mp4
```

### 进程数和并行大小

**公式**：进程数 = `tensor_p_size` × `seq_p_size` × `cfg_p_size`

Config 中的 `parallel` 部分：
```jsonc
{
    "parallel": {
        "tensor_p_size": 2,    // 张量并行
        "seq_p_size": 2,       // 序列并行
        "cfg_p_size": 1        // CFG 并行（通常 1 或 2）
    }
}
```

例：`tensor_p_size=2, seq_p_size=2, cfg_p_size=1` → 需要 4 个进程

### 现有 SP/SP+TP 示例的 CCL 调优变量

| 变量 | 示例值 | 用途 |
|------|--------|------|
| `CCL_SYCL_ALLTOALL_ARC_LL` | 1 | Ulysses all-to-all 通信优化 |
| `CCL_SYCL_ALLTOALL_TMP_BUF` | 1 | 临时缓冲区 |
| `CCL_SYCL_CCL_BARRIER` | 1 | 屏障同步 |
| `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` | 4294967296 | allreduce 阈值 |
| `CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD` | 4294967296 | reduce-scatter 阈值 |
| `CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD` | 4294967296 | allgatherv 阈值 |

这些值不是 XCCL 的通用默认或能力条件；按实际版本与通信实测决定是否保留。

---

## 核对实际设备与算子

在导入 LightX2V 前设置 `PLATFORM=intel_xpu`，确认模型计算 tensor 在 XPU。CPU 预处理/保存和设备空闲不直接代表配置错误；offload 可能受加载或同步限制，需结合阶段日志与 timeline。

- `vae_cpu_offload` 和 encoder offload 通常控制上下卡，计算设备看各组件的 encode/decode/forward。
- `intel_xpu_flash_attn` 缺少 ESIMD 时回退到输入 tensor 所在设备的 torch SDPA，不自动回退到 CPU。
- `intel_xpu_cute_attn` 缺少 CUTE 构建会报错；CUDA 专用 `flash_attn` 也不能被描述为自动 CPU fallback。
- 两个 XPU adapter 不传递 causal/mask。需要因果语义时选择 `torch_sdpa`，并检查调用方实际传递的 `causal` / `attn_mask`。

---

## 常见报错速查

| 报错 / 现象 | 原因 | 处理 |
|------------|------|------|
| `FileNotFoundError` | checkpoint、缓存或输入路径缺失 | 按 traceback 定位组件及其实际路径字段 |
| `KeyError: block_0.safetensors` | INT8/FP8 未转换或路径错 | 检查 `dit_quantized_ckpt`，重跑 `tools\convert\converter.py`（`--linear_type int8\|fp8`） |
| `AttributeError: 'NoneType' object has no attribute ...` | Config 中 `attn_type` / `rms_type` / `rope_type` 错误或缺失 | 对照推荐配置逐字段检查 |
| Windows 构建拒绝 CUTE | 当前 CMake 因已记录的正确性问题禁用该组合 | 选择已验证契约的 ESIMD 或 torch SDPA backend |
| 因果模型结果异常 | adapter 未传递 causal/mask，可能不会主动报错 | 检查实际调用，使用支持该语义的 backend |
| 首次调用长时间无输出 | 可能是 JIT、加载或通信等待 | 查各 rank 日志、CPU/设备活动及编译进度，不以固定等待时间判定正常 |
| `intel_xpu_cute_attn requires sycl-kernels built with CUTE FMHA` | CUTE 扩展未加载 | 核对 Linux 构建及依赖；换 backend 前验证形状与数值契约 |
| 并行初始化失败，world size 不匹配 | 进程数与并行大小不匹配 | 调整 `--nproc_per_node` 或 config 中的 `parallel` 大小 |
| `RuntimeError: mismatched dimensions` / TP 场景 `torch.addmm` 报错 | 列并行 layer 的 1-D bias/scale 未跟着权重一起切分 | 见主文 Case 7 |
| `RuntimeError: sycl_kernels.sla_block_map/sparse_block_attention/sla_sparse_attention is unavailable` | `sycl_kernels` 编译时未打开 SLA/CUTE sparse attention 支持 | 重新编译 `lightx2v_kernel/intel_xpu`（见步骤 1 的构建链接），或改用非 `dynamic_sparse_attn` 的普通 attn_type |
| 多卡初始化卡住 | 进程数、设备数或并行大小不匹配，或 oneCCL 不兼容 | 确认设备数、检查 `ZE_AFFINITY_MASK`、验证进程数公式 |
| All-to-all/all-reduce 卡住（SP/SP+TP） | oneCCL 设置不正确或驱动不兼容 | 从 2 卡验证，检查 oneCCL 环境变量是否设置，先验证纯 TP 和纯 SP |
| 某个 rank 提前退出 | 不同进程状态不同步或配置不一致 | 设置 `PYTHONFAULTHANDLER=1` 查看所有 rank 的日志，确保配置和模型路径一致 |
| OOM / SIGABRT / 进程静默消失 | 内存或驱动问题 | 见主文 Case 2/3/4 |
