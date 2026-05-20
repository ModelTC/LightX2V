/xpu-exec — Intel XPU 执行诊断
用于帮助用户在 Intel Arc XPU 上首次成功运行模型。 崩溃排查（OOM / SIGABRT / 驱动问题）→ /xpu。

步骤 1：确认环境
运行以下两条命令，确认输出符合预期：

python -c "import torch; print(torch.__version__, '| XPU:', torch.xpu.is_available(), '| mem:', torch.xpu.get_device_properties(0).total_memory/1e9, 'GB')"
# 期望：版本含 "xpu"，XPU: True，mem ~16.46 GB
若 XPU: False：Arc 驱动未正确安装，重装驱动后重启。 若 import 报错：PyTorch XPU 版本未安装，按顺序执行：

pip install --no-cache-dir -r requirements_win.txt
pip install --no-cache-dir torch==2.9.1+xpu torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install --no-cache-dir -e .
步骤 2：找 config 文件
所有 XPU 配置在 configs/platforms/intel_xpu/，文件名格式 {模型}_{任务}.json。

dir configs\platforms\intel_xpu\
config 顶部 model_cls 字段 = 推理命令的 --model_cls 参数值。

--task 合法值：t2v 文生视频 / i2v 图生视频 / t2i 文生图 / i23d 图生3D

步骤 3：确认组件完整
一个可运行的模型需要：DiT（主模型）、文本编码器、VAE。

检查 config 里的 text_encoder_path，确认路径存在且含权重文件。路径不存在时推理报 FileNotFoundError，不自动下载。

模型	文本编码器	是否随模型打包	缺失时下载来源
Wan2.1 / Wan2.2	UMT5-XXL	✓	—
HunyuanVideo-1.5	CLIP + Qwen2.5-VL-7B	✗ Qwen 需单独下载	HuggingFace: Qwen/Qwen2.5-VL-7B-Instruct
Qwen-Image-2512	Qwen2.5-VL-7B	✓	—
CogVideoX	T5-XXL	✓	—
步骤 4：判断是否需要 FP8 转换
查看 DiT 权重总大小：

python -c "
import os, glob
d = 'D:/yanran/models/YOUR_MODEL'
fs = glob.glob(d+'/**/*.safetensors', recursive=True) + glob.glob(d+'/**/*.bin', recursive=True)
for f in sorted(fs): print(f'{os.path.getsize(f)/1e9:.1f}GB  {os.path.basename(f)}')
"
DiT BF16 总大小	操作	Config 关键字段
< 8 GB	直接加载	无需额外设置
8–16 GB	FP8 转换后加载	dit_quantized: true
> 16 GB	FP8 转换 + lazy offload	dit_quantized: true, cpu_offload: true, lazy_load: true
FP8 转换（只需运行一次）：

python tools\convert\<model>_fp8_convert.py --source <原始目录> --output <fp8目录>
转换后确认输出目录包含 block_0.safetensors … block_N.safetensors 和 non_block.safetensors。

无对应转换脚本 → 该模型尚未接入 LightX2V，见 /xpu 接入新模型流程。

步骤 5：推理命令
每次开新终端必须先设环境变量（关闭终端即丢失）：

$env:PLATFORM = "intel_xpu"    # PowerShell
# set PLATFORM=intel_xpu       # CMD
方式 A：CLI（推荐，参数来自 config 文件）

python lightx2v/infer.py `
    --model_cls   <见 config 顶部> `
    --task        <t2v/i2v/t2i/i23d> `
    --model_path  <模型根目录> `
    --config_json configs\platforms\intel_xpu\<config>.json `
    --prompt      "your prompt" `
    --save_result_path output\result.mp4
图生视频 / 图生3D 加：--image_path <图片或视频路径>

方式 B：Python API

from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path=r"D:\yanran\models\Wan2.1-T2V-1.3B",
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
Config 关键参数速查
{
    // ── 注意力（二选一）───────────────────────────────────
    "attn_type": "intel_xpu_flash_attn",  // config 文件中的写法（XPU 专用）
    // attn_mode="torch_sdpa"             // Python API create_generator() 中的等价写法
    // 不可用 "flash_attn"（CUDA only）
    "rope_type": "torch",                 // XPU 必须用 torch

    "cpu_offload": true,                  // DiT > 16 GB 时必须
    "offload_granularity": "block",       // 推荐 "block"
    "lazy_load": true,                    // 大模型必须，block 按需从磁盘加载
    "num_disk_workers": 4,

    "dit_quantized": true,
    "dit_quant_scheme": "fp8-intel-xpu",
    "dit_quantized_ckpt": "D:/yanran/models/xxx/fp8",

    "vae_cpu_offload": true,              // VAE 在 CPU 运行（节省 XPU 内存，属正常）
    "unload_modules": true,               // 文本编码器推理后释放，为 DiT 腾空间
    "feature_caching": "NoCaching"        // 首次运行推荐
}
各模块设备分工（正常行为参考）
模块	运行设备	备注
图像/视频预处理、结果保存	CPU	正常
文本编码器（推理期间）	XPU	超大模型自动 INT8 量化后上 XPU
文本编码器（DiT 推理后）	已释放	unload_modules: true
DiT（无 offload）	XPU	GPU 使用率持续高位
DiT（有 offload）	计算在 XPU，权重从磁盘/CPU 流入	GPU 使用率呈间歇脉冲 + CPU 有搬运负荷，均正常
VAE	CPU	vae_cpu_offload: true 时，正常
判断标准：DiT 推理阶段（占总时间 80%+）GPU 应有明显活动。若全程 GPU 为零 → 见下方异常诊断。

异常：GPU 全程闲置
按顺序排查：

① PLATFORM 未设置（最常见）

echo $env:PLATFORM   # 若为空则未设置
$env:PLATFORM = "intel_xpu"
② XPU 不可用

python -c "import torch; print(torch.xpu.is_available())"
# False → 重装 Arc 驱动后重启
③ Config 字段错误

确认 config 包含：

"attn_type": "intel_xpu_flash_attn",
"rope_type": "torch"
缺失或使用 "flash_attn" → 注意力回落到 CPU。

常见报错速查
报错 / 现象	原因	处理
FileNotFoundError	文本编码器路径不存在	下载后在 config text_encoder_path 填写
KeyError: block_0.safetensors	FP8 未转换或路径错	检查 dit_quantized_ckpt，重跑转换脚本
推理卡住 2–5 分钟无输出	XPU JIT 编译 oneDNN kernel	正常，等待即可
ImportError: No module named flash_attn	正常 fallback	确认 config attn_type: intel_xpu_flash_attn
OOM / SIGABRT / 进程静默消失	内存或驱动问题	→ /xpu Case 2/3/4
