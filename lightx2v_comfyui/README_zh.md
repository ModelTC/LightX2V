# LightX2V ComfyUI

[English](README.md) | 简体中文

通过 ComfyUI 调用本机的 LightX2V server。首次执行加载模型，后续请求复用同一进程；默认空闲 600 秒释放。管理器随插件加载，无需单独运行管理服务或手动分配后端端口。

本目录可以直接软链接到 `ComfyUI/custom_nodes`。只需 ComfyUI 和包含本目录的 LightX2V checkout，不再单独安装 ComfyUI-LightX2V-Server、Wrapper 或 MAINodes。

## 节点与目录

| 节点 | 输入 → 输出 | 用途 |
| --- | --- | --- |
| LightX2V Service | 源码、模型、config、task、GPU → `LX2V_SERVER` | 描述服务；执行推理时才加载模型；浏览/编辑 config、释放服务 |
| LightX2V Generate Video | server、prompt、尺寸、帧数、seed、可选首尾 `IMAGE` → `VIDEO` | H3、Wan 等视频生成 |
| LightX2V Generate Image | server、prompt、尺寸、seed、可选 `IMAGE` → `IMAGE` | 文生图或单图编辑 |
| LightX2V Upscale Video | server、`VIDEO`、scale、seed → `VIDEO` | RelayVSR 超分 |

四类节点统一显示蓝底 `LightX2V-Server: 类型` 标签，节点标题仍可自由修改。图片/视频的加载、保存和预览使用 Comfy 原生节点。没有独立的 Config Path、Release VRAM 或暂停节点。

```text
lightx2v_comfyui/
├── __init__.py                  # Comfy 入口、节点与路由注册、WEB_DIRECTORY
├── lightx2v_comfy_server/
│   ├── nodes.py                 # 四类节点、IMAGE/VIDEO 与文件转换
│   ├── service.py               # 启动参数、配置指纹、服务身份和启动命令
│   ├── manager.py               # HTTP 请求、服务池、执行锁、取消和空闲释放
│   ├── supervisor.py            # 独立进程守护、父进程退出检测、进程树清理
│   ├── models.py                # 静态读取模型与 task 声明、H3 variant 推断
│   ├── gpus.py                  # GPU 占用查询、编号与 UUID 归一化
│   ├── files.py                 # 路径、文件浏览、JSON 校验、备份与原子保存
│   └── routes.py                # Comfy 的 /lightx2v/* HTTP 接口
├── web/                         # Service、GPU、文件浏览、Config 编辑界面
├── examples/                    # 8 个画布工作流 JSON；不放模型 config
├── README.md
└── README_zh.md
```

模型 config 统一使用主仓库 [`configs/`](../configs)，本机修改另存到持久目录。旧版本工作流转换工具和历史部署记录留在原独立仓库，不属于本插件的运行组件。

## Linux + CUDA 部署

以下使用 Bash，系统包命令以 Ubuntu 24.04 为例。默认复用 Comfy 的 Python；Docker 可选。H3 768p 和 RelayVSR 需要足够的 GPU 显存、主机内存和磁盘，CPU offload 不会消除视频激活的显存占用。

### 1. 准备 ComfyUI 和 Python

先确认 NVIDIA 驱动可用：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
export COMFY_DIR="$HOME/ComfyUI"
export LX_HOME="$HOME/lightx2v-local"
export LIGHTX2V_DIR="$LX_HOME/LightX2V"
mkdir -p "$LX_HOME/models" "$LX_HOME/configs"
```

这些变量用于下面的安装命令，不是插件运行时的必填设置。

**已有 ComfyUI**：停止准备升级依赖的 Comfy，激活原环境。例如原来使用 `.venv`：

```bash
source "$COMFY_DIR/.venv/bin/activate"
python -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip freeze > "$LX_HOME/comfy-before-lightx2v.txt"
```

Conda 用户使用原来的 `conda activate`。LightX2V 要求 Python 3.10+，建议使用与 Comfy、PyTorch 共同兼容的 Python 3.12。Comfy 需提供原生 `VIDEO`、`LoadVideo`、`SaveVideo`；旧环境先升级。

**没有 ComfyUI**：执行这个分支，创建 Comfy 自己的环境：

```bash
sudo apt-get update
sudo apt-get install -y git curl python3.12 python3.12-venv \
  build-essential pkg-config ffmpeg libgl1 libglib2.0-0t64
git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
python3.12 -m venv "$COMFY_DIR/.venv"
source "$COMFY_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
# CUDA 13.0 wheel 示例；已有可用的 CUDA PyTorch 时跳过。
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r "$COMFY_DIR/requirements.txt"
```

CUDA wheel 必须匹配驱动和 GPU，其他组合参考 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)。`nvidia-smi` 的 CUDA Version 不代表 Python 已装 CUDA 版 PyTorch；编译额外 CUDA 扩展才需要对应 Toolkit。检查当前解释器：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(), "下面的 H3 配置使用 BF16"
print(torch.cuda.get_device_name(0))
PY
```

### 2. 安装 LightX2V 并链接插件

已有 LightX2V 可直接将 `LIGHTX2V_DIR` 指向它。首次安装克隆包含本目录的版本；本目录尚未合入所用分支时，先将整个 `lightx2v_comfyui/` 复制到该 checkout。

```bash
# 已有 checkout 时不重复 clone。
git clone https://github.com/ModelTC/LightX2V.git "$LIGHTX2V_DIR"
python -m pip install -v -e "$LIGHTX2V_DIR"
python -m pip install httpx
python -m pip check
cd "$LIGHTX2V_DIR"
python -m lightx2v.server --help

test -f "$LIGHTX2V_DIR/lightx2v_comfyui/__init__.py"
mkdir -p "$COMFY_DIR/custom_nodes"
ln -s "$LIGHTX2V_DIR/lightx2v_comfyui" "$COMFY_DIR/custom_nodes/lightx2v_comfyui"
```

插件没有额外 pip 依赖，无需对本目录执行 `pip install`。`httpx` 是 LightX2V server 的依赖，显式安装可避免依赖间接引入失效。算子安装参考[主仓库安装文档](../docs/EN/source/getting_started/quickstart.md)。本教程使用已有的 `torch_sdpa`，不要求 FlashAttention 3。

从独立插件迁移时，先停止 Comfy，将旧 `custom_nodes/ComfyUI-LightX2V-Server` **移到 custom_nodes 外备份**，再建立上面的链接，避免重复注册。当前四类节点的 class ID、输入顺序和工作流格式保留；旧图中的源码、模型和 config 路径需改成新机器上的实际路径。不要只复制内部 Python 包，外层 `__init__.py` 和 `web/` 也必须存在。

Linux 上某些 `decord==0.6.0` wheel 的内部标签与发布文件名不一致。若 `pip check` 仅报告 `decord 0.6.0 is not supported on this platform`，按[上游问题](https://github.com/dmlc/decord/issues/356)核对，并实际执行 CPU 解码测试；不要修改包元数据或忽略其他依赖错误。例如：

```bash
ffmpeg -nostdin -f lavfi -i testsrc2=size=128x128:rate=8 -t 2 \
  -c:v libx264 -pix_fmt yuv420p "$LX_HOME/decord-check.mp4"
python - <<'PY'
import os
import decord
reader = decord.VideoReader(os.path.join(os.environ['LX_HOME'], 'decord-check.mp4'), ctx=decord.cpu(0))
assert len(reader) == 16
assert reader.get_batch([0, 7, 15]).asnumpy().shape == (3, 128, 128, 3)
PY
```

### 3. 下载 H3 和 RelayVSR

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir "$LX_HOME/models/MiniMax-H3"
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --local-dir "$LX_HOME/models/H3-Turbo"
hf download kopper/RelayVSR wan.safetensors flash.safetensors \
  --local-dir "$LX_HOME/models/RelayVSR"
```

`hf` 来自 `huggingface_hub`，缺少时执行 `python -m pip install huggingface_hub`；有访问限制的模型先 `hf auth login`。已有权重可直接复用。

H3 的 model_path 是包含 transformer、transformer_ref、文本编码器、VAE 等组件的完整模型根目录；不是单个权重文件。这里使用非 `_comfyui_` 的 LightX2V Turbo LoRA。RelayVSR x4 目录需同时有 `wan.safetensors` 和 `flash.safetensors`。详见 [H3](../scripts/minimax_h3/README_zh.md)、[RelayVSR](../scripts/relayvsr/README_zh.md)。

### 4. 从主仓库 config 创建本机配置

以下脚本创建 H3 四步、H3 ref2av 29 步和 RelayVSR 三份配置。只创建新文件，不覆盖已有配置；H3 ref2av 不使用 fl2av Turbo LoRA。

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['LX_HOME'])
source = Path(os.environ['LIGHTX2V_DIR']) / 'configs'
for template, name in [
    ('minimax_h3/dmd/minimax_h3_bf16_4step.json', 'h3.json'),
    ('minimax_h3/minimax_h3.json', 'h3-ref2av.json'),
    ('relayvsr/relayvsr_x4.json', 'relay.json'),
]:
    config = json.loads((source / template).read_text())
    config.update(use_compile=False, warmup=False, cpu_offload=True, offload_granularity='block')
    if name.startswith('h3'):
        config.update(attn_type='torch_sdpa', rms_type='torch_native', rope_type='torch_real_rope',
                      size=[768, 1344], num_frames=124, text_encoder_cpu_offload=True,
                      text_encoder_offload_granularity='block', vae_cpu_offload=True,
                      use_adaln_cache=True, adaln_cache_dir=str(Path.home() / '.cache/lightx2v/adaln'))
    if name == 'h3.json':
        config['lora_configs'][0]['path'] = str(root / 'models/H3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors')
    elif name == 'h3-ref2av.json':
        config['reference_image_resize_mode'] = 'match'
    else:
        config.update(attention_backend='torch_sdpa', cross_attention_backend='torch_sdpa')
    target = root / 'configs' / name
    with target.open('x', encoding='utf-8') as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(target)
PY
```

之后可用 Service 的 **Browse / Edit config** 修改。LoRA 放在 `lora_configs` 中，`path` 可 Browse 权重，`strength/alpha` 按模型要求配置，`name` 是 Advanced 内的可选项。配置中的 `${ENV_VAR}` 会展开；模型专用字段由 LightX2V 校验。

启用 H3 AdaLN 缓存时必须保留非空 `adaln_cache_dir` 并生成缓存。`~/.cache/lightx2v/adaln` 是惯用路径，不是删除字段后的自动回退；fl2av 与 ref2av 可共享根目录，但使用各自的缓存子目录。

### 5. 生成 H3 AdaLN 缓存

使用后端相同的 Python、模型、LoRA、config 和 DTYPE。下面实际使用 GPU 0，确认可用后运行；完成后不会留下 server：

```bash
cd "$LIGHTX2V_DIR"
PYTHONPATH="$LIGHTX2V_DIR" DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 CUDA_VISIBLE_DEVICES=0 \
  python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$LX_HOME/models/MiniMax-H3" --model-variant fl2av \
  --config_json "$LX_HOME/configs/h3.json"
# 需要 ref2av 时再单独准备。
PYTHONPATH="$LIGHTX2V_DIR" DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 CUDA_VISIBLE_DEVICES=0 \
  python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$LX_HOME/models/MiniMax-H3" --model-variant ref2av \
  --config_json "$LX_HOME/configs/h3-ref2av.json"
```

出现 `AdaLN cache saved to` 后再生成视频。改变 variant、步数、flow shift、模型或 LoRA 后，按 [H3 文档](../scripts/minimax_h3/README_zh.md)重新检查缓存要求；不要把其他配置生成的缓存改名混用。

### 6. 启动与导入工作流

```bash
cd "$COMFY_DIR"
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 \
  python main.py --cpu --listen 127.0.0.1 --port 8188
```

推理 GPU 统一在每个 Service 节点的 **CUDA_VISIBLE_DEVICES** 下拉框中选择，Comfy 启动命令无需配置选卡环境变量。Docker 启动使用 `--gpus all`，Comfy 命令保留 `--cpu`：容器开放设备，具体用哪张卡由 Service 决定。启动 Comfy 或打开工作流不会加载模型。第 5 步的显式选卡仅用于独立运行的 AdaLN 缓存命令。

打开 [ComfyUI](http://127.0.0.1:8188/)，导入 [h3_relay_workflow.json](examples/h3_relay_workflow.json)。`--cpu` 让 Comfy 本身不创建 CUDA 上下文，独立后端仍使用节点选择的 GPU；需要其他 Comfy GPU 节点时可去掉它。远端部署在浏览器电脑使用 `ssh -N -L 8188:127.0.0.1:8188 用户@服务器`，只转发 Comfy 端口。

| Service 字段 | H3 | RelayVSR |
| --- | --- | --- |
| lightx2v_path | 本机 LightX2V 目录 | 相同目录 |
| model_path | `$LX_HOME/models/MiniMax-H3` | `$LX_HOME/models/RelayVSR` |
| model_cls | minimax_h3 | relayvsr_x4 |
| config_json | `$LX_HOME/configs/h3.json` | `$LX_HOME/configs/relay.json` |
| task | t2av | sr |
| CUDA_VISIBLE_DEVICES | 下拉选择空闲 GPU | 下拉选择同一张或另一张空闲 GPU |

表中变量用于说明路径；用 Browse 选择实际目录/文件即可。保存的工作流优先使用自身路径。新建 Service 会自动识别所在 LightX2V checkout，可用 `LIGHTX2V_PATH` 覆盖初始值。

H3 示例为 1344×768、124 帧，Relay 为 2×。点击 Run 后依次生成 H3 视频、Save Video 保存并预览、Relay 超分、再次 Save Video。中间预览不暂停，超分继续使用原始 VIDEO。输出通常在 `ComfyUI/output/video/`。H3 宽高须为 32 的倍数，帧数为 `17*n+5` 且在 124–362；`relayvsr_x4` 的 2× 是内部 4× 后缩放。

## 工作流与其他模型

本目录只保留 8 个 `*_workflow.json` 画布文件，可直接拖入 ComfyUI 打开、编辑和运行。所有路径和 Load Image/Video 文件名都是占位值，需要替换。需要脚本调用时，可从 ComfyUI 导出 API 格式，用于提交到 `/prompt`。

| 画布 | 模型 / task | 配置来源 |
| --- | --- | --- |
| [H3 → 预览 → Relay](examples/h3_relay_workflow.json) | minimax_h3/t2av + relayvsr_x4/sr | 上面的 h3.json + relay.json |
| [H3 连续生成](examples/h3_t2av_workflow.json) | minimax_h3/t2av | h3.json |
| [H3 首尾帧](examples/h3_fl2av_workflow.json) | minimax_h3/fl2av | h3.json |
| [H3 参考图](examples/h3_ref2av_workflow.json) | minimax_h3/ref2av | h3-ref2av.json |
| [Wan](examples/wan_t2v_workflow.json) | wan2.1/t2v | [wan_t2v.json](../configs/wan/wan_t2v.json) |
| [Qwen 文生图](examples/qwen_t2i_workflow.json) | qwen_image/t2i | [2512 distill](../configs/qwen_image/qwen_image_t2i_2512_distill.json) |
| [Qwen 单图编辑](examples/qwen_i2i_workflow.json) | qwen_image/i2i | [2511 distill](../configs/qwen_image/qwen_image_i2i_2511_distill.json) |
| [纯视频超分](examples/relay_video_workflow.json) | relayvsr_x4/sr | relay.json |

H3 `t2av` 不接图片；`i2av` 只接 image；`l2av` 只接 last_frame；`fl2av` 同时接二者；`ref2av` 接单张参考 image，使用 transformer_ref 和匹配配置。连接图片不会自动切换 task。节点会在模型加载前检查这些组合，并弹窗显示具体错误。ref2av 参考图不等于固定首帧。

只运行 Wan、Qwen 或 Relay 时可以跳过 H3 权重与缓存。按需下载：

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$LX_HOME/models/Wan2.1-T2V-1.3B"
hf download Qwen/Qwen-Image-2512 --local-dir "$LX_HOME/models/Qwen-Image-2512"
hf download lightx2v/Qwen-Image-2512-Lightning \
  Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors --local-dir "$LX_HOME/models/Qwen-LoRA/qwen2512"
hf download Qwen/Qwen-Image-Edit-2511 --local-dir "$LX_HOME/models/Qwen-Image-Edit-2511"
hf download lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors --local-dir "$LX_HOME/models/Qwen-LoRA/qwen2511"
```

将表中主仓库 config 复制到 `$LX_HOME/configs/wan.json`、`qwen.json`、`qwen-edit.json`（目标应不存在）。Wan 的 `self_attn_1_type/cross_attn_1_type`、Qwen 的 `attn_type` 改为 `torch_sdpa`；设置 `use_compile=false`、`warmup=false`。Qwen 的 `lora_configs[0].path` 改为实际下载的对应文件，包括 Edit 示例下载的 bf16 文件名；保留 8 步设置，按显存需要配置 `cpu_offload`、`offload_granularity=block`、`text_encoder_cpu_offload`、`vae_cpu_offload`。Wan 示例 832×480、81 帧、50 步，可先用 17 帧验证。

纯视频超分使用原生 Load Video，选择自己的 MP4。原生 x2 模型需下载 [lightx2v/RelayVSR_X2](https://huggingface.co/lightx2v/RelayVSR_X2) 的 `wan_x2.safetensors` 和 `flash_x2.safetensors`，选择 `relayvsr_x2`、[x2 config](../configs/relayvsr/relayvsr_x2.json) 和 scale=2；不要混用 x4 权重。

兼容现有视频/图片请求协议的新模型只需换 Service 参数、config 和工作流 JSON。模型/task 从 runner 声明静态读取，不导入推理运行时。图片目前支持 t2i/单图 i2i，视频支持已有的 image/last_frame/video 输入；多参考、mask、音频输入、分层输出等新契约仍需补媒体适配。

## Config 编辑和路径

**Browse LightX2V / Browse model** 选目录。**Browse / Edit config** 默认打开当前 Service 选中的文件；Browse 选择已有 JSON，并更新 Service 的 config_json 路径。读取时自动格式化，修改 JSON 或 LoRA 后点击 **Save**。切换文件或关闭时，有未保存修改会先确认是否丢弃。

- Save 写回所选文件：校验打开时的内容哈希，将原始字节备份到同目录 `.lightx2v-config-backups`，再原子替换并保留权限。文件被其他编辑器修改时，需通过 Browse 重新打开后再编辑。
- 下次 Run 读取所选文件已保存的内容，配置有变化时替换对应服务。未保存修改不参与运行；共享同一文件的 Service 会读到相同修改，保存不会中断正在运行的模型。
- LoRA 的 Browse 只选择 `.safetensors` 路径，不读取权重、不加载模型；名称在 Advanced，未知 JSON 字段保留。

所有路径都是 **Comfy 所在机器/容器可见的路径**。远端网页的 Browse 不是浏览器电脑的文件上传。没有 Docker 挂载或路径转换逻辑；容器使用者自行挂载源码、权重、配置和输出。`config_json` 相对路径按 lightx2v_path 解析，config 内的相对资源路径也使用后端工作目录。JSON 上限 1 MiB，需为 UTF-8 object。

## 常驻、GPU 与取消

| 环境变量 | 默认值 / 用途 |
| --- | --- |
| LIGHTX2V_PATH | 从本目录真实位置推断 LightX2V 根目录；只作为新建节点初始值 |
| LIGHTX2V_COMFY_PYTHON | Comfy 的 sys.executable；可指定独立后端 Python 的绝对可执行路径 |
| LIGHTX2V_COMFY_RUNTIME | Comfy temp 下的 lightx2v-server |
| LIGHTX2V_COMFY_IDLE_TTL | 600 秒；0 关闭自动释放；需为有限非负数，空字符串无效 |
| LIGHTX2V_COMFY_SAME_GPU_POLICY | exclusive；coexist 允许同卡多个模型保留 |

这些设置在 Comfy 启动前配置，修改后重启。`LIGHTX2V_COMFY_KEEP_ALIVE` 和 profiles 已移除。节点不暴露 TTL、release_after_run、nproc_per_node 或 lora_dir；节点使用单进程，LoRA 全在 config 中。

| 布局 | 两个 Service 的 GPU | 策略 | 行为 |
| --- | --- | --- | --- |
| 同卡轮换 | 0 / 0 | exclusive（默认） | 启动 Relay 前回收 H3；下一轮 H3 重新加载 |
| 双卡常驻 | 0 / 1 | 任一策略 | 各自保留；未超时即可复用 |
| 同卡常驻 | 0 / 0 | coexist | 显存必须容纳二者；不自动驱逐外部模型 |

无限常驻示例（先退出原 Comfy）：

```bash
cd "$COMFY_DIR"
LIGHTX2V_COMFY_IDLE_TTL=0 LIGHTX2V_COMFY_SAME_GPU_POLICY=coexist \
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 python main.py --cpu --listen 127.0.0.1 --port 8188
```

GPU 下拉每 5 秒在页面可见时刷新显存和利用率，不自动选卡、不预留资源。已有的 GPU UUID 选择会保留，因此容器从只开放部分 GPU 改成 `--gpus all` 时，UUID 工作流无需修改。编号按当前环境可见设备的 PCI 地址排序，容器内可能不同于宿主机；改变可见设备后，使用数字编号的工作流需重新核对选卡。MIG 尚不支持。填多张卡不等于启动模型并行；多个服务可以常驻，但管理器仍串行执行推理。

空闲时间从请求完成且媒体交接结束后计算，每个服务独立计时。加载/推理不计入空闲；浏览页面或命中 Comfy 输出缓存不延长 TTL。CPU offload 仍按 config 搬运权重；进程常驻与上游 `warmup` 是不同概念。

**Release** 先确认，只释放当前 Service 最后执行的受管实例；多个相同描述共享一个后端。忙碌时显示 queued，等任务完成后停止；正常停止先 TERM，10 秒后 KILL 残留。**Comfy 取消生成** 则立即向当前任务所属 Supervisor 发出强制终止命令，杀掉该 server 及其子进程，保留其他服务。监听间隔约 50 ms，实际显存回收仍取决于进程退出和驱动。关闭网页不释放服务；退出 Comfy 时 Supervisor 清理其子进程。

连续两次改变 seed，在 TTL 内运行，并检查实例 ID、PID 不变、completed_tasks 增加，才能证明实际复用：

```bash
python - <<'PY'
import json
from urllib.request import urlopen

def get(url):
    with urlopen(url, timeout=10) as response:
        return json.load(response)

for item in get('http://127.0.0.1:8188/lightx2v/services')['instances']:
    status = get(item['url'] + '/v1/service/status')
    print(item['server'], item['instance_id'], item.get('pid'), status.get('completed_tasks'))
PY
```

## 运行边界与排错

```text
ComfyUI → Manager（服务池、串行执行、空闲计时）
            ├─ Supervisor → LightX2V server → 模型 / CUDA / 子进程
            └─ Supervisor → LightX2V server → 另一模型
```

HTTP client 使用 loopback，绕过代理；每次请求唯一 ID，POST 超时只查询原 ID，不重复提交。图片走 `/v1/tasks/image/`，视频走 `/v1/tasks/video/`。未裁切的原生 VIDEO 可文件直通，编辑后的视频通过 Comfy API 导出；输出交回原生 IMAGE/VIDEO，保留视频 FPS 和音轨。

Supervisor 持有 runtime 文件锁，父进程正常退出或崩溃都会清理自己的进程树；Linux subreaper 收养脱离进程组的子进程。无法确认退出时阻止启动下一模型，不会杀外部任务。启动超时 900 秒、任务超时 7200 秒、HTTP 超时 5 秒。当前不提供跨机器后端、逐帧流式输出、多租户或整机 GPU 调度。

日志位于 `ComfyUI/temp/lightx2v-server/instances/<id>/`：`server.log` 是后端日志，`supervisor.log` 是守护日志，`config.json` 是启动快照，`state.json` 含 PID。`/lightx2v/services` 返回实际目录。释放服务保留媒体与日志；Comfy 重启可能清理 temp，长期结果使用 Save Image/Video 保存。

| 问题 | 检查 |
| --- | --- |
| 节点缺失/重复 | 检查软链接、外层 __init__.py、旧插件是否仍在 custom_nodes；重启 Comfy 并刷新网页 |
| 缺少 VIDEO 节点 | 更新 Comfy 和它的 requirements |
| 缺少 lightx2v/httpx | 核对后端 Python，安装依赖后执行 python -m lightx2v.server --help |
| 路径/LoRA 不存在 | Browse 选实际路径；示例占位符不会自动转换 |
| AdaLN 缺失/无效 | 核对缓存目录、用户、variant、步数、flow shift，重新生成匹配缓存 |
| H3 图片输入与 task 不符 | 按首帧/尾帧/参考图选择 i2av/l2av/fl2av/ref2av；t2av 断开图片 |
| OOM/进程被系统杀掉 | 检查 GPU 和主机 RAM、同卡策略、offload、模型允许的尺寸/帧数 |
| FA3 或其他算子不可用 | 配置改为支持的 torch_sdpa，或安装匹配 GPU/PyTorch/CUDA 的算子 |
| 第二次没有后端任务 | 改 seed 避免 Comfy 缓存；检查 TTL、配置变化或同卡切换 |
| 视频无法预览 | Save Video 选择 MP4/H.264，检查 PyAV/FFmpeg 编解码支持 |

已有环境依赖冲突时，可单独创建后端环境，安装匹配的 CUDA PyTorch、`-e "$LIGHTX2V_DIR"` 和 httpx，再在启动 Comfy 前设置 `LIGHTX2V_COMFY_PYTHON=/absolute/backend-venv/bin/python`。AdaLN 也使用该解释器。Comfy、插件和 Supervisor 仍使用原 Comfy Python，无需额外管理服务。

更新前保存工作流和本机 config，释放服务并停止 Comfy；更新后重新检查依赖、模型和缓存。原地更新代码或权重后应重启 Comfy，服务指纹不哈希整个源码库或大权重。

## 语法与格式检查

```bash
cd /path/to/LightX2V/lightx2v_comfyui
python -m compileall -q lightx2v_comfy_server
for file in web/*.js; do node --input-type=module --check < "$file"; done
cd ..
ruff check --config pyproject.toml lightx2v_comfyui
ruff format --config pyproject.toml --check lightx2v_comfyui
```

迁移前的实现曾在 H100 验证 H3 1344×768/124 帧 → 预览 → Relay 2688×1536、连续复用，以及 Wan/Qwen/独立视频链路；H3→Relay 的干净环境使用 SDPA。部署后仍需在目标 GPU 上验证显存需求与模型运行。保存源码提交、模型 revision 和 `pip freeze` 才能复现具体部署。
