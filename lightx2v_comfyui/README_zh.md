# LightX2V ComfyUI

[English](README.md) | 简体中文

通过 ComfyUI 调用本机的 LightX2V server。首次执行加载模型，后续请求复用同一进程。

本目录可以直接软链接到 `ComfyUI/custom_nodes`。

## 节点与目录

| 节点                      | 输入 → 输出                                         | 用途                                |
| ----------------------- | ----------------------------------------------- | --------------------------------- |
| LightX2V Service        | 源码、模型、config、task、GPU → `LX2V_SERVER`           | 描述服务；执行推理时才加载模型；浏览/编辑 config、释放服务 |
| LightX2V Generate Video | server、prompt、尺寸、帧数、seed、可选首尾 `IMAGE` → `VIDEO` | H3、Wan 等视频生成                      |
| LightX2V Generate Image | server、prompt、尺寸、seed、可选 `IMAGE` → `IMAGE`      | 文生图或单图编辑                          |
| LightX2V Upscale Video  | server、`VIDEO`、scale、seed → `VIDEO`             | 视频超分                              |

四类节点统一显示蓝底 `LightX2V-Server: 类型` 标签。图片/视频的加载、保存和预览使用 Comfy 原生节点。

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

模型 config 统一使用主仓库 [`configs/`](../configs)，本机修改另存到持久目录。

## Linux + CUDA 部署

以下使用 Bash，系统包命令以 Ubuntu 24.04 为例。默认复用 Comfy 的 Python；Docker 可选。

### 1. 准备 ComfyUI 和 Python

**已有 ComfyUI**：激活原环境。例如原来使用 `.venv`，Conda 用户使用原来的 `conda activate`。LightX2V 要求 Python 3.10+，建议使用与 Comfy、PyTorch 共同兼容的 Python 3.12。
```bash
export COMFY_DIR="/path/to/ComfyUI"
```

**没有 ComfyUI**：创建 Comfy 自己的环境：

```bash
export COMFY_DIR="$HOME/ComfyUI"
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

CUDA wheel 必须匹配驱动和 GPU，其他组合参考 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)。

### 2. 安装 LightX2V 并链接插件

已有 LightX2V 可直接将 `LIGHTX2V_DIR` 指向它。

```bash
export LIGHTX2V_DIR="$HOME/LightX2V"
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

插件没有额外 pip 依赖，无需对本目录执行 `pip install`。`httpx` 是 LightX2V server 的依赖，显式安装可避免依赖间接引入失效。算子安装参考[主仓库安装文档](../docs/EN/source/getting_started/quickstart.md)。

### 3. 启动与导入工作流

```bash
cd "$COMFY_DIR"
python main.py --cpu --listen 127.0.0.1 --port 8188
```

推理 GPU 统一在每个 Service 节点的 **CUDA_VISIBLE_DEVICES** 下拉框中选择，Comfy 启动命令无需配置选卡环境变量。Docker 启动使用 `--gpus all`，Comfy 命令保留 `--cpu`：容器开放设备，具体用哪张卡由 Service 决定。启动 Comfy 或打开工作流不会加载模型。

打开 [ComfyUI](http://127.0.0.1:8188/)，导入 [h3_t2av_workflow.json](examples/h3_t2av_workflow.json)。`--cpu` 让 Comfy 本身不创建 CUDA 上下文，独立后端仍使用节点选择的 GPU；需要其他 Comfy GPU 节点时可去掉它。远端部署在浏览器电脑使用 `ssh -N -L 8188:127.0.0.1:8188 用户@服务器`，只转发 Comfy 端口。

## 从官方 Docker 镜像部署

这是上面本机安装流程的另一种方式。Linux 宿主机需已安装 NVIDIA 驱动、Docker 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。下面使用[官方快速开始](../docs/ZH_CN/source/getting_started/quickstart.md)中的 `lightx2v/lightx2v:26052801-cu130`；其他 CUDA 版本从 [Docker Hub](https://hub.docker.com/r/lightx2v/lightx2v/tags) 选择与驱动、GPU 兼容的 tag。

### 1. 在宿主机准备目录并进入容器

整个 `/workspace` 挂载到宿主机的 `$LX_HOME`，用于保存源码、ComfyUI、模型、配置、缓存和生成结果。已有 LightX2V checkout 时可放到 `$LX_HOME/LightX2V`，跳过 clone；所用版本需包含本目录。

```bash
export LX_HOME="$HOME/lightx2v-comfyui"
export LX_IMAGE="lightx2v/lightx2v:26052801-cu130"
mkdir -p "$LX_HOME"/{models,configs,cache}
git clone https://github.com/ModelTC/LightX2V.git "$LX_HOME/LightX2V"
test -f "$LX_HOME/LightX2V/lightx2v_comfyui/__init__.py"
docker pull "$LX_IMAGE"
docker run --gpus all -it --init --ipc=host --name lightx2v-comfyui \
  -p 127.0.0.1:8188:8188 \
  -v "$LX_HOME":/workspace \
  -e HF_HOME=/workspace/cache/huggingface \
  -e XDG_CACHE_HOME=/workspace/cache \
  -w /workspace --entrypoint /bin/bash "$LX_IMAGE"
```

以下第 2–3 步均在这个容器终端执行。容器开放全部 GPU，Comfy 启动时不固定选卡，也不加载模型。

### 2. 复用镜像的 Python，安装 ComfyUI 和插件

不创建额外 venv。镜像提供 CUDA、PyTorch 和推理算子；安装当前 LightX2V 源码的依赖及 ComfyUI 所需包。

```bash
export LIGHTX2V_DIR=/workspace/LightX2V
export COMFY_DIR=/workspace/ComfyUI
git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
python -m pip install -r "$COMFY_DIR/requirements.txt" httpx huggingface_hub
mkdir -p "$COMFY_DIR/custom_nodes"
ln -s "$LIGHTX2V_DIR/lightx2v_comfyui" "$COMFY_DIR/custom_nodes/lightx2v_comfyui"
```

### 3. 启动 ComfyUI

```bash
cd /workspace/ComfyUI
python main.py --cpu --listen 0.0.0.0 --port 8188
```

容器内监听 `0.0.0.0` 才能接收 Docker 转发，宿主机仍只通过 `127.0.0.1:8188` 开放页面。打开 [ComfyUI](http://127.0.0.1:8188/)，导入 [h3_t2av_workflow.json](examples/h3_t2av_workflow.json)，修改 Service 中的占位值。

远端部署时，在浏览器电脑运行 `ssh -N -L 8188:127.0.0.1:8188 用户@服务器`。无需暴露 LightX2V server 端口，服务由插件按需启动。

在容器终端按 `Ctrl+C` 停止 Comfy，再执行 `exit` 退出并停止容器。以后在宿主机执行 `docker start -ai lightx2v-comfyui`，再运行本节的 Comfy 启动命令即可。

## 工作流与其他模型

本目录保留 8 个 `*_workflow.json` 画布文件，可直接拖入 ComfyUI 打开、编辑和运行。所有路径和 Load Image/Video 文件名都是占位值，需要替换。

| 画布                                                 | 模型 / task                        | 配置来源                                                                   |
| -------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| [H3 → 预览 → Relay](examples/h3_relay_workflow.json) | minimax_h3/t2av + relayvsr_x4/sr | h3.json + relay.json                                                   |
| [H3 连续生成](examples/h3_t2av_workflow.json)          | minimax_h3/t2av                  | h3.json                                                                |
| [H3 首尾帧](examples/h3_fl2av_workflow.json)          | minimax_h3/fl2av                 | h3.json                                                                |
| [H3 参考图](examples/h3_ref2av_workflow.json)         | minimax_h3/ref2av                | h3-ref2av.json                                                         |
| [Wan](examples/wan_t2v_workflow.json)              | wan2.1/t2v                       | [wan_t2v.json](../configs/wan/wan_t2v.json)                            |
| [Qwen 文生图](examples/qwen_t2i_workflow.json)        | qwen_image/t2i                   | [2512 distill](../configs/qwen_image/qwen_image_t2i_2512_distill.json) |
| [Qwen 单图编辑](examples/qwen_i2i_workflow.json)       | qwen_image/i2i                   | [2511 distill](../configs/qwen_image/qwen_image_i2i_2511_distill.json) |
| [纯视频超分](examples/relay_video_workflow.json)        | relayvsr_x4/sr                   | relay.json                                                             |

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

这些设置在 Comfy 启动前配置，修改后重启。

| 布局 | 两个 Service 的 GPU | 策略 | 行为 |
| --- | --- | --- | --- |
| 同卡轮换 | 0 / 0 | exclusive（默认） | 启动 Relay 前回收 H3；下一轮 H3 重新加载 |
| 双卡常驻 | 0 / 1 | 任一策略 | 各自保留；未超时即可复用 |
| 同卡常驻 | 0 / 0 | coexist | 显存必须容纳二者；不自动驱逐外部模型 |

无限常驻示例（先退出原 Comfy）：

```bash
cd "$COMFY_DIR"
LIGHTX2V_COMFY_IDLE_TTL=0 LIGHTX2V_COMFY_SAME_GPU_POLICY=coexist \
python main.py --cpu --listen 127.0.0.1 --port 8188
```
