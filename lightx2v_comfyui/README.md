# LightX2V ComfyUI

English | [简体中文](README_zh.md)

Run local LightX2V servers from ComfyUI. The first inference loads a model; later requests reuse its process.

This directory can be symlinked into `ComfyUI/custom_nodes`.

## Nodes and layout

| Node | Inputs → output | Purpose |
| --- | --- | --- |
| LightX2V Service | Source, model, config, task, GPU → `LX2V_SERVER` | Describe a service; load only when inference runs; browse/edit config and release the service |
| LightX2V Generate Video | server, prompt, dimensions, frames, seed, optional first/last `IMAGE` → `VIDEO` | H3, Wan and compatible video models |
| LightX2V Generate Image | server, prompt, dimensions, seed, optional `IMAGE` → `IMAGE` | Text-to-image or single-image editing |
| LightX2V Upscale Video | server, `VIDEO`, scale, seed → `VIDEO` | Video upscaling |

All four use the same blue `LightX2V-Server: type` badge. Use native Comfy nodes to load, save and preview media.

```text
lightx2v_comfyui/
├── __init__.py                  # Comfy entry point, nodes, routes, WEB_DIRECTORY
├── lightx2v_comfy_server/
│   ├── nodes.py                 # Four nodes and IMAGE/VIDEO file conversion
│   ├── service.py               # Startup settings, fingerprints, identity, commands
│   ├── manager.py               # HTTP, service pool, execution lock, cancellation, TTL
│   ├── supervisor.py            # Parent death detection and process-tree cleanup
│   ├── models.py                # Static model/task discovery and H3 variant selection
│   ├── gpus.py                  # GPU usage, device indices and UUID normalization
│   ├── files.py                 # Paths, browsing, JSON validation, backups, atomic saves
│   └── routes.py                # Comfy /lightx2v/* HTTP endpoints
├── web/                         # Service, GPU, file browser and config editor UI
├── examples/                    # Eight canvas workflows; no model configs
├── README.md
└── README_zh.md
```

Use model configs from the main repository's [`configs/`](../configs), saving local edits to a persistent directory.

## Linux + CUDA installation

Commands use Bash, with Ubuntu 24.04 for system packages. Reuse Comfy's Python by default; Docker is optional.

### 1. Prepare ComfyUI and Python

**Existing ComfyUI:** activate its original environment, such as `.venv`, or use the original `conda activate` command for Conda. LightX2V requires Python 3.10+; Python 3.12 is recommended when compatible with your Comfy and PyTorch versions.
```bash
export COMFY_DIR="/path/to/ComfyUI"
```

**No ComfyUI yet:** create its environment:

```bash
export COMFY_DIR="$HOME/ComfyUI"
sudo apt-get update
sudo apt-get install -y git curl python3.12 python3.12-venv \
  build-essential pkg-config ffmpeg libgl1 libglib2.0-0t64
git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
python3.12 -m venv "$COMFY_DIR/.venv"
source "$COMFY_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
# CUDA 13.0 wheel example; skip if CUDA PyTorch already works.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r "$COMFY_DIR/requirements.txt"
```

Match the CUDA wheel to your driver and GPU; use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for other combinations.

### 2. Install LightX2V and link the plugin

For an existing LightX2V checkout, set `LIGHTX2V_DIR` to its location.

```bash
export LIGHTX2V_DIR="$HOME/LightX2V"
# Skip clone when reusing a checkout.
git clone https://github.com/ModelTC/LightX2V.git "$LIGHTX2V_DIR"
python -m pip install -v -e "$LIGHTX2V_DIR"
python -m pip install httpx
cd "$LIGHTX2V_DIR"
python -m lightx2v.server --help

test -f "$LIGHTX2V_DIR/lightx2v_comfyui/__init__.py"
mkdir -p "$COMFY_DIR/custom_nodes"
ln -s "$LIGHTX2V_DIR/lightx2v_comfyui" "$COMFY_DIR/custom_nodes/lightx2v_comfyui"
```

The plugin adds no pip dependencies and does not need its own `pip install`. Install `httpx` explicitly for the LightX2V server instead of relying on another package to supply it. See the [main installation guide](../docs/EN/source/getting_started/quickstart.md) for optional kernels.

### 3. Start Comfy and load the workflow

```bash
cd "$COMFY_DIR"
python main.py --cpu --listen 127.0.0.1 --port 8188
```

Choose the inference GPU in each Service node's **CUDA_VISIBLE_DEVICES** dropdown. The Comfy startup command needs no GPU assignment variables. For Docker, use `--gpus all` and keep `--cpu` on the Comfy command; the container exposes the devices and each Service selects its GPU. Starting Comfy or opening a workflow does not load a model.

Open [ComfyUI](http://127.0.0.1:8188/) and import [h3_t2av_workflow.json](examples/h3_t2av_workflow.json). `--cpu` keeps Comfy itself off CUDA while the independent servers use the GPUs selected on Service. Remove it if other Comfy nodes need the GPU. For a remote host, run `ssh -N -L 8188:127.0.0.1:8188 user@server` on the browser computer. Only the Comfy port needs forwarding.

## Deploy from the official Docker image

This is an alternative to the native installation above. The Linux host needs an NVIDIA driver, Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The commands use `lightx2v/lightx2v:26052801-cu130` from the [official quick start](../docs/EN/source/getting_started/quickstart.md). For other CUDA versions, choose a tag compatible with your driver and GPU from [Docker Hub](https://hub.docker.com/r/lightx2v/lightx2v/tags).

### 1. Prepare the host directory and enter the container

The entire `/workspace` directory maps to `$LX_HOME` on the host, keeping source, ComfyUI, models, configs, caches and generated files there. To reuse a LightX2V checkout, place it at `$LX_HOME/LightX2V` and skip clone. The checkout must contain this plugin directory.

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

Run steps 2–3 below in this container shell. All GPUs are visible to the container; starting Comfy does not assign a GPU or load a model.

### 2. Use the image's Python to install ComfyUI and the plugin

No extra venv is needed. The image provides CUDA, PyTorch and inference kernels; install the current LightX2V checkout's dependencies and ComfyUI packages.

```bash
export LIGHTX2V_DIR=/workspace/LightX2V
export COMFY_DIR=/workspace/ComfyUI
git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
python -m pip install -r "$COMFY_DIR/requirements.txt" httpx huggingface_hub
mkdir -p "$COMFY_DIR/custom_nodes"
ln -s "$LIGHTX2V_DIR/lightx2v_comfyui" "$COMFY_DIR/custom_nodes/lightx2v_comfyui"
```

### 3. Start ComfyUI

```bash
cd /workspace/ComfyUI
python main.py --cpu --listen 0.0.0.0 --port 8188
```

Listening on `0.0.0.0` inside the container accepts Docker's forwarded connections; the host publishes the page only at `127.0.0.1:8188`. Open [ComfyUI](http://127.0.0.1:8188/), import [h3_t2av_workflow.json](examples/h3_t2av_workflow.json) and update the Service placeholders.

For a remote host, run `ssh -N -L 8188:127.0.0.1:8188 user@server` on the browser computer. No LightX2V server ports need publishing; the plugin starts servers as needed.

Press `Ctrl+C` in the container shell to stop Comfy, then run `exit` to leave and stop the container. To return, run `docker start -ai lightx2v-comfyui` on the host and repeat this section's Comfy startup command.

## Workflows and other models

This directory contains eight `*_workflow.json` canvas files. Drag them into ComfyUI to open, edit and run them. All paths and Load Image/Video filenames are placeholders and need to be replaced.

| Canvas | Model / task | Config source |
| --- | --- | --- |
| [H3 → preview → Relay](examples/h3_relay_workflow.json) | minimax_h3/t2av + relayvsr_x4/sr | h3.json + relay.json |
| [Repeated H3 generation](examples/h3_t2av_workflow.json) | minimax_h3/t2av | h3.json |
| [H3 first/last frames](examples/h3_fl2av_workflow.json) | minimax_h3/fl2av | h3.json |
| [H3 reference image](examples/h3_ref2av_workflow.json) | minimax_h3/ref2av | h3-ref2av.json |
| [Wan](examples/wan_t2v_workflow.json) | wan2.1/t2v | [wan_t2v.json](../configs/wan/wan_t2v.json) |
| [Qwen text-to-image](examples/qwen_t2i_workflow.json) | qwen_image/t2i | [2512 distill](../configs/qwen_image/qwen_image_t2i_2512_distill.json) |
| [Qwen editing](examples/qwen_i2i_workflow.json) | qwen_image/i2i | [2511 distill](../configs/qwen_image/qwen_image_i2i_2511_distill.json) |
| [Standalone video upscaling](examples/relay_video_workflow.json) | relayvsr_x4/sr | relay.json |

## Config editing and paths

**Browse LightX2V / Browse model** select directories. **Browse / Edit config** opens the file selected by this Service. Browse selects an existing JSON file and updates the Service's config_json path. JSON is formatted when read; edit it or the LoRA fields, then click **Save**. Switching files or closing with unsaved edits asks before discarding them.

- Save writes to the selected file. It checks the content hash captured when opened, backs up the original bytes in `.lightx2v-config-backups`, then atomically replaces the file while preserving permissions. If another editor changed the file, Browse it again before editing.
- The next Run reads the selected file's saved contents and replaces the affected service if its config changed. Unsaved edits are not used. Services sharing the file see the same saved changes; saving does not interrupt a running model.
- LoRA Browse selects a `.safetensors` path without loading weights or a model. `name` is under Advanced. Unknown JSON fields are preserved.

All paths refer to the **filesystem visible to the Comfy process**, including inside containers. A remote browser's Browse dialog does not upload files from the browser computer. There is no Docker mount creation or path translation. Mount source, weights, configs and outputs yourself. Relative config_json paths resolve against lightx2v_path; relative resources inside configs use the backend working directory too. Configs must be UTF-8 JSON objects, up to 1 MiB.

## Residency, GPUs and cancellation

| Environment variable | Default / purpose |
| --- | --- |
| LIGHTX2V_PATH | LightX2V root inferred from this directory's real location; initial value for new nodes |
| LIGHTX2V_COMFY_PYTHON | Comfy's sys.executable; optionally an absolute executable path to a separate backend Python |
| LIGHTX2V_COMFY_RUNTIME | lightx2v-server inside Comfy's temp directory |
| LIGHTX2V_COMFY_IDLE_TTL | 600 seconds; 0 disables idle release; finite and nonnegative, empty values are invalid |
| LIGHTX2V_COMFY_SAME_GPU_POLICY | exclusive; coexist permits multiple resident models on one GPU |

Set these before starting Comfy and restart after changing them.

| Layout | Service GPUs | Policy | Behavior |
| --- | --- | --- | --- |
| Rotate on one GPU | 0 / 0 | exclusive, the default | Release H3 before Relay starts; reload H3 next round |
| Keep both on two GPUs | 0 / 1 | Either | Independent residency and reuse until timeout |
| Keep both on one GPU | 0 / 0 | coexist | Both must fit; external models are not evicted |

To disable automatic release, stop the previous Comfy process and start it with:

```bash
cd "$COMFY_DIR"
LIGHTX2V_COMFY_IDLE_TTL=0 LIGHTX2V_COMFY_SAME_GPU_POLICY=coexist \
python main.py --cpu --listen 127.0.0.1 --port 8188
```
