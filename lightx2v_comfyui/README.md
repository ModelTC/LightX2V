# LightX2V ComfyUI

English | [简体中文](README_zh.md)

Run local LightX2V servers from ComfyUI. The first inference loads a model; later requests reuse its process. Each idle service is released after 600 seconds by default. The manager runs inside the plugin, so there is no separate management service or backend port configuration.

Symlink this directory into `ComfyUI/custom_nodes`. You need ComfyUI and a LightX2V checkout containing this directory. The standalone ComfyUI-LightX2V-Server repository, Wrapper and MAINodes are not required.

## Nodes and layout

| Node | Inputs → output | Purpose |
| --- | --- | --- |
| LightX2V Service | Source, model, config, task, GPU → `LX2V_SERVER` | Describe a service; load only when inference runs; browse/edit config and release the service |
| LightX2V Generate Video | server, prompt, dimensions, frames, seed, optional first/last `IMAGE` → `VIDEO` | H3, Wan and compatible video models |
| LightX2V Generate Image | server, prompt, dimensions, seed, optional `IMAGE` → `IMAGE` | Text-to-image or single-image editing |
| LightX2V Upscale Video | server, `VIDEO`, scale, seed → `VIDEO` | RelayVSR upscaling |

All four use the same blue `LightX2V-Server: type` badge. Node titles remain editable. Use native Comfy nodes to load, save and preview media. There are no separate Config Path, Release VRAM or pause nodes.

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

Use model configs from the main repository's [`configs/`](../configs), saving local edits outside the checkout. Legacy workflow converters and historical deployment records remain in the standalone repository; they are not runtime components.

## Linux + CUDA installation

Commands use Bash, with Ubuntu 24.04 for system packages. Reuse Comfy's Python by default; Docker is optional. H3 at 768p and RelayVSR require sufficient VRAM, host RAM and disk space. CPU offload does not remove video activation memory costs.

### 1. Prepare ComfyUI and Python

Check the NVIDIA driver and set paths:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
export COMFY_DIR="$HOME/ComfyUI"
export LX_HOME="$HOME/lightx2v-local"
export LIGHTX2V_DIR="$LX_HOME/LightX2V"
mkdir -p "$LX_HOME/models" "$LX_HOME/configs"
```

These variables organize installation commands; they are not required plugin runtime settings.

**Existing ComfyUI:** stop the instance whose dependencies you will update and activate its original environment. For a `.venv` installation:

```bash
source "$COMFY_DIR/.venv/bin/activate"
python -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip freeze > "$LX_HOME/comfy-before-lightx2v.txt"
```

For Conda, use the original `conda activate` command. LightX2V requires Python 3.10+; Python 3.12 is a suitable starting point when supported by your Comfy/PyTorch versions. Comfy must provide native `VIDEO`, `LoadVideo` and `SaveVideo`; update older installations first.

**No ComfyUI yet:** create its environment:

```bash
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

Match the CUDA wheel to your driver and GPU; use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for other combinations. The CUDA Version in `nvidia-smi` does not establish that Python has CUDA PyTorch. A matching Toolkit is needed when building additional CUDA extensions. Check the interpreter:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(), "The H3 configuration below uses BF16"
print(torch.cuda.get_device_name(0))
PY
```

### 2. Install LightX2V and link the plugin

For an existing checkout, set `LIGHTX2V_DIR` to its location. For a new installation, use a revision containing this directory. If your branch does not include it yet, copy the complete `lightx2v_comfyui/` directory into that checkout before creating the link.

```bash
# Skip clone when reusing a checkout.
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

The plugin adds no pip dependencies and does not need its own `pip install`. Install `httpx` explicitly for the LightX2V server instead of relying on another package to supply it. See the [main installation guide](../docs/EN/source/getting_started/quickstart.md) for optional kernels. This guide uses the existing `torch_sdpa` backend and does not require FlashAttention 3.

When replacing the standalone plugin, stop Comfy and **move the old `custom_nodes/ComfyUI-LightX2V-Server` outside custom_nodes as a backup** before linking the new directory. This avoids duplicate registration. The four current class IDs, input ordering and workflow format are preserved. Update existing workflows' source, model and config paths for the new machine. The outer `__init__.py` and `web/` are required; copying only the inner Python package is insufficient.

Some Linux `decord==0.6.0` wheels have inconsistent internal compatibility tags. If the only `pip check` error is `decord 0.6.0 is not supported on this platform`, consult the [upstream issue](https://github.com/dmlc/decord/issues/356) and verify actual CPU decoding. Do not edit package metadata or ignore other dependency errors. For example:

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

### 3. Download H3 and RelayVSR

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir "$LX_HOME/models/MiniMax-H3"
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --local-dir "$LX_HOME/models/H3-Turbo"
hf download kopper/RelayVSR wan.safetensors flash.safetensors \
  --local-dir "$LX_HOME/models/RelayVSR"
```

`hf` comes from `huggingface_hub`; install it with `python -m pip install huggingface_hub` if missing. Use `hf auth login` for restricted models. Existing weights can be reused.

H3's model_path must point to the complete model directory, including transformer, transformer_ref, text encoders and VAE, rather than a single weight file. Use the LightX2V Turbo LoRA without `_comfyui_` in its name. RelayVSR x4 needs both `wan.safetensors` and `flash.safetensors`. See the [H3](../scripts/minimax_h3/README.md) and [RelayVSR](../scripts/relayvsr/README.md) guides.

### 4. Create local configs from the main repository

This creates four-step H3, 29-step H3 ref2av and RelayVSR configs. It refuses to overwrite existing files. The ref2av config does not use the fl2av Turbo LoRA.

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

Use **Browse / Edit config** on Service for later edits. LoRAs belong in `lora_configs`: Browse selects the `path`; set `strength/alpha` as required by the model. `name` is optional under Advanced. Config strings support `${ENV_VAR}` expansion; LightX2V validates model-specific fields.

When H3 AdaLN caching is enabled, `adaln_cache_dir` must be nonempty and the cache must be generated. `~/.cache/lightx2v/adaln` is a conventional path, not a fallback when the field is removed. fl2av and ref2av may share the root directory but use separate cache subdirectories.

### 5. Build H3 AdaLN caches

Use the same backend Python, model, LoRA, config and DTYPE as inference. The commands below use GPU 0; select an available GPU before running them. They leave no resident server:

```bash
cd "$LIGHTX2V_DIR"
PYTHONPATH="$LIGHTX2V_DIR" DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 CUDA_VISIBLE_DEVICES=0 \
  python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$LX_HOME/models/MiniMax-H3" --model-variant fl2av \
  --config_json "$LX_HOME/configs/h3.json"
# Prepare this separate cache when using ref2av.
PYTHONPATH="$LIGHTX2V_DIR" DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 CUDA_VISIBLE_DEVICES=0 \
  python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$LX_HOME/models/MiniMax-H3" --model-variant ref2av \
  --config_json "$LX_HOME/configs/h3-ref2av.json"
```

Wait for `AdaLN cache saved to` before generating video. Recheck the [H3 cache requirements](../scripts/minimax_h3/README.md) after changing variant, steps, flow shifts, model or LoRA. Do not rename another configuration's cache files to reuse them.

### 6. Start Comfy and load the workflow

```bash
cd "$COMFY_DIR"
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 \
  python main.py --cpu --listen 127.0.0.1 --port 8188
```

Choose the inference GPU in each Service node's **CUDA_VISIBLE_DEVICES** dropdown. The Comfy startup command needs no GPU assignment variables. For Docker, use `--gpus all` and keep `--cpu` on the Comfy command; the container exposes the devices and each Service selects its GPU. Starting Comfy or opening a workflow does not load a model. The explicit GPU selection in step 5 applies only to the separate AdaLN cache command.

Open [ComfyUI](http://127.0.0.1:8188/) and import [h3_relay_workflow.json](examples/h3_relay_workflow.json). `--cpu` keeps Comfy itself off CUDA while the independent servers use the GPUs selected on Service. Remove it if other Comfy nodes need the GPU. For a remote host, run `ssh -N -L 8188:127.0.0.1:8188 user@server` on the browser computer. Only the Comfy port needs forwarding.

| Service field | H3 | RelayVSR |
| --- | --- | --- |
| lightx2v_path | Local LightX2V checkout | Same checkout |
| model_path | `$LX_HOME/models/MiniMax-H3` | `$LX_HOME/models/RelayVSR` |
| model_cls | minimax_h3 | relayvsr_x4 |
| config_json | `$LX_HOME/configs/h3.json` | `$LX_HOME/configs/relay.json` |
| task | t2av | sr |
| CUDA_VISIBLE_DEVICES | Select an available GPU in the dropdown | Select the same GPU or another available GPU |

Variables in the table describe locations; use Browse to select actual paths. Saved workflow paths take precedence. A new Service automatically finds the enclosing LightX2V checkout; `LIGHTX2V_PATH` overrides this initial value.

The example generates 1344×768, 124-frame H3 video and upscales 2×. Run executes H3, Save Video with preview, RelayVSR, then a second Save Video. Preview does not pause execution, and Relay receives the original VIDEO. Outputs normally go to `ComfyUI/output/video/`. H3 dimensions must be multiples of 32, with `17*n+5` frames in the 124–362 range. `relayvsr_x4` computes 4× internally before resizing to 2×.

## Workflows and other models

This directory contains eight `*_workflow.json` canvas files. Drag them into ComfyUI to open, edit and run them. Paths and Load Image/Video filenames are placeholders. For scripted execution, export the API format from ComfyUI and submit it to `/prompt`.

| Canvas | Model / task | Config source |
| --- | --- | --- |
| [H3 → preview → Relay](examples/h3_relay_workflow.json) | minimax_h3/t2av + relayvsr_x4/sr | h3.json + relay.json above |
| [Repeated H3 generation](examples/h3_t2av_workflow.json) | minimax_h3/t2av | h3.json |
| [H3 first/last frames](examples/h3_fl2av_workflow.json) | minimax_h3/fl2av | h3.json |
| [H3 reference image](examples/h3_ref2av_workflow.json) | minimax_h3/ref2av | h3-ref2av.json |
| [Wan](examples/wan_t2v_workflow.json) | wan2.1/t2v | [wan_t2v.json](../configs/wan/wan_t2v.json) |
| [Qwen text-to-image](examples/qwen_t2i_workflow.json) | qwen_image/t2i | [2512 distill](../configs/qwen_image/qwen_image_t2i_2512_distill.json) |
| [Qwen editing](examples/qwen_i2i_workflow.json) | qwen_image/i2i | [2511 distill](../configs/qwen_image/qwen_image_i2i_2511_distill.json) |
| [Standalone video upscaling](examples/relay_video_workflow.json) | relayvsr_x4/sr | relay.json |

H3 `t2av` takes no images; `i2av` takes image; `l2av` takes last_frame; `fl2av` takes both; `ref2av` takes one reference image with transformer_ref and its matching config. Connecting images does not switch task automatically. Nodes validate these combinations before loading a model and show specific errors. A ref2av reference is not a fixed first frame.

Skip H3 weights and caching when only using Wan, Qwen or Relay. Download only the models you need:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$LX_HOME/models/Wan2.1-T2V-1.3B"
hf download Qwen/Qwen-Image-2512 --local-dir "$LX_HOME/models/Qwen-Image-2512"
hf download lightx2v/Qwen-Image-2512-Lightning \
  Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors --local-dir "$LX_HOME/models/Qwen-LoRA/qwen2512"
hf download Qwen/Qwen-Image-Edit-2511 --local-dir "$LX_HOME/models/Qwen-Image-Edit-2511"
hf download lightx2v/Qwen-Image-Edit-2511-Lightning \
  Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors --local-dir "$LX_HOME/models/Qwen-LoRA/qwen2511"
```

Copy the table's configs to new files at `$LX_HOME/configs/wan.json`, `qwen.json` and `qwen-edit.json`. Set Wan's `self_attn_1_type/cross_attn_1_type` and Qwen's `attn_type` to `torch_sdpa`, with `use_compile=false` and `warmup=false`. Set each Qwen `lora_configs[0].path` to its downloaded file, including the bf16 filename downloaded for Edit. Keep eight steps. Configure `cpu_offload`, `offload_granularity=block`, `text_encoder_cpu_offload` and `vae_cpu_offload` to suit memory. Wan uses 832×480, 81 frames and 50 steps; start with 17 frames for an initial check.

Standalone upscaling uses native Load Video and your own MP4. For native x2, download `wan_x2.safetensors` and `flash_x2.safetensors` from [lightx2v/RelayVSR_X2](https://huggingface.co/lightx2v/RelayVSR_X2), choose `relayvsr_x2`, the [x2 config](../configs/relayvsr/relayvsr_x2.json) and scale=2. Do not mix x4 weights with x2.

A new model compatible with the existing image/video request contracts needs only Service settings, config and workflow JSON changes. Model/task discovery reads runner declarations without importing inference code. Image support currently covers t2i and single-image i2i; video supports the existing image/last_frame/video inputs. Multiple references, masks, audio inputs and layered outputs require additional media adapters.

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

Set these before starting Comfy and restart after changing them. `LIGHTX2V_COMFY_KEEP_ALIVE` and profiles have been removed. Nodes do not expose TTL, release_after_run, nproc_per_node or lora_dir. Nodes use one backend process; LoRAs are configured in JSON.

| Layout | Service GPUs | Policy | Behavior |
| --- | --- | --- | --- |
| Rotate on one GPU | 0 / 0 | exclusive, the default | Release H3 before Relay starts; reload H3 next round |
| Keep both on two GPUs | 0 / 1 | Either | Independent residency and reuse until timeout |
| Keep both on one GPU | 0 / 0 | coexist | Both must fit; external models are not evicted |

To disable automatic release, stop the previous Comfy process and start it with:

```bash
cd "$COMFY_DIR"
LIGHTX2V_COMFY_IDLE_TTL=0 LIGHTX2V_COMFY_SAME_GPU_POLICY=coexist \
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 python main.py --cpu --listen 127.0.0.1 --port 8188
```

The GPU dropdown refreshes VRAM usage and utilization every five seconds while the page is visible. It neither selects nor reserves a GPU automatically. Existing GPU UUID selections are preserved, so changing the container from selected GPUs to `--gpus all` does not require editing UUID-based workflows. Indices follow visible devices sorted by PCI address and may differ from host indices inside a container; recheck numeric selections after changing device visibility. MIG is not supported. Listing several GPUs does not enable model parallelism. Multiple services can remain resident, but inference is serialized by the manager.

Idle timing starts after the request and media handoff finish, independently for each service. Loading and inference are not idle. Browsing the UI or hitting Comfy's output cache does not reset TTL. CPU offload still moves weights as configured; process residency and upstream `warmup` are separate settings.

**Release** asks for confirmation and targets the last service executed by that node. Identical Service descriptions share a backend. A busy service reports queued and stops after its current request. Normal cleanup sends TERM, then KILL after ten seconds if needed. **Comfy cancellation** immediately asks the current task's Supervisor to force-stop its server and descendants, preserving other services. The interrupt watcher checks about every 50 ms; actual VRAM release still depends on process exit and the driver. Closing the browser does not release services; exiting Comfy triggers Supervisor cleanup.

Change the seed and run twice within TTL. Reuse is demonstrated by unchanged instance ID/PID and an increased completed_tasks count:

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

## Runtime boundaries and troubleshooting

```text
ComfyUI → Manager (service pool, serialized execution, idle timers)
            ├─ Supervisor → LightX2V server → model / CUDA / descendants
            └─ Supervisor → LightX2V server → another model
```

The HTTP client uses loopback and bypasses proxies. Every request has a unique ID; a timed-out POST is queried by that same ID, never resubmitted. Images use `/v1/tasks/image/`, videos use `/v1/tasks/video/`. An untrimmed native VIDEO can pass its file directly; edited video is exported through Comfy's API. Outputs return as native IMAGE/VIDEO, retaining video FPS and audio.

Supervisors hold the runtime file lock and clean their own process trees on normal parent exit or a crash. Linux subreapers adopt descendants that leave the original process group. Unconfirmed cleanup blocks a new launch; external tasks are not killed. Defaults are 900 seconds for startup, 7200 for a task and five for HTTP. Remote backends, frame streaming, multi-tenancy and host-wide GPU scheduling are not implemented.

Logs live in `ComfyUI/temp/lightx2v-server/instances/<id>/`: `server.log` for inference, `supervisor.log` for supervision, `config.json` for the startup snapshot and `state.json` for process state/PID. `/lightx2v/services` returns the actual directory. Releasing a service keeps media and logs; Comfy may clear temp at restart. Use Save Image/Video for durable outputs.

| Symptom | Check |
| --- | --- |
| Missing or duplicate nodes | Symlink, outer __init__.py and an old plugin still in custom_nodes; restart Comfy and refresh |
| Missing native VIDEO nodes | Update Comfy and its requirements |
| Missing lightx2v/httpx | Check backend Python, install dependencies, run python -m lightx2v.server --help |
| Missing paths/LoRAs | Browse actual files; example placeholders are not translated |
| Missing/invalid AdaLN cache | Check directory, user, variant, steps and flow shifts; regenerate the matching cache |
| H3 image/task mismatch | Select i2av/l2av/fl2av/ref2av for the connected inputs; disconnect images for t2av |
| OOM or OS-killed process | Check VRAM, RAM, GPU policy, offload and model-supported dimensions/frame counts |
| FA3 or another kernel unavailable | Use supported torch_sdpa or install kernels matching GPU/PyTorch/CUDA |
| No backend task on second run | Change seed to avoid Comfy caching; check TTL, config changes and GPU switching |
| Preview cannot play | Use MP4/H.264 in Save Video and check PyAV/FFmpeg codec support |

If dependencies conflict, create a separate backend environment with compatible CUDA PyTorch, `-e "$LIGHTX2V_DIR"` and httpx. Set `LIGHTX2V_COMFY_PYTHON=/absolute/backend-venv/bin/python` before starting Comfy and use that interpreter for AdaLN generation too. Comfy, the plugin and Supervisors remain in the original environment; no extra manager service is needed.

Before updates, save workflows and local configs, release services and stop Comfy. Recheck dependencies, models and caches afterward. Restart Comfy after modifying code or weights in place: fingerprints do not hash the entire source tree or large model files.

## Syntax and formatting

```bash
cd /path/to/LightX2V/lightx2v_comfyui
python -m compileall -q lightx2v_comfy_server
for file in web/*.js; do node --input-type=module --check < "$file"; done
cd ..
ruff check --config pyproject.toml lightx2v_comfyui
ruff format --config pyproject.toml --check lightx2v_comfyui
```

Before migration, the implementation was tested on H100 with H3 at 1344×768/124 frames → preview → Relay at 2688×1536, repeated service reuse, and Wan/Qwen/standalone video workflows. The clean H3→Relay deployment used SDPA. Validate model execution and VRAM requirements on the target GPU after deployment. Record source commits, model revisions and `pip freeze` to reproduce a specific deployment.
