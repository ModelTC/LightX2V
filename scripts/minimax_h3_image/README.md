# MiniMax-H3-Image

[English](README.md) | [简体中文](README_zh.md)

Supports text-to-image (`t2i`) and reference-image generation/editing (`i2i`) with the same MiniMax-H3-Image weights and six-step PDD schedule. I2I accepts one or multiple reference images.

## Setup

Install LightX2V and the dependencies for your device first. The Qwen3-VL vision encoder requires `transformers>=5.15.0`, including the `get_vision_interpolation_indices_and_weights` interface.

Run the commands below from the repository root. In each new terminal, set the paths and load the base environment:

```bash
lightx2v_path=/data/nvme1/yongyang/dan/x2v_infer/h3-image/LightX2V
model_path=/data/nvme1/models/MiniMaxAI/MiniMax-H3-Image

cd "$lightx2v_path"
source "$lightx2v_path/scripts/base/base.sh"
```

Set `model_path` to the complete model root containing the Transformer, Qwen3-VL, VAE, tokenizer/processor, `pdd_config.json`, and `pdd_fused_heads.safetensors`.

Before using an inference script or the cache generation script, update its `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES`. The current single-GPU inference scripts use GPU 7, the four-GPU scripts use GPUs 4, 5, 6, and 7, and the cache generation script uses GPU 0.

## AdaLN cache

**Example configs enable AdaLN caching by default. Generate a matching cache before your first inference run or server startup; skip this step if a matching cache already exists.**

```bash
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh
```

The [cache generation script](../../tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh) needs only one GPU. The log message `AdaLN cache saved to` confirms completion. Inference and serving locate the cache automatically based on the model and schedule.

Default settings:

```json
{
  "use_adaln_cache": true,
  "adaln_cache_dir": "~/.cache/lightx2v/adaln"
}
```

The default cache directory for the current model is:

```text
~/.cache/lightx2v/adaln/minimax_h3_image/
└── image_06steps_1.000_0.9675_0.8875_0.7500_0.5328_0.2320/
    ├── manifest.json
    └── adaln_cache.safetensors
```

The directory name includes the inference step count and the sigma for each step, formatted to four significant digits, including trailing zeros. The terminal `0.0`, which marks the integration endpoint, is excluded. `~` refers to the home directory of the user running the process.

A cache for the same model and schedule can be reused across T2I, I2I, prompts, reference images, seeds, output sizes, and TP/SP configurations. Regenerate it after changing schedule timesteps or the AdaLN-related weights used to build it. `manifest.json` validates exact timesteps and cache structure; it does not record or check checkpoint file fingerprints, PDD configuration hashes, or runtime environment information.

The generation tool raises `FileExistsError` if the target directory already exists. To rebuild, move the old cache directory aside or change `adaln_cache_dir`. Cache generation and inference must use the same cache root.

## Offline CLI inference

### Example scripts

| Task | Single GPU | Four GPUs, TP2 × SP2 |
| --- | --- | --- |
| Text-to-image | [run_minimax_h3_image_t2i.sh](run_minimax_h3_image_t2i.sh) | [run_minimax_h3_image_t2i_tp_sp.sh](run_minimax_h3_image_t2i_tp_sp.sh) |
| Reference-image generation/editing | [run_minimax_h3_image_i2i.sh](run_minimax_h3_image_i2i.sh) | [run_minimax_h3_image_i2i_tp_sp.sh](run_minimax_h3_image_i2i_tp_sp.sh) |

Choose a script for your task:

```bash
# Single-GPU text-to-image
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i.sh

# Single-GPU reference-image generation/editing
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i.sh

# Four-GPU text-to-image
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i_tp_sp.sh

# Four-GPU reference-image generation/editing
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_tp_sp.sh
```

All four scripts currently generate `4096 × 4096` PNG files in the repository's `save_results/` directory, with filenames corresponding to the task and parallel mode. The default I2I reference image is `$model_path/examples/reference1.png`.

The four-GPU I2I script accepts an `IMAGE_PATH` override. For multiple references, use comma-separated paths and update the script's `prompt` accordingly:

```bash
IMAGE_PATH="/absolute/path/reference1.png,/absolute/path/reference2.png" \
  bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_tp_sp.sh
```

### Direct CLI commands

Single-GPU T2I, with explicit prompt, size, seed, and output path:

```bash
CUDA_VISIBLE_DEVICES=7 python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task t2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image.json" \
  --prompt "A photorealistic photograph of a cat sitting by a sunlit window." \
  --size 2048 2048 \
  --seed 42 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_t2i.png"
```

Four-GPU I2I:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task i2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image_tp_sp.json" \
  --image_path "$model_path/examples/reference1.png" \
  --prompt "Keep the scene in <Picture 1> and change the lighting to warm sunset light." \
  --size 2048 2048 \
  --seed 1234 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_i2i_tp_sp.png"
```

The CLI's `--image_path` also accepts comma-separated paths for up to 16 reference images. Images correspond, in input order, to `<Picture 1>`, `<Picture 2>`, and subsequent markers in the prompt.

### Parallel configurations

The CUDA configs in [configs/minimax_h3_image](../../configs/minimax_h3_image) have the following defaults and can be used for T2I, I2I, and serving:

| Config | GPUs | DiT | Qwen3-VL | VAE |
| --- | --- | --- | --- | --- |
| [minimax_h3_image.json](../../configs/minimax_h3_image/minimax_h3_image.json) | 1 | Single GPU | Single GPU | Single GPU |
| [minimax_h3_image_tp.json](../../configs/minimax_h3_image/minimax_h3_image_tp.json) | 2 | TP2 | Rank 0 | Rank 0 |
| [minimax_h3_image_sp.json](../../configs/minimax_h3_image/minimax_h3_image_sp.json) | 2 | SP2 (Ulysses) | Rank 0 | Rank 0 |
| [minimax_h3_image_tp_sp.json](../../configs/minimax_h3_image/minimax_h3_image_tp_sp.json) | 4 | TP2 × SP2 (Ulysses) | TP4 | Spatial tile parallelism for both encoding and decoding |

The four-GPU config enables `text_encoder_tensor_parallel`, `vae_encode_parallel`, and `vae_decode_parallel` by default. The VAE encoder distributes tiles from one or multiple reference images across ranks, then uses all-gather so every rank receives the complete encoding results. The decoder computes tiles across ranks, and rank 0 collects, stitches, and outputs the final image. T2I has no reference images and skips VAE encoding.

When changing configurations, also adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` so the process count matches the DiT TP × SP size. Qwen3-VL TP uses the entire inference process group, giving TP4 in the four-GPU mixed configuration.

## Serving and POST requests

### Start the server

Complete the path setup and AdaLN cache preparation first. Omit `--task` at server startup to support both T2I and I2I, selected by each request's `task`. Setting `--task t2i` or `--task i2i` at startup restricts the server to that task. H3 Image does not require `--model-variant`.

Set the following variable to store cached server inputs and asynchronous outputs under the repository's `save_results/server_cache/` directory:

```bash
export LIGHTX2V_CACHE_DIR="$lightx2v_path/save_results/server_cache"
```

Single-GPU server:

```bash
CUDA_VISIBLE_DEVICES=7 python -m lightx2v.server \
  --model_cls minimax_h3_image \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image.json" \
  --host 0.0.0.0 \
  --port 8000
```

Four-GPU server, using the same Qwen3-VL, DiT, and VAE parallel configuration as the CLI:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m lightx2v.server \
  --model_cls minimax_h3_image \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image_tp_sp.json" \
  --host 0.0.0.0 \
  --port 8000
```

Choose either the single-GPU or four-GPU startup command. Wait for model loading, compilation, warmup, and HTTP startup to complete before sending requests from another terminal. In multi-GPU mode, only rank 0 exposes the HTTP API; the other ranks participate in inference.

```bash
curl --fail http://127.0.0.1:8000/v1/service/status
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. For remote access, replace `127.0.0.1` below with the server's address.

### Synchronous requests: return PNG bytes

`POST /v1/tasks/image/sync` waits for generation and returns `image/png` bytes. Save them with `-o`. These examples do not require a server-side output file; the image is saved on the client running curl.

T2I:

```bash
mkdir -p save_results
curl --fail --max-time 610 \
  http://127.0.0.1:8000/v1/tasks/image/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "t2i",
    "prompt": "A photorealistic photograph of a cat sitting by a sunlit window.",
    "seed": 42,
    "size": [2048, 2048]
  }' \
  -o save_results/minimax_h3_image_t2i_http.png
```

I2I with a reference image on the server:

```bash
mkdir -p save_results
curl --fail --max-time 610 \
  http://127.0.0.1:8000/v1/tasks/image/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "i2i",
    "prompt": "Keep the scene in <Picture 1> and change the lighting to warm sunset light.",
    "image_path": "/data/nvme1/models/MiniMaxAI/MiniMax-H3-Image/examples/reference1.png",
    "seed": 1234,
    "size": [2048, 2048]
  }' \
  -o save_results/minimax_h3_image_i2i_http.png
```

`image_path` is a string. A single image can also be provided as a URL accessible to the server or as a Base64-encoded image file. For client-local images, encode the file contents as Base64 or place the file where the server can access it. For multiple references, use comma-separated **server-local paths**, for example `"image_path": "/absolute/path/a.png,/absolute/path/b.png"`, and refer to them in order as `<Picture 1>` and `<Picture 2>` in the prompt.

The synchronous endpoint has a default timeout of 600 seconds. For longer requests, set a query parameter such as `?timeout_seconds=1200` and increase curl's `--max-time` accordingly.

### Asynchronous requests: submit, query, and download

The asynchronous endpoint returns a task ID immediately. Explicitly set `save_result_path` if you want to download the result later:

```bash
curl --fail http://127.0.0.1:8000/v1/tasks/image/ \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "t2i",
    "prompt": "A photorealistic photograph of a cat sitting by a sunlit window.",
    "seed": 42,
    "size": [2048, 2048],
    "save_result_path": "minimax_h3_image_t2i_async.png"
  }'
```

Replace the value below with the returned `task_id`, then query its status:

```bash
task_id="replace-with-returned-task-id"
curl --fail "http://127.0.0.1:8000/v1/tasks/$task_id/status"
```

Successful tasks progress through `pending`, `processing`, and `completed`. Failed tasks report `failed` with error details in the status response. Once the status is `completed`, download the image:

```bash
mkdir -p save_results
curl --fail "http://127.0.0.1:8000/v1/tasks/$task_id/result" \
  -o save_results/minimax_h3_image_t2i_async.png
```

Relative `save_result_path` values are resolved under the server cache's `outputs/` directory. With the environment variable above, this example saves to `$lightx2v_path/save_results/server_cache/outputs/minimax_h3_image_t2i_async.png`. Asynchronous I2I uses the same endpoint; set `task` to `"i2i"` and add `image_path`.

## Request parameters and startup configuration

| Setting | Location and meaning |
| --- | --- |
| Model, parallelism, offload, operators, compilation, and warmup | Startup arguments and `--config_json`; shared by the CLI and server. |
| `task` | Selected with `--task` in the CLI; the server commands above select `t2i`/`i2i` through each request's `task`. |
| `prompt`, `image_path`, `seed` | Set per request. An omitted or null seed defaults to 42; an explicit 0 is valid. |
| `size` | `[height, width]`; use `--size height width` in the CLI. An explicit request size overrides the JSON; when omitted, the config default is `[2048, 2048]`. |
| `save_result_path` | Required to save PNG files through the CLI. Synchronous HTTP requests can receive PNG bytes directly; asynchronous downloads require a server-side save path. |

All four CUDA configs for this directory enable `use_adaln_cache`, `use_compile`, and `warmup` by default. The single-GPU, TP2, and SP2 configs enable CPU offload. The four-GPU TP2 × SP2 config sets `cpu_offload`, `text_encoder_cpu_offload`, and `vae_cpu_offload` to `false`, keeping DiT, Qwen3-VL, and VAE weights resident on the GPUs. Warmup runs at `[2048, 2048]` and `[3552, 4736]`; when serving both tasks, both are warmed up. Warmup sizes are independent of individual request output sizes.

Current inference uses the six-step PDD schedule and BF16 weights, preserving sensitive computations in FP32. The model is guidance-distilled: keep `enable_cfg: false` and `sample_guide_scale: 1.0`, and omit negative prompts.
