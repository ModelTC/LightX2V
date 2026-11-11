# MiniMax-H3-Image

[English](README.md) | [简体中文](README_zh.md)

Supports text-to-image (`t2i`) and reference-image generation/editing (`i2i`) with the same MiniMax-H3-Image weights and six-step PDD schedule. I2I accepts one or multiple reference images.

## Setup

Model weights: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).

Install LightX2V and the dependencies for your device first. The Qwen3-VL vision encoder requires `transformers>=5.15.0`, including the `get_vision_interpolation_indices_and_weights` interface.

Run the commands below from the repository root. In each new terminal, set the paths and load the base environment:

```bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3-Image

cd "$lightx2v_path"
source "$lightx2v_path/scripts/base/base.sh"
```

Set `model_path` to the complete model root containing the Transformer, Qwen3-VL, VAE, tokenizer/processor, `pdd_config.json`, and `pdd_fused_heads.safetensors`.

Before using an inference script, server startup script, or the cache generation script, update its `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES`.

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

| Task | Single GPU | Four GPUs, SP4 | Four GPUs, TP2 × SP2 |
| --- | --- | --- | --- |
| Text-to-image | [run_minimax_h3_image_t2i.sh](run_minimax_h3_image_t2i.sh) | [run_minimax_h3_image_t2i_sp4.sh](run_minimax_h3_image_t2i_sp4.sh) | [run_minimax_h3_image_t2i_tp_sp.sh](run_minimax_h3_image_t2i_tp_sp.sh) |
| Reference-image generation/editing | [run_minimax_h3_image_i2i.sh](run_minimax_h3_image_i2i.sh) | [run_minimax_h3_image_i2i_sp4.sh](run_minimax_h3_image_i2i_sp4.sh) | [run_minimax_h3_image_i2i_tp_sp.sh](run_minimax_h3_image_i2i_tp_sp.sh) |

Choose a script for your task:

```bash
# Single-GPU text-to-image
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i.sh

# Single-GPU reference-image generation/editing
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i.sh

# Four-GPU SP4 text-to-image
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i_sp4.sh

# Four-GPU SP4 reference-image generation/editing
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_sp4.sh

# Four-GPU TP2 × SP2 text-to-image
bash scripts/minimax_h3_image/run_minimax_h3_image_t2i_tp_sp.sh

# Four-GPU TP2 × SP2 reference-image generation/editing
bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_tp_sp.sh
```

All six scripts currently generate `4096 × 4096` PNG files in the repository's `save_results/` directory, with filenames corresponding to the task and parallel mode. The default I2I reference image is `$model_path/examples/reference1.png`.

Both four-GPU I2I scripts accept an `IMAGE_PATH` override. For multiple references, use comma-separated paths and update the script's `prompt` accordingly:

```bash
IMAGE_PATH="/path/to/reference1.png,/path/to/reference2.png" \
  bash scripts/minimax_h3_image/run_minimax_h3_image_i2i_sp4.sh
```

### Parallel configurations

The CUDA configs in [configs/minimax_h3_image](../../configs/minimax_h3_image) have the following defaults and can be used for T2I, I2I, and serving:

| Config | GPUs | DiT | Qwen3-VL | VAE |
| --- | --- | --- | --- | --- |
| [minimax_h3_image.json](../../configs/minimax_h3_image/minimax_h3_image.json) | 1 | Single GPU | Single GPU | Single GPU |
| [minimax_h3_image_sp.json](../../configs/minimax_h3_image/minimax_h3_image_sp.json) | 2 | SP2 (Ulysses) | Rank 0 | Rank 0 |
| [minimax_h3_image_sp4.json](../../configs/minimax_h3_image/minimax_h3_image_sp4.json) | 4 | SP4 (Ulysses) | TP4 | Spatial tile parallelism for both encoding and decoding |
| [minimax_h3_image_tp_sp.json](../../configs/minimax_h3_image/minimax_h3_image_tp_sp.json) | 4 | TP2 × SP2 (Ulysses) | TP4 | Spatial tile parallelism for both encoding and decoding |

Both four-GPU configs enable `text_encoder_tensor_parallel`, `vae_encode_parallel`, and `vae_decode_parallel` by default. SP4 uses sequence parallelism without DiT tensor parallelism; both four-GPU configs disable CPU offload and enable DiT and VAE compilation. The VAE encoder distributes tiles from one or multiple reference images across ranks, then uses all-gather so every rank receives the complete encoding results. The decoder computes tiles across ranks, and rank 0 collects, stitches, and outputs the final image. T2I has no reference images and skips VAE encoding.

When changing configurations, also adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` so the process count matches the DiT TP × SP size. Qwen3-VL TP uses the entire inference process group, giving TP4 in both four-GPU configs.

## Serving and POST requests

### Start the server

Prepare the AdaLN cache, then update `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the selected startup script. All three scripts support T2I and I2I on port 8000.

Single GPU ([start_server.sh](server/start_server.sh)):

```bash
bash scripts/minimax_h3_image/server/start_server.sh
```

Four GPUs, SP4 ([start_server_sp4.sh](server/start_server_sp4.sh)):

```bash
bash scripts/minimax_h3_image/server/start_server_sp4.sh
```

Four GPUs, TP2 × SP2 ([start_server_parallel.sh](server/start_server_parallel.sh)):

```bash
bash scripts/minimax_h3_image/server/start_server_parallel.sh
```

### POST clients

Wait for model loading, compilation, and warmup to finish, then run the client for your task in another terminal. Update the `url` in [post_t2i.py](server/post_t2i.py) or [post_i2i.py](server/post_i2i.py) for your server. For I2I, also set `image_path` to a client-local reference image; the client sends it as Base64.

```bash
python scripts/minimax_h3_image/server/post_t2i.py
python scripts/minimax_h3_image/server/post_i2i.py
```

The clients return a task ID and request `2048 × 2048` PNG output by default. Completed images are saved on the server under `$lightx2v_path/save_results/server_cache/outputs/`. Edit `prompt`, `size`, `seed`, and `save_result_path` in the clients as needed.
