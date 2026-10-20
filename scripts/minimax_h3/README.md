# MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

Before running, set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in the selected script.

## Weights and tasks

Download Diffusers-format weights from the official Hugging Face repository [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and set `model_path` to the local model directory.

| Task | Request inputs | Transformer |
| --- | --- | --- |
| `t2av` | `prompt` | `transformer` |
| `i2av` | `prompt`, `image_path` (first frame) | `transformer` |
| `l2av` | `prompt`, `last_frame_path` | `transformer` |
| `fl2av` | `prompt`, `image_path`, `last_frame_path` | `transformer` |
| `ref2av` | `prompt`, reference images and/or videos, optional reference audio | `transformer_ref` |

Set `--model-variant` at startup: `fl2av` supports the first four tasks in the table; `ref2av` supports reference generation. Select the request's `task`; switching base tasks requires no reload. Both variants can share an inference JSON, but LoRA and quantized weights must match the variant.

## AdaLN cache

The AdaLN cache stores reusable computation results to avoid repeating work during inference. **Example configs already enable caching. Generate the cache before your first inference run or server startup; skip this step if a matching cache exists.**

1. **Check the cache settings.** Keep these fields in your inference JSON (already included in the example configs):

   ```json
   {
     "use_adaln_cache": true,
     "adaln_cache_dir": "~/.cache/lightx2v/adaln"
   }
   ```

2. **Edit the [cache generation script](../../tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh).** Set the following parameters:

   | Parameter | Setting |
   | --- | --- |
   | `lightx2v_path`, `model_path` | Use the same repository and model paths as inference. |
   | `CUDA_VISIBLE_DEVICES` | Select a GPU, e.g. `0`; cache generation only needs one GPU. |
   | `--config_json` | Use the inference JSON. |
   | `--model-variant` | Use `fl2av` for `t2av`, `i2av`, `l2av`, and `fl2av`; use `ref2av` for reference generation. |

3. **Generate the cache.** Run from the repository root:

   ```bash
   bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
   ```

   The log message `AdaLN cache saved to` confirms completion. You can then run offline inference or start a server as described below. Files are stored in a subdirectory of `adaln_cache_dir` and loaded automatically during inference; no cache file path is needed.

> **Cache reuse**: regenerate the cache when changing `--model-variant`, `infer_steps`, `video_flow_shift`, or `audio_flow_shift`; other changes can reuse the existing cache. For example, generate separate caches for the 29-step base config and a four-step LoRA config.

## Offline inference

The five task scripts share `configs/minimax_h3/minimax_h3.json`: one GPU, BF16 weights, and 124 frames at `[height, width] = [544, 960]`.

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
bash scripts/minimax_h3/run_minimax_h3_i2av.sh
bash scripts/minimax_h3/run_minimax_h3_l2av.sh
bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

The ordinary config uses SageAttention2 and SGL kernels. Select a config whose attention, quantization, and parallel settings match your installed kernels and devices.

Select a config through `--config_json`; the table lists representative configs under `configs/minimax_h3/`. For one GPU, use the matching task script; for SP, TP, or mixed parallelism, use `run_minimax_h3_t2av_parallel.sh` as a reference and adjust the variant, task, and inputs for other tasks.

| Configuration under `configs/minimax_h3/` | Mode |
| --- | --- |
| `minimax_h3.json` | Single-GPU BF16 |
| `minimax_h3_compile.json` | Compile with startup warmup |
| `minimax_h3_block_offload.json` | Single-GPU BF16 block offload |
| `minimax_h3_sp.json` | SP4 |
| `minimax_h3_tp.json` | TP2 |
| `minimax_h3_tp_sp.json` | TP2 × SP2 |
| `fp8/minimax_h3.json` | Single-GPU DiT FP8 |
| `dmd/minimax_h3_bf16_4step.json` | Single-GPU BF16, 4-step LoRA |
| `dmd/minimax_h3_fp8_4step.json` | SP4, FP8 + 4-step LoRA |
| `dmd/minimax_h3_ref2av_4step.json` | SP8, reference-task 4-step LoRA |

The multi-GPU script defaults to SP4. When switching configs, adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` to match the configured parallel size.

Use `minimax_h3_compile.json` for compilation, enabling `use_compile` and `warmup` and setting `rope_type: "torch_real_rope"`. Service launchers can use it too.

Add `video_codec_options` to the inference JSON to configure MP4 encoding, for example:

```json
{
  "video_codec_options": {
    "preset": "ultrafast",
    "crf": "18"
  }
}
```

## Python API

The [four-step DMD T2AV example](inference/distill/minimax_h3_t2av_dmd.py) uses
`configs/minimax_h3/dmd/minimax_h3_bf16_4step.json`. After installation, set
`MODEL_PATH` and `OUTPUT_PATH` in the script and the local LoRA path in that JSON,
then run from the repository root:

```bash
python scripts/minimax_h3/inference/distill/minimax_h3_t2av_dmd.py
```

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md)
for generator setup and configuration precedence.

## Low-cost candidate generation

For low-resolution drafts, candidate selection, and higher-resolution regeneration, see the [MiniMax-H3 Roll documentation](../minimax_h3_roll/README.md).

## LoRA

| DiT weights | `lora_dynamic_apply` | Behavior |
| --- | --- | --- |
| Original BF16 | `false` or `true` | `false` merges the adapter when loading weights; `true` applies it during inference. |
| Quantized FP8 or INT8, including ConvRot | `true` | Apply the adapter during inference; merging into quantized DiT weights is unsupported. |

Base-task 4-step v1.0 LoRA example: add or update these fields in `minimax_h3.json`, replacing `path` with your local LoRA file:

```json
{
  "infer_steps": 4,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "h3_step_update": "training_euler",
  "lora_dynamic_apply": true,
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ]
}
```

## Server and POST

The base service starts with `--model-variant fl2av` and no startup `--task`. Select a launcher by GPU count, then submit requests from another terminal after the model loads.

Single GPU:

```bash
bash scripts/minimax_h3/server/start_server.sh
```

Four-GPU SP4 uses `minimax_h3_sp.json`:

```bash
bash scripts/minimax_h3/server/start_server_parallel.sh
```

Both services use the following POST clients:

```bash
python scripts/minimax_h3/server/post_t2av.py
python scripts/minimax_h3/server/post_i2av.py
python scripts/minimax_h3/server/post_l2av.py
python scripts/minimax_h3/server/post_fl2av.py
```

Each base request must specify `task`; see the task table for inputs. The image examples encode client-local files as Base64; server-accessible absolute paths also work.

The reference service uses `--model-variant ref2av`:

```bash
bash scripts/minimax_h3/server/start_server_ref2av.sh
```

Set the reference image path in the client, then send a request from another terminal:

```bash
python scripts/minimax_h3/server/post_ref2av.py
```

## Request and startup settings

- **Startup and requests**: set model paths, variant, weights/LoRA, kernels, offload, parallelism, compile, and warmup at startup. Pass `prompt`, media paths, `seed`, and the output path through Python, CLI, or POST.
- **Output settings**: JSON provides defaults for `num_frames` and `size`; requests can override them. `size` is `[height, width]`, with both dimensions divisible by 32. Frame counts round up to `17*n+5` and must fall within 124–362 after alignment, e.g. 125 → 141. Output is MP4 with 24 FPS video and 32 kHz stereo audio.
- **Random seed**: an omitted or `null` `seed` uses 42. You can specify any non-negative integer, including 0.
- **Saving results**: CLI skips saving when no output path is supplied. For service requests, prefer a relative `save_result_path` such as `./minimax_h3_t2av.mp4`, resolved under the service output directory. Omitted or `null` paths skip saving; saved results can be downloaded.
