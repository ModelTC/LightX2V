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

> **Cache reuse**: regenerate when changing the base variant/checkpoint, timestep grid, or any LoRA affecting AdaLN/final norm (including strength, alpha, mode or `LORA_DTYPE`). Export and inference must share these settings. LoRAs without modulation updates can share the base cache.

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
| `distill/4steps/minimax_h3_bf16_4step.json` | Single-GPU BF16, 4-step LoRA |
| `distill/4steps/minimax_h3_fp8_4step.json` | SP4, FP8 + 4-step LoRA |
| `distill/4steps/minimax_h3_ref2av_4step.json` | SP8, reference-task 4-step LoRA |

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
`configs/minimax_h3/distill/4steps/minimax_h3_bf16_4step.json`. After installation, set
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

Set `lora_mode` in the inference JSON (`merge` by default). All `distill/` configs select `consumer`:

```json
{
  "lora_mode": "consumer"
}
```

Precision remains an environment setting for **both cache export and inference**:

```bash
export LORA_DTYPE=BF16  # BF16, FP16, FP32; unset follows DTYPE
```

| DiT weights | `lora_mode` | Behavior |
| --- | --- | --- |
| Original BF16 | `merge` (default), `separate`, `consumer` | Merge at load time, compute independent branches, or fuse branch addition into H3's activation/residual consumers. |
| Quantized FP8 or INT8, including ConvRot | `separate`, `consumer` | Retain the existing quantized GEMM and a floating-point LoRA branch; merging into quantized DiT weights is unsupported. |

`consumer` is implemented only for `minimax_h3` on CUDA. It supports eager and
`use_compile: true`, but not tensor parallelism or GEMM calibration; sequence parallelism
and model/block offload remain available. Unsupported combinations raise an
error rather than switching modes. The fused kernels retain intermediate dtype
rounding, including BF16 projections with FP32 gates/residuals. Numerical parity
and speed must be checked on the chosen backend; no universal speedup is implied.
Compilation can change rounding elsewhere in the block, so eager/compiled outputs
are not guaranteed to be bitwise equal. Quantized compilation also requires the
selected GEMM/attention backend to support tracing; consumer does not replace it.

`separate` and `consumer` retain request-time adapter switching. This is distinct
from the execution mode: both leave base weights unchanged.

Set `export LORA_DTYPE=FP32` (or `BF16` / `FP16`) before launching. Unset means
follow `DTYPE`, not `SENSITIVE_LAYER_DTYPE`. This controls LoRA factors and
computation in all three modes. Merge restores the stored base weight dtype;
separate/consumer restore the base projection output dtype **after** adding the
LoRA result. Base precision and quantized GEMMs are unchanged. FP32 merge still
has final weight rounding and is not equivalent to separate/consumer.
The old JSON `lora_merge_force_fp32` is removed; no JSON `lora_dtype` is needed.

Base-task 4-step v1.0 LoRA example: add or update these fields in `minimax_h3.json`, replacing `path` with your local LoRA file:

```json
{
  "lora_mode": "consumer",
  "infer_steps": 4,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ]
}
```

Native (non-ComfyUI) H3 Turbo LoRAs use per-layer alpha tensors first, file
metadata alpha second, and `lora_configs[].alpha` only if neither is embedded.
The loader logs each alpha's source, value and layer count. Rank comes from the
factor shapes; AdaLN and `norm_out.linear` need no special configuration.
All BF16, FP8 and INT8 ConvRot configs under `configs/minimax_h3/distill/8steps/`
use `minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors`: eight Euler steps,
shifts 12/3 and strength=1. Replace the LoRA path with your local file. Alpha is
read per layer from the checkpoint, without a global JSON override.
These configs set `lora_mode: "consumer"`; `separate` is also supported. Quantized bases do not support merge.
The adapter remains BF16; FP8/INT8 refers only to the base weights. Before the
first run, export the matching AdaLN cache as below, using the same config and
`lora_mode` config and `LORA_DTYPE` environment for export and inference.
The earliest FL2V v0.1 has no embedded alpha and requires an explicit configuration.
The repository's FP8 full transformer checkpoint is not a LoRA.

### Scheduler and LoRA-aware AdaLN cache

`scheduler_type` selects `euler`, `lcm`, or the base model's default
`reference_blend`. No legacy field aliases are accepted. LCM uses a clean prediction
followed by re-noising, matching Pika H3's sampler; it is not Euler with fewer steps.
For a matching 6-step LCM adapter, set:

```json
{
  "scheduler_type": "lcm",
  "infer_steps": 6,
  "distilled_sigma_values": [1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0],
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0
}
```

These are **unshifted** sigma knots, including both endpoints. There must be
`infer_steps + 1` values strictly descending from 1 to 0. Each modality applies
its own shift once. Do not change an Euler-trained adapter to LCM arbitrarily.

Generate the cache with the **same inference JSON (including `lora_mode`) and `LORA_DTYPE` environment**, including LoRA and scheduler:

```bash
PYTHONPATH=. python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path /path/to/MiniMax-H3 --model-variant fl2av \
  --config_json configs/minimax_h3/distill/8steps/minimax_h3_bf16_8step.json
```

Use `--model-variant ref2av` for the reference transformer. Cache identity includes
the base checkpoint, timestep grid, affected LoRA projections, strength, and
merge arithmetic. `separate`/`consumer` share modulation arithmetic and caches;
`merge` has its own. Existing H3 caches need regeneration once. Cached modulation
LoRAs are not applied a second time online. Before switching/removing a LoRA,
export its corresponding cache; a missing cache fails before changing the active
adapter. LoRAs without modulation updates can share the base cache.

### Pika-H3-Flex six-step LoRAs

The [official implementation](https://github.com/Pika-Labs/Pika-H3-Flex) releases separate
FL2VA and Ref2VA adapters: rank/alpha=128/128, strength=1, six-step LCM with shifts 12/3.
The files have no embedded alpha, so these configs explicitly set `alpha: 128`.
No checkpoint conversion is needed.

- [FL2AV config](../../configs/minimax_h3/distill/6steps/minimax_h3_pika_fl2av_6step.json): `transformer/`, for T2AV/I2AV/L2AV/FL2AV.
- [Ref2AV config](../../configs/minimax_h3/distill/6steps/minimax_h3_pika_ref2av_6step.json): `transformer_ref/`; use the matching adapter.

Download the adapters and update `lora_configs[0].path` in each config. Use the MiniMax-H3 base:

```bash
hf download Pika-Labs/Pika-H3-Flex fl2va/adapter.safetensors ref2va/adapter.safetensors --local-dir /path/to/pika-h3-flex

export DTYPE=BF16
export LORA_DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export PYTHONPATH=.
model_path=/path/to/MiniMax-H3
config_path=configs/minimax_h3/distill/6steps/minimax_h3_pika_fl2av_6step.json

python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$model_path" --model-variant fl2av --config_json "$config_path"
python -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant fl2av --task t2av \
  --model_path "$model_path" --config_json "$config_path" \
  --prompt 'integrated_multimodal_description: A fox walks through a snowy forest. overall_soundscape: Wind and crunching snow. non_diegetic_music: N/A' \
  --seed 42 --save_result_path output_pika_t2av.mp4
```

Change `--task` and add `--image_path` for I2AV, `--last_frame_path` for L2AV,
or both for FL2AV. For Ref2AV, use its own config and cache:

```bash
config_path=configs/minimax_h3/distill/6steps/minimax_h3_pika_ref2av_6step.json
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$model_path" --model-variant ref2av --config_json "$config_path"
python -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant ref2av --task ref2av \
  --model_path "$model_path" --config_json "$config_path" \
  --image_path /path/to/reference.png \
  --prompt 'Animate the subject in <Picture 1>, preserving its appearance and setting. Natural motion and ambient sound, no music.' \
  --seed 42 --save_result_path output_pika_ref2av.mp4
```

Examples use 768×1344, 124 frames, single-GPU block offload and unquantized weights.
Set `LORA_DTYPE=BF16` to match the official adapter compute dtype. `lora_mode: "separate"`
is also supported; merging introduces different weight rounding.
The Ref config uses `reference_image_resize_mode: "match"`: cap reference image
area to the output canvas while preserving aspect ratio, without upscaling.
This is our selected preprocessing policy, not the official 768p tier's reference
short edge of 2048. Attention/kernel differences may also change numerical results.

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
