# MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

MiniMax-H3 generates video with synchronized stereo audio. Run the commands below from the LightX2V repository root. Set `lightx2v_path` and `model_path` in the selected shell script before running it.

## Weights and tasks

Use the released Diffusers component layout under `model_path`:

```text
MiniMax-H3/
├── transformer/       # t2av, i2av, l2av, fl2av
├── transformer_ref/   # ref2av
├── text_encoder/
├── tokenizer/
├── processor/
├── vae/
└── audio_vae/
```

Each transformer directory needs its `config.json`, weight index, and checkpoint shards. Download the components for the task family you use; downloading only the original `FL2VA/` or `Ref2VA/` checkpoint directories does not provide this layout. FP8/INT8 presets still use the original component configs, tokenizer, and VAEs, and specify additional local quantized checkpoints in JSON.

| Task | Request inputs | Transformer |
| --- | --- | --- |
| `t2av` | `prompt` | `transformer` |
| `i2av` | `prompt`, `image_path` (first frame) | `transformer` |
| `l2av` | `prompt`, `last_frame_path` | `transformer` |
| `fl2av` | `prompt`, `image_path`, `last_frame_path` | `transformer` |
| `ref2av` | `prompt`, reference images and/or videos, optional reference audio | `transformer_ref` |

The base transformer can serve all four base tasks without reloading. Reference generation uses a separate transformer and service. These checkpoints are CFG-distilled: do not send `negative_prompt`, including an empty string.

## Offline inference

The five task scripts share `configs/minimax_h3/minimax_h3.json`: one GPU, BF16 weights, model CPU offload, and 124 frames at `[height, width] = [544, 960]`. The script's `--task` selects the transformer and input handling; the JSON filename does not select a task.

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
bash scripts/minimax_h3/run_minimax_h3_i2av.sh
bash scripts/minimax_h3/run_minimax_h3_l2av.sh
bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

The ordinary config uses SageAttention2 and SGL kernels. Select a config whose attention, quantization, and parallel settings match your installed kernels and devices. Paths, prompt, seed, and output path are visible in each script; each JSON remains a complete startup config.

To select another mode, change `--config_json` in the listed script to the corresponding file under `configs/minimax_h3/`. The single-GPU script also launches FP8, compile, and four-step LoRA modes; `run_minimax_h3_t2av_parallel.sh` provides a shared launcher for SP, TP, and mixed parallel configs. The table lists each preset's configured parallel sizes. In filenames, `encoder` refers to the text encoder and `vae` to the video VAE.

| Configuration under `configs/minimax_h3/` | Launch script | Mode |
| --- | --- | --- |
| `minimax_h3.json` | Any of the five task scripts above | Single-GPU BF16 |
| `minimax_h3_compile.json` | Any of the five task scripts above | Compile with startup warmup |
| `minimax_h3_block_offload.json` | `run_minimax_h3_t2av.sh` | Single-GPU BF16 block offload |
| `minimax_h3_sp.json` | `run_minimax_h3_t2av_parallel.sh` | SP4 |
| `minimax_h3_tp.json` | `run_minimax_h3_t2av_parallel.sh` | TP2 |
| `minimax_h3_tp_sp.json` | `run_minimax_h3_t2av_parallel.sh` | TP2 × SP2 |
| `minimax_h3_sol_block_offload.json` | `run_minimax_h3_t2av.sh` | Single-GPU Sol-Attn |
| `fp8/minimax_h3.json` | `run_minimax_h3_t2av.sh` | Single-GPU DiT FP8 |
| `fp8/minimax_h3_encoder_fp8.json` | `run_minimax_h3_t2av.sh` | Single-GPU DiT + text-encoder FP8 |
| `fp8/minimax_h3_vae_fp8.json` | `run_minimax_h3_t2av.sh` | Single-GPU DiT + video-VAE FP8 |
| `fp8/minimax_h3_sp_5090.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, 5090 FP8 config |
| `dmd/minimax_h3_bf16_4step.json` | `run_minimax_h3_t2av.sh` | Single-GPU BF16, 4-step LoRA |
| `dmd/minimax_h3_bf16_4step_sol.json` | `run_minimax_h3_t2av.sh` | Single-GPU 4-step Sol-Attn |
| `dmd/minimax_h3_fp8_4step.json` | `run_minimax_h3_t2av_parallel.sh` | SP4, FP8 + 4-step LoRA |
| `dmd/minimax_h3_int8_4step.json` | `run_minimax_h3_t2av_parallel.sh` | SP4, INT8 + 4-step LoRA |
| `dmd/minimax_h3_fp8_8step.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, FP8 + 8-step LoRA |
| `dmd/minimax_h3_int8_convrot_8step.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, INT8 ConvRot + 8-step LoRA |
| `dmd/minimax_h3_fp8_4step_5090.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, 5090, 4-step LoRA |
| `dmd/minimax_h3_fp8_4step_5090_vae_fp8.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, FP8 VAE + 4-step LoRA |
| `dmd/minimax_h3_fp8_4step_5090_vae_fp8_sla.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, FP8 VAE + matching SLA LoRA |
| `dmd/minimax_h3_fp8_4step_5090_vae_fp8_sol.json` | `run_minimax_h3_t2av_parallel.sh` | SP8, FP8 text encoder/VAE + Sol-Attn + 4-step LoRA |
| `dmd/minimax_h3_ref2av_4step.json` | `run_minimax_h3_ref2av.sh`, with 8 processes as described below | Reference-task 4-step LoRA |

`run_minimax_h3_t2av_parallel.sh` defaults to SP4. To select another parallel preset, change `--config_json` and keep `torchrun --nproc_per_node` and `CUDA_VISIBLE_DEVICES` consistent with the JSON. The process count is `tensor_p_size × seq_p_size` for these presets; an omitted parallel size is 1.

| Parallel mode in the selected JSON | `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` |
| --- | --- | --- |
| TP2 | `0,1` | `2` |
| SP4 or TP2 × SP2 | `0,1,2,3` | `4` |
| SP8, including the 5090 presets | `0,1,2,3,4,5,6,7` | `8` |

For the 8-GPU reference LoRA config, set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` in `run_minimax_h3_ref2av.sh` and replace `python -m lightx2v.infer` with `torchrun --standalone --nproc_per_node=8 -m lightx2v.infer`. Keep its `--task ref2av` and reference input arguments.

Both single-GPU Sol configs default to 362 frames at `[768, 1344]`. Select the corresponding Sol JSON through `--config_json` in `run_minimax_h3_t2av.sh` to use these output defaults.

All MiniMax-H3 configs that enable compilation also set `warmup: true`. The ordinary and compile configs share the same output defaults and can be used by both service launch scripts as well.

### Video encoding

To reproduce the 362-frame output encoder example, update these fields in a copy of `minimax_h3.json` and point the launch script at that complete JSON:

```json
{
  "target_video_length": 362,
  "video_codec_options": {
    "preset": "ultrafast",
    "crf": "18"
  }
}
```

`video_codec_options` is a startup setting consumed by the existing MP4 encoder. Other dimensions and frame counts remain request overrides; no separate config is needed for each output shape.

## LoRA and Python

Replace `/path/to/...` in every selected quantized checkpoint and LoRA field with an existing local file. The loader does not download LoRAs from a repository ID. Use a LoRA for the correct task family and keep `alpha` consistent with that checkpoint's training configuration; do not interchange base, reference, SLA, or differently versioned LoRAs merely because their shapes match.

The named base 4-step v1.0 LoRA uses `alpha=128`, `video_flow_shift=6`, and `audio_flow_shift=3`. Other presets retain their own alpha and shift settings.

Choose `lora_dynamic_apply` in the selected JSON according to the DiT weights:

| DiT weights | Supported setting | Behavior |
| --- | --- | --- |
| Original BF16 | `false` or `true` | `false` merges the adapter when loading weights; `true` applies it during inference. |
| Quantized FP8 or INT8, including ConvRot | `true` | Apply the adapter during inference; merging into quantized DiT weights is unsupported. |

Quantizing only the encoder or VAE does not impose this restriction. Dynamic LoRA currently accepts one adapter and requires its `alpha` to be explicitly configured. The existing model loader rejects unsupported combinations.

The shared `dmd/minimax_h3_bf16_4step.json` defaults to `lora_dynamic_apply: true`, with 362 frames at `[768, 1344]`. Set the same field to `false` to use merging. Select this JSON in `run_minimax_h3_t2av.sh`; for a shorter, smaller output, add `--target_shape 544 960` and `--num_frames 124` to its inference command.

Set `infer_steps` in JSON to the number of model evaluations: `4` for 4-step inference and `8` for 8-step inference. The scheduler includes the terminal zero automatically. When migrating an older MiniMax-H3 config, subtract one from its `infer_steps` value (`5` → `4`, `9` → `8`, `30` → `29`) to preserve the original sampling schedule. The bundled configs already use this convention.

For Python, set `MODEL_PATH` in [the example](../../examples/minimax_h3/minimax_h3_t2av_dmd.py) and the local LoRA path in its selected JSON, then run:

```bash
python examples/minimax_h3/minimax_h3_t2av_dmd.py
```

## Server and POST

Start the base service, then submit a request from another terminal:

```bash
bash scripts/minimax_h3/server/start_server.sh
```

```bash
python scripts/minimax_h3/server/post_t2av.py
python scripts/minimax_h3/server/post_i2av.py
python scripts/minimax_h3/server/post_l2av.py
python scripts/minimax_h3/server/post_fl2av.py
```

Each base request must specify `task`, because the loaded transformer supports four tasks. The image examples encode client-local files as Base64.

For reference generation, use the separate reference service instead:

```bash
bash scripts/minimax_h3/server/start_server_ref2av.sh
```

```bash
python scripts/minimax_h3/server/post_ref2av.py
```

Both launch scripts default to port 8000. To run both services simultaneously, choose separate GPUs, `--port`, and `--metric_port` values, and update the POST URLs. The reference service accepts only `ref2av`, so `task` may be omitted there; the example includes it for clarity.

Reference requests can supply `image_path`, `video_path`, and `audio_path` together. For multiple files of one kind, use comma-separated server-local paths, for example:

```json
{
  "task": "ref2av",
  "prompt": "Generate an audio-video scene following the references.",
  "image_path": "/path/to/character.jpg,/path/to/scene.jpg",
  "video_path": "/path/to/motion.mp4",
  "audio_path": "/path/to/voice.wav",
  "seed": 42,
  "save_result_path": "./minimax_h3_references.mp4"
}
```

Audio must be accompanied by an image or video. The runner accepts at most 9 images, 3 videos, and 12 references in total; the audio-reference limit is 3, including video soundtracks. Single image inputs also accept Base64 and HTTP(S) URLs; input videos use server-local paths. The default reference-image preprocessing follows the released Diffusers sizing; `reference_image_resize_mode: "match"` in JSON is an explicit alternative that limits reference-image area to the output canvas.

POST returns a task ID before inference completes. Query status, then download the completed result:

```bash
curl http://localhost:8000/v1/tasks/TASK_ID/status
curl --fail http://localhost:8000/v1/tasks/TASK_ID/result -o minimax_h3.mp4
```

Use a relative path such as `./minimax_h3_t2av.mp4` for `save_result_path`; the service resolves it relative to its output directory and can return it through the download endpoint. Omitting the path or sending `null` skips saving; that request has no downloadable result.

## Request and startup settings

- Startup: model paths, default task, weights/LoRA, kernels, offload, parallelism, compile, and warmup. Prompt, media paths, seed, and output path belong in the Python call, CLI command, or POST body.
- Output defaults: JSON sets `target_video_length` and `target_height`/`target_width`. Requests can override them with `num_frames` and `target_shape` (`[height, width]`). Dimensions must be multiples of 32. Frame counts align upward to `17*n+5`, with supported aligned counts from 124 to 362; for example, 125 becomes 141.
- Seed: omitted or `null` defaults to 42; an explicit non-negative value, including 0, is used as supplied.
- Saving: every CLI example explicitly provides an output path. Removing it skips file saving. The service follows the same rule. Output is MP4 with 24 FPS video and 32 kHz stereo audio.
