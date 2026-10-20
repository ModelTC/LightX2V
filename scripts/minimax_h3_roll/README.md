# MiniMax-H3 Roll: Low-cost candidate generation

[English](README.md) | [简体中文](README_zh.md)

Generate low-resolution drafts to select prompts and seeds, then regenerate selected drafts at a higher resolution. Set `draft_steps` to reduce the cost of each draft further. Run the commands below from the repository root.

Regeneration takes `resample_source_video`: decode its RGB frames, resize them with Lanczos, encode video and audio with their VAEs, add noise using the video/audio sigmas at `resample_start_step`, and run the remaining original sampling steps. No latent files or spatial latent interpolation are involved. RGB resizing is skipped when the source already has the target dimensions.

This workflow supports the offline CLI, Python `LightX2VPipeline.generate()`, inference JSON, and HTTP server requests. For `fl2av`, first/last-frame conditions remain independent inputs and can still be supplied during regeneration.

## T2AV launchers

The two independent scripts default to eight-GPU SP8 with the 8-step LoRA config:

- [run_minimax_h3_t2av_draft.sh](run_minimax_h3_t2av_draft.sh): generate a 480p draft.
- [run_minimax_h3_t2av_resample.sh](run_minimax_h3_t2av_resample.sh): regenerate a selected draft at 768p.

Before running, edit `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in both scripts, set the local LoRA path in the config selected by `--config_json`, and prepare a matching AdaLN cache.

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_draft.sh
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_resample.sh
```

The default outputs are `save_results/minimax_h3_roll_t2av/draft.mp4` and `resample.mp4`. Drafts use the configured full schedule by default (currently 8 steps); regeneration starts at `resample_start_step=1` (currently the remaining 7 steps). Both stages use the same prompt, seed, and frame count. Each script starts its own `lightx2v.infer` process and loads the model separately. To reuse a loaded model, use the Python example below or an HTTP server.

Edit inputs and generation parameters directly in the scripts:

| Script variable or argument | Example value | Purpose |
| --- | --- | --- |
| `lightx2v_path` / `model_path` | `/path/to/LightX2V` / `/path/to/MiniMax-H3` | Repository and base-model directories; replace with local paths. |
| `--config_json` | `configs/minimax_h3_roll/minimax_h3_bf16_8step.json`, relative to the repository root | Startup config, including LoRA, shifts, and inference-step count. |
| `CUDA_VISIBLE_DEVICES` / `--nproc_per_node` | `0,1,2,3,4,5,6,7` / `8` | GPUs and process count; must match the config's parallel size. |
| `--prompt` / `--seed` / `--num_frames` | Yellow tram in rain / `42` / `124` | Prompt, seed, and frame count shared by both stages. |
| `--draft_steps` | Omitted | Use the configured schedule; add `--draft_steps 3` to the draft command for a 3-step draft. |
| `--size` | Draft: `480 864`; regeneration: `768 1344` | `[height, width]`; use `1088 1920` for a 1080p delivery canvas. |
| `--resample_start_step` | `1` | Regeneration's starting index in the original schedule. |
| `--resample_source_video` | `save_results/minimax_h3_roll_t2av/draft.mp4`, relative to the repository root | Selected draft with an audio track. |
| `--save_result_path` | `save_results/minimax_h3_roll_t2av/{draft,resample}.mp4`, relative to the repository root | Output file. Reusing the path overwrites the corresponding output. |
| `DTYPE` / `SENSITIVE_LAYER_DTYPE` | `BF16` / `BF16` | Inference precision, matching the existing H3 offline scripts. |

For a K3 draft, add `--draft_steps 3` to the draft script's `torchrun` command. Inspect its output before running the resample script:

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_draft.sh
# Inspect save_results/minimax_h3_roll_t2av/draft.mp4, then regenerate the selected draft.
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_resample.sh
```

Change `--save_result_path` for each seed to retain candidates, then point the resample script's `--resample_source_video` to the selected draft. For roughly 15 seconds, change `--num_frames` to `362` in both scripts. To regenerate an existing video, edit `--resample_source_video` and set the matching frame count and desired prompt.

## Parameters and workflow

| Field | Default | Meaning |
| --- | --- | --- |
| `resample_source_video` | None | Draft file with audio. RGB frames are resized to the target dimensions before encoding. Without a source, use draft or ordinary generation. |
| `resample_start_step` | `1` | Integer index in `[0,infer_steps]`. `1` skips the first step, `0` runs the full regeneration schedule, and `infer_steps` performs only VAE encoding/decoding. |
| `draft_steps` | Configured `infer_steps` | An explicit positive K below `infer_steps` runs K forwards and estimates clean audio/video from the last velocity. Omitted values or K at/above `infer_steps` use the full original schedule. |

Fields can be set in the startup config or passed to `generate()`. Non-`None` request values override the config, including `resample_start_step=0`. CLI flags use `--field_name`. The workflow preset does not set `draft_steps`, so an 8-, 4-, or 29-step config defaults to its own full schedule. Early termination is enabled only for `1 <= draft_steps < infer_steps` and cannot be combined with `resample_source_video`. It also supports `ref2av`: only target audio/video rows are projected; image, video, and audio reference rows retain their conditions. Larger K values do not extend the schedule or trigger clean projection.

For Ref2AV, supply the original `image_path` / `video_path` / `audio_path` references in both stages. Generate the draft with, for example, `draft_steps=3`, then omit `draft_steps` and set `resample_source_video` for regeneration. Passing the draft only as `video_path` does not enable regeneration. Listen to early-terminated draft audio separately when judging quality.

Regeneration always uses **suffix sampling**: index the original schedule with `resample_start_step` and run its remaining steps without constructing a new schedule. Video and audio use their own sigma and timestep at the same index. Logs report the sigmas, starting index, and remaining NFE. The old `resample_strength` parameter has been removed.

```text
resample_source_video → RGB decode → Lanczos resize → Video VAE encode
                      + source audio → Audio VAE encode
                      → noise with separate video/audio sigmas
                      → remaining original steps → AV VAE decode → output
```

Video VAE encoding retains the existing seed-42 posterior sample. Audio VAE encoding uses the source soundtrack and posterior mean. A source without an audio track raises an error; silence is not substituted.

## Load once: draft and RGB regeneration

Run this Python example under eight-GPU `torchrun` with the matching config. `size` is `[height,width]`.

```python
from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="/root/lightx2v_h3_world/models/MiniMax-H3-native",
    model_cls="minimax_h3", model_variant="fl2av",
)
pipe.create_generator(config_json="configs/minimax_h3_roll/minimax_h3_bf16_8step.json")
prompt = "A red fox walks through a snowy clearing, one locked-off shot. Sound: paws on snow and soft wind."
pipe.generate(
    task="t2av", prompt=prompt, seed=42, size=[480,864], num_frames=124,
    save_result_path="/root/h3_example/draft.mp4",  # Full configured 8 steps.
)
pipe.generate(
    task="t2av", prompt=prompt, seed=2026, size=[1088,1920], num_frames=124,
    resample_source_video="/root/h3_example/draft.mp4",  # Default index 1: last 7 steps.
    save_result_path="/root/h3_example/rgb.native.mp4",
)
```

With this config, omitting `draft_steps` from the first request produces a full 8-step draft. For a shorter draft, explicitly pass a positive integer below 8, such as `draft_steps=3`. `draft_steps=8` or larger still uses the original 8-step schedule. Using `[1088,1920]` for both stages skips RGB resizing but still performs VAE re-encoding and decoding. For FL2AV, pass `task="fl2av"`, `image_path`, and `last_frame_path` to both requests and explicitly specify their sizes.
