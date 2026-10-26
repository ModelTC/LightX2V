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

The default outputs are `save_results/minimax_h3_roll_t2av/draft.mp4` and `resample.mp4`. The draft script explicitly sets `--draft_steps 3` to generate a K3 draft; remove this argument to use the full configured 8-step schedule. Regeneration starts at `resample_start_step=1` (currently the remaining 7 steps). Both stages use the same prompt, seed, and frame count. Each script starts its own `lightx2v.infer` process and loads the model separately. To reuse a loaded model, use the Python example below or an HTTP server.

Edit inputs and generation parameters directly in the scripts:

| Script variable or argument | Example value | Purpose |
| --- | --- | --- |
| `lightx2v_path` / `model_path` | `/path/to/LightX2V` / `/path/to/MiniMax-H3` | Repository and base-model directories; replace with local paths. |
| `--config_json` | `configs/minimax_h3_roll/minimax_h3_bf16_8step.json`, relative to the repository root | Startup config, including LoRA, shifts, and inference-step count. |
| `CUDA_VISIBLE_DEVICES` / `--nproc_per_node` | `0,1,2,3,4,5,6,7` / `8` | GPUs and process count; must match the config's parallel size. |
| `--prompt` / `--seed` / `--num_frames` | Yellow tram in rain / `42` / `124` | Prompt, seed, and frame count shared by both stages. |
| `--draft_steps` | `3` | Generate a 3-step draft; remove this argument to use the full configured schedule. |
| `--size` | Draft: `480 864`; regeneration: `768 1344` | `[height, width]`; use `1088 1920` for a 1080p delivery canvas. |
| `--resample_start_step` | `1` | Regeneration's starting index in the original schedule. |
| `--resample_source_video` | `save_results/minimax_h3_roll_t2av/draft.mp4`, relative to the repository root | Selected draft with an audio track. |
| `--save_result_path` | `save_results/minimax_h3_roll_t2av/{draft,resample}.mp4`, relative to the repository root | Output file. Reusing the path overwrites the corresponding output. |
| `DTYPE` / `SENSITIVE_LAYER_DTYPE` | `BF16` / `BF16` | Inference precision, matching the existing H3 offline scripts. |

The draft script already sets `--draft_steps 3`. Generate a K3 draft and inspect its output before running the resample script:

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_draft.sh
# Inspect save_results/minimax_h3_roll_t2av/draft.mp4, then regenerate the selected draft.
bash scripts/minimax_h3_roll/run_minimax_h3_t2av_resample.sh
```

Change `--save_result_path` for each seed to retain candidates, then point the resample script's `--resample_source_video` to the selected draft. For roughly 15 seconds, change `--num_frames` to `362` in both scripts. To regenerate an existing video, edit `--resample_source_video` and set the matching frame count and desired prompt.

## FL2AV launchers

FL2AV takes both a first-frame image and a last-frame image. The two independent scripts use the same eight-GPU SP8, BF16/BF16, and 8-step LoRA config as above:

- [run_minimax_h3_fl2av_draft.sh](run_minimax_h3_fl2av_draft.sh): generate a 480p K3 draft conditioned on the first and last frames.
- [run_minimax_h3_fl2av_resample.sh](run_minimax_h3_fl2av_resample.sh): regenerate a selected draft at 768p while retaining both frame conditions.

Before running, edit `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in both scripts, set the local LoRA path in `configs/minimax_h3_roll/minimax_h3_bf16_8step.json`, and prepare a matching AdaLN cache.

The example uses bird images already included in the repository; no additional image download is needed:

| First frame: `--image_path` | Last frame: `--last_frame_path` |
| --- | --- |
| ![Standing bird](../../assets/inputs/imgs/flf2v_input_first_frame-fs8.png) | ![Flying bird](../../assets/inputs/imgs/flf2v_input_last_frame-fs8.png) |

Both scripts use the same prompt, describing the bird taking off with chirping, wingbeats, and wind:

```text
A fluffy blue bird stands on a wooden ledge, looks up, spreads its wings and takes off into a bright blue sky. The camera follows the bird upward in one continuous shot. Sound: soft chirping, fluttering wings and a gentle breeze.
```

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_fl2av_draft.sh
# Inspect save_results/minimax_h3_roll_fl2av/draft.mp4, then regenerate the selected draft.
bash scripts/minimax_h3_roll/run_minimax_h3_fl2av_resample.sh
```

The default outputs are `save_results/minimax_h3_roll_fl2av/draft.mp4` and `resample.mp4`. Both stages pass `--task fl2av` and use the same `--image_path`, `--last_frame_path`, prompt, seed `42`, and frame count `124`. The draft uses `--size 480 864 --draft_steps 3`; regeneration uses `--size 768 1344 --resample_start_step 1` and runs the last 7 steps of the original 8-step schedule.

To use different inputs, update both frame paths and the prompt in both scripts. Point regeneration's `--resample_source_video` to the selected draft with audio and keep both frame arguments. Remove `--draft_steps 3` from the draft script to generate a full 8-step draft. For roughly 15 seconds, change `--num_frames` to `362` in both scripts. Each script loads the model separately.

## Ref2AV launchers

Ref2AV uses reference media to guide generation. The two independent scripts default to eight-GPU SP8 and BF16/BF16, using the base model's `transformer_ref/` weights and a separate [Ref2AV 8-step config](../../configs/minimax_h3_roll/minimax_h3_ref2av_bf16_8step.json):

- [run_minimax_h3_ref2av_draft.sh](run_minimax_h3_ref2av_draft.sh): generate a 480p K3 draft from a reference image.
- [run_minimax_h3_ref2av_resample.sh](run_minimax_h3_ref2av_resample.sh): regenerate a selected draft at 768p while retaining the reference image.

Before running, edit `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` in both scripts. The base-model directory must include `transformer_ref/`; set the config's LoRA path to your local [minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors). This example uses video/audio shifts `12/3`, `alpha=8`, and `reference_image_resize_mode="match"`.

Before the first run, prepare a matching AdaLN cache with the Ref2AV config and `--model-variant ref2av`. Skip this step if a matching cache already exists:

```bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_ref2av_bf16_8step.json" \
  --model-variant ref2av
```

The example uses a bird image already included in the repository as an appearance reference. It does not lock the first or last frame, and no additional media download is needed:

![Bird reference](../../assets/inputs/imgs/flf2v_input_first_frame-fs8.png)

Both scripts use the same prompt, describing the reference bird hopping along a garden fence, tilting its head, and flapping its wings:

```text
The fluffy blue bird from the reference image hops along a wooden garden fence, tilts its head and flaps its small wings. Keep its round body, blue feathers and expressive eyes consistent. A gentle camera pan follows the bird in warm morning sunlight. Sound: cheerful chirping, soft wingbeats and a light breeze.
```

```bash
bash scripts/minimax_h3_roll/run_minimax_h3_ref2av_draft.sh
# Inspect save_results/minimax_h3_roll_ref2av/draft.mp4, then regenerate the selected draft.
bash scripts/minimax_h3_roll/run_minimax_h3_ref2av_resample.sh
```

The default outputs are `save_results/minimax_h3_roll_ref2av/draft.mp4` and `resample.mp4`. Both stages pass `--model-variant ref2av --task ref2av` and use the same `--image_path`, prompt, seed `42`, and frame count `124`. The draft uses `--size 480 864 --draft_steps 3`; regeneration uses `--size 768 1344 --resample_start_step 1` and runs the last 7 steps of the original 8-step schedule. Remove `--draft_steps 3` from the draft script to generate a full 8-step draft. For roughly 15 seconds, change `--num_frames` to `362` in both scripts. Each script loads the model separately.

To use different references, update `--image_path` and the prompt in both scripts. You can also add `--video_path` / `--audio_path`, with comma-separated paths for multiple files of the same type; audio references require an image or video reference as well. Retain the original references during regeneration and pass the selected draft with audio as `--resample_source_video`. The `--video_path` argument supplies a reference video and does not replace that parameter.

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

## Server: sequential draft and resample

The startup follows the [H3 server documentation](../minimax_h3/README.md#server-and-post). The example directly uses the committed `lightx2v.server` entry point, Roll config, and images, with the complete client code inline below. The service loads the model once and reuses it for subsequent draft and regeneration requests.

Set the repository and base-model paths, configure the local LoRA path, and prepare a matching AdaLN cache before starting the eight-GPU service. Use `--model-variant fl2av` at startup; each request selects T2AV or FL2AV through `task`:

```bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc_per_node=8 -m lightx2v.server \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_bf16_8step.json" \
  --host 0.0.0.0 \
  --port 8000
```

For four GPUs, select four devices in `CUDA_VISIBLE_DEVICES`, set `--nproc_per_node=4`, and set `parallel.seq_p_size` to `4` in the config you use.

Once startup completes, run this client from the repository root in another terminal. It defaults to the T2AV tram example. Change `task` to `"fl2av"` to use the bird example above, uploading both frame images as Base64. Change `url` if the client and server run on different machines.

```bash
python - <<'PY'
import base64
import time
from pathlib import Path

import requests

url = "http://127.0.0.1:8000"
task = "t2av"  # Change to "fl2av" for the first/last-frame example.

def submit_and_wait(message):
    response = requests.post(f"{url}/v1/tasks/video/", json=message, timeout=30)
    response.raise_for_status()
    task_id = response.json()["task_id"]
    print(f"Submitted {task_id}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        response = requests.get(f"{url}/v1/tasks/{task_id}/status", timeout=30)
        response.raise_for_status()
        result = response.json()
        if result["status"] == "completed":
            if not result.get("save_result_path"):
                raise RuntimeError(f"Task {task_id} completed without a saved video")
            print(result, flush=True)
            return result["save_result_path"]
        if result["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Task {task_id}: {result}")
        time.sleep(1)
    raise TimeoutError(f"Task {task_id}: check its status before submitting another request")

common = {
    "task": task,
    "prompt": "A yellow tram passes along a rainy street. Sound: rainfall and a tram bell.",
    "seed": 42,
    "num_frames": 124,
}
if task == "fl2av":
    common.update(
        prompt="A fluffy blue bird stands on a wooden ledge, looks up, spreads its wings and takes off into a bright blue sky. The camera follows the bird upward in one continuous shot. Sound: soft chirping, fluttering wings and a gentle breeze.",
        image_path=base64.b64encode(Path("assets/inputs/imgs/flf2v_input_first_frame-fs8.png").read_bytes()).decode("ascii"),
        last_frame_path=base64.b64encode(Path("assets/inputs/imgs/flf2v_input_last_frame-fs8.png").read_bytes()).decode("ascii"),
    )
output_dir = f"minimax_h3_roll/{task}_{time.time_ns()}"
source_video = submit_and_wait({
    **common,
    "size": [480, 864],
    "draft_steps": 3,
    "save_result_path": f"{output_dir}/draft.mp4",
})
submit_and_wait({
    **common,
    "size": [768, 1344],
    "resample_source_video": source_video,
    "resample_start_step": 1,
    "save_result_path": f"{output_dir}/resample.mp4",
})
PY
```

The requests run sequentially: regeneration is submitted only after the draft reaches `completed`; failure or cancellation stops the client. The draft uses K3 and regeneration runs the last 7 steps. Remove `draft_steps` from the first request for a K8 draft. Both FL2AV requests retain the same first/last frames, prompt, seed, and frame count.

Relative `save_result_path` values resolve under the service output directory. Regeneration uses the server's absolute path returned when the draft completes. `resample_source_video` must point to a video with audio that the server can read. The client prints task IDs and completion results; use `GET /v1/tasks/{task_id}/result` to download either video.
