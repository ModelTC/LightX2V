# EdenLive

[English](README.md) | [简体中文](README_zh.md)

EdenLive renders video from a proxy recording, a target first frame and a text prompt. This page covers the Wan2.2 5B autoregressive path. Its public task is `v2v`: pass the recording with `--video_path` and the first frame with `--image_path`.

## Prepare the model

Follow the LightX2V [Quick Start](../../docs/EN/source/getting_started/quickstart.md), download the [EdenLive release](https://huggingface.co/VAST-AI/EdenLive), then convert it from the repository root:

```bash
hf download VAST-AI/EdenLive --local-dir ./models/EdenLive
python tools/convert/convert_edenlive.py --source ./models/EdenLive --output ./models/EdenLive-x2v --all
```

For the RTX 5090 FP8 profile, prepare FP8-FP16 DiT/T5 weights and the INT8 ConvRot Proxy checkpoint:

```bash
EDEN_MODEL="$PWD/models/EdenLive-x2v" QUANT_DEVICE=cuda:0 bash scripts/edenlive/prepare_fp8_f16_accum_5090.sh
```

The script validates existing checkpoints before reusing them. The final profile quantizes DiT/T5 to FP8-FP16 accumulation and keeps Proxy INT8 ConvRot because Proxy FP8 caused larger generation error. It requires the SM120 FP8-FP16 operator and comfy-kitchen for the Proxy conversion.

## Run

Choose one of the two 5090 configurations:

- [FP8 final profile](../../configs/edenlive/realtime/wan22_5b_ar_fp8_f16_accum_sage_lighttae_async_5090.json)
- [BF16 comparison profile](../../configs/edenlive/realtime/wan22_5b_ar_bf16_sage_lighttae_async_5090.json)

```bash
export MODEL_PATH="$PWD/models/EdenLive-x2v"
export IMAGE_PATH=/path/to/cover.png
export VIDEO_PATH=/path/to/recording.mp4
export PROMPT="A cinematic scene with detailed, realistic lighting."
export OUTPUT_PATH="$PWD/edenlive_output.mp4"
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_5090.sh
```

`run_5090.sh` selects the FP8 profile. To run BF16, set `CONFIG_PATH` to the BF16 JSON in the same command. Update the model, quantized checkpoint and LightTAE paths stored in a copied JSON when moving to another machine. Both configurations use Sage, warmup, DiT compile/CUDA Graph, streaming LightTAE and top-level asynchronous decode; `vae_use_compile` controls decoder compilation independently. KV offload is disabled.

## RTX 5090 results

Complete official cryo-crossing case: 717 published frames, 1280×704 at 24 FPS, seven timed prompts, four denoising steps, three latent frames per chunk, window 18 and sink 6. Each row uses one formal request after excluded warmup.

| Implementation / GPU | Generation FPS ↑ | Inference FPS ↑ | First-chunk latency ms ↓ | Inter-chunk gap P95 ms ↓ | RTF@fps24 ↓ | Peak allocated GiB ↓ | Peak reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official source BF16 / RTX 5090 | 13.4663 | 11.6319 | 9083.1971 | 908.8465 | 2.0633 | 21.3601 | 25.3223 |
| LightX2V FP8 / RTX 5090 | **25.2811** | **25.1732** | **509.6956** | **511.9061** | **0.9534** | **23.3308** | **24.9082** |
