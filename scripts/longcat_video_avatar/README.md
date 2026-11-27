# LongCat-Video-Avatar 1.6

[简体中文](README_zh.md)

The `longcat_video_avatar` model uses the `s2v` task to generate videos with one or two speakers. Each segment defaults to 93 frames at 25 fps.

## Setup

Place both checkpoints under the same parent directory. Set `model_path` to `LongCat-Video-Avatar-1.6`. Its sibling `LongCat-Video` supplies `text_encoder/`, `tokenizer/`, and `vae/`. [Avatar weights](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) · [Shared weights](https://huggingface.co/meituan-longcat/LongCat-Video)

The H800 examples use BF16, FlashAttention 3, and FFmpeg. Run `pip install -e '.[longcat-avatar]'` from the repository root to install the audio dependencies. Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` at the top of each Bash script you use.

The [four-GPU H800 configuration](../../configs/longcat_video_avatar/h800/s2v_bf16_sp4.json) enables `parallel.seq_p_nvlink` for direct Ulysses transfers. It uses Triton and PyTorch CUDA symmetric memory on peer-accessible GPUs in the same node.

Enabling `audio_vocal_separation` requires the CUDA provider from `onnxruntime-gpu`. Choose a build compatible with the CUDA and cuDNN versions used by PyTorch. When upgrading an environment with the CPU-only `onnxruntime` package, uninstall it before installing the extra.

## HTTP server

Start the server. For four GPUs, use [start_server_sp4.sh](server/start_server_sp4.sh).

```bash
bash scripts/longcat_video_avatar/server/start_server.sh
```

Send a single-speaker request from another terminal. For two speakers, use [post_s2v_multi.py](server/post_s2v_multi.py).

```bash
python scripts/longcat_video_avatar/server/post_s2v_single.py
```

## CLI

Run one example from the repository root.

```bash
# One speaker
bash scripts/longcat_video_avatar/run_longcat_video_avatar_single.sh
# Two speakers in one image
bash scripts/longcat_video_avatar/run_longcat_video_avatar_multi.sh
# Four GPUs with Ulysses sequence parallelism
bash scripts/longcat_video_avatar/run_longcat_video_avatar_s2v_sp4.sh
```

## Shared parameters

For example, pass `--num_segments 2` on the command line or `"num_segments": 2` in the HTTP request.

| Goal | Arguments |
| --- | --- |
| Drive one speaker | Supply a prompt and one audio track. The examples include a reference image by default. Omit `image_path` to generate from the prompt and audio alone. |
| Drive two speakers | Supply a shared photo and list both tracks in the `audio_path` array, in left-to-right speaker order by default. The default `audio_type`, `para`, plays both tracks up to the end of the shorter one. Use `add` to play them in order. |
| Continue a video | Set `num_segments` to 2 in any input mode. Each new segment reuses the previous 13 frames, giving 173 frames (6.92 seconds) for two segments at the default length. |
| Choose a resolution preset | Set `resolution` to `480p` (default) or `720p`. With a reference image, the preset selects the closest aspect ratio. Without one, the outputs are 832 × 480 and 1280 × 768 respectively (width × height). |
| Set exact dimensions | CLI `--size 512 768` or HTTP `"size": [512, 768]` gives a height of 512 and a width of 768. Height must be divisible by 16 × grid rows and width by 16 × grid columns. For eight GPUs the multiples are 32 and 64, respectively. |

An explicit `size` takes priority, followed by an explicit `resolution`, then the deployment's `request_defaults`. [Single-GPU config](../../configs/longcat_video_avatar/h800/s2v_bf16.json) · [Four-GPU config](../../configs/longcat_video_avatar/h800/s2v_bf16_sp4.json)
