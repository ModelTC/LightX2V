# SwiftVR Deployment Guide

[English](README.md) | [简体中文](README_zh.md)

[技术博客](https://light-ai.top/LightX2V-BLOG/posts/SwiftVR/)

SwiftVR supports image and video restoration and super-resolution. Both use `--model_cls swiftvr --task sr`.

## 1. Model Download and Conversion

After setting up the environment, download and convert the official weights from [H-oliday/SwiftVR](https://huggingface.co/H-oliday/SwiftVR), running these commands from the LightX2V repository root:

```bash
hf download H-oliday/SwiftVR --local-dir /path/to/SwiftVR
python tools/convert/examples/convert_swiftvr.py \
  --source /path/to/SwiftVR \
  --output /path/to/SwiftVR_lightx2v
```

## 2. Environment Setup

Choose the [LightX2V Docker image](https://hub.docker.com/r/lightx2v/lightx2v/tags) for your GPU. Pull the corresponding image in a working directory on the host:

```bash
# H100, A100, etc.
docker pull lightx2v/lightx2v:26062001

# RTX 5090
docker pull lightx2v/lightx2v:26062001-cu130-5090-fix-260921
```

Start a container using the selected image:

```bash
docker run --gpus all -itd --ipc=host --name [container_name] -v [mount_settings] --entrypoint /bin/bash [image_id]
```

Inside the container, clone the source and install the dependencies:

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
python -m pip install "av>=17" decord imageio-ffmpeg huggingface_hub
```

To set up your own environment, see the [installation guide](../../docs/EN/source/getting_started/quickstart.md).

## 3. Offline Inference

Before running, set `lightx2v_path` (repository path), `model_path` (converted model path), `config_path` (configuration path), and `CUDA_VISIBLE_DEVICES` (GPU IDs) in the corresponding script, along with the input path and `output_path`.

Select a configuration under `configs/swiftvr/` for your GPU:

| GPU | Configuration |
| --- | --- |
| H100 | `h100/swiftvr.json` |
| A800 | `a800/swiftvr.json` |
| RTX 5090 | `5090/swiftvr.json` |

To enable compilation, add `"use_compile": true` to the configuration file or use the corresponding `swiftvr_compile.json`.

```bash
# Image super-resolution
bash scripts/swiftvr/inference/run_swiftvr_image_sr.sh

# Video super-resolution
bash scripts/swiftvr/inference/run_swiftvr_video_sr.sh
```

Provide exactly one of `--image_path` or `--video_path`, with a `.png` or `.mp4` output file, respectively. The default scale factor is 2 (`--sr_ratio 2`); change it as needed or use `--size 1440 2560` to specify the output dimensions (height, width). If both are set, `size` takes precedence.

For multi-GPU video inference, use `h100/swiftvr_parallel.json` as a reference. For four GPUs, set `parallel.chunk_p_size=4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, and `--nproc_per_node 4`, then run:

```bash
bash scripts/swiftvr/inference/run_swiftvr_video_sr_4gpu.sh
```

## 4. Service Deployment and API Usage

Set the repository path, model path, configuration path, and GPU IDs in the corresponding server script, then choose a single-GPU or four-GPU launch:

```bash
# Single GPU; uses the H100 compile configuration by default
bash scripts/swiftvr/server/start_server.sh

# Four GPUs; match the parallel settings as described above
bash scripts/swiftvr/server/start_server_4gpu.sh
```

The default port is `8000`. One `sr` service supports both image and video super-resolution.

Before sending requests, edit `url` and `message` in the client scripts: use absolute input paths accessible to the server and set `save_result_path` to a server-side output file. Use `sr_ratio` to set the scale factor or `size: [height, width]` to specify output dimensions, following the same rules as offline inference.

Once startup and warmup finish, run the corresponding request script from another terminal:

```bash
# Image super-resolution
python scripts/swiftvr/server/post_image.py

# Video super-resolution
python scripts/swiftvr/server/post_video.py
```
