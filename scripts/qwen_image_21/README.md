# Qwen-Image-2.1 Deployment Guide

## 1. Model Download

Download the official weights from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1):

```bash
hf download Qwen/Qwen-Image-2.1 --local-dir Qwen/Qwen-Image-2.1
```

## 2. Environment Setup

Use the [LightX2V Docker image](https://hub.docker.com/r/lightx2v/lightx2v/tags). Run these commands in a working directory on the host:

```bash
docker pull lightx2v/lightx2v:26062001
docker run --gpus all -itd --ipc=host --name [container_name] -v [mount_settings] --entrypoint /bin/bash [image_id]
```

Inside the container, clone the LightX2V source. It can run directly without installing the package:

```bash
git clone https://github.com/ModelTC/LightX2V.git
```

Run the following commands from the LightX2V repository root inside the container.

## 3. Offline Inference

Set `lightx2v_path` (repository path), `model_path` (model root directory), and `CUDA_VISIBLE_DEVICES` (GPU ID) in the corresponding bash script.

```bash
# Text-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i.sh

# Image-to-image
bash scripts/qwen_image_21/qwen_image_21_i2i.sh
```

Edit `--prompt`, `--image_path`, `--size`, `--seed`, and `--save_result_path` directly in the script as needed.

Use `--size HEIGHT WIDTH`. Both dimensions must be at least 32; non-multiples of 32 are rounded down automatically.

For image-to-image, remove the script's `--size` argument to determine the output size from the reference image's aspect ratio. The pixel count is approximately `resolution²` (`resolution` defaults to `1024` in the config), with each dimension rounded to the nearest multiple of 32.

## 4. Service Deployment and API Usage

Set the repository path, model path, and GPU ID in `server/start_server.sh`, then start the server:

```bash
bash scripts/qwen_image_21/server/start_server.sh
```

The default port is `8000`. One server loads a single set of weights and supports both text-to-image and image-to-image requests. No `task` is needed at startup.

Once the server is ready, open a new terminal in the same container and send requests:

```bash
# Text-to-image
python scripts/qwen_image_21/server/post_t2i.py

# Image-to-image
python scripts/qwen_image_21/server/post_i2i.py
```

Edit `url`, `message`, and `output_path` directly; for image-to-image, also set `image_path`. The scripts already specify `task: "t2i"` and `task: "i2i"`, respectively.

Service requests follow the same size rules. For automatic image-to-image sizing, remove `size` from `message`.

The image-to-image script reads a local image on the client, encodes it as Base64, and uploads it in the request's `image_path` field. Both scripts wait for generation to finish, receive PNG binary data from the server, and save it to the client's `output_path`.
