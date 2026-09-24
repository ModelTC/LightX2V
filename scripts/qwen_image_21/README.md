# Qwen-Image-2.1 Deployment Guide

[English](README.md) | [简体中文](README_zh.md)

## 1. Model Download

Download the official weights from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1):

```bash
hf download Qwen/Qwen-Image-2.1 --local-dir Qwen/Qwen-Image-2.1
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

Clone the LightX2V source inside the container:

```bash
git clone https://github.com/ModelTC/LightX2V.git
```

Run the following commands from the LightX2V repository root inside the container.

## 3. Offline Inference

### 3.1 General examples

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

The default config already enables the general optimization path: FlashAttention3, FlashInfer RoPE, Triton LayerNorm and modulation, fused QK RMSNorm, fused transformer-block operators, CFG-disabled inference, and OpenCV result saving.

### 3.2 RTX 5090 examples

The examples use these optimization techniques:

- FP8 DiT linears with FP16 accumulation and FP8 QwenVL language weights.
- SageAttention2 and FlashInfer RoPE.
- Triton LayerNorm/modulation, fused QK RMSNorm and transformer-block operators.
- Text encoder offload and VAE torch.compile.
- Ulysses SP, FP8 communication, grouped head parallelism, language TP and VAE parallelism.

Convert the DiT weights:

```bash
python tools/convert/converter.py \
    --source /path/to/Qwen-Image-2.1/transformer \
    --output /path/to/Qwen-Image-2.1-fp8-f16-accum \
    --output_name qwen_image_21_fp8_f16_accum \
    --model_type qwen_image_21_dit \
    --quantization_profile qwen-image-21-fp8-f16-accum \
    --quantized --linear_type fp8 --device cuda:0 --single_file
```

For QwenVL FP8 conversion, see the example in `convert_qwen_image_21_text_encoder_fp8` in [converter.py](../../tools/convert/converter.py). Set `dit_quantized_ckpt` and `text_encoder_quantized_ckpt` in the corresponding JSON, and set the repository path, original model path and GPU IDs in the scripts.

```bash
# Single GPU, 1K: text-to-image / image-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i_5090_1k.sh
bash scripts/qwen_image_21/qwen_image_21_i2i_5090_1k.sh

# Single GPU, 2K: text-to-image / image-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i_5090_2k.sh
bash scripts/qwen_image_21/qwen_image_21_i2i_5090_2k.sh

# Dual GPU, 2K: text-to-image / image-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i_5090_2k_sp2.sh
bash scripts/qwen_image_21/qwen_image_21_i2i_5090_2k_sp2.sh
```

| GPU | Task | Output resolution | End-to-end latency |
| --- | --- | ---: | ---: |
| RTX 5090 ×1 | T2I | 1024×1024 | **5.586 s** |
| RTX 5090 ×1 | I2I | 1024×1024 | **6.804 s** |
| RTX 5090 ×1 | T2I | 2048×2048 | **30.333 s** |
| RTX 5090 ×1 | I2I | 2048×2048 | **42.787 s** |
| RTX 5090 ×2 | T2I | 2048×2048 | **16.868 s** |
| RTX 5090 ×2 | I2I | 2048×2048 | **25.212 s** |

Tests use PyTorch 2.11.0+cu130, 40 steps, seed 42, CFG disabled, and the example prompts. I2I uses `assets/inputs/imgs/girl.png`, preprocessed at the corresponding resolution.

Latency is the median of three consecutive requests after the first request in the same instance; dual-GPU samples use the slower rank. End-to-end timing includes image saving and excludes model loading and one-time initialization.

Search the runtime log for **`RUN pipeline cost`** to find the pipeline latency in seconds.

## 4. Service Deployment and API Usage

Set the repository path, model path, and GPU ID in `server/start_server.sh`, then start the server:

```bash
bash scripts/qwen_image_21/server/start_server.sh
```

The default port is `8000`. The server supports both text-to-image and image-to-image requests.

Once the server is ready, open a new terminal in the same container and send requests:

```bash
# Text-to-image
python scripts/qwen_image_21/server/post_t2i.py

# Image-to-image
python scripts/qwen_image_21/server/post_i2i.py
```

Edit `url`, `message`, and `output_path` directly; for image-to-image, also set `image_path`. The scripts already specify `task: "t2i"` and `task: "i2i"`, respectively.

Service requests follow the same size rules. For automatic image-to-image sizing, remove `size` from `message`.
