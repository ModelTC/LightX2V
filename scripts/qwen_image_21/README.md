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

Inside the container, clone the LightX2V source. It can run directly without installing the package:

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

### 3.2 RTX 5090 FP8 example

This example uses FP8 linear layers with FP16 accumulation. Convert the DiT weights first:

```bash
python tools/convert/converter.py \
    --source /path/to/Qwen-Image-2.1/transformer \
    --output /path/to/Qwen-Image-2.1-fp8-f16-accum \
    --output_name qwen_image_21_fp8_f16_accum \
    --model_type qwen_image_21_dit \
    --quantization_profile qwen-image-21-fp8-f16-accum \
    --quantized --linear_type fp8 --device cuda:0 --single_file
```

Set `dit_quantized_ckpt` in the selected JSON config to the converted `.safetensors` file, then set `lightx2v_path` and `model_path` in the script. `model_path` must still point to the original model directory. The FP16-accumulation path requires lightx2v-kernel with the SM120 operator.

```bash
# Text-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i_fp8_f16_accum_5090.sh

# Image-to-image
bash scripts/qwen_image_21/qwen_image_21_i2i_fp8_f16_accum_5090.sh
```

The RTX 5090 config retains the applicable general optimizations and uses:

- stage-level text-encoder CPU offload;
- FP8 DiT block-linear weights with qmax 14, dynamic FP8 activations with qmax 7, and FP16 accumulation;
- dense SageAttention2.

| GPU | Task | Output resolution | End-to-end latency |
| --- | --- | ---: | ---: |
| RTX 5090 ×1 | T2I | 1024×1024 | **5.914 s** |
| RTX 5090 ×1 | I2I | 1024×1024 | **7.191 s** |

Both tasks were measured on the current code with 40 steps, seed 42, CFG disabled, and the median latency of three consecutive requests. I2I uses one 1024×1024 reference image and produces a 1024×1024 image. End-to-end latency covers input encoding, condition-KV prefill, denoising, VAE decoding, post-processing, and PNG saving; it excludes model loading and one-time runner initialization.

### 3.3 Dual RTX 5090 sequence-parallel example

The SP2 presets distribute the target image-token sequence across two GPUs with Ulysses, use FP8 sequence-parallel communication, and use an overlap-based spatial VAE decode: its global low-resolution attention remains exact, while a two-latent halo around each upsampling shard may introduce small boundary differences from serial decode. They retain the FP8-FP16-accumulation and SageAttention2 settings from the single-GPU RTX 5090 example. Set `dit_quantized_ckpt`, `lightx2v_path`, and `model_path` as described above, then run:

```bash
# Text-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i_fp8_f16_accum_5090_sp2.sh

# Image-to-image
bash scripts/qwen_image_21/qwen_image_21_i2i_fp8_f16_accum_5090_sp2.sh
```

For 2048×2048 output, use the dedicated 2K presets. They process four local attention heads per pipelined stage, which reduces communication-launch overhead at the longer sequence length:

```bash
# Text-to-image
bash scripts/qwen_image_21/qwen_image_21_t2i_fp8_f16_accum_5090_sp2_2k.sh

# Image-to-image
bash scripts/qwen_image_21/qwen_image_21_i2i_fp8_f16_accum_5090_sp2_2k.sh
```

The scripts launch two processes on GPUs 0 and 1. Keep `CUDA_VISIBLE_DEVICES`, `torchrun --nproc_per_node`, and `parallel.seq_p_size` consistent when changing the GPU count. The target-image token count must be divisible by `seq_p_size`.

| GPU | Task | Output resolution | End-to-end latency |
| --- | --- | ---: | ---: |
| RTX 5090 ×2 | T2I | 1024×1024 | **3.954 s** |
| RTX 5090 ×2 | I2I | 1024×1024 | **4.802 s** |
| RTX 5090 ×2 | T2I | 2048×2048 | **17.298 s** |
| RTX 5090 ×2 | I2I | 2048×2048 | **25.488 s** |

These results use the same request settings and measurement boundary as the single-GPU table above, with the median latency of three consecutive requests. The 2K I2I measurement also uses `resolution: 2048` for reference-image preprocessing.

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
