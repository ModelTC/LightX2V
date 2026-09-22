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
| RTX 5090 | T2I | 1024×1024 | **5.930 s** |
| RTX 5090 | I2I | 1024×1024 | **7.144 s** |

Both tasks were measured on the current code with 40 steps, seed 42, CFG disabled, and the median latency of three consecutive requests. I2I uses one 1024×1024 reference image and produces a 1024×1024 image. End-to-end latency covers input encoding, condition-KV prefill, denoising, VAE decoding, post-processing, and PNG saving; it excludes model loading and one-time runner initialization.

### 3.3 AMD ROCm GPUs (Radeon AI PRO R9700 / Radeon PRO W7900)

The 5090 stack above (FlashAttention3, FlashInfer RoPE, CUTLASS FP8, SageAttention2 CUDA kernels) is CUDA-only. The configs below run Qwen-Image-2.1 on AMD RDNA GPUs with a torch-op inference path (`torch_sdpa`, `torch_real_rope`, torch LayerNorm/modulation) plus ROCm-friendly quantization, `torch.compile`, and SageAttention2. They quantize the **released BF16 weights on load** — no `tools/convert` step and no `dit_quantized_ckpt`.

Validated in the `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` image (PyTorch 2.10, Triton 3.6) with the `transformers` 4.57.x line (`Qwen3VLProcessor` needs a recent release; pin to a version you have tested rather than only a lower bound). Install the runtime deps not baked into that image:

```bash
pip install "transformers>=4.57" diffusers ftfy accelerate loguru omegaconf einops \
    qtorch langdetect tqdm imageio imageio-ffmpeg opencv-python-headless comfy-kitchen \
    peft gguf prometheus_client fastapi uvicorn pydantic aiohttp pyzmq python-multipart \
    PyJWT jsonschema sageattention
```

Run through the platform scripts under `scripts/platforms/amd_rocm/`, which `export PLATFORM=amd_rocm` and source the common env. This activates the AMD ROCm platform backend under `lightx2v_platform/` (registers the ROCm FP8/INT8 GEMM and applies the SageAttention Triton workaround). A single GPU is enough — the text encoder is CPU-offloaded, and no `aiter` build is required.

Set `lightx2v_path` and `model_path` at the top of the script, then run:

```bash
# R9700 (FP8)
bash scripts/platforms/amd_rocm/qwen_image_21_r9700_t2i.sh
# W7900 (INT8)
bash scripts/platforms/amd_rocm/qwen_image_21_w7900_t2i.sh
```

Each script selects the fastest config for that GPU; edit `--config_json` in the script to pick another. Per-GPU config (all under `configs/platforms/amd_rocm/`):

| GPU (arch) | Quantization | Configs |
| --- | --- | --- |
| R9700 (gfx1201 / RDNA4) | FP8 via `torch._scaled_mm` | `qwen_image_21_r9700_fp8_compile_sage.json` (fastest), `..._fp8_compile.json`, `..._fp8_compile_sage_25steps.json`, `qwen_image_21_r9700.json` (BF16 eager baseline) |
| W7900 (gfx1100 / RDNA3) | INT8 via `torch._int_mm` (no FP8 on RDNA3) | `qwen_image_21_w7900_int8_compile_sage.json` (fastest), `..._int8_compile.json`, `..._int8_compile_sage_25steps.json`, `qwen_image_21_w7900_int8.json` (INT8 eager baseline) |

The `dit_quant_scheme` is `fp8-rocm` (R9700) or `int8-rocm` (W7900), registered by the AMD ROCm platform ops in `lightx2v_platform/ops/mm/amd_rocm/`: per-channel symmetric weight + per-token dynamic activation, quantized from BF16 on load. Both use `use_compile` and `attn_type: sage_attn2`. The VAE runs native convolution — the AMD ROCm platform disables cuDNN/MIOpen, which sidesteps the nondeterministic-NaN convolution path observed on gfx1201. The SageAttention/Inductor Triton `num_stages` workaround (needed on gfx1201 / gfx1100) is installed automatically, and only when `attn_type: sage_attn2` is selected.

| GPU | Quant | Steps | End-to-end |
| --- | --- | ---: | ---: |
| R9700 | FP8 | 40 | ~24 s |
| R9700 | FP8 | 25 | ~17 s |
| W7900 | INT8 | 40 | ~46 s |
| W7900 | INT8 | 25 | ~29 s |

Measured at 1024×1024, seed 42, CFG disabled, single GPU, steady state after warmup (median of a few runs). End-to-end covers text encoding, condition-KV prefill, denoising, VAE decode and PNG save; it excludes model load and one-time initialization. In the tested prompts and seeds, 25 steps (a common ComfyUI setting for this model) showed no obvious visual degradation versus 40 — this is not a systematic quality evaluation. Compilation is a one-time warmup cost (~1.5–3 min); `TORCHINDUCTOR_COMPILE_THREADS=1` is set automatically on the SageAttention path.

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
