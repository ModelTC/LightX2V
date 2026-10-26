# EdenLive

[English](README.md) | [简体中文](README_zh.md)

EdenLive is a Wan-based neural renderer that transforms a **proxy or reference video stream**, a **target first frame**, and a **text prompt** into refined or stylized video. References can come from 2D/3D scenes, existing engines, or recordings. The upstream application maintains interaction state; EdenLive handles visual rendering.

Control is injected through a **post-trained DINOv3 vision encoder**, which embeds each proxy frame and conditions the diffusion transformer. The proxy provides scene layout, camera motion, and timing, while the first frame and text prompt specify the target appearance.

The [EdenLive project](https://vast-eden.ai/?p=edenlive) demonstrates generation above 24 FPS, minute-scale sequences, and 480p/720p output. The 5B model supports single-GPU deployment at 720p. LightX2V provides optimized inference for both bidirectional and autoregressive variants, including quantization, streaming decoding, compilation, and CUDA Graph acceleration.

## Supported Models

| Backbone | Inference mode | Weight directory | Launcher |
| --- | --- | --- | --- |
| Wan2.1 14B | Bidirectional | `dit-14b-bidir` | [wan21_14b_bidir.sh](wan21_14b_bidir.sh) |
| Wan2.1 14B | Four-step autoregressive | `dit-14b` | [wan21_14b_few_step.sh](wan21_14b_few_step.sh) |
| Wan2.2 5B | Bidirectional | `dit-5b-bidir` | [wan22_5b_bidir.sh](wan22_5b_bidir.sh) |
| Wan2.2 5B | Four-step autoregressive | `dit-5b` | [wan22_5b_few_step.sh](wan22_5b_few_step.sh) |

Bidirectional inference jointly denoises the complete clip. Autoregressive inference generates consecutive chunks with a persistent KV cache and supports streaming decoded output.

## Setup

Prepare the environment using the LightX2V [Quick Start](../../docs/EN/source/getting_started/quickstart.md). The real-time profiles require SageAttention2, FlashInfer, and SGL kernels compatible with the target GPU. The RTX 5090 profile additionally requires comfy-kitchen and a CUDA compiler for INT8 ConvRot.

Run the following commands from the LightX2V repository root.

### Download and convert

Download the complete [VAST-AI/EdenLive](https://huggingface.co/VAST-AI/EdenLive) release and convert it to the LightX2V format:

```bash
hf download VAST-AI/EdenLive --local-dir ./models/EdenLive

export EDEN_RELEASE="$PWD/models/EdenLive"
export EDEN_MODEL="$PWD/models/EdenLive-x2v"

python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE" --output "$EDEN_MODEL" --all
```

Full conversion prepares all four DiTs and the T5/CLIP encoders, and preserves the DINO, VAE, tokenizer, and LightTAE assets. Set `MODEL_PATH` to the converted model root for all inference modes, including quantized inference.

## Inference

Run the Wan2.2 5B autoregressive model in BF16:

```bash
MODEL_PATH="$EDEN_MODEL" \
IMAGE_PATH=/path/to/cover.png \
VIDEO_PATH=/path/to/recording.mp4 \
PROMPT="A cinematic scene with detailed, realistic lighting." \
SAVE_RESULT_PATH="$PWD/save_results/edenlive_5b.mp4" \
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/wan22_5b_few_step.sh
```

Use the corresponding launcher in the model table for the other variants. All launchers accept `MODEL_PATH`, `IMAGE_PATH`, `VIDEO_PATH`, `PROMPT`, `SAVE_RESULT_PATH`, and `CUDA_VISIBLE_DEVICES`.

The four base launchers also accept `CONFIG_JSON` to select a configuration. Additional inference arguments are forwarded to LightX2V, for example `--seed 123`. Base configurations, including AR prompt travel, are available in [`configs/edenlive`](../../configs/edenlive/).

## Real-Time Deployment

The hardware presets target **Wan2.2 5B AR at 1280×704 and 24 FPS**, with four denoising steps and three latent frames per chunk. Both use FP8 T5 resident on GPU, synchronous LightTAE streaming decode, compilation, and warmup.

| Hardware | DiT | Proxy encoder | Launcher |
| --- | --- | --- | --- |
| H100 / H200 | FP8-SGL | BF16 | [run_hopper.sh](run_hopper.sh) |
| RTX 5090 | INT8 ConvRot | INT8 ConvRot | [run_5090.sh](run_5090.sh) |

Configuration presets: [Hopper FP8-SGL](../../configs/edenlive/realtime/wan22_5b_ar_fp8_sgl_hopper.json) · [RTX 5090 INT8 ConvRot](../../configs/edenlive/realtime/wan22_5b_ar_int8_convrot_5090.json).

### Prepare quantized weights

Both presets use an FP8 T5 checkpoint converted from the LightX2V model:

```bash
python tools/convert/converter.py \
  --source "$EDEN_MODEL/umt5/text_encoder/model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name t5-fp8 \
  --model_type wan_t5 --quantized --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 --device cuda:0 --single_file
```

For **H100/H200**, quantize the DiT to FP8:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8 \
  --quantize fp8 --device cuda:0
```

For **RTX 5090**, quantize the DiT and proxy encoder to INT8 ConvRot:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-int8-convrot \
  --quantize int8-convrot --device cuda:0

python tools/convert/convert_edenlive_proxy.py \
  --source "$EDEN_MODEL/vision-encoder/projection.safetensors" \
  --output "$EDEN_MODEL/quantized/proxy-dino-int8-convrot.safetensors" \
  --device cuda:0 --group-size 256
```

### Run a hardware preset

Set the model, input, and decoder paths:

```bash
export MODEL_PATH="$EDEN_MODEL"
export IMAGE_PATH=/path/to/cover.png
export VIDEO_PATH=/path/to/recording.mp4
export PROMPT="A cinematic scene with detailed, realistic lighting."
export TAE_PATH="$EDEN_MODEL/decoder-5b/lighttaew2_2.pth"
export CUDA_VISIBLE_DEVICES=0
```

On **H100/H200**:

```bash
bash scripts/edenlive/run_hopper.sh
```

On **RTX 5090**:

```bash
bash scripts/edenlive/run_5090.sh
```

The converted release includes the LightTAE checkpoint used in this example. `TAE_PATH` also accepts a fine-tuned `.safetensors` checkpoint. The original Wan VAE encodes the target first frame; LightTAE decodes the generated chunks.

The launchers use the quantized files prepared above. Checkpoints stored elsewhere can be selected through these environment variables:

| Variable | Default checkpoint |
| --- | --- |
| `DIT_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/dit-5b-fp8.safetensors` on Hopper; `dit-5b-int8-convrot.safetensors` on RTX 5090 |
| `T5_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/t5-fp8.safetensors` |
| `PROXY_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/proxy-dino-int8-convrot.safetensors`, RTX 5090 only |

`SAVE_RESULT_PATH` selects the output MP4. `T5_CPU_OFFLOAD=true` reduces resident GPU memory by offloading T5; the performance results below use `false`.

## LightX2V Acceleration

| Component | Optimization |
| --- | --- |
| Quantized projections | FP8-SGL or INT8 ConvRot GEMMs, shared activation quantization, packed QKV projections, and fused GELU/quantization. |
| Attention and small operators | Architecture-specific SageAttention2, FlashInfer RoPE, SGL RMSNorm, and Triton LayerNorm/modulation. |
| AR cache | Persistent sink-and-ring KV storage, reusable attention workspaces, and cached historical RoPE data. |
| Conditioning | Reuse text K/V, text/time projections, and repeated proxy feature interpolation. |
| KV commit | Preserve the clean forward while skipping unused attention, FFN, and output computation after the final required cache update. |
| Proxy encoder | GPU preprocessing, fused RoPE/normalization, and CUDA Graph replay. |
| Compilation | Compile DiT blocks and the streaming LightTAE decoder; warm up the required shapes and cache states. |
| Streaming | Prefetch proxy frames and decode each generated chunk with persistent decoder state. |

The RTX 5090 preset supports CUDA Graph execution for the DiT stack and proxy encoder. The Hopper preset uses compiled DiT blocks and CUDA Graph execution for the proxy encoder.

## Performance

**Wan2.2 5B AR · 1280×704 · 24 FPS · batch size 1 · synchronous LightTAE decode.** Both presets use four denoising steps, three latent frames per chunk, an attention window of 18 latent frames, and six sink frames.

| GPU | Generation FPS ↑ | Inference FPS ↑ | First-chunk latency ms ↓ | Inter-chunk gap P95 ms ↓ | RTF@fps24 ↓ | Peak allocated GiB ↓ | Peak reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA H200 | 36.12 | 35.95 | 358.19 | 351.20 | 0.6676 | 29.72 | 32.23 |
| NVIDIA RTX 5090 | 24.87 | 24.69 | 500.02 | 492.63 | 0.9722 | 28.38 | 28.95 |

- **Generation FPS** measures AR generation through the final decoded chunk ready on GPU. **Inference FPS** additionally includes request input preparation and encoding. Model loading, compilation, warmup, final video concatenation, and MP4 saving are excluded.
- **First-chunk latency** starts at the request boundary. **Inter-chunk gap P95** measures the interval between successive decoded chunks; the H200 value pools all three requests. FPS is total frames divided by total measured time.
- **RTF@fps24** is inference time divided by playback duration at 24 FPS. Values below 1 indicate generation faster than playback.
- **Peak memory** covers the full request, including final video concatenation and saving.
