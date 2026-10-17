# EdenLive Deployment Guide

[English](README.md) | [简体中文](README_zh.md)

EdenLive generates video from a cover image, a recording, and a text prompt. The examples support Wan2.2 5B and Wan2.1 14B, with bidirectional and autoregressive (AR) inference. Run the commands below from the LightX2V repository root.

## 1. Environment and Model Files

Follow the repository's [Quick Start](../../docs/EN/source/getting_started/quickstart.md). Install attention and linear kernels built for the destination GPU: the realtime profiles use SageAttention2, FlashInfer, SGL kernels, and, on RTX 5090, comfy-kitchen INT8 ConvRot. The ConvRot GELU fusion also needs a CUDA compiler for its first build.

Keep the original release and converted model in separate directories:

```bash
export EDEN_RELEASE=/path/to/EdenLive
export EDEN_MODEL=/path/to/EdenLive-x2v
export CUDA_VISIBLE_DEVICES=0
export PROFILING_DEBUG_LEVEL=0
```

The full-release converter expects all four DiTs and both encoders:

```text
EdenLive/
├── dit-5b/
├── dit-5b-bidir/
├── dit-14b/
├── dit-14b-bidir/
├── umt5/text_encoder/
├── umt5/tokenizer/
├── clip/image_encoder/
├── vision-encoder/backbone/
├── vision-encoder/projection.safetensors
├── vae-5b/
└── vae-14b/
```

DiT directories contain `config.json` and safetensors weights. Keep the release's tokenizer and auxiliary files. `model_path` always points to the complete converted model root, including when a quantized DiT checkpoint is loaded separately.

## 2. Model Conversion and Quantization

### 2.1 Convert the original release

Convert all DiTs and the T5/CLIP encoder tensor names, and copy the remaining DINO, VAE, tokenizer, and auxiliary assets:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE" --output "$EDEN_MODEL" --all
```

`--all` requires the complete release above. The output must differ from the source and must not be inside it. Converted DiT weights are BF16. Configuration examples are also copied into the output's `configs/` directory.

To convert only one DiT:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_RELEASE/dit-5b" --output "$EDEN_MODEL/dit-5b"
```

This command does not prepare encoders or other model assets. Use the corresponding source and output directories for `dit-5b-bidir`, `dit-14b`, or `dit-14b-bidir`.

### 2.2 Quantize the DiT

FP8 weights for the Hopper FP8-SGL profile:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8 \
  --quantize fp8 --device cuda:0
```

INT8 ConvRot weights for the RTX 5090 profile:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-int8-convrot \
  --quantize int8-convrot --device cuda:0
```

Conversion uses `fp8`; the runtime scheme is `fp8-sgl`. The respective outputs are `dit-5b-fp8.safetensors` and `dit-5b-int8-convrot.safetensors`. Set `dit_quantized`, `dit_quant_scheme`, and `dit_quantized_ckpt` together when adapting a base configuration.

### 2.3 Quantize T5 and ProxyDINO

Convert the already remapped native T5 checkpoint to FP8:

```bash
python tools/convert/converter.py \
  --source "$EDEN_MODEL/umt5/text_encoder/model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name t5-fp8 \
  --model_type wan_t5 --quantized --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 --device cuda:0 --single_file
```

Use `t5_quantized=true`, `t5_quant_scheme="fp8-sgl"`, and the resulting `t5_quantized_ckpt`. Both realtime profiles keep T5 on the GPU with `t5_cpu_offload=false`.

The RTX 5090 realtime profile also uses an independent INT8 ConvRot ProxyDINO checkpoint:

```bash
python tools/convert/convert_edenlive_proxy.py \
  --source "$EDEN_MODEL/vision-encoder/projection.safetensors" \
  --output "$EDEN_MODEL/quantized/proxy-dino-int8-convrot.safetensors" \
  --device cuda:0 --group-size 256
```

Here `--output` is a file path. The converter quantizes linear weights, keeps other tensors, and writes a quantization-error report beside the checkpoint. It refuses existing checkpoint/report files unless `--overwrite` is supplied. The Hopper profile uses the original ProxyDINO weights.

## 3. Inference Scripts

The input sample directory must contain `cover.png`, `recording.mp4`, and `prompts.json` with a `staticPrompt` field. Set model and input paths before launching:

```bash
MODEL_PATH="$EDEN_MODEL" \
SAMPLE_DIR=/path/to/cryo-crossing \
SAVE_RESULT_PATH=/path/to/output.mp4 \
CUDA_VISIBLE_DEVICES=0 bash scripts/ed/wan22_5b_few_step.sh
```

| Script | Model | Execution / default configuration |
| --- | --- | --- |
| `wan22_5b_bidir.sh` | Wan2.2 5B | Bidirectional, base BF16 profile |
| `wan22_5b_few_step.sh` | Wan2.2 5B | Four-step AR, base BF16 profile |
| `wan21_14b_bidir.sh` | Wan2.1 14B | Bidirectional, base BF16 profile |
| `wan21_14b_few_step.sh` | Wan2.1 14B | Four-step AR, base BF16 profile |
| `run_5090.sh` | Wan2.2 5B | Single RTX 5090, INT8 ConvRot realtime profile |
| `run_hopper.sh` | Wan2.2 5B | Single H100/H200, FP8-SGL realtime profile |

All six scripts accept `MODEL_PATH`, `SAMPLE_DIR`, `SAVE_RESULT_PATH`, and `CUDA_VISIBLE_DEVICES`. The four base scripts also accept `CONFIG_JSON`; use it to select another retained base profile, including prompt travel. Extra CLI arguments are forwarded, for example `--seed 123` or `--prompt 'A frozen cavern'`.

The realtime scripts accept these additional environment overrides:

| Variable | Default |
| --- | --- |
| `DIT_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/dit-5b-int8-convrot.safetensors` (5090), or `dit-5b-fp8.safetensors` (Hopper) |
| `T5_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/t5-fp8.safetensors` |
| `PROXY_QUANTIZED_CKPT` | `$MODEL_PATH/quantized/proxy-dino-int8-convrot.safetensors`, 5090 only |
| `TAE_PATH` | `lighttaew2_2.safetensors` beside the repository |
| `T5_CPU_OFFLOAD` | `false`; set `true` for a lower resident-memory comparison |

For example:

```bash
MODEL_PATH="$EDEN_MODEL" SAMPLE_DIR=/path/to/cryo-crossing \
TAE_PATH=/path/to/lighttaew2_2.safetensors \
CUDA_VISIBLE_DEVICES=0 bash scripts/ed/run_5090.sh

MODEL_PATH="$EDEN_MODEL" SAMPLE_DIR=/path/to/cryo-crossing \
TAE_PATH=/path/to/lighttaew2_2.safetensors \
CUDA_VISIBLE_DEVICES=0 bash scripts/ed/run_hopper.sh
```

The scripts resolve the repository automatically and create temporary configurations for checkpoint-path overrides. Saved realtime JSON files remain unchanged. Profiling defaults to `0`; set `PROFILING_DEBUG_LEVEL=1` when diagnostic timing is needed.

## 4. Realtime Profiles and LightTAE

The two JSON files in [`configs/edenlive/realtime`](../../configs/edenlive/realtime/) use 1280×704 output, 24 FPS, four denoising steps, three latent frames per chunk, local attention 18, sink 6, FP KV storage, and synchronous LightTAE streaming decode.

Here 24 FPS sets video playback and recording sampling; generation throughput is measured separately.

| Profile | DiT attention: self / cross 1 / cross 2 | ProxyDINO |
| --- | --- | --- |
| `wan22_5b_ar_int8_convrot_5090.json` | SM120 cached / SM120 cached / Sage2 | INT8 ConvRot, Sage2 |
| `wan22_5b_ar_fp8_sgl_hopper.json` | SM90 fused / SM90 fused / SM90 fused | Original weights, Sage2 |

Select the profile matching the GPU architecture. The operators are explicitly named by `self_attn_1_type`, `cross_attn_1_type`, and `cross_attn_2_type`.

For the supplied Wan2.2 LightTAE checkpoint, decoding uses:

```json
{
  "use_tae": true,
  "tae_path": "/path/to/lighttaew2_2.safetensors",
  "need_scaled": true,
  "streaming_vae": true,
  "use_cuda_graph": true
}
```

The original Wan VAE still encodes the cover frame. `use_cuda_graph` controls graph use for both DiT and ProxyDINO; unsupported execution paths use their regular compiled/eager implementation.

## 5. Optimization Coverage

| Area | Optimization and scope |
| --- | --- |
| DiT linear kernels | FP8-SGL or INT8 ConvRot quantized linears; BF16 non-linear tensors. |
| QKV projections | Share input activation quantization; pack self-attention QKV into one GEMM where supported. Cross-attention K/V can share quantization. |
| FFN | Fuse GELU with activation quantization while preserving the supported backend's BF16 rounding boundary. |
| AR KV storage | Fixed sink plus recent-frame ring avoids shifting the entire retained history. |
| Attention / RoPE | Reuse historical K/V workspaces, persistent transposed V, and fixed-window RoPE data across steps. |
| Cross-attention | Cache projected text K/V; SM120 cached Sage2 also reuses prepared quantized K/V. |
| Conditioning | Cache fixed text/time projections and repeated Proxy feature interpolation within eligible requests. |
| KV commit | In the final layer's cache-only pass, compute required K/V and skip the unused attention/FFN/output suffix. |
| ProxyDINO | GPU bilinear preprocessing, fused Q/K RoPE and normalization, model-dtype residuals, dense batched attention, and optional shared INT8 QKV quantization. |
| Recording input | Decode sequentially and prefetch the next RGB uint8 chunk in a CPU worker. |
| Compile / warmup | Compile eligible blocks and warm their execution before timed requests; first startup includes compilation. |
| CUDA Graph | Reuse eligible fixed-shape DiT and bilinear ProxyDINO graphs to reduce launch overhead. |
| Decode / memory | Stream LightTAE chunks; release completed-request references and KV pools before assembling the final video. |

Shared projection and GELU/quant fusion wrappers automatically select supported `fp8-sgl` and `int8-convrot` paths. They fall back when shapes, dtype, GPU, adapters, offload, or parallel execution do not meet their requirements. INT8 sharing targets resident SM120 group-256 projections. INT8 GELU fusion requires BF16 input, hidden width 14336, at least three tokens, and an output width divisible by 16; FP8 GELU fusion accepts two-dimensional BF16 input with hidden width 1–16384. FFN fusion requires the supported compiled path with precision casts preserved (`use_compile=true`, default compile mode, `emulate_precision_casts=true`).

| Attention implementation | Persistent transposed V | Prepared quantized cross-KV | DiT stack CUDA Graph |
| --- | --- | --- | --- |
| SM120 cached Sage2 | Yes | Yes | Eligible for the resident single-GPU realtime profile |
| SM90 fused Sage2 | Yes | No | Falls back to compiled blocks; ProxyDINO Graph remains available |

Hopper currently lacks the prepared cross-KV interface required by the DiT stack graph. Quantization, SageAttention, ProxyDINO dtype changes, and LightTAE can affect output quality; evaluate with representative inputs. These profiles retain the specified denoising steps and attention context.
