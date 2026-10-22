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

Full conversion prepares all four DiTs and the T5/CLIP encoders, and preserves the DINO, VAE, tokenizer, and LightTAE assets. Set `model_path` in the launcher to the converted model root for all inference modes, including quantized inference.

## Inference

Edit `lightx2v_path` and `model_path` at the top of the launcher, then set its input image/video paths, prompt and output path, following the same layout as [`scripts/wan`](../wan). Run the Wan2.2 5B autoregressive model in BF16:

```bash
IMAGE_PATH=/path/to/cover.png VIDEO_PATH=/path/to/recording.mp4 bash scripts/edenlive/wan22_5b_few_step.sh
```

Use the corresponding launcher in the model table for other variants. Select a different configuration by changing its `--config_json` argument; base configurations, including AR prompt travel, are available in [`configs/edenlive`](../../configs/edenlive/). Edit inference options such as `--seed` directly in the launcher.

### Local cryo-crossing run

`wan22_5b_few_step.sh` uses this machine's converted BF16 weights, seed 2026092901, and a cryo-crossing prompt. Set `IMAGE_PATH` and `VIDEO_PATH` to the desired local inputs before running it. Its JSON enables compile, warmup, the shared CUDA Graph stack and fused RoPE/cache preparation.

```bash
IMAGE_PATH=/path/to/cover.png VIDEO_PATH=/path/to/recording.mp4 bash scripts/edenlive/wan22_5b_few_step.sh
```

The launcher saves its output under this checkout's `save_results`. Profiling level 1 exposes `RUN pipeline` timing and adds per-step synchronization; `Total Cost` also includes loading and warmup. For steady-state timing and exact latent/pixel comparisons, use the current [pipeline benchmark](../self_forcing/benchmark_edenlive_pipeline.py) with `--execution compiled_graph --requests 3` and set `PROFILING_DEBUG_LEVEL=0` explicitly; it saves full tensors.

## Real-Time Deployment

The hardware presets target **Wan2.2 5B AR at 1280×704 and 24 FPS**, with four denoising steps and three latent frames per chunk. Both use FP8 T5 resident on GPU, synchronous LightTAE streaming decode, compilation, and warmup.

| Hardware | DiT | Proxy encoder | Launcher |
| --- | --- | --- | --- |
| H100 / H200 | FP8-SGL | BF16 | [run_hopper.sh](run_hopper.sh) |
| RTX 5090 | INT8 ConvRot | INT8 ConvRot | [run_5090.sh](run_5090.sh) |

Configuration presets: [Hopper FP8-SGL](../../configs/edenlive/realtime/wan22_5b_ar_fp8_sgl_hopper.json) · [RTX 5090 INT8 ConvRot](../../configs/edenlive/realtime/wan22_5b_ar_int8_convrot_5090.json) · [SM120 FP8-F16 accumulation (pending device validation)](../../configs/edenlive/realtime/wan22_5b_ar_fp8_f16_accum_5090.json).

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

For **SM120 FP8-F16 accumulation**, create a separate reduced-range DiT checkpoint. The ordinary FP8-SGL checkpoint has a different quantization range and must not be reused:

```bash
python tools/convert/convert_edenlive.py \
  --source "$EDEN_MODEL/dit-5b/diffusion_pytorch_model.safetensors" \
  --output "$EDEN_MODEL/quantized" --output_name dit-5b-fp8-f16-accum \
  --quantize fp8-f16-accum --device cuda:0
```

Select `wan22_5b_ar_fp8_f16_accum_5090.json` in the launcher and set its checkpoint paths. This path has not yet been measured on SM120.

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

Set `lightx2v_path`, `model_path` and the input/output arguments in the selected launcher. Set the checkpoint paths in its JSON profile to the files prepared above.

On **H100/H200**:

```bash
bash scripts/edenlive/run_hopper.sh
```

On **RTX 5090**:

```bash
bash scripts/edenlive/run_5090.sh
```

The converted release includes the LightTAE checkpoint used in this example. The JSON field `tae_path` also accepts a fine-tuned `.safetensors` checkpoint. The original Wan VAE encodes the target first frame; LightTAE decodes the generated chunks.

Configure these fields in the JSON selected by `--config_json`; the launcher passes the JSON directly without generating a temporary configuration:

| JSON field | Checkpoint under the converted model directory |
| --- | --- |
| `dit_quantized_ckpt` | `quantized/dit-5b-fp8.safetensors` on Hopper; `quantized/dit-5b-int8-convrot.safetensors` for INT8 ConvRot; `quantized/dit-5b-fp8-f16-accum.safetensors` for the SM120 FP8-F16 profile |
| `t5_quantized_ckpt` | `quantized/t5-fp8.safetensors` |
| `proxy_quantized_ckpt` | `quantized/proxy-dino-int8-convrot.safetensors`, RTX 5090 only |
| `tae_path` | `decoder-5b/lighttaew2_2.pth` |

Replace the JSON's `/path/to/EdenLive-x2v` prefix with the actual model directory. The script's `model_path` does not rewrite checkpoint paths in the JSON. Set `t5_cpu_offload=true` in the JSON to reduce resident GPU memory; the performance results below use `false`. Set the output MP4 through the launcher's `--save_result_path` argument.

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

The earlier resident BF16 FA3/Sage2 validation used the shared DiT CUDA Graph path. Its 717-frame Sage2 cryo-crossing run matched original main latent and video bytes, with 275 whole-stack graph replays after warmup. The current FA3 SP2/offload result is recorded separately below. Hardware presets keep their compiled blocks and proxy Graphs; the Sage2 path uses the ordinary ordered RoPE workspace and has no specialized prepared-KV cache.

### Historical BF16 FlashAttention-3 validation (2026-10-19/20)

For the measured 5B AR path, use resident BF16 weights and native `flash_attn3` self/cross attention with `use_compile=true`, `use_cuda_graph=true`, and `warmup=true`. Keep `ar_config.kv_cache_scheme="rolling"`. This retains raw keys, the original logical-position RoPE, reused history workspaces, and the original dense FA3 reduction order while using the shared [autoregressive CUDA Graph stack](../../lightx2v/common/transformer_infer/autoregressive_cuda_graph.py).

The real cryo-crossing test used the first 249 frames at 832×480 / 24 FPS, including the original first three dynamic prompts. An independent archive of pre-change HEAD `d48a6173` averaged **9.347 s** over three requests; the final shared-graph version averaged **9.042 s**, 3.27% faster, with identical complete latent and uncompressed pixel tensors across all six requests. DiT event time decreased from 3208.50 to 2903.71 ms. The original BF16 FA3 path supported compilation but its manual graph guard did not enable DiT graphs; the new version replayed the fixed-window graph 80 times per request with no formal capture, recompile, or fallback. All three production warmup shapes were retained. Startup plus the first request measured 57.151 s before and 57.619 s after in the single startup samples, so this is a steady-request improvement, not a cold-start improvement. The [pipeline benchmark](../self_forcing/benchmark_edenlive_pipeline.py) measures this dense path.

Resident FA3 with interleaved FlashInfer RoPE automatically fuses history gather, RoPE and ordered workspace writes. Another kernel rotates current Q/K and writes the tail while retaining raw K/V for clean commit. History is prepared once per chunk and reused during denoising, preserving workspace and dense attention order. Other RoPE formats use ordinary workspace preparation. The fusion switch and Sage-specific transposed V and prepared/quantized cross-KV branches have been removed; ordinary text cross-KV caching remains.

A same-source fused/unfused comparison on 2026-10-20 used the real BF16 5B cryo-crossing case, 480×832 / 249 frames, all three acceleration features enabled, and three formal requests per setting on one H200. Mean DiT CUDA-event time decreased **2881.911 → 2846.562 ms (1.227%)**; complete pipeline time **8.948430 → 8.922167 s (0.293%)**. All three paired requests improved. Original main, the pre-fusion source and the fused source produced 32 byte-exact full latent/pixel comparisons; eager and compiled execution each matched original main in the same execution mode. Formal requests used 80 graph replays, with no new compilation, capture or fallback. The kernel matches FlashInfer's FP32 cosine FMA and BF16/FP16 rounding. History workspace remains **2.4101 GiB across 30 layers** in this case, so fusion reduces passes and launches rather than removing historical copies or workspace memory. Startup/warmup is excluded from these steady-request gains.

EdenLive uses [`RoPEKVWorkspace`](../../lightx2v/common/kvcache/attn/rope.py) for reusable logical history buffers; `prepare_history` / `update_rope` receive visible ranges and frequencies. SF obtains physical ring views from the rolling pool and keeps its attention backends independent of this workspace. Models retain their position policy and commit timing.

EdenLive clean commits automatically use a final K/V-only block when eligible. The capability check reads active adapter flags; the default zero `weight_diff` on RMSNorm does not disable this path.

EdenLive AR combines `EdenLiveTransformerInfer` with `WanCausalTransformerInfer`, which derives from `WanAutoregressiveTransformerInfer`. It shares KV-only compilation dispatch, request cleanup, and Graph lifetime with SF. EdenLive retains its RoPE, proxy, and scheduler flow; its runner shares warmup shape ordering and request cleanup. See [shared execution interfaces](../self_forcing/README.md#14-shared-execution-interfaces).

The 2026-10-20 shared execution cleanup enabled KV-only commits on the real 5B model. Two same-GPU groups reduce DiT time by about 1.3–1.9% and clean commits by 3.7–4.2%; a controlled reverse-order repeat reduces pipeline time by 1.4%, with output-transfer timing varying across groups. All 48 full latent/video byte comparisons have zero differences. Formal requests use 80 replays with no compilation, capture, or fallback. The separate clean Graph increases peak allocated memory by about 618 MiB.

## Performance

The table retains earlier hardware-preset measurements. The dated BF16 / FA3 validation above covers that earlier cache cleanup; current shared-execution results follow below.

**Wan2.2 5B AR · 1280×704 · 24 FPS · batch size 1 · synchronous LightTAE decode.** Both presets use four denoising steps, three latent frames per chunk, an attention window of 18 latent frames, and six sink frames.

| GPU | Generation FPS ↑ | Inference FPS ↑ | First-chunk latency ms ↓ | Inter-chunk gap P95 ms ↓ | RTF@fps24 ↓ | Peak allocated GiB ↓ | Peak reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA H200 | 36.12 | 35.95 | 358.19 | 351.20 | 0.6676 | 29.72 | 32.23 |
| NVIDIA RTX 5090 | 24.87 | 24.69 | 500.02 | 492.63 | 0.9722 | 28.38 | 28.95 |

- **Generation FPS** measures AR generation through the final decoded chunk ready on GPU. **Inference FPS** additionally includes request input preparation and encoding. Model loading, compilation, warmup, final video concatenation, and MP4 saving are excluded.
- **First-chunk latency** starts at the request boundary. **Inter-chunk gap P95** measures the interval between successive decoded chunks; the H200 value pools all three requests. FPS is total frames divided by total measured time.
- **RTF@fps24** is inference time divided by playback duration at 24 FPS. Values below 1 indicate generation faster than playback.
- **Peak memory** covers the full request, including final video concatenation and saving.

## Current AR cache and validation scope

FA3 and standard `sage_attn2` share `RoPERingAttention`: raw K/V ring → original logical-position RoPE and reusable ordered K/V workspace → dense attention. EdenLive does not use the removed segmented Sage/LSE cache. It avoids rebuilding history on each denoising step while retaining history copies and workspace memory. Resident layers retain their workspaces; KV offload shares one workspace and refreshes it after each layer load. `use_compile`, `use_cuda_graph` and `warmup` can run together for resident, block weight/KV offload, SP and TP. Current warmup uses the configured target resolution only.

The 2026-10-21 FA3 SP2 + weight block/KV offload cryo-crossing case generated 717 frames at 1280×704. Full latent and float-video SHA matched untouched main in the same mode. The formal request had **0 captures and 9000 per-block replays**; pipeline time decreased 132.736 → 127.807 s and DiT 91.046 → 89.410 s. This is one formal request per version after warmup, not a stable throughput study.

The separate 85-frame TP2 resident check also matched full latent/video SHA, against an independent main oracle that only repairs the original runner's TP cache-head count. Untouched main fails before inference in this mode;. Earlier Sage2 resident 717-frame validation used 275 whole-stack replays; it is a different backend/mode and does not certify every Sage2 offload/distributed combination. RTX 40/50 have no new hardware acceptance for these cache changes; the earlier preset table is historical. The installed FA2 runtime's uncached long-sequence NaNs remain a validation limitation. See [backend notes](../self_forcing/SAGE_ATTENTION.md).
