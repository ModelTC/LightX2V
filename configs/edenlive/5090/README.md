# EdenLive Wan2.2 5B AR on RTX 5090

These local presets use the converted EdenLive model in
`/mnt/devsft_afs_2/gushiqiao/vast/EdenLive-x2v` and the separately downloaded
`/mnt/devsft_afs_2/gushiqiao/vast/lighttaew2_2.safetensors`.

The original FP8 benchmark presets retain 1280×704, 16 FPS, four denoising steps, an 18 latent-frame
attention window, FP8 DiT, BF16 sensitive layers, compile, warmup, and synchronous
LightTAE decoding. T5 block offload is enabled; DiT and KV remain resident.

- `wan22_5b_ar_fp8_lighttae_sync.json`: standard SM120 SageAttention 2 baseline.
- `wan22_5b_ar_fp8_lighttae_cached.json`: identical SageAttention quantization and
  attention kernels, with persistent transposed V storage for self-attention
  and per-prompt quantized text K/V for cross-attention.
- `wan22_5b_ar_fp8_lighttae_sp2.json` / `sp4.json`: sequence parallel deployment
  profiles; they split each chunk's DiT tokens across ranks, retaining the full
  logical KV history on each rank. They do not represent independent requests.

Run from the repository root:

```bash
bash scripts/edenlive/run_5090.sh
GPU_COUNT=2 bash scripts/edenlive/run_5090.sh
GPU_COUNT=4 bash scripts/edenlive/run_5090.sh
```

`CUDA_VISIBLE_DEVICES` can select different cards. Extra CLI arguments override
request fields, for example `--num_frames 97` or `--save_result_path /path/out.mp4`.
The default example is cryo-crossing, seed 2026092901; videos are saved under
`save_results/edenlive_5090_<N>gpu_16fps.mp4`.

The cached backend requires SM120, head dimension 64 or 128, and the tested
SageAttention private CUDA kernel interfaces. Tested environment: PyTorch
2.11.0+cu130, Triton 3.6.0, SageAttention 2.2.0, FlashInfer 0.6.12, NVIDIA driver
580.95.05. It keeps standard SM120 per-warp Q / per-block K INT8 quantization and
FP8 V scaling with scale_max=2.25. Self-attention recomputes V quantization on
each attention call and reuses the BF16 transposed layout. Raw history KV
contents are unchanged.

The single-card cached preset selects `sage_attn2_sm120_cached` for both
self-attention and text cross-attention, and enables
`cache_cross_attn_quantized_kv`. Text K/V are projected, centered, quantized and
transposed once per layer, prompt and CFG branch, then reused across denoising
steps and chunks. Prompt changes refresh the cache. Request completion and
failure clear it. Segmented text prompts retain per-segment quantization through
the ordinary attention path. Quantized text caching currently requires a single
card, resident DiT/KV and the default compile backend.

The standard `sync.json` baseline and SP presets retain their earlier settings.
`scripts/edenlive/wan22_5b_few_step.sh` also selects the optimized single-card
preset and disables profiling synchronizations.

A same-card comparison on GPU1 with 477 frames, one discarded request and three
measured requests improved 19.89 to 20.10 inference FPS (+1.05%); mean chunk
interval decreased 594.64 to 588.44 ms. Peak allocated memory decreased from
27.37 to 27.28 GiB. The full latent and MP4 are bit-identical. See
`/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090_cross_cache/REPORT.txt`.

Local timing logs, metrics, parity checks, and videos are in
`/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090/`.

Projection and FFN experiments are available as independent single-card flags:

- `share_qkv_activation_quant=true`: quantize the common self-attention input
  once, then run the original three FP8 GEMMs. Text K/V also share one activation
  quantization when their prompt cache is refreshed.
- `pack_qkv=true`: additionally concatenate loaded Q/K/V weights, scales and
  biases for one GEMM, then split its output. This implies shared activation
  quantization. The original operator weights remain available for fallback;
  the packed copies use about 812 MiB for the 30-layer 5B model. No checkpoint
  conversion or requantization is needed.
- `fuse_ffn_gelu_quant=true`: fuse compiled BF16 GELU(tanh) and SGL per-token
  FP8 quantization before the FFN output GEMM, retaining the original BF16
  rounding and SGL scale rules. This currently targets the default compiler,
  resident single-card FP8-SGL path with precision-cast emulation enabled.

The default cached preset enables shared activation quantization without
packing, avoiding duplicate weights. Same-card full-video three-trial means were
20.15 FPS for the previous baseline, 20.20 for shared quantization, and 20.17 for
packed QKV. These are small differences close to run-to-run variation; they do
not establish a material end-to-end speedup. Packed QKV and FFN fusion remain
disabled by default. The shared+FFN and packed+FFN means were 20.13 and 20.06 FPS.
All five complete latent tensors and MP4 bytes match exactly, with 95 regression
tests passing. The optional `wan22_5b_ar_fp8_lighttae_packqkv.json` enables packed projections:

```bash
bash scripts/edenlive/run_5090.sh \
  --config_json configs/edenlive/5090/wan22_5b_ar_fp8_lighttae_packqkv.json \
  --save_result_path save_results/edenlive_5090_packqkv_16fps.mp4
```

Packed Q/K normalization uses the existing compiled FlashInfer kernel's row
stride support. Transposed-V updates also honor the source strides, avoiding
materialization of packed output slices. Unsupported projection types,
CPU/lazy offload, TP/SP and adapter/diff changes use the existing projection
path. Weight reloads rebuild the groups.

Profiler counts confirm self QKV activation quantization changes from 3 calls
per layer/pass to 1; packing also changes FP8 GEMM calls from 3 to 1, including
under `torch.compile`. Full measurements and exact-output checks are in
`/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090_packqkv/REPORT.txt`.

The optional `wan22_5b_ar_fp8_lighttae_sla_sage2_cached.json` meets a **chunk
P95 below 500 ms** on the fixed cryo-crossing benchmark. It keeps 1280x704,
16 FPS, the original FP8-SGL weights, four denoising steps, the 18-latent-frame
attention window and six sink frames. It uses SLA with 80% block sparsity,
SM120 sparse SageAttention, persistent transposed V, whole-stack CUDA Graphs,
shared K centering, and overlapping KV rerun with LightTAE decoding.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/edenlive/run_5090.sh \
  --config_json configs/edenlive/5090/wan22_5b_ar_fp8_lighttae_sla_sage2_cached.json \
  --save_result_path save_results/edenlive_5090_sla_16fps.mp4
```

On the same physical GPU1, three measured 477-frame / 40-chunk requests, after
warmup and one discarded request, produced P95 values of **494.18, 494.80 and
497.03 ms**. Mean inference throughput is **24.01 FPS**, mean chunk interval
**489.59 ms**, and peak allocated memory **28.40 GiB**. The merged 117 intervals
have P95 **496.29 ms**; the largest interval is **523.23 ms**. This meets the
P95 target for the measured workload; it does not guarantee every chunk or
every other prompt stays below 500 ms. Formal trials have no compile
recompilations and reuse one captured DiT graph.

**SLA is approximate attention.** The sparse result visibly differs from dense
attention, including foreground object shape and scene geometry. These timings
are not a quality-equivalent dense comparison. Engineering changes to the same
SLA Sage2 path preserve the full latent and MP4 bytes, including cached V and
KV/decoder overlap. The original dense cached preset remains the default.

`wan22_5b_ar_fp8_lighttae_sla_triton.json` tests the requested
`sla_mask_generator` + `sla_triton_operator` configuration using the existing
AR kernel. That kernel now handles tail padding safely and selects 8 warps /
2 stages for SM120, D128, 128x128 blocks; its three full-video P95 values were
511.84, 512.67 and 514.67 ms. The Sage2 cached preset above is the measured
choice for the 500 ms target. Other GPUs retain the previous launch parameters.

SLA routing directly passes its LUT to the operator, avoiding repeated top-k
and CPU synchronization. Centered K pooling is fused while preserving BF16
subtraction rounding; pooled K, masks and LUTs match the original router. The
complete SLA operator is a custom op so varying KV lengths do not create new
compile specializations. CUDA Graphs stage x, timestep modulation and text KV,
retain history/RoPE storage across requests, and keep cursor updates outside
the graph. Weight reloads invalidate captures; unsupported/offloaded/parallel
or segmented-prompt paths retain ordinary execution.

The graph design was informed by the official
[FlashDreams implementation](https://github.com/NVIDIA/flashdreams/blob/fe48cfaec3ddeac500acf47a1a44b6939c6cb07a/flashdreams/flashdreams/infra/cuda_graph.py).
This is a LightX2V implementation, with no FlashDreams runtime dependency.
Validation: **225 passed, 3 SM90 tests skipped, 1 FA3 test deselected**.
Benchmark commands, raw metrics, source hashes, videos and a visual comparison
are in `/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090_chunk500/`.

## Dense INT8 ConvRot and ProxyDINO (24 FPS)

The following single-card presets use 1280×704, 24 FPS, four steps, the original
18-frame attention context and six sink frames, resident INT8 ConvRot DiT,
LightTAE and **dense attention without SLA**:

- `wan22_5b_ar_int8_convrot_lighttae_dense.json` keeps the original ProxyDINO
  weights, FP32 residuals and Torch SDPA. It enables ConvRot shared/packed QKV,
  DiT CUDA Graphs, precision-preserving GELU/rotation/quantization fusion,
  request-local text/timestep projections and per-chunk proxy interpolation
  caching, overlapping KV commit with decoding, and the commit-only final-layer
  K/V path. These engineering changes preserve the complete latent and MP4
  bytes of the original user-supplied INT8 ConvRot configuration.
- `wan22_5b_ar_int8_convrot_proxy_int8_lighttae.json` additionally quantizes
  ProxyDINO and uses BF16 residuals, embeddings, LayerScale, norms and output,
  SageAttention 2, shared QKV activation quantization, fused Q/K RoPE and fused
  output normalization. **This profile changes numerical results.**

Convert ProxyDINO separately from DiT; the source is preserved and an existing
destination is rejected unless `--overwrite` is specified:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/convert/convert_edenlive_proxy.py \
  --source ../EdenLive-x2v/vision-encoder/projection.safetensors \
  --output ../EdenLive-x2v/quantized/proxy-dino-int8-convrot.safetensors \
  --device cuda:0 --group-size 256
```

The converted file has already been generated locally. It contains 145 INT8
linear weights with per-output FP32 scales and group size 256, 121 BF16 linear
biases, and the original 153 non-linear tensors. Size: 298 MiB. Patch convolution
remains BF16. No installed package is modified.

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/edenlive/run_5090.sh \
  --config_json configs/edenlive/5090/wan22_5b_ar_int8_convrot_proxy_int8_lighttae.json \
  --num_frames 477 \
  --save_result_path save_results/edenlive_5090_proxy_int8_24fps.mp4
```

ProxyDINO-only measurements use three 704×1280 RGB frames per chunk, resized
internally to 480×832 (1565 tokens), including preprocessing, all 24 transformer
blocks, projection and output layout. Three graph-trial means on GPU3:

| ProxyDINO configuration | ms/chunk |
|---|---:|
| Original weights / Torch SDPA | 25.03 |
| Original weights / FA2 | 25.07 |
| Original weights / Sage2 | 23.72 |
| ConvRot INT8 / original precision / Torch SDPA | 18.09 |
| ConvRot INT8 / BF16 / Sage2 / shared QKV / fused ops | **11.64** |

The fastest encoder saves 13.39 ms/chunk (53.5%, 2.15×). Its profile contains
145 native CUTLASS INT8 GEMMs, 24 shared QKV quantizations, 24 fused RoPE calls,
one fused normalization and zero Torch SDPA calls. Dtype copies decrease from
149 to three; arithmetic inside fused RoPE/normalization still uses FP32.
The removed `images.float()` was already a no-op on FP32 lookup-table pixels;
the speedup comes from eliminating materialized casts and intermediate tensors.

On the fixed random three-frame test, the fastest features have relative L2
error 0.01596 and mean cosine 0.999873 against the original encoder. These are
feature comparisons, not a video-quality score. Fused RoPE preserves its
original arithmetic and BF16 output exactly; ConvRot quantization, Sage2 and
the BF16 precision mode introduce the measured differences. Keeping FP32
residuals with quantization/Sage2/fused RoPE costs 12.52 ms/chunk, and its fused
and unfused features are byte-identical.

Proxy flags are independent: `proxy_quantized_ckpt`, `proxy_attn_type`,
`proxy_residual_dtype`, `proxy_fuse_ops`, `proxy_share_qkv_activation_quant`.
Omitting them preserves the original encoder. Weight reloads and attention
backend changes clear Proxy CUDA Graphs; reloads rebuild shared projection
groups. The DiT FFN fusion builds its small SM120 CUDA quantizer once through
PyTorch's extension cache; unavailable compiler/backend support retains the
original linear path.

Full measurements, profiler traces, source hashes and feature comparisons are
in `/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090_proxy_int8/`;
full-video measurements are in
`/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090_convrot_dense/`.

On physical GPU1, 477 frames / 40 chunks, one discarded request and three
measured requests, the supplied ConvRot configuration improves from 22.78 to
**24.54 inference FPS**. Mean chunk interval changes from 518.89 to **480.93 ms**;
mean per-request P95 changes from 529.94 to **491.16 ms**. The three optimized
P95 values are **488.29, 491.49 and 493.69 ms**, all below 500 ms. Merged P95
across 117 intervals is 493.26 ms, maximum 493.87 ms, first chunk mean 683.60 ms,
peak allocated memory 28.99 GiB. In the full pipeline the original encoder
costs 25.49 ms/chunk versus 11.82 ms/chunk for the optimized encoder.

DiT-only engineering with the original proxy encoder produces 23.83 FPS and
P95 505.43 ms, preserving the full latent and MP4 bytes. The quantized-proxy
profile produces different latent/video contents, with visible hand, weapon
and scene-detail changes in the checked frames; it has not had a systematic
quality evaluation. Compare `visual_comparison/comparison.jpg` and both
complete videos in the full-video artifact directory. Formal trials reuse one
DiT graph and have no compile recompilations. Validation: 491 passed, three
SM90 tests skipped, one FA3 test deselected; the separate FA3-only quantized
model integration file is not run on SM120.

The current proxy-INT8 JSON has `ar_config.overlap_kv_vae=false`. A separate
2026-10-17 GPU1 run measures that exact configuration at 24 FPS, 477 frames /
40 chunks, one discarded request followed by three formal requests:
**Generation FPS 24.64**, Inference FPS **24.25**, first chunk **683.58 ms**,
mean interval **486.84 ms**, mean per-request P95 **496.89 ms**, decoder CUDA
time **13.90 ms/chunk**, RTF **0.9897**, peak allocated **28.99 GiB**.
The three P95 values are **494.34 / 497.01 / 499.33 ms**. Full latents and MP4
bytes match the earlier overlap-enabled result exactly. The earlier 24.54 FPS /
491.16 ms P95 result above uses `overlap_kv_vae=true`.

Generation FPS now has an explicit timer: all decoded frames divided by time
from `_run_diffusion` entry after CUDA synchronization to the final decoded
chunk callback. This includes the first chunk, per-chunk recording/DINO, DiT,
KV commit and decoding; initial input encoding and scheduler preparation are
excluded. Inference FPS also includes initial request/input preparation. This
local definition is documented rather than assumed identical to the paper's
undisclosed timing boundaries. `post_first_chunk_fps` remains a separate metric.
Raw measurements, timing instrumentation, output parity and reproduction are in
`/mnt/devsft_afs_2/gushiqiao/vast/edenlive_benchmarks/5090_convrot_sync_generation_24fps_20261017/`.
