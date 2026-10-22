
# Self-Forcing (Wan2.1-T2V-1.3B)

This document shows how to run the Self-Forcing acceleration for `Wan2.1-T2V-1.3B` in LightX2V.

## Prepare the Environment

Please refer to [Quick Start](../../docs/EN/source/getting_started/quickstart.md)

## Download

### 1. BF16 original model (Wan2.1-T2V-1.3B)

Download the original BF16 model from the Wan-AI Hugging Face repository:

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```

### 2. Self-Forcing checkpoint

Download the Self-Forcing checkpoint (`self_forcing_dmd.pt`):

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### 3. Quantized models (FP8 / NVFP4)

Download the quantized models from the LightX2V HuggingFace collection:

https://huggingface.co/collections/lightx2v/ar-lightx2v

## Start Running

We provide three scripts:

1. `scripts/self_forcing/run_wan_t2v_sf.sh`: BF16 original model + Self-Forcing checkpoint
2. `scripts/self_forcing/run_wan_t2v_sf_fp8.sh`: FP8 quantized model
3. `scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh`: NVFP4 quantized model

Before running any script below, you need to fill in the path variables in the script.

### 1) BF16 + Self-Forcing checkpoint

Edit and run:

```
bash scripts/self_forcing/run_wan_t2v_sf.sh
```

Key fields in `run_wan_t2v_sf.sh`:

- `lightx2v_path`: your LightX2V repo path
- `model_path`: path to the original BF16 `Wan2.1-T2V-1.3B` directory
- `dit_original_ckpt` in the config file: path to `self_forcing_dmd.pt` (downloaded from `gdhe17/Self-Forcing`)

The script runs:

- `python -m lightx2v.infer`
- `--model_cls wan2.1_sf`: use the Wan2.1 Self-Forcing pipeline
- `--task t2v`: text-to-video
- `--config_json configs/self_forcing/wan_t2v_sf.json`: Self-Forcing runtime config

Note: In `configs/self_forcing/wan_t2v_sf.json`, `enable_cfg` is set to `false` by default.

`ar_config.kv_cache_scheme` defaults to `"rolling"`, the sink-and-ring cache policy. Use `"rolling"` in configs that previously specified `"fp"`; this setting does not select the model or cache tensor dtype.

### 2) FP8 quantized model

Edit and run:

```
bash scripts/self_forcing/run_wan_t2v_sf_fp8.sh
```

Additional notes for FP8:

- The script uses `configs/self_forcing/wan_t2v_sf_fp8.json`.
- You must change `dit_quantized_ckpt` in that config file to the **absolute path** of the downloaded FP8 weights (from the HuggingFace collection).
- `dit_quant_scheme` is `fp8-vllm`.

### 3) NVFP4 quantized model

Edit and run:

```
bash scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh
```

Additional notes for NVFP4:

- The script uses `configs/self_forcing/wan_t2v_sf_nvfp4.json`.
- You must change `dit_quantized_ckpt` in that config file to the **absolute path** of the downloaded NVFP4 weights (from the HuggingFace collection).
- `dit_quant_scheme` is `nvfp4`.

### 4) FA2/FA3 physical ring views and Sage2 segments

Use `configs/self_forcing/wan_t2v_sf_ring_fa3.json`, fill in `dit_original_ckpt`, and pass it to `--config_json`. The example uses FA3 self/cross attention, a 12-frame local window, a 3-frame sink and 3-frame chunks; these sizes count **latent frames**.

The persistent ring stores keys after RoPE. A GPU slot kernel writes current K/V. When attention covers the valid physical ring, dense FA2/FA3 reads a view without an ordered workspace, history gather or KV concatenation. Jointly permuting K/V preserves unmasked attention mathematically but changes floating-point reduction order. A narrower attention window gathers its live spans into reusable K/V buffers; that path is not zero-copy. Current KV writes and metadata uploads remain in either case.

`use_cuda_graph=true` captures variants for the first chunk, growing/full windows, reference prefill and clean commits. Slot and RoPE frame metadata use stable GPU addresses, allowing compatible variants to replay across wraps and requests. Resident execution captures the block stack; weight block offload or KV staging captures each block after data arrives. All of these paths can run `use_compile=true` at the same time. SP/TP, step cache banks, image cross-attention and narrow windows no longer select a separate unoptimized CUDA executor.

Validation is backend- and configuration-specific. The current H200 multi-GPU/offload real-model records use FA3; the SF Sage2 real videos in section 16 are earlier resident tests. Sage2 cache/Graph synthetic coverage does not certify every full-model SP/TP/offload combination. The installed FA2 runtime also produces NaNs in uncached long-sequence baselines, so the current Flash multi-GPU evidence uses FA3. RTX 40/50 hardware remains untested. See section 19 for current scope; sections 9–16 preserve dated experiments.

`WanSFModel` inherits `WanAutoregressiveModel`. `WanSFTransformerInfer`, LingbotFast, MG4 and Audio AR share `WanCausalTransformerInfer`, which derives from `WanAutoregressiveTransformerInfer`; SF retains its input projection hook, while LingbotFast retains camera injection. Shared code manages ring planning, cross-attention KV lifetime and graph execution. Models retain their own position encoding and clean-commit behavior. Slot writes live in [`common/kvcache/kernel.py`](../../lightx2v/common/kvcache/kernel.py); [`RollingKVOffload`](../../lightx2v/common/kvcache/offload.py) separately manages CPU pinned buffers, staging, prefetch and writeback.

The full DiT benchmark accepts `--mode eager` or `--mode graph`:

```bash
PYTHONPATH=. python scripts/self_forcing/benchmark_ring_cache.py \
    --model-path /path/to/Wan2.1-T2V-1.3B \
    --checkpoint /path/to/self_forcing_dmd.pt \
    --mode graph --compile --warmup \
    --reference-latents /path/to/old/latents.pt \
    --output-dir /path/to/benchmark/graph
```

The benchmark defaults to 480×832, 12 chunks and three repeats. Each chunk includes four denoising forwards and one clean rerun. Reference latents measure full first-rollout error before formal repeats; physical permutation does not require bitwise equality. Startup, warmup, formal DiT work and optional decoding are recorded separately.

### 5) Startup warmup

Set `"warmup": true` in the startup JSON. Warmup runs once after runner initialization, before the first request, using real text encoding, all denoising steps, clean cache commits and cached VAE decoding. SF and LingbotFast now warm the configured target resolution only. With the example window, five chunks cover the first chunk, growing window, full window and wraparound, so formal requests can reuse the warmed cache and Graph layouts.

Warmup resets valid KV lengths and ring positions, clears scheduler state, latents, inputs and the generator, and restores CPU/CUDA RNG state. Compatible cache allocations, compiled callables and Graph bindings remain reusable. The same lifecycle supports weight block offload, KV offload and SP/TP; actual backend/model acceptance is listed separately below. Seko AR still rejects `lazy_load` warmup, and MG4's streaming input encoder requires a resident VAE. Complete-model unload/reload invalidates Graphs because GPU weight addresses change.

For the full DiT benchmark, add `--warmup` to `scripts/self_forcing/benchmark_ring_cache.py`. The report separates startup `warmup_seconds`, the first formal request and repeated requests. The product CLI reads the JSON setting; it does not have a `--warmup` flag.

### 6) torch.compile with CUDA Graph

`configs/self_forcing/wan_t2v_sf_ring_fa3_compile.json` enables `warmup`, `use_compile` and `use_cuda_graph`. Compilation alone is supported by disabling `use_cuda_graph`; no additional cache-order flag is needed.

The compilation boundary is one transformer block. Resident SF keeps callables by layer; staging execution binds the reusable GPU block buffers. Ring planning, metadata uploads and CPU/GPU transfers stay outside compiled computation. FA2/FA3 use physical views or a reusable narrow-window buffer; Sage2 binds persistent quantized segments for the full-ring path. [`AutoregressiveCUDAGraphStack`](../../lightx2v/common/transformer_infer/autoregressive_cuda_graph.py) captures growing/full windows and prefill as distinct variants, either for the whole stack or for staged blocks.

The Wan causal base enables `emulate_precision_casts=true` to preserve BF16 intermediate rounding. Dynamic RoPE, slot writes and native LayerNorm retain their kernel boundaries. Manual CUDA Graph sets `triton.cudagraphs=false` so Inductor does not also manage graphs. The production block API does not force `fullgraph=True`; actual compiler counters and capture/replay counts establish whether compilation and graphs ran. The real-weight comparisons below confirm no formal recompilation or recapture for their tested configurations.

SF keeps raw KV exclusively in [`RollingKVCachePool`](../../lightx2v/common/kvcache/rolling.py); `prepare_attention` exposes stable physical views and slot metadata. `RingAttention` writes slots and dispatches FA2/FA3; `SageRingAttention` maintains quantized segments over the same storage. EdenLive uses the independent [`RoPEKVWorkspace`](../../lightx2v/common/kvcache/attn/rope.py) for logical post-RoPE history, with no workspace inheritance in SF attention backends.

### 7) Autoregressive KV methods and current scope

The reusable optimization depends on cache lifetime and position encoding:

| Family / model | Current implementation and optimization scope |
| --- | --- |
| RoPE already applied to cached K: Self-Forcing / realtime-video 14B / LingbotFast | Persistent sink-and-ring storage; a slot kernel writes current KV and FA2/FA3 read physical views, while Sage2 uses persistent quantized segments and LSE merging. No history gather, cat, or ordered workspace for the complete visible ring; narrow windows use reusable gather buffers. Permutation and quantization differences require video/numerical validation. |
| Raw K with local position re-anchoring: EdenLive | Keep raw K/V in the ring. Fuse history gather, RoPE and ordered workspace writes once per chunk, reuse that history through denoising, and update the current tail each forward. Resident layers retain their own history workspace; KV offload shares one workspace and refreshes history after each layer load. |
| Raw K: Wan Audio / Seko AR | Step/hybrid rolling pools retain raw K. `PhysicalRoPERingAttention` rotates K into a physical-order workspace while V remains a ring view. The 2-step global/capped and 4-step local position policies remain distinct, including SP semantics. |
| Raw K: MiniMax H3 causal | Retain step/layer banks and full-history RoPE, including condition prefixes, first-chunk sinks and joint video/audio attention. |
| FIFO: LingbotVA | Retain the original live-token gather and dense attention. |
| Contiguous history plus temporary action tokens: DreamZero | Retain original history movement and KV packing; only video history is committed. |
| Per-chunk memory-window reconstruction: WorldPlay AR / AR Distill | Retain original text prefill, context rebuild and `[text, context, current]` concatenation, including normal/ProPE streams. |

SF FA2/FA3 and EdenLive use dense attention; full-ring SF Sage2 batches physical segments and merges LSE. EdenLive retains logical window order, while Audio/Seko uses a physical-order K workspace. H3, LingbotVA, DreamZero and WorldPlay retain their separate position policies. Shared compile/Graph execution does not make these cache layouts interchangeable. Real-model precision and latency claims apply only to the configurations recorded in each report.

### 8) Ordinary cross-attention K/V caching (P1)

`ar_config.use_cross_attn_kv_cache` defaults to `true`; set it to `false` for an uncached comparison. This stores ordinary projected K/V, including K normalization required by the path, independently of Sage quantization or prepared-KV. Shared [`ConditionKVCache`](../../lightx2v/common/kvcache/condition.py) separates sources, CFG/named branches, modalities, layers and variants. Changes to source storage range, version, dtype or device invalidate entries; weight/LoRA updates must clear entries through the model's cache-reset lifecycle. Inference tensors without a trackable mutation version are not reused.

This optional cache applies to Self-Forcing/realtime-video, LingbotFast, Wan Audio AR and EdenLive. Dynamic audio adapter conditions are still computed on each call. H3 uses a joint-attention condition prefix rather than separate cross-attention KV. LingbotVA, DreamZero and WorldPlay retain their original condition/text cache behavior after restoration.

For the retained paths, `AutoregressiveModelMixin` invalidates state on weight changes and `AutoregressiveTransformerInferMixin` manages condition KV lifetime. Self-Forcing implements `WanAutoregressiveModel`; its inference and LingbotFast share `WanCausalTransformerInfer`. Audio and EdenLive retain their own attention and scheduling calculations.

[`benchmark_cross_kv_cache.py`](benchmark_cross_kv_cache.py) needs no weight files and defaults to the SF shape:

```bash
PYTHONPATH=. python scripts/self_forcing/benchmark_cross_kv_cache.py \
    --output-dir /path/to/benchmark/cross-kv
```

Measurements on 2026-10-19 used H200, BF16, FA3, seed 42, random weights/inputs and actual MM, RMSNorm, FA3, `ConditionKVCache` and Wan KV helpers. Each request runs 12 chunks with four denoising steps and one clean rerun per chunk. Three warmup requests precede three rounds of 30 requests; first-use KV creation is included. Values below are CUDA-event medians for **one layer's Q/K/V/O projections, Q/K RMSNorm and FA3 subpath**, not a full DiT. These are operator microbenchmarks for the retained Wan shapes; they do not establish support or full-model speedups for other architectures.

| Synthetic representative shape | Uncached | Project every step in chunk 0, then reuse | Once per condition/variant | Reduction vs uncached |
| --- | ---: | ---: | ---: | ---: |
| Wan SF 1536 | 13.911 ms | 10.212 ms | 10.399 ms | 25.25% |
| Wan 3072 (Eden5B shape) | 14.636 ms | 11.838 ms | 11.648 ms | 20.41% |
| Wan 5120 text | 57.078 ms | 52.042 ms | 51.684 ms | 9.45% |
| Wan 5120 image, two CFG branches | 107.167 ms | 100.201 ms | 100.264 ms | 6.44% |

All measured outputs were bitwise identical to uncached outputs. For one static branch, K/V projection pairs per layer decrease from 60 to 1. SF already reused KV after the first chunk, so its additional benefit is insignificant here, and some cases are slightly slower. Reductions relative to fully uncached execution are not additional SF gains and cannot be extrapolated to full DiT latency. The middle column is a policy reference, not a replay of an old complete model.

Ordinary KV capacity is `2 × context_tokens × heads × head_dim × element_bytes × layers × branches × variants`, excluding weights, self KV and temporary tensors. SF's existing text KV allocation is not entirely new overhead.

Sections 9–14 retain historical validation from before the physical-view change. Their byte-equality and ordered-scratch descriptions apply to those versions. Section 15 documents the current physical-view results.

### 9) Exact acceptance: Self-Forcing 1.3B

On 2026-10-19, the original X2V source and final logical-order compile + warmup + CUDA Graph path ran sequentially on the same **H200 GPU1**, BF16/FA3, all 30 layers, seed 42 and the same prompt. Both use a 12/3/3 latent-frame window/sink/chunk and generate 165 frames at 832×480, 16fps (10.3125 seconds). Each has one complete first request followed by three measured repeats; the old runner has no startup warmup, so that first request warms its formal signatures.

| Implementation | 14-chunk request median | Full-window chunk median |
| --- | ---: | ---: |
| Original X2V, HEAD source | 3.84615 s | 290.417 ms |
| Final logical-order compile + warmup + CUDA Graph | 3.69740 s | 278.172 ms |

Warm formal DiT request latency decreases **3.87%** and the full-window chunk median decreases **4.22%**. All 42 latent frames match the original source **byte for byte**, with max abs/RMSE 0; repeated requests are also identical. Startup retains 30 compiled block callables and two graph captures; every formal request adds 55 replays with empty compile-counter deltas and no recapture. Warmup costs 9.499 s; startup and first-request costs are separate, so this is not a claim of faster one-shot cold start. Timings exclude text/VAE/MP4 work.


### 10) Exact acceptance: Krea Realtime 14B

The actual `/data/nvme4/models/realtime-video/krea-realtime-video-14b.safetensors` is a T2V model with **40 layers, width 5120, 40×128 heads and 14.288B parameters**, not the 1.3B model. Its existing `model.`-prefixed checkpoint loads directly; stock Wan tokenizer/T5/VAE are shared assets. The old-source snapshot matches all 1,775 tracked source/config files at HEAD `d48a6173bc15a88b06cdcfb211183b43d8038846`.

Both implementations ran on the same idle **H200 GPU5**, BF16/FA3, seed 42, identical prompt, 832×480, 14 chunks and 12/3/3 latent-frame cache settings. This controls the attention backend and visible set; it does not compare against the example's different Sage2/unlimited-window defaults.

| Implementation | 14-chunk request median | Full-window chunk median |
| --- | ---: | ---: |
| Original X2V, verified HEAD snapshot | 21.84255 s | 1626.146 ms |
| Final logical-order compile + warmup + CUDA Graph | 21.33125 s | 1588.597 ms |

Warm formal DiT latency decreases **2.34% / 2.31%**, respectively. All three final repeats are faster than the fastest of the three old repeats. Complete latents match HEAD byte for byte, max abs/RMSE 0; full MP4 files also have identical SHA256. There are 40 compiled block callables, four Dynamo graphs, two fixed captures and 55 additional replays per formal request, with no formal recompilation. A single 365.625 MiB logical K/V scratch pair is shared across layers; this path still copies history and is not zero-copy.

Startup including warmup costs 47.041 s, of which warmup is 21.566 s; the first request is 21.223 s. Peak formal allocated/reserved memory is 52.954/58.570 GiB. VAE decode and saving both full/preview MP4s cost another 3.543/2.001 s and are excluded from the table. Startup costs are not included in the warm-latency acceptance.


These 2026-10-19 real-weight results are historical acceptance records. The shared-cache regression below checks the later SF integration separately.


### 11) Exact acceptance: Lingbot World Fast cam

The 2026-10-19 test used the actual 16-shard `lingbot-world-base-cam/lingbot_world_fast` checkpoint, 40 layers, resident BF16, FA3, 480×832 and 14 chunks / 165 frames. Inputs were the official example 03 photo, camera poses and intrinsics, with seed 42, a 21-frame local window and 3-frame sink. Compile, production warmup and CUDA Graph were all active.

All 70 noisy inputs, 70 noise predictions, complete latents and all decoded video frames matched original HEAD `d48a6173` bitwise. All three repeated final latents matched as well. On the same H200 GPU6, steady pipeline time decreased **41.500 → 35.844 s (13.63%)**, and GPU DiT **30.798 → 29.898 s (2.92%)**. Formal requests added 40 graph replays each, with two retained warmup captures and unchanged compilation counters. Pipeline improvement includes replacing unconditional per-chunk allocator clearing with the existing pressure-aware policy; it is not attributable only to the KV kernel.

Both sides used an external test adapter to force the exact 480×832 VAE input: the ordinary CLI has an existing resize rounding issue that can produce height 464. This acceptance does not claim that issue is fixed. The base-act MoE weights were not tested as an AR fast model.

LingbotFast now derives its VAE and warmup latent grid from the configured output size with the same integer alignment. A standard 480×832 request uses 60×104 latent pixels and reuses the matching warmup Graphs; the historical result above still describes the earlier adapter-based test.

On 2026-10-22, the fixed runner generated 165 frames from official example 03 on H200 GPU1: first and warm DiT requests took 28.465 and 28.545 s with 40 compiled blocks, 21 retained captures and 84 replays per request, without new captures. Original main `d48a6173` with the same exact-size VAE input took 31.768 and 29.944 s; warm DiT latency improved 4.7%. Both paths produced identical latents for the first seven chunks. After the ring window rolled, their final videos had YUV PSNR 31.37 dB and SSIM 0.9565. A ring oracle confirmed identical logical K/V across wraps; the physical FA3 reduction order produces small per-layer rounding differences.

### 12) Shared-cache plugin regression

The 2026-10-20 SF1.3B real-weight regression compared the frozen pre-integration source and shared `AttentionKVCache` integration on H200 GPU6, 480×832, 14 chunks, local window 12 / sink 3, with compile, warmup and CUDA Graph enabled. Each side ran a first request and three formal repeats. Every complete latent matched the saved original-main reference byte for byte. Mean DiT time was **3.719586 → 3.719567 s**, effectively unchanged; interface unification is not claimed as an additional speedup. Each formal request added 55 graph replays, with no new compilation or capture. This regression skipped video decoding. Realtime14B and LingbotFast were not rerun in that plugin regression; their full-model evidence is the earlier acceptance above.

### 13) Clean commit with a K/V-only final layer

SF1.3B, realtime-video14B and LingbotFast share this optimization. It is enabled automatically for supported cache commits, without a configuration switch. It applies only when the scheduler sets `is_rerun=true`. Denoising remains unchanged.

The first N−1 layers run fully to produce the final layer's input. The final layer retains its input norm/modulation, K/V projections, K normalization/RoPE and ring write, then skips Q, self/cross attention, FFN, output head/post-processing and the unused clean noise-prediction assignment. SF's input `before_proj` and Lingbot's camera injections in earlier layers are preserved.

Resident denoising uses the full block-stack graph; clean commit uses a separate first N−1 layer variant and a compiled K/V-only final layer. Weight/KV staging uses the same policy through per-block graphs. Graph-only execution also passes the GPU RoPE frame offset. Reference prefill and models requiring the final post-adapter output retain full computation inside the shared optimized executor; Audio AR sets `cache_commit_enabled=False` for that reason. Parallelism, offload and per-step banks do not disable compilation or Graph.

H200, BF16/FA3, 480×832, 14 chunks, three formal requests per side on the same GPU, with compilation, production warmup and CUDA Graph actually enabled:

| Model | Mean request DiT ms | Mean clean total ms | DiT time reduction | Clean time reduction |
| --- | ---: | ---: | ---: | ---: |
| SF1.3B, local12/sink3 | 3681.887 → 3658.085 | 735.914 → 711.734 | 0.65% | 3.29% |
| realtime-video14B, local12/sink3 | 21225.324 → 21132.581 | 4244.832 → 4149.106 | 0.44% | 2.26% |
| LingbotFast cam, local21/sink3 | 29655.453 → 29527.478 | 5903.611 → 5773.922 | 0.43% | 2.20% |

Every chunk's complete valid final-layer K/V and ring/cursor metadata were byte-identical. Complete latents from the first and all three formal requests also matched original main. Lingbot additionally matched its full video, all 70 inputs and 56 denoising predictions; unused clean predictions are excluded. Both Lingbot runs used the external adapter that fixes height to 480; base-act is unvalidated. SF/14B video decoding was not repeated in this regression.

Formal requests added no compilation or capture, with 55 replays per SF/14B request and 40 for Lingbot. Retaining another graph adds startup and memory costs; this is a steady-state optimization.

### 14) Shared execution interfaces

`WanCausalTransformerInfer` derives from `WanAutoregressiveTransformerInfer`; `EdenLiveARTransformerInfer` combines `EdenLiveTransformerInfer` with that causal base. They share condition KV caches, KV-only compilation dispatch, and CUDA Graph lifetime. The causal base owns the rolling manager/backend, cache initialization/reset, Wan block/cross-attention/FFN/head calculation, and weight/KV offload. SF, LingbotFast, MG4 and Audio AR extend this base; EdenLive keeps its independent RoPE/workspace calculation. Full blocks use `infer_block` / `run_block`; the final cache projection uses `infer_kv_block` / `run_kv_block`; commits use `update_self_attn_cache`. Models retain their own RoPE and commit timing. `can_use_kv_only` selects the model-safe clean shortcut; CUDA Graph eligibility no longer requires resident weights, a full window or single-GPU execution.

SF and LingbotFast runners share chunk denoising, clean commits, streaming decode, and warmup through `WanAutoregressiveRunnerMixin`. EdenLive keeps its proxy and scheduler flow while sharing warmup shape ordering and request cleanup through `AutoregressiveRunnerMixin`. Request cleanup retains compiled blocks, Graphs, and reusable buffers while clearing condition caches; weight or storage changes use `reset_cuda_graph`. `use_compile`, `use_cuda_graph`, and `warmup` still select execution modes; exact cache reuse and eligible clean commits are automatic.

The 2026-10-20 shared-interface cleanup passed 201 distinct regression cases and repeats real SF1.3B, realtime14B, LingbotFast, and EdenLive runs with compile, warmup, and Graphs. Full outputs remain byte exact; SF/Lingbot throughput is unchanged.

### 15) 2026-10-20 physical-view validation

Native FA3 reads the valid physical view of the post-RoPE ring. The SF ordered workspace and history gather are removed without a cache-order switch. EdenLive is unchanged. The permutation property is not exclusive to FA3, but this earlier validation covered native FA3 only; existing fallbacks such as narrow windows are outside this zero-cat path.

H200, BF16, 832×480, 14 chunks, four denoising steps plus clean per chunk, with compile, production warmup and CUDA Graph actually enabled. Three formal requests follow one precision request per side. Times below are CUDA-event DiT GPU totals including clean, excluding VAE. SF/14B use local12/sink3; Lingbot uses local21/sink3. Each decoded video has 165 frames at 16 fps.

| Model | Before | After | Time reduction | Peak allocated reduction | Video PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| SF 1.3B | 3.656 s | 3.565 s | 2.51% | 109.69 MiB | 28.27 dB |
| Realtime14B | 21.186 s | 20.716 s | 2.22% | 365.62 MiB | 24.65 dB |
| LingbotFast cam/fast | 29.491 s | 28.908 s | 1.98% | 639.84 MiB | 28.15 dB |

All formal requests have zero new compilation/capture, with 55 replays per SF/14B request and 40 per Lingbot request. Ring attention workspace is zero after the change. Subjects, scenes and overall motion remain similar, but floating-point differences accumulate after wrap and visibly alter later poses, textures and local boundaries. This is not exact precision alignment. PSNR uses complete normalized video tensors, not MP4. Only one scene/seed per model is tested; Lingbot uses an external 480-high shape adapter and validates cam/fast only.

No cold-start improvement is claimed. GPU regression: 219 passed, zero skipped.

### 16) FA2 and SageAttention2

SF1.3B, Realtime14B and LingbotFast now allow FA2, FA3 and Sage2 self/cross combinations with `use_compile=true`, `use_cuda_graph=true` and `warmup=true` together. For the complete visible ring, FA2/FA3 reuse physical KV views and Sage2 self-attention uses persistent quantized segments and LSE merging, without a new switch or history concatenation. Narrow windows use the reusable gather buffers described in section 4. EdenLive uses standard `sage_attn2` with a raw-key ring adapter. Ordinary projected condition KV is retained; backend-specific `prepare_kv` / `apply_prepared` caches are removed.

`SageRingAttention` keeps segmented INT8 K, K means/scales, FP8 V and V scales alongside the original rolling storage. Slot writes update only touched segments, including the final clean KV-only layer. SM89, SM90 and SM120 use their respective native kernels, quantization granularities and FP8 scales. Resident rolling storage no longer contains offload fields or branches; separate offloaded rolling/step/hybrid classes share transfer lifecycle.

The earlier 2026-10-20 resident Sage2 comparison uses original main (commit `d48a6173`) as its primary precision reference, with `sage_attn2` directly used for both self and cross attention. H200, 480×832, 14 chunks, three formal GPU DiT repeats: SF1.3B 4.195→3.704s (11.71% less time), Realtime14B 23.095→20.843s (9.75%), and LingbotFast 32.986→28.437s (13.79%). Main runs eager with its first complete request excluded as warmup; the new path actually runs compile, Graph and production warmup. These combined gains include earlier shared execution improvements and cannot be attributed solely to quantized caching. Immediate-before and mixed-backend measurements remain auxiliary, not the primary precision reference.

Full 165-frame video PSNR is 16.06dB / 16.76dB / 16.36dB for SF1.3B / Realtime14B / LingbotFast. Subjects, scenes and overall motion remain close, while poses, waterlines and local Lingbot geometry visibly drift. This is not exact precision alignment or proof of equivalent quality. The quantized companion also increases memory usage. Only H200 hardware was available; RTX40/50 implementation and quantization layouts still need native hardware acceptance. See [Sage setup and boundaries](SAGE_ATTENTION.md).

### 17) EdenLive runner lifecycle

Ordinary `EdenLiveRunner` derives from `RunnerLifecycleMixin`, which owns warmup resolution ordering and request-state cleanup. `EdenLiveARRunner` explicitly adds `AutoregressiveRunnerMixin`. The registered Wan2.1/Wan2.2 ordinary runners therefore have no AR mixin in their MRO. This inheritance correction changes no EdenLive attention or KV arithmetic; 62 focused lifecycle, warmup and runner tests pass.

FA2 full-video validation passed in a healthy Torch2.8/cu128 runtime with upstream BF16 compiler fixes applied only to the validation process. SF1.3B/Realtime14B/LingbotFast DiT time decreases 4.88%/3.83%/4.19%, with full-video PSNR26.60/28.81/29.57dB. The first wrap starts the floating-point differences; the preceding4/4/7 chunks match old eager byte for byte. This is a separate baseline/runtime from original-main Sage2. See [compiler boundaries](SAGE_ATTENTION.md).

### 18) Cache ownership and backend selection

Transformer Infer owns and initializes the SF / Realtime14B / LingbotFast / MG4 cache. The runner calls `initialize_kv_cache(latent_shape)` and updates scheduler frame counts; warmup uses `reset_kv_cache()` without constructing managers or binding pools.

FA2/FA3 use `RollingKVCachePool` plus `RingAttention` over physical views. `sage_attn2` uses the same rolling pool with `SageRingAttention` for incremental quantization and LSE merging. These direct/segmented paths cover the complete visible ring; narrower windows gather into reusable buffers as described in section 4. No separate cache-backend option is required; omit `kv_cache_scheme` and `step_kv_cache` for the default rolling cache shared across steps. SF models use `sage_attn2` for automatic SM89/SM90/SM120 dispatch. EdenLive also uses standard `sage_attn2`, with a raw-key RoPE adapter; the old SM90/SM120 backend entries have been removed. MG4 keeps only its sink position policy in `rope.py` and reuses the same storage and attention backends.

### 19) Current shared execution and validation scope

The causal executor now shares prepared-cache dispatch across resident execution, weight block offload, KV offload, SP/TP, prefill and per-step banks. Computation remains compiled and Graph-captured; staging and cache planning run outside the graph. SF, LingbotFast, MG4, Audio/Seko AR and EdenLive AR inherit the common causal base, with model-specific positional adapters. Non-CUDA compatibility paths remain available.

The 2026-10-21 validation records distinguish FA3 real multi-GPU/offload comparisons from Sage2 cache tests and the earlier SF Sage2 resident videos. They do not certify every backend/model/parallel combination. On EdenLive, the current FA3 SP2 + weight block/KV offload 717-frame run has exact full latent/video SHA against the same-mode main reference, with 0 formal captures and 9000 replays. The earlier Sage2 resident 717-frame result in [SAGE_ATTENTION.md](SAGE_ATTENTION.md) had 275 replays and is a separate experiment.

The current installed FA2 runtime fails four long-sequence numerical cases even without KV caching; its physical-view implementation remains present, while current Flash full-model multi-GPU results use FA3. Earlier Torch 2.8/cu128 FA2 results depend on the documented process-local compiler fixes. No RTX 40/50 hardware was available. Most current speed comparisons contain one formal request after warmup and are regression measurements, not stable throughput estimates.
