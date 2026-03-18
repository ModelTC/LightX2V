# Diffusion Model Disaggregated Deployment Guide

For large-scale generative models (e.g. Wan, Qwen Image), Text Encoder, Image Encoder, and VAE encoder/decoder often reside in GPU memory and squeeze the space available for the core DiT, leading to OOM under high resolution or long-sequence generation.

LightX2V provides a native **Disaggregation Mode** that splits the inference pipeline into **Encoder, Transformer, and Decoder** stages and deploys them on different GPUs or nodes, using the **Mooncake** transport (RDMA / TCP). **Wan** and **Qwen Image** both support full **three-stage** disaggregation, including a dedicated VAE Decoder node. Each stage can process different requests concurrently to improve throughput.

---

## Comparison

| Aspect | **Baseline (single process)** | **Disagg Mode (split microservices)** |
|--------|-------------------------------|----------------------------------------|
| **Architecture** | All models in one process | Three stages: **Encoder** → **Transformer** → **Decoder** (VAE decode on its own node) |
| **Memory** | Very high | **Per-node**: each node loads only its subset (Decoder = VAE Decoder only) |
| **Transport** | In-process tensors | **Mooncake** (Phase1: Encoder→Transformer; Phase2: Transformer→Decoder) |
| **Use case** | Single machine, quick validation | **Memory-constrained**, long video, multi-node, high concurrency |

---

## Quick Start

#### Wan2.1-T2V-14B (50 steps, 480×832, 81 frames)

```bash
# Optional: set GPU for each stage
# GPU_ENCODER=0 GPU_TRANSFORMER=1 GPU_DECODER=0

# Start three-stage disagg services
bash scripts/server/disagg/wan/start_wan_t2v_disagg.sh

# After services are up, run test
python scripts/server/disagg/wan/post_wan_t2v.py
```

#### Wan2.1-I2V-14B-480P (40 steps, 480×832, 81 frames)

```bash
bash scripts/server/disagg/wan/start_wan_i2v_disagg.sh
python scripts/server/disagg/wan/post_wan_i2v.py
```

#### Qwen Image (T2I)

```bash
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg.sh
python scripts/server/disagg/qwen/post_qwen_t2i.py
```

#### Qwen Image (I2I)

```bash
bash scripts/server/disagg/qwen/start_qwen_i2i_disagg.sh
python scripts/server/disagg/qwen/post_qwen_i2i.py
```

---

## Benchmark (summary)

Environment: NVIDIA H100 SXM5 80 GB, `sage_attn2`, BF16, Mooncake RDMA. Baseline = single GPU; Disagg = Encoder / Transformer / Decoder on separate GPUs (three-stage further reduces Transformer card memory).

- **RTX 4090 24GB (memory/PCIe-constrained)**: On memory-constrained GPUs (e.g. 4090 24GB), baseline often requires `cpu_offload=block` to run, while Disagg reduces per-GPU resident memory by splitting Encoder/DiT/VAE Decoder across processes/GPUs, which can reduce offload pressure and improve latency/throughput.

### RTX 4090 24GB (memory/PCIe-constrained)

This section highlights baseline vs disagg behavior on **memory-constrained GPUs**. Numbers below are collected on RTX 4090 24GB.

#### Wan2.1-I2V-14B (BF16, 40 steps)

| Metric | Baseline (1×4090, block offload) | Disagg (2×4090, block offload) | Notes |
| --- | --- | --- | --- |
| Peak VRAM | 480P: 9GB; 720P: 14GB | — | Baseline relies on offload to run |
| DiT step | 480P: 24.24 s/step; 720P: 90.71 s/step | 480P: 19.02 s/step; 720P: 62.80 s/step | DiT is more likely to be memory/PCIe limited on 4090 |
| Image Encoder | 480P: 0.57s; 720P: 0.52s | 480P: 0.28s; 720P: 0.20s | Encoder is more stable after resource decoupling |
| VAE Encoder | 480P: 3.02s; 720P: 6.83s | 480P: 3.22s; 720P: 7.36s | Strongly affected by resolution |
| Text Encoder | 480P: 2.14s; 720P: 1.90s | 480P: 0.20s; 720P: 0.20s | Baseline is more impacted by offload/scheduling |
| Encoder total | 480P: 6.09s; 720P: 9.78s | 480P: 4.08s; 720P: 8.32s |  |

#### Wan2.1-I2V-14B (INT8)

| Metric | Baseline (block offload) | Disagg (no offload) | Disagg (block offload) |
| --- | --- | --- | --- |
| Notes | `use_offload=false` will OOM | — | — |
| DiT step | 12.94 s/step | 12.55 s/step | 12.91 s/step |
| Image Encoder | 0.79s | 0.19s | 0.19s |
| VAE Encoder | 3.11s | 2.93s | 2.93s |
| Text Encoder | 1.67s | 0.20s | 0.20s |
| Encoder total | 5.90s | 3.69s | 3.70s |

#### 4090 concurrency (BF16 + block offload, Wan2.1 480P, 4 steps)

This table is used to observe whether Disagg improves QPS and tail latency under **multi-request contention**.

| N (concurrency) | mode | ok/total | QPS | P50 (s) | P95 (s) | P99 (s) | note |
| ---:| --- | ---:| ---:| ---:| ---:| ---:| --- |
| 1 | baseline | 1 | 0.0091 | 109.45 | 109.45 | 109.45 | wan2.1 480P 4step |
| 2 | baseline | 2 | 0.0092 | 162.52 | 211.01 | 215.32 | wan2.1 480P 4step |
| 4 | baseline | 4 | 0.0093 | 269.85 | 414.94 | 427.83 | wan2.1 480P 4step |
| 8 | baseline | 8 | 0.0093 | 485.82 | 824.24 | 854.32 | wan2.1 480P 4step |
| 1 | disagg | 1 | 0.0117 | 85.38 | 85.38 | 85.38 | wan2.1 480P 4step |
| 2 | disagg | 2 | 0.0122 | 125.83 | 160.18 | 163.23 | wan2.1 480P 4step |
| 4 | disagg | 4 | 0.0126 | 201.85 | 305.73 | 314.94 | wan2.1 480P 4step |
| 8 | disagg | 8 | 0.0129 | 358.20 | 595.12 | 616.48 | wan2.1 480P 4step |

#### 4090 concurrency (Qwen Image 2512, T2I, 5 steps)

| N (concurrency) | mode | ok/total | QPS | P50 (s) | P95 (s) | P99 (s) | note |
| ---:| --- | ---:| ---:| ---:| ---:| ---:| --- |
| 1 | baseline | 1 | 0.0207 | 48.30  | 48.30  | 48.30  | qwen-image-2512 5step |
| 2 | baseline | 2 | 0.0207 | 72.57  | 94.31  | 94.24  | qwen-image-2512 5step |
| 4 | baseline | 4 | 0.0212 | 118.77 | 181.44 | 187.33 | qwen-image-2512 5step |
| 8 | baseline | 8 | 0.0216 | 208.64 | 354.94 | 367.90 | qwen-image-2512 5step |
| 1 | disagg | 1 | 0.0452 | 22.11  | 22.11  | 22.11  | qwen-image-2512 5step |
| 2 | disagg | 2 | 0.0510 | 29.68  | 38.25  | 39.02  | qwen-image-2512 5step |
| 4 | disagg | 4 | 0.0528 | 48.52  | 73.00  | 75.17  | qwen-image-2512 5step |
| 8 | disagg | 8 | 0.0534 | 85.78  | 143.37 | 148.62 | qwen-image-2512 5step |

- **Wan2.1-T2V-1.3B**: Disagg reduces peak memory (e.g. Transformer ~9.2 GB) and can slightly improve DiT step time; end-to-end ~3 s faster.
- **Wan2.1-14B (T2V/I2V)**: End-to-end latency on par with baseline; Disagg allows Encoder and Transformer to serve different requests concurrently.
- **Qwen Image T2I/I2I**: Disagg with `lightllm_kernel` Text Encoder gives ~30% T2I encoder speedup and ~19% I2I encoder speedup; I2I end-to-end ~5 s lower.

---

## 1. Three-stage architecture

The pipeline is split by `disagg_mode` into three roles, with two Mooncake phases:

- **Encoder (`disagg_mode="encoder"`)**
  - Loads only Text Encoder, Image Encoder (for I2V/I2I), and VAE Encoder; no DiT or VAE Decoder.
  - Sends `context`, `clip_encoder_out`, `vae_encoder_out`, `latent_shape` via **Phase1** to the Transformer.

- **Transformer (`disagg_mode="transformer"`)**
  - Loads only DiT; no Encoder or VAE Decoder (Decoder node does decode in three-stage).
  - Waits for Phase1 data, runs denoising; if `decoder_engine_rank` is set, sends latents via **Phase2** to Decoder and does **not** run VAE decode locally.

- **Decoder (`disagg_mode="decode"`)**
  - Loads only **VAE Decoder**; no Text/Image Encoder or DiT.
  - Waits for Phase2, runs VAE decode, and saves output; **task completion and result path are on the Decoder node**.

---

## 2. Configuration

All disagg options live under `disagg_config` in the config JSON.

### 2.1 T2V (Wan)

**Encoder** (`configs/disagg/wan/wan_t2v_disagg_encoder.json`): `disagg_mode: "encoder"`, plus `bootstrap_addr`, `bootstrap_room`, `sender_engine_rank`, `receiver_engine_rank`, `protocol`, `local_hostname`, `metadata_server`.

**Transformer** (`configs/disagg/wan/wan_t2v_disagg_transformer.json`): `disagg_mode: "transformer"` and Phase2 fields:
- `decoder_engine_rank`: Decoder rank for Phase2.
- `decoder_bootstrap_room`: Phase2 room; must match Decoder’s `bootstrap_room`.

**Decoder** (`configs/disagg/wan/wan_t2v_disagg_decode.json`): `disagg_mode: "decode"`; `bootstrap_room` must equal Transformer’s `decoder_bootstrap_room`.

### 2.2 I2V (Wan)

I2V uses CLIP ViT-H/14; you must set **`clip_embed_dim`: 329216** (257×1280) so Mooncake buffers are sized correctly. Both Encoder and Transformer configs need this.

### 2.3 Decoder example (Qwen Image I2I)

`configs/disagg/qwen/qwen_image_i2i_disagg_decode.json`: `task: "i2i"`, `disagg_mode: "decode"`, plus `vae_z_dim`, `vae_stride`, `target_video_length`, `target_height`, `target_width`, and `disagg_config` with `bootstrap_room` equal to Transformer’s `decoder_bootstrap_room`.

### 2.4 Parameter reference

| Parameter | Description |
|-----------|-------------|
| `disagg_mode` | Role: `"encoder"`, `"transformer"`, or `"decode"` |
| `bootstrap_addr` | IP of the peer (Encoder→Transformer; Decoder→Transformer) |
| `bootstrap_room` | Phase1/Phase2 room; **must match** on sender and receiver |
| `sender_engine_rank` | Phase1: Encoder rank; Phase2: Transformer rank |
| `receiver_engine_rank` | Phase1: Transformer rank; Phase2: Decoder rank |
| `decoder_engine_rank` | **Transformer only**: Decoder rank for Phase2 |
| `decoder_bootstrap_room` | **Transformer only**: Phase2 room; must match Decoder’s `bootstrap_room` |
| `protocol` | `"rdma"` (recommended) or `"tcp"` |
| `local_hostname` | This node’s hostname/IP for Mooncake P2P handshake |
| `metadata_server` | Use `"P2PHANDSHAKE"` for single-node |
| `clip_embed_dim` | **I2V only**: 329216 for ViT-H/14 |

---

## 3. Startup and request flow

Use `lightx2v.server` to start the HTTP API. For **three-stage** disagg:

**Start order (recommended):**
1. Start **Decoder** first (Phase2 receiver).
2. Start **Transformer** (Phase1 receiver + Phase2 sender).
3. Start **Encoder** last.

**Request and polling order:**
1. **POST to Decoder** → get `decoder_task_id` (Decoder starts waiting for Phase2).
2. **POST to Transformer** with same payload (waits Phase1, runs DiT, sends Phase2 to Decoder).
3. **POST to Encoder** with same payload (encodes and sends Phase1 to Transformer).
4. **Poll Decoder** for status: completion and result path are on the Decoder; use Decoder’s `task_id` and Decoder URL.

### 3.1 Wan T2V/I2V

Scripts: `scripts/server/disagg/wan/`. Start Decoder (e.g. port 8004), then Transformer (8003), then Encoder (8002). Use `/v1/tasks/video/`. Request order: POST Decoder → POST Transformer → POST Encoder → poll Decoder.

### 3.2 Qwen Image T2I/I2I

Scripts: `scripts/server/disagg/qwen/`. Use `/v1/tasks/image/`. Same start order (Decoder → Transformer → Encoder). Example: `post_qwen_t2i.py`, `post_qwen_i2i.py` (they send to Decoder first, then Transformer, then Encoder, then poll Decoder). Set `IMAGE_PATH` in the script for I2I.

### 3.3 API summary

| Model | Task | Endpoint | Completion / result |
|-------|------|----------|----------------------|
| Wan2.1 | T2V / I2V | `/v1/tasks/video/` | **Decoder** node |
| Qwen Image | T2I / I2I | `/v1/tasks/image/` | **Decoder** node |

---

## 4. RDMA vs TCP

| Aspect | **RDMA** | **TCP** |
|--------|----------|---------|
| Transport | Zero-copy, kernel bypass | Kernel network stack |
| CPU | Very low | Higher |
| Latency | Microseconds | Milliseconds |
| Hardware | InfiniBand / RoCE | Any network |
| When to use | Production, multi-node | Single-node or no RDMA |

Without RDMA hardware, set `protocol` to `"tcp"` in `disagg_config`.
