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

## 4. Decentralized Queue Mode

### 4.1 Overview

In the standard three-stage deployment, the client must send requests to Decoder → Transformer → Encoder in order, which makes the caller logic relatively complex. The decentralized queue mode simplifies this to:

- **Controller** only maintains RDMA metadata ring buffers (request ring / phase1 ring / phase2 ring) and does not participate in inference computation;
- **Encoder** receives client requests via HTTP API (`lightx2v.server`), runs inference, and writes phase1 metadata into the RDMA ring for the matching Transformer worker to pull;
- **Transformer** and **Decoder** run as pull-based workers (`qwen_t2i_queue_workers.py`), consuming dispatch packets from the RDMA ring in a loop without exposing any HTTP endpoints;
- **Multiple Transformer workers** can be deployed on different GPUs. The client request includes `disagg_phase1_receiver_engine_rank` to specify which Transformer handles the request, enabling multi-worker concurrency.

```text
┌──────────┐  HTTP POST   ┌──────────┐ Phase1 RDMA ┌─────────────┐ Phase2 RDMA ┌──────────┐
│  Client  │ ──────────→  │ Encoder  │ ──────────→ │ Transformer │ ──────────→ │ Decoder  │
└──────────┘              │ (GPU 0)  │             │ (GPU 1/2/3) │             │ (GPU 0)  │
                          └──────────┘             └─────────────┘             └──────────┘
                                ↑                        ↑                          ↑
                          lightx2v.server          pull worker ×N              pull worker
                          HTTP port 8002           (qwen_t2i_queue_workers)    (qwen_t2i_queue_workers)
                                │
                          ┌──────────┐
                          │Controller│  ← RDMA metadata ring buffer (always-on)
                          └──────────┘
```

**Comparison with the standard three-stage mode:**

| Aspect              | Standard three-stage                                    | Decentralized queue                                            |
| ------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| **Client calls**    | Must POST to Decoder → Transformer → Encoder separately | Single POST to **Encoder HTTP**                                |
| **Transformer**     | HTTP server, processes one request at a time             | Pull worker, multiple instances consume in parallel            |
| **Decoder**         | HTTP server                                             | Pull worker, auto-consumes Phase2                              |
| **Request routing** | Client explicitly specifies                             | Encoder writes RDMA ring, workers pull by rank                 |
| **Result retrieval**| Poll Decoder HTTP status                                | Poll **Encoder HTTP** `/v1/tasks/{task_id}/status`             |

### 4.2 Configuration

Decentralized mode configs are in `configs/disagg/qwen/`. The key difference is that each component's `disagg_config` must set `"decentralized_queue": true` along with RDMA ring handshake ports.

#### Controller (`qwen_image_t2i_disagg_controller.json`)

The controller does not load any model; it only initializes three RDMA ring buffers:

```json
{
  "task": "t2i",
  "model_cls": "qwen_image",
  "disagg_mode": "controller",
  "disagg_config": {
    "bootstrap_addr": "127.0.0.1",
    "bootstrap_room": 0,
    "encoder_engine_rank": 0,
    "transformer_engine_rank": 1,
    "decoder_engine_rank": 4,
    "protocol": "rdma",
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "rdma_buffer_slots": 128,
    "rdma_buffer_slot_size": 4096,
    "rdma_request_handshake_port": 5566,
    "rdma_phase1_handshake_port": 5567,
    "rdma_phase2_handshake_port": 5568
  }
}
```

#### Encoder (`qwen_image_t2i_disagg_encoder_decentralized.json`)

The Encoder runs as an HTTP service, loads the Text Encoder, and writes features and metadata into the Phase1 RDMA ring after inference:

```json
{
  "disagg_mode": "encoder",
  "text_encoder_type": "lightllm_kernel",
  "disagg_config": {
    "decentralized_queue": true,
    "rdma_phase1_host": "127.0.0.1",
    "rdma_phase1_handshake_port": 5567,
    "rdma_buffer_slots": 128,
    "rdma_buffer_slot_size": 4096,
    "bootstrap_addr": "127.0.0.1",
    "bootstrap_room": 0,
    "sender_engine_rank": 0,
    "receiver_engine_rank": 1,
    "protocol": "rdma",
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE"
  }
}
```

#### Transformer (`qwen_image_t2i_disagg_transformer_decentralized.json`)

The Transformer runs as a pull worker, consuming tasks from the Phase1 ring and writing results to the Phase2 ring. Each worker's `receiver_engine_rank` and `transformer_engine_rank` must correspond to a different rank:

```json
{
  "disagg_mode": "transformer",
  "disagg_config": {
    "decentralized_queue": true,
    "encoder_engine_rank": 0,
    "transformer_engine_rank": 1,
    "decoder_engine_rank": 4,
    "decoder_bootstrap_room": 0,
    "rdma_phase1_host": "127.0.0.1",
    "rdma_phase1_handshake_port": 5567,
    "rdma_phase2_host": "127.0.0.1",
    "rdma_phase2_handshake_port": 5568,
    "rdma_buffer_slots": 128,
    "rdma_buffer_slot_size": 4096,
    "bootstrap_addr": "127.0.0.1",
    "bootstrap_room": 0,
    "sender_engine_rank": 0,
    "receiver_engine_rank": 1,
    "protocol": "rdma",
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE"
  }
}
```

> When deploying multiple Transformer workers, generate a separate config for each worker with `receiver_engine_rank` and `transformer_engine_rank` set to the worker's rank (e.g. 1, 2, 3). The startup script generates these automatically.

#### Decoder (`qwen_image_t2i_disagg_decode_decentralized.json`)

The Decoder runs as a pull worker, consuming tasks from the Phase2 ring, performing VAE decode, and saving results:

```json
{
  "disagg_mode": "decode",
  "disagg_config": {
    "decentralized_queue": true,
    "encoder_engine_rank": 0,
    "transformer_engine_rank": 1,
    "decoder_engine_rank": 4,
    "rdma_phase2_host": "127.0.0.1",
    "rdma_phase2_handshake_port": 5568,
    "rdma_buffer_slots": 128,
    "rdma_buffer_slot_size": 4096,
    "bootstrap_addr": "127.0.0.1",
    "bootstrap_room": 0,
    "sender_engine_rank": 1,
    "receiver_engine_rank": 4,
    "protocol": "rdma",
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE"
  }
}
```

#### Parameter reference

| Parameter                       | Description                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| `decentralized_queue`           | Set to `true` to enable decentralized queue mode                                         |
| `rdma_request_handshake_port`   | RDMA handshake port for the controller request ring (controller only)                    |
| `rdma_phase1_handshake_port`    | RDMA handshake port for the Phase1 ring; must be consistent across Encoder / Transformer / Controller |
| `rdma_phase2_handshake_port`    | RDMA handshake port for the Phase2 ring; must be consistent across Transformer / Decoder / Controller |
| `rdma_buffer_slots`             | Number of slots in the RDMA ring buffer                                                  |
| `rdma_buffer_slot_size`         | Byte size of each slot                                                                   |
| `receiver_engine_rank`          | Rank of the Transformer worker; must differ across workers                                |
| `transformer_engine_rank`       | Same as `receiver_engine_rank`; keep consistent                                           |
| `decoder_engine_rank`           | Rank of the Decoder; must be consistent across all component configs                      |

### 4.3 Starting the services

Example: Qwen Image T2I on 4 GPUs (GPU 0: Controller + Encoder + Decoder; GPU 1/2/3: one Transformer worker each):

```bash
bash scripts/server/disagg/qwen/start_qwen_t2i_decentralized.sh
```

**Service startup order (handled automatically by the script):**

1. **Controller** (RDMA ring buffer, always-on process):

```bash
# Controller only initializes RDMA ring buffers, no model loading
python3 -c "
import json
from lightx2v.disagg.services.controller import ControllerService
with open('configs/disagg/qwen/qwen_image_t2i_disagg_controller.json') as f:
    cfg = json.load(f)
dc = cfg.get('disagg_config', {})
cfg['data_bootstrap_addr'] = dc.get('bootstrap_addr', '127.0.0.1')
ControllerService().serve_rdma_dispatch_only(cfg)
"
```

2. **Encoder HTTP service** (GPU 0, port 8002):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m lightx2v.server \
  --model_cls qwen_image \
  --task t2i \
  --model_path /path/to/qwen-2512/ \
  --config_json configs/disagg/qwen/qwen_image_t2i_disagg_encoder_decentralized.json \
  --host 0.0.0.0 \
  --port 8002
```

3. **Decoder pull worker** (GPU 0, shared with Encoder):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m lightx2v.disagg.examples.qwen_t2i_queue_workers \
  --role decoder \
  --model_path /path/to/qwen-2512/ \
  --config_json configs/disagg/qwen/qwen_image_t2i_disagg_decode_decentralized.json
```

4. **Transformer pull workers ×3** (GPU 1, 2, 3, each with its own rank config):

```bash
# Each worker's config has a different receiver_engine_rank / transformer_engine_rank
CUDA_VISIBLE_DEVICES=1 python3 -m lightx2v.disagg.examples.qwen_t2i_queue_workers \
  --role transformer \
  --model_path /path/to/qwen-2512/ \
  --config_json /tmp/qwen_t2i_decentralized_cfg/transformer_r1.json

CUDA_VISIBLE_DEVICES=2 python3 -m lightx2v.disagg.examples.qwen_t2i_queue_workers \
  --role transformer \
  --model_path /path/to/qwen-2512/ \
  --config_json /tmp/qwen_t2i_decentralized_cfg/transformer_r2.json

CUDA_VISIBLE_DEVICES=3 python3 -m lightx2v.disagg.examples.qwen_t2i_queue_workers \
  --role transformer \
  --model_path /path/to/qwen-2512/ \
  --config_json /tmp/qwen_t2i_decentralized_cfg/transformer_r3.json
```

> The startup script automatically generates `transformer_r1.json` / `transformer_r2.json` / `transformer_r3.json` from the template config, setting `receiver_engine_rank` to 1, 2, 3 respectively.

### 4.4 Sending requests

In decentralized mode, the client **only needs to send a single request to the Encoder HTTP endpoint**. The request uses `disagg_phase1_receiver_engine_rank` to specify the target Transformer rank:

```python
import requests

ENCODER_URL = "http://127.0.0.1:8002"

resp = requests.post(f"{ENCODER_URL}/v1/tasks/image/", json={
    "prompt": "A cute cat on a table, cinematic lighting.",
    "negative_prompt": "blurry, low quality",
    "seed": 42,
    "aspect_ratio": "16:9",
    "save_result_path": "/tmp/output.png",
    "data_bootstrap_room": 100,                       # unique room ID for this request
    "disagg_phase1_receiver_engine_rank": 1,           # target Transformer rank (1/2/3)
})
task_id = resp.json()["task_id"]

# Poll Encoder for completion status (Decoder saves the file asynchronously)
import time
while True:
    st = requests.get(f"{ENCODER_URL}/v1/tasks/{task_id}/status").json()
    if st["status"] == "completed":
        print(f"Done: {st.get('save_result_path')}")
        break
    if st["status"] == "failed":
        raise RuntimeError(st)
    time.sleep(2)
```

**Request parameters:**

| Parameter                            | Description                                                                         |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `data_bootstrap_room`               | Unique room ID per request, used for Mooncake data transfer channel matching         |
| `disagg_phase1_receiver_engine_rank` | Target Transformer rank; must be within the range of running worker ranks (e.g. 1, 2, 3) |
| `save_result_path`                   | Output file path; the Decoder worker saves the final result to this path             |

### 4.5 Benchmarking

Use the built-in benchmark script to send concurrent requests across multiple Transformer workers:

```bash
# Default: 6 requests, 3 concurrency, round-robin across ranks 1,2,3
bash scripts/server/disagg/qwen/bench.sh

# Custom parameters
REQUESTS=24 CONCURRENCY=6 TIMEOUT=600 RANKS=1,2,3 \
  bash scripts/server/disagg/qwen/bench.sh
```

The benchmark script distributes requests round-robin to different Transformer ranks, waits for all output files to be saved to disk, and reports end-to-end QPS / P50 / P95 / P99 latency.

---

## 5. RDMA vs TCP

| Aspect | **RDMA** | **TCP** |
|--------|----------|---------|
| Transport | Zero-copy, kernel bypass | Kernel network stack |
| CPU | Very low | Higher |
| Latency | Microseconds | Milliseconds |
| Hardware | InfiniBand / RoCE | Any network |
| When to use | Production, multi-node | Single-node or no RDMA |

Without RDMA hardware, set `protocol` to `"tcp"` in `disagg_config`.
