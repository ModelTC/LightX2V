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

These examples launch three stages on one machine. Choose one model at a time. Use the LightX2V environment and repository root in both terminals.

### Shared setup

In **terminal 1**, install Mooncake and set the repository path and GPUs:

```bash
pip install mooncake-transfer-engine
export LIGHTX2V_PATH="$PWD"
export GPU_ENCODER=0 GPU_TRANSFORMER=1 GPU_DECODER=0
```

This places Encoder and Decoder on GPU 0 and Transformer on GPU 1. Adjust IDs and placement for available memory. Prepare the DiT, encoder, and VAE weights required by your model.

Stage configs are under `configs/disagg/wan/` or `configs/disagg/qwen/`. Install their selected backends: Wan uses `sage_attn2`; Qwen Encoder uses `lightllm_kernel` and `flash_attn3`, while the T2I/I2I Transformer examples use `flash_attn2`/`flash_attn3` respectively. When changing a backend, update the config and verify model and GPU support.

The launcher stays running. Wait until all three services are ready, then send requests from **terminal 2**. Pressing `Ctrl+C` in terminal 1 stops the entire group.

### Wan2.1-T2V-14B (50 steps, 480×832, 81 frames)

Terminal 1: replace the absolute model path and start the services.

```bash
WAN_T2V_MODEL_PATH="/absolute/path/to/Wan2.1-T2V-14B" \
  bash scripts/server/disagg/wan/start_wan_t2v_disagg.sh
```

Terminal 2:

```bash
python scripts/server/disagg/wan/post_wan_t2v.py
```

### Wan2.1-I2V-14B-480P (40 steps, 480×832, 81 frames)

Terminal 1:

```bash
WAN_I2V_MODEL_PATH="/absolute/path/to/Wan2.1-I2V-14B-480P" \
  bash scripts/server/disagg/wan/start_wan_i2v_disagg.sh
```

Terminal 2: set `IMAGE_PATH` in the request script to a local image before running.

```bash
python scripts/server/disagg/wan/post_wan_i2v.py
```

(qwen-t2i-disagg-quickstart)=
### Qwen Image (T2I)

First edit `configs/disagg/qwen/qwen_image_t2i_disagg_transformer.json`: set `dit_original_ckpt` to an existing DiT checkpoint matching your model and inference config. This field takes precedence over the model directory; changing only `QWEN_IMAGE_MODEL_PATH` does not override it.

Terminal 1:

```bash
QWEN_IMAGE_MODEL_PATH="/absolute/path/to/qwen-model" \
  bash scripts/server/disagg/qwen/start_qwen_t2i_disagg.sh
```

Terminal 2:

```bash
python scripts/server/disagg/qwen/post_qwen_t2i.py
```

### Qwen Image Edit 2511 (I2I)

Terminal 1:

```bash
QWEN_IMAGE_MODEL_PATH="/absolute/path/to/Qwen-Image-Edit-2511" \
  bash scripts/server/disagg/qwen/start_qwen_i2i_disagg.sh
```

Terminal 2: set `IMAGE_PATH` in the request script to a local image before running.

```bash
python scripts/server/disagg/qwen/post_qwen_i2i.py
```

### Addresses and ports

| Example | API ports (Encoder / Transformer / Decoder) | Metrics ports |
| --- | --- | --- |
| Wan T2V | 8002 / 8003 / 8006 | 18002 / 18003 / 18006 |
| Wan I2V | 8004 / 8005 / 8007 | 18004 / 18005 / 18007 |
| Qwen T2I | 8002 / 8003 / 8008 | 18002 / 18003 / 18008 |
| Qwen I2I | 8012 / 8013 / 8014 | 18012 / 18013 / 18014 |

If you change API addresses or ports, update `ENCODER_URL`, `TRANSFORMER_URL`, and `DECODER_URL` in the request script. When running multiple groups, avoid conflicts in both API and metrics ports.

## Benchmark (summary)

Environment: NVIDIA H100 SXM5 80 GB, `sage_attn2`, BF16, Mooncake RDMA. Baseline = single GPU; Disagg = Encoder / Transformer / Decoder on separate GPUs (three-stage further reduces Transformer card memory).

- **RTX 4090 24GB (memory/PCIe-constrained)**: On memory-constrained GPUs (e.g. 4090 24GB), baseline often requires `cpu_offload=block` to run, while Disagg reduces per-GPU resident memory by splitting Encoder/DiT/VAE Decoder across processes/GPUs, which can reduce offload pressure and improve latency/throughput.

### RTX 4090 24GB (memory/PCIe-constrained)

This section highlights baseline vs disagg behavior on **memory-constrained GPUs**. Numbers below are collected on RTX 4090 24GB.

### Wan2.1-I2V-14B (BF16, 40 steps)

| Metric | Baseline (1×4090, block offload) | Disagg (2×4090, block offload) | Notes |
| --- | --- | --- | --- |
| Peak VRAM | 480P: 9GB; 720P: 14GB | — | Baseline relies on offload to run |
| DiT step | 480P: 24.24 s/step; 720P: 90.71 s/step | 480P: 19.02 s/step; 720P: 62.80 s/step | DiT is more likely to be memory/PCIe limited on 4090 |
| Image Encoder | 480P: 0.57s; 720P: 0.52s | 480P: 0.28s; 720P: 0.20s | Encoder is more stable after resource decoupling |
| VAE Encoder | 480P: 3.02s; 720P: 6.83s | 480P: 3.22s; 720P: 7.36s | Strongly affected by resolution |
| Text Encoder | 480P: 2.14s; 720P: 1.90s | 480P: 0.20s; 720P: 0.20s | Baseline is more impacted by offload/scheduling |
| Encoder total | 480P: 6.09s; 720P: 9.78s | 480P: 4.08s; 720P: 8.32s |  |

### Wan2.1-I2V-14B (INT8)

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

Set routing and queue parameters in `disagg_config`. Mooncake reads the file selected by `MOONCAKE_CONFIG_PATH` (`configs/mooncake_config.json` in the scripts); corresponding `MOONCAKE_*` environment variables take precedence.

### 2.1 T2V (Wan)

**Encoder** (`configs/disagg/wan/wan_t2v_disagg_encoder.json`): `disagg_mode: "encoder"`, plus `bootstrap_addr`, `bootstrap_room`, `sender_engine_rank`, `receiver_engine_rank`.

**Transformer** (`configs/disagg/wan/wan_t2v_disagg_transformer.json`): `disagg_mode: "transformer"` and Phase2 fields:
- `decoder_engine_rank`: Decoder rank for Phase2.
- `decoder_bootstrap_room`: Phase2 room; must match Decoder’s `bootstrap_room`.

**Decoder** (`configs/disagg/wan/wan_t2v_disagg_decode.json`): `disagg_mode: "decode"`; `bootstrap_room` must equal Transformer’s `decoder_bootstrap_room`.

### 2.2 I2V (Wan)

I2V uses CLIP ViT-H/14; you must set **`clip_embed_dim`: 329216** (257×1280) so Mooncake buffers are sized correctly. Both Encoder and Transformer configs need this.

### 2.3 Decoder example (Qwen Image I2I)

`configs/disagg/qwen/qwen_image_i2i_disagg_decode.json`: `task: "i2i"`, `disagg_mode: "decode"`, plus `vae_z_dim`, `vae_stride`, `num_frames`, `size`, and `disagg_config` with `bootstrap_room` equal to Transformer’s `decoder_bootstrap_room`.

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

Scripts: `scripts/server/disagg/wan/`. Start Decoder, then Transformer, then Encoder. When launching stages manually, assign nonconflicting API and metrics ports, for example Decoder `--port 8004 --metric_port 18004`, Transformer `--port 8003 --metric_port 18003`, and Encoder `--port 8002 --metric_port 18002`. Metrics start by default for each independent service. Use `/v1/tasks/video/` and match the request URLs to the API ports. Request order: POST Decoder → POST Transformer → POST Encoder → poll Decoder.

### 3.2 Qwen Image T2I/I2I

Scripts: `scripts/server/disagg/qwen/`. Use `/v1/tasks/image/`. Same start order (Decoder → Transformer → Encoder). Example: `post_qwen_t2i.py`, `post_qwen_i2i.py` (they send to Decoder first, then Transformer, then Encoder, then poll Decoder). Set `IMAGE_PATH` in the script for I2I.

For T2I, first complete the model directory, JSON checkpoint path, and GPU settings in the [Qwen Image T2I quick start](#qwen-t2i-disagg-quickstart). Its API ports are Encoder `8002`, Transformer `8003`, and Decoder `8008`; the I2I scripts use `8012`, `8013`, and `8014`, respectively.

### 3.3 API summary

| Model | Task | Endpoint | Completion / result |
|-------|------|----------|----------------------|
| Wan2.1 | T2V / I2V | `/v1/tasks/video/` | **Decoder** node |
| Qwen Image | T2I / I2I | `/v1/tasks/image/` | **Decoder** node |

---

## 4. Decentralized queues: implementation and entry points

The repository retains RDMA queues, the Controller's ring service, Runner queue communication methods, and Qwen configs. Qwen queue mode cannot currently run end to end because its standalone pull-worker module is missing.

### 4.1 Components and data flow

The queue design separates dispatch from inference: Encoder writes task metadata to the Phase1 ring, Transformer consumes tasks and writes the next stage's metadata to the Phase2 ring, and Decoder consumes it for decoding. Controller maintains the rings and handshake services.

```text
Encoder → Phase1 RDMA ring → Transformer → Phase2 RDMA ring → Decoder
                   ↑                              ↑
                   └──── Controller ring service ─┘
```

| Component | Current implementation and role |
|-----------|---------------------------------|
| RDMA ring | [RDMABuffer](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/disagg/rdma_buffer.py) provides metadata buffers |
| Controller | `serve_rdma_dispatch_only()` in [ControllerService](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/disagg/services/controller.py) serves request, Phase1, and Phase2 rings |
| Runner communication | [DisaggMixin](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/disagg/disagg_mixin.py) implements queue connections, request snapshots, and Transformer/Decoder dispatch preparation |

The `DisaggMixin` source also documents a multi-consumer ordering limitation: the current FAA implementation is not a true remote atomic operation across processes. Correctness with concurrent workers needs separate validation; distinct ranks alone do not establish it.

### 4.2 Configuration reference

These configs remain in the current repository. Adjust model paths and backends for the actual environment:

| Role | Configuration file |
|------|--------------------|
| Controller | [qwen_image_t2i_disagg_controller.json](https://github.com/ModelTC/LightX2V/blob/main/configs/disagg/qwen/qwen_image_t2i_disagg_controller.json) |
| Encoder | [qwen_image_t2i_disagg_encoder_decentralized.json](https://github.com/ModelTC/LightX2V/blob/main/configs/disagg/qwen/qwen_image_t2i_disagg_encoder_decentralized.json) |
| Transformer | [qwen_image_t2i_disagg_transformer_decentralized.json](https://github.com/ModelTC/LightX2V/blob/main/configs/disagg/qwen/qwen_image_t2i_disagg_transformer_decentralized.json) |
| Decoder | [qwen_image_t2i_disagg_decode_decentralized.json](https://github.com/ModelTC/LightX2V/blob/main/configs/disagg/qwen/qwen_image_t2i_disagg_decode_decentralized.json) |

Encoder, Transformer, and Decoder select queue communication with `decentralized_queue: true` in `disagg_config`; the Controller config defines the ring service. Key fields are:

| Field | Meaning |
|-------|---------|
| `rdma_request_handshake_port` | Controller request-ring handshake port; 5566 in the example |
| `rdma_phase1_handshake_port` | Phase1 handshake port shared by the relevant components; 5567 in the example |
| `rdma_phase2_handshake_port` | Phase2 handshake port shared by the relevant components; 5568 in the example |
| `rdma_phase1_host` / `rdma_phase2_host` | Handshake host for the corresponding ring |
| `rdma_buffer_slots` / `rdma_buffer_slot_size` | Metadata-ring slot count and bytes per slot; 128 and 4096 in the examples |
| `receiver_engine_rank` / `transformer_engine_rank` | Transformer's receiving and execution ranks; keep role assignments consistent and use different ranks for different workers |
| `decoder_engine_rank` | Decoder rank, shared consistently by the relevant components |

### 4.3 Current entry points

| Use case | Existing entry point and scope |
|----------|--------------------------------|
| Qwen Image three-stage HTTP services | Use the launch and request scripts in the [T2I quick start](#qwen-t2i-disagg-quickstart), after configuring the model directory, DiT checkpoint, GPUs, and dependencies |
| Dynamic Wan disaggregation | See the [Disagg guide](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/disagg/README.md) and [run_dynamic.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/disagg/run_dynamic.sh); configure paths, devices, and networking for the topology. The launcher cleans up old services |
| Individual Wan service roles | [run_service.py](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/disagg/examples/run_service.py) accepts `--service encoder/transformer/decoder/controller`; the service implementations currently load Wan components |

`run_service.py` supports Wan services; it cannot replace the Qwen pull-worker.

### 4.4 Request and dispatch field boundaries

The internal request snapshot in `DisaggMixin` uses `data_bootstrap_room` to identify a transfer room and can carry `disagg_phase1_receiver_engine_rank` to select a Transformer rank; `save_result_path` specifies the output path. These are internal configuration/dispatch fields, not automatically HTTP request fields.

The current [HTTP schema](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/server/schema.py) forbids extra fields, and `ImageTaskRequest` does not declare the room and rank fields above. For HTTP, use the existing request scripts in section 3. Queue-mode completion reporting, polling endpoints, and Decoder result delivery still need validation with a complete worker implementation.

## 5. RDMA vs TCP

| Aspect | **RDMA** | **TCP** |
|--------|----------|---------|
| Transport | Zero-copy, kernel bypass | Kernel network stack |
| CPU | Very low | Higher |
| Latency | Microseconds | Milliseconds |
| Hardware | InfiniBand / RoCE | Any network |
| When to use | Production, multi-node | Single-node or no RDMA |

Set `export MOONCAKE_PROTOCOL=tcp` before startup to use TCP for Mooncake transfers. Decentralized queues still require RDMA hardware.
