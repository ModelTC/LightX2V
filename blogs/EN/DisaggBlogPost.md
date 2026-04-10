# Scaling Diffusion Model Serving with Disaggregated Deployment in LightX2V

By LightX2V and Mooncake

Diffusion models for image and video generation—such as Wan and Qwen Image—have been surging in capability, but their adoption in production is often blocked by a simple reality: **the models don't fit**. A single Qwen-Image-2512 pipeline demands ~58 GB of GPU memory in BF16, overflowing even an A100-40GB. On consumer GPUs like the RTX 4090 (24 GB), a common workaround is CPU offloading, but offloading also introduces significant additional inference latency.

In this post, we introduce **disaggregated deployment** in [LightX2V](https://github.com/ModelTC/LightX2V), a three-stage architecture that splits the inference pipeline into Encoder, Transformer (DiT), and Decoder microservices. The stages are connected via the [Mooncake](https://github.com/kvcache-ai/Mooncake) RDMA engine. Combined with a novel **decentralized queue scheduler**, this approach delivers:

- **32.2x** Text Encoder speedup on RTX 4090 (offload eliminated)
- Up to **1.54x** DiT per-step speedup
- Up to **1.89x** throughput (QPS) on 8 GPUs
- **< 0.2%** network overhead (Mooncake RDMA)

**Table of contents:**

- [Background: The Memory Wall in Diffusion Inference](#background-the-memory-wall-in-diffusion-inference)
- [Three-Stage Disaggregated Architecture](#three-stage-disaggregated-architecture)
- [Mooncake Engine: Near-Zero Communication Overhead with RDMA](#mooncake-engine-near-zero-communication-overhead-with-rdma)
- [Theoretical Analysis: Optimal Encoder-Transformer Ratios](#theoretical-analysis-optimal-encoder-transformer-ratios)
- [Decentralized Queue Scheduling](#decentralized-queue-scheduling)
- [Benchmark Results](#benchmark-results)
- [Conclusion](#conclusion)

---

## Background: The Memory Wall in Diffusion Inference

Modern diffusion pipelines are composed of several heavyweight components that must all reside in GPU memory simultaneously. The table below shows the parameter counts and memory footprints of the major components:

| Component | Parameters | BF16 Weight |
|---|---|---|
| Qwen-Image-2512 DiT | 20.43B | 40.9 GB |
| Qwen2.5-VL (Text Encoder) | 8.29B | 16.6 GB |
| Qwen Image VAE | 0.13B | 0.3 GB |
| Wan2.1 DiT-14B | 14.0B | 28.0 GB |
| T5-XXL (Wan Text Encoder) | ~4.7B | 9.4 GB |
| CLIP ViT-H/14 (Wan I2V) | ~0.63B | 1.3 GB |
| Wan VAE | ~0.10B | 0.2 GB |

When these components are loaded together, the total memory requirement quickly exceeds what most GPUs can provide:

| Configuration | Weight + Activation | RTX 4090 (24 GB) | A100 (40 GB) | A100 (80 GB) | H100 (80 GB) |
|---|---|---|---|---|---|
| Qwen-2512 BF16 Baseline | ~62–68 GB | OOM | OOM | Tight | OK |
| Qwen-2512 BF16 Disagg Encoder | ~18–20 GB | **OK** | OK | OK | OK |
| Wan 14B BF16 Baseline | ~42–48 GB | OOM | OOM | OK | OK |
| Wan 14B BF16 Disagg Encoder | ~11–13 GB | **OK** | OK | OK | OK |
| Wan 14B INT8 Disagg DiT | ~17–20 GB | **OK** | OK | OK | OK |

The standard workaround on memory-constrained GPUs is `cpu_offload`, which swaps model weights between CPU and GPU memory on-the-fly. While this makes inference *possible*, it comes at a severe performance cost: on the RTX 4090, the Qwen-2512 Text Encoder latency balloons from **0.40 s** (disaggregated, no offload needed) to **12.89 s** (baseline with offload)—a **32x** slowdown. The larger the model, the greater this latency increase.

---

## Three-Stage Disaggregated Architecture

LightX2V splits the monolithic pipeline into three independent services, each loading only its own subset of model weights:

```
┌──────────┐  HTTP POST   ┌──────────────────┐ Phase1 RDMA  ┌───────────────┐ Phase2 RDMA  ┌──────────────┐
│  Client  │ ──────────→  │     Encoder      │ ──────────→  │  Transformer  │ ──────────→  │   Decoder    │
└──────────┘              │ (Text/Image/VAE) │              │    (DiT)      │              │ (VAE Decode) │
                          │   ~17–20 GB      │              │  ~28–41 GB    │              │   ~0.3 GB    │
                          └──────────────────┘              └───────────────┘              └──────────────┘
```

- **Encoder** (`disagg_mode="encoder"`): Loads Text Encoder, Image Encoder (for I2V/I2I), and VAE Encoder. Sends feature tensors to Transformer via Mooncake Phase1.
- **Transformer** (`disagg_mode="transformer"`): Loads only the DiT model. Receives Phase1 data, runs denoising, sends latents to Decoder via Mooncake Phase2.
- **Decoder** (`disagg_mode="decode"`): Loads only the VAE Decoder. Receives latents, decodes to pixels, and saves output.

### Measured Peak Memory

| Model | Mode | GPU | Peak Memory |
|---|---|---|---|
| Qwen-2512 BF16 | Baseline | H100 | ~58 GB |
| Qwen-2512 BF16+offload | Disagg Encoder | H100 | **~18 GB** |
| Qwen-2512 BF16+offload | Disagg Transformer | H100 | **~40 GB** |
| Wan T2V 14B | Baseline | H100 | ~39 GB |
| Wan T2V 14B | Disagg Encoder | H100 | ~11 GB |
| Wan T2V 14B | Disagg Transformer | H100 | ~28 GB |
| Wan I2V 14B 480P | Baseline | H100 | ~48 GB |
| Wan I2V 14B 480P | Disagg Encoder | H100 | ~13 GB |
| Wan I2V 14B 480P | Disagg Transformer | H100 | ~32 GB |

By splitting components across GPUs, each node's memory footprint drops to a fraction of the baseline. Critically, the Encoder stage fits comfortably on an RTX 4090 **without any offloading**, unlocking the full 32x Text Encoder speedup.

---

## Mooncake Engine: Near-Zero Communication Overhead with RDMA

A natural concern with disaggregated deployment is communication overhead. LightX2V integrates the [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake), which provides zero-copy RDMA transport with microsecond-level latency.

### End-to-End Latency Breakdown

We profiled a single Qwen-2512 T2I request (50 steps) on H100 to measure where time is spent:

| Stage | Latency | % of Total |
|---|---|---|
| Encoder: Text Encoder computation | ~0.26 s | 1.0% |
| Encoder: Phase1 send (serialize + RDMA) | ~0.025 s | 0.1% |
| Transformer: DiT inference (50 steps) | ~25.3 s | 96.2% |
| Transformer: Phase2 send (serialize + RDMA) | ~0.024 s | 0.09% |
| Decoder: VAE Decode | ~0.31 s | 1.2% |
| **Network total (Phase1 + Phase2)** | **~0.05 s** | **0.2%** |
| **Pipeline total** | **~26.3 s** | |

The network overhead is dominated by serialization (~4 ms) and Mooncake RDMA transfer (~20 ms) per phase. On H100 with InfiniBand (400 GB/s theoretical bandwidth), even 100 MB transfers complete in sub-millisecond time.

### Transfer Sizes Across Models

| Transfer Phase | Data Content | Typical Size | Model |
|---|---|---|---|
| Phase1 (Wan T2V) | context (512×4096 BF16) | ~4 MB | Wan2.1-14B |
| Phase1 (Wan I2V) | context + clip_out + vae_enc | ~20 MB | Wan2.1-14B |
| Phase1 (Qwen T2I) | prompt_embeds (4096×3584 BF16) | ~28 MB | Qwen-2512 |
| Phase2 (Wan 480P 81f) | latent (16×21×60×104 BF16) | ~42 MB | Wan2.1-14B |
| Phase2 (Wan 720P 81f) | latent (16×21×90×160 BF16) | ~97 MB | Wan2.1-14B |
| Phase2 (Qwen T2I 16:9) | latent (16×1×104×58 BF16) | ~0.2 MB | Qwen-2512 |

### Cross-Model Network Overhead

| Model | Encoder | DiT Total | VAE Decoder | Pipeline Total | Network Overhead |
|---|---|---|---|---|---|
| Qwen-2512 T2I 50 step (H100) | 0.30 s | 22.04 s | ~2.7 s | ~25.0 s | < 0.1% |
| Wan 14B T2V 50 step (H100) | 0.89 s | 252.8 s | 2.38 s | ~253.5 s | < 0.01% |
| Wan 14B I2V 40 step (H100) | 3.17 s | 207.7 s | 2.22 s | ~210.9 s | < 0.02% |

The communication cost is negligible across all models and configurations. Mooncake's RDMA integration ensures that disaggregation introduces virtually zero latency overhead.

| Aspect | **RDMA** | **TCP** |
|---|---|---|
| Transport | Zero-copy, kernel bypass | Kernel network stack |
| CPU overhead | Very low | Higher |
| Latency | Microseconds | Milliseconds |
| Hardware | InfiniBand / RoCE NIC | Any network |
| Recommendation | Production, multi-node | Single-node testing, no RDMA hardware |

---

## Theoretical Analysis: Optimal Encoder-Transformer Ratios

For simplicity, the Encoder and Decoder are currently deployed on the same GPU, sharing compute resources.

### Throughput Model

Consider a system with E Encoder GPUs and T Transformer GPUs (the Decoder shares a GPU with the Encoder since its footprint is negligible). Let t_e be the Encoder processing time per request and t_t be the Transformer processing time per request.

The throughput of each stage is:

$$R_e = \frac{E}{t_e}, \quad R_t = \frac{T}{t_t}$$

The system throughput is bounded by the bottleneck:

$$R_{\text{system}} = \min(R_e, R_t)$$

To maximize resource utilization, we balance the two stages:

$$R_e = R_t \implies \frac{E}{t_e} = \frac{T}{t_t} \implies \boxed{\frac{T}{E} = \frac{t_t}{t_e}}$$

**The optimal Transformer-to-Encoder ratio equals the ratio of their per-request processing times.**

### How Disaggregation Changes the Ratio

A critical insight is that disaggregation itself dramatically shifts this ratio by accelerating the Encoder stage (eliminating offload overhead):

| Scenario | Text Encoder | DiT (8 steps) | DiT:Enc Ratio |
|---|---|---|---|
| Baseline (4090 + offload) | 12.89 s | 23.2 s | **1.8 : 1** |
| Disagg (4090) | 0.40 s | 15.0 s | **37.5 : 1** |
| Baseline (4090 + offload) | 12.89 s | 287.5 s (50 steps) | **22.3 : 1** |
| Disagg (4090) | 0.40 s | 188 s (50 steps) | **470 : 1** |

By reducing Encoder time by 32x, disaggregation makes the ratio far more extreme, underscoring the importance of allocating as many GPUs as possible to the Transformer stage.

### Ratio vs. Number of Inference Steps

The step count is the primary factor driving the optimal ratio (Qwen-2512, 4090):

| Steps | T\_encoder | T\_DiT | DiT:Encoder Ratio | Recommended |
|---|---|---|---|---|
| 50 | 0.4 s | 188 s | 470 : 1 | Max Transformer GPUs |
| 8 | 0.4 s | 15.04 s | 37.6 : 1 | ~38 : 1 |
| 4 | 0.4 s | 7.52 s | 18.8 : 1 | ~19 : 1 |
| 2 | 0.4 s | 3.76 s | 9.4 : 1 | ~10 : 1 |
| 1 | 0.4 s | 1.88 s | 4.7 : 1 | ~5 : 1 |

### Multi-Scale Optimal Ratios

**8 GPUs** (common single-node setup):

| Model | Steps | Theoretical Ratio | Optimal Config | Analysis |
|---|---|---|---|---|
| Qwen-2512 | 8 | 37.5 : 1 | **7T : 1E** | 1 Encoder @ 2.5 req/s >> 7T bottleneck @ 0.47 req/s |
| Qwen-2512 | 50 | 470 : 1 | **7T : 1E** | Encoder is never the bottleneck |

**16 GPUs** (8-step distilled model):

| Config | T Throughput | E Throughput | Bottleneck | E Utilization |
|---|---|---|---|---|
| 15T : 1E | 1.0 req/s | 2.5 req/s | Transformer | 40% |
| 14T : 2E | 0.93 req/s | 5.0 req/s | Transformer | 18.7% (wasteful) |

Conclusion: **15:1** is optimal—adding a second Encoder wastes a GPU.

**32 GPUs** (50-step model):

31T : 1E yields 0.165 req/s throughput. The single Encoder (2.5 req/s capacity) runs at only 6.6% utilization—still not the bottleneck.

**800 GPUs** (8-step distilled model)—the first scale where Encoder matters:

| Config | T Throughput | E Throughput | System Throughput | E Utilization | T Utilization |
|---|---|---|---|---|---|
| 790T : 10E | 52.67 req/s | 25 req/s | 25 req/s | 100% | 47.5% |
| 780T : 20E | 52 req/s | 50 req/s | 50 req/s | 100% | 96.2% |
| **779T : 21E** | **51.93 req/s** | **52.5 req/s** | **51.93 req/s** | **98.9%** | **99.8%** |

The theoretical optimum is $E = N / (1 + t_t/t_e) = 800 / 38.5 \approx 21$.

### Practical Configuration Guide

| Scale | 8-step Distilled | 50-step Full |
|---|---|---|
| **8 GPUs** | 7T : 1E | 7T : 1E |
| **16–64 GPUs** | 1 Encoder per 30–40 GPUs | 1 Encoder per 60–100 GPUs |
| **100+ GPUs** | $E = \lceil N / (1 + t_t/t_e) \times 1.2 \rceil$ | Same formula |

**When to add more Encoders:**

1. Measured Encoder utilization exceeds 80%
2. P95 latency is significantly higher than P50 (queuing is severe)
3. Concurrent request count exceeds the number of Transformer workers

Otherwise, **always prioritize adding more Transformer GPUs**.

---

## Decentralized Queue Scheduling

In the standard three-stage deployment, the client must send requests to Decoder → Transformer → Encoder in sequence, which is operationally complex and introduces scheduling overhead. LightX2V introduces a **decentralized queue scheduler** that simplifies this to a single HTTP POST:

```
┌──────────┐  HTTP POST   ┌──────────┐ Phase1 RDMA ┌─────────────┐ Phase2 RDMA ┌──────────┐
│  Client  │ ──────────→  │ Encoder  │ ──────────→ │ Transformer │ ──────────→ │ Decoder  │
└──────────┘              │ (GPU 0)  │             │ (GPU 1/2/3) │             │ (GPU 0)  │
                          └──────────┘             └─────────────┘             └──────────┘
                                ↑                        ↑                          ↑
                          lightx2v.server          pull worker ×N              pull worker
                          HTTP port 8002
                                │
                          ┌──────────┐
                          │Controller│  ← RDMA metadata ring buffer (always-on)
                          └──────────┘
```

### Key Design Decisions

- **Controller**: Maintains three RDMA ring buffers (request / phase1 / phase2) for metadata dispatch. Loads no models and performs no inference.
- **Encoder**: Runs as an HTTP service, performs Text Encoder inference, then writes dispatch metadata to the Phase1 RDMA ring.
- **Transformer** and **Decoder**: Run as pull-based workers that consume from their respective RDMA rings automatically.
- **Multiple Transformer workers** can be deployed on different GPUs. The client specifies `disagg_phase1_receiver_engine_rank` to target a specific worker, enabling round-robin or explicit routing.

### Comparison with Standard Three-Stage

| Aspect | Standard Three-Stage | Decentralized Queue |
|---|---|---|
| Client calls | Must POST to Decoder → Transformer → Encoder | **Single POST to Encoder HTTP** |
| Transformer | HTTP server, one request at a time | Pull worker, multiple instances in parallel |
| Decoder | HTTP server | Pull worker, auto-consumes Phase2 |
| Request routing | Client explicitly specifies all endpoints | Encoder writes RDMA ring, workers pull by rank |
| Result retrieval | Poll Decoder HTTP | Poll Encoder HTTP |
| Scaling | Fixed 1:1:1 ratio | **Flexible N Transformer : M Encoder** |

### Impact on Throughput and Tail Latency

The decentralized scheduler eliminates sequential request dispatch overhead and reduces queue contention. On 8× RTX 4090 with 7:1 ratio:

| Scheduling | QPS | P50 | P95 | P99 |
|---|---|---|---|---|
| Centralized three-stage (7:1) | 0.24 | 17 s | 25 s | 28 s |
| **Decentralized queue (7:1)** | **0.34** | **17 s** | **20 s** | **22 s** |

The decentralized scheduler improves QPS by **1.42x** over the centralized three-stage approach (and **1.89x** over baseline) while significantly reducing tail latency—P95 drops from 25 s to 20 s, and P99 from 28 s to 22 s.

---

## Benchmark Results

### How to Run (Qwen-2512 T2I Decentralized Deployment)

Taking a 4-GPU (1 Encoder + 3 Transformer) Qwen-2512 T2I decentralized deployment as an example, a single command starts all services:

```bash
git clone git@github.com:ModelTC/LightX2V.git
cd LightX2V
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg_decentralized.sh
```

The script launches Controller → Encoder → Decoder → Transformer×3 (5 processes total) with the default GPU layout:

| Role | GPU | Notes |
|---|---|---|
| Controller | CPU | RDMA ring buffer only, no GPU needed |
| Encoder + Decoder | GPU 0 | Shared GPU (both have small memory footprint) |
| Transformer 1 | GPU 1 | DiT worker |
| Transformer 2 | GPU 2 | DiT worker |
| Transformer 3 | GPU 3 | DiT worker |

GPU assignments and the number of Transformers can be customized via environment variables:

```bash
GPU_ENCODER=0 GPU_DECODER=0 \
GPU_TRANSFORMER_1=1 GPU_TRANSFORMER_2=2 GPU_TRANSFORMER_3=3 \
NUM_TRANSFORMERS=3 \
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg_decentralized.sh
```

Once the services are up, send requests and query results via the Controller's HTTP API:

```bash
# Send a generation request
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cute cat on a table", "seed": 42, "aspect_ratio": "16:9"}'

# Check result status
curl http://localhost:8080/v1/status/{room_id}
```

### Configuration Walkthrough

Decentralized deployment configs are in `configs/disagg/qwen/`, with one JSON file per role. Below are the key fields for each:

**Controller** (`qwen_image_t2i_disagg_controller.json`):

```json
{
  "disagg_mode": "controller",
  "disagg_config": {
    "protocol": "rdma",
    "rdma_buffer_slots": 128,
    "rdma_buffer_slot_size": 4096,
    "rdma_request_handshake_port": 5566,
    "rdma_phase1_handshake_port": 5567,
    "rdma_phase2_handshake_port": 5568
  }
}
```

The Controller loads no model weights—it only initializes three RDMA ring buffers (request / phase1 / phase2) for metadata dispatch. `rdma_buffer_slots` controls queue depth, and `rdma_*_handshake_port` specifies the RDMA handshake port for each ring.

**Encoder** (`qwen_image_t2i_disagg_encoder_decentralized.json`):

```json
{
  "disagg_mode": "encoder",
  "disagg_config": {
    "decentralized_queue": true,
    "sender_engine_rank": 0,
    "receiver_engine_rank": 1,
    "rdma_phase1_host": "127.0.0.1",
    "rdma_phase1_handshake_port": 5567
  }
}
```

`decentralized_queue: true` activates the decentralized scheduling mode. After inference, the Encoder writes feature metadata into the Phase1 RDMA ring for Transformer workers to pull.

**Transformer** (`qwen_image_t2i_disagg_transformer_decentralized.json`):

```json
{
  "disagg_mode": "transformer",
  "disagg_config": {
    "decentralized_queue": true,
    "transformer_engine_rank": 1,
    "decoder_engine_rank": 4,
    "rdma_phase1_handshake_port": 5567,
    "rdma_phase2_handshake_port": 5568
  }
}
```

The Transformer runs as a pull worker, consuming tasks from the Phase1 ring. After DiT inference, it writes latents to the Phase2 ring. When deploying multiple Transformers, each worker uses a different `transformer_engine_rank` (handled automatically by the startup script).

**Decoder** (`qwen_image_t2i_disagg_decoder_decentralized.json`):

```json
{
  "disagg_mode": "decode",
  "disagg_config": {
    "decoder_engine_rank": 2,
    "rdma_phase2_handshake_port": 5568
  }
}
```

The Decoder loads only the VAE Decoder (~0.3 GB), receiving latents from the Phase2 ring and decoding them into the final image.

### Per-Stage Speedup on Memory-Constrained GPUs

The most dramatic speedups occur on memory-constrained GPUs where baseline inference requires CPU offloading. On RTX 4090 with Qwen-2512 (BF16 + block offload):

| Component | Baseline (offload) | Disagg (no offload on Encoder) | Speedup |
|---|---|---|---|
| Text Encoder | 12.89 s | 0.40 s | **32.22x** |
| DiT per step (50 steps) | 5.75 s/step | 3.76 s/step | **1.53x** |
| DiT per step (8 steps, distilled) | 2.90 s/step | 1.88 s/step | **1.54x** |

The Text Encoder speedup comes from eliminating weight offloading entirely—the Encoder node's ~17 GB footprint fits on a 4090 without any offloading. The DiT speedup comes from exclusive VRAM bandwidth: in baseline mode, DiT competes with other components for PCIe and memory bandwidth during offload transfers.

For Wan2.1-I2V-14B on RTX 4090 (BF16, 40 steps, block offload):

| Metric | Baseline (1×4090) | Disagg (2×4090) |
|---|---|---|
| DiT per step (480P) | 24.24 s/step | 19.02 s/step |
| DiT per step (720P) | 90.71 s/step | 62.80 s/step |
| Text Encoder | 2.14 s | 0.20 s |
| Image Encoder (480P) | 0.57 s | 0.28 s |

**As models grow larger, the offload penalty increases proportionally, making disaggregation's advantage more pronounced.** Qwen-2512 (58 GB baseline) shows a 32x Text Encoder speedup, compared to 10x for the smaller Wan 14B (39 GB baseline).

### Multi-GPU Throughput Scaling

We benchmarked 100 requests of Qwen-2512 T2I on RTX 4090 GPUs, comparing baseline (all components on each GPU) against disaggregated deployment with different DiT:Encoder ratios.

**4 GPUs — Centralized Three-Stage Scheduling:**

| Mode | DiT:Enc | Concurrency | Total Time | P50 | P95 | P99 | QPS |
|---|---|---|---|---|---|---|---|
| Baseline | 4:0 | 4 | 1079 s | 41 s | 55 s | 61 s | 0.092 |
| Disagg | 2:2 | 2 | 1037 s | 35 s | 62 s | 63 s | 0.096 |
| Disagg | **3:1** | **3** | **705 s** | **35 s** | **38 s** | **39 s** | **0.15** |

**8 GPUs — Decentralized Queue Scheduling:**

| Mode | DiT:Enc | Concurrency | Total Time | P50 | P95 | P99 | QPS |
|---|---|---|---|---|---|---|---|
| Baseline | 8:0 | 8 | 550 s | 22 s | 28 s | 30 s | 0.18 |
| Disagg | 4:4 | 4 | 497 s | 20 s | 23 s | 25 s | 0.20 |
| Disagg | 5:3 | 5 | 402 s | 19 s | 20 s | 23 s | 0.25 |
| Disagg | 6:2 | 6 | 335 s | 18 s | 20 s | 22 s | 0.30 |
| Disagg | **7:1** | **7** | **294 s** | **17 s** | **20 s** | **22 s** | **0.34** |

*RTX 4090, Qwen-2512 T2I, BF16 + block offload, 100 requests. Decentralized scheduling for disagg rows.*

At the optimal 7:1 ratio on 8 GPUs, disaggregated deployment achieves **0.34 QPS** vs baseline **0.18 QPS**—a **1.89x throughput improvement**. The decentralized scheduler also significantly reduces tail latency: P95 drops from 28 s (baseline) to 20 s (disagg 7:1), and P99 from 30 s to 22 s.

### Sensitivity to Inference Parameters

**Prompt length** has minimal impact on the DiT:Encoder ratio, since Encoder latency stays well below DiT latency across all lengths (Qwen-2512, H100):

| Prompt Length | Encoder Latency | DiT 4-step Total | Ratio |
|---|---|---|---|
| 16 tokens | 35 ms | 1702 ms | 49:1 |
| 256 tokens | 53 ms | 1735 ms | 33:1 |
| 1024 tokens | 102 ms | 1855 ms | 18:1 |
| 4096 tokens | 81 ms | 1877 ms | 23:1 |

**Resolution** affects DiT latency but not Encoder latency, widening the ratio for larger outputs (Qwen-2512, 4090 + offload, 50 steps):

| Resolution | Aspect Ratio | DiT Total | DiT Per-Step |
|---|---|---|---|
| 1664×928 | 16:9 | 188 s | 3.76 s |
| 1328×1328 | 1:1 | 204 s | 4.07 s |
| 1472×1140 | 4:3 | 215 s | 4.29 s |

---

## Conclusion

In this post, we presented disaggregated deployment in LightX2V—a three-stage architecture that physically separates Encoder, Transformer, and Decoder onto independent GPU nodes. By integrating Mooncake's RDMA transport (< 0.2% overhead) and a decentralized queue scheduler, we achieve:

1. **Memory decoupling**: Each node loads only its own component, enabling models like Qwen-2512 (58 GB) to run on RTX 4090 (24 GB) without offloading.
2. **Massive Encoder acceleration**: Eliminating offload yields a 32x Text Encoder speedup on memory-constrained GPUs.
3. **Flexible scaling**: Optimal Encoder:Transformer ratios follow $T:E = t_t : t_e$, with practical configs ranging from 7:1 (8 GPUs) to 779:21 (800 GPUs).
4. **Production-ready throughput**: 1.89x QPS improvement with decentralized scheduling on 8 GPUs.

As diffusion models continue to grow—20B, 50B, and beyond—the gap between monolithic and disaggregated deployment will only widen. The Encoder stage stays small, while the DiT stage grows with model parameters, making the case for disaggregation stronger with every generation.
