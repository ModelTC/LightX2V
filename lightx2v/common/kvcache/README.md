KV storage, positional policy and attention execution have separate lifetimes. The transformer infer owns `KVCacheManager` and selects its attention adapter from the configured backend. A Sage backend adds quantized companion buffers; it does not replace the raw KV pool.

| Module | Responsibility |
| --- | --- |
| `base.py`, `static.py`, `fifo.py` | Common tensor storage and existing model-specific policies |
| `layout.py` | Sink/recent ring indices, spans and cache update plans |
| `rolling.py` | Resident rolling, per-step and hybrid storage; physical views |
| `slot.py` | Persistent CPU/GPU slot metadata and ordered uploads |
| `manager.py` | Request layout, pool allocation, reuse and reset |
| `offload.py`, `rolling_offload.py` | CPU backing, one-layer GPU staging and transfer ordering |
| `condition.py` | Projected condition KV, source identity and invalidation |
| `kernel.py` | Slot writes, fused RoPE workspace updates and physical-slot Q/K rotation |
| `attn/ring.py` | `RingAttention`: direct physical-view attention |
| `attn/sage.py` | `SageRingAttention`: incremental quantization and segmented Sage attention |
| `attn/rope.py` | `RoPERingAttention`: logical RoPE workspace; `PhysicalRoPERingAttention`: physical-order K workspace |
| `attn/kernels/` | Sage ring quantization, reduction and merge kernels |

| Model policy | Raw KV storage | Attention adapter and positional handling |
| --- | --- | --- |
| SF / RealtimeVideo / LingbotFast | `RollingKVCachePool`, shared by denoising steps | K receives causal RoPE before storage. FA2/FA3 use `RingAttention`; Sage2 uses `SageRingAttention`. The full ring is read in physical slot order without concatenation or an ordered workspace. A narrower attention window gathers its live spans into reusable K/V buffers outside the full-ring path. |
| Matrix Game 4 | The same rolling pool | The same FA2/FA3/Sage2 adapters, with model-owned `SinkRoPE` handling its fixed or absolute sink anchor. |
| EdenLive | Raw-key rolling pool, shared by denoising steps | FA2/FA3/Sage2 use `RoPERingAttention`: gather and rotate history into reusable logical K/V workspaces once per chunk, then call the configured dense attention backend. Sage2 retains main's full-window quantization. |
| Seko AR 2-step / 4-step | `StepRollingKVCachePool`; 2-step hybrid windows use `HybridStepRollingKVCachePool` | `PhysicalRoPERingAttention` rotates raw K into a physical-order K workspace; V is a direct ring view. The 2-step model uses global positions capped at 40 frames, while the 4-step model uses local window positions. Audio sequence parallel retains its head-sharded positional policy. |
| Other autoregressive models | Their existing FIFO/static/rolling policy | Existing model-specific positional and attention paths remain in place. They do not automatically inherit the optimized physical-view policy. |

`ar_config.kv_offload` selects the corresponding `OffloadedRollingKVCachePool`, `OffloadedStepRollingKVCachePool` or `OffloadedHybridStepRollingKVCachePool`. They preserve each policy's layout while sharing one GPU staging layer. Transfers initially copy only the initialized prefix; kernel writes persist their exact physical spans. EdenLive shares one K/V workspace across offloaded layers and refreshes it after each layer load, for both FlashAttention and Sage2. Weight block offload and KV offload are independent; their transfers remain outside captured computation.

`KVCacheManager.initialize(latent_shape, ...)` reuses compatible allocations and resets request lengths and positions. Changed layouts invalidate Graph bindings before reallocation. Adapters retain stable buffers and physical write views for mutation tracking and Graph capture. Full-block and clean KV-only functions are compiled before binding the block argument, so their Dynamo caches remain independent.

Text and image cross-attention use separate projected-KV buffers, invalidated by condition identity. EdenLive's `prepare_condition_cache`, `prepare_rope_cache` and `finish_rope_cache` express its window-position policy; the inherited causal `prepare_self_attn_cache(pool, layer_id, update)` and `prepare_cross_attn_cache(blocks, context)` keep their common contracts.

SF-family Ulysses SP preserves the configured causal RoPE precision. With the default float64 setting, the dynamic-offset kernel rounds frequencies to complex64 and performs the rotation in FP64, including the FP32 conversion before BF16 storage used by the original PyTorch path. Single-GPU and other models keep their existing FP32 kernel. This changes only current-token rotation; ring storage, direct physical views and CUDA Graph slot updates remain the same.

Distributed Ring Attention remains in `common/ops/attn/ring_attn.py`; it is separate from rolling KV storage. Current-token writes, RoPE/quantization work and slot metadata uploads are still required even when historical concatenation is removed.

The causal Wan executor uses the same prepared-cache interface for resident weights, block weight offload, KV offload, SP and TP. Cache planning, metadata uploads and CPU/GPU transfers happen outside compiled and captured computation. Resident execution captures the block stack; staging execution captures each block after its weights and KV have arrived. First chunks, growing windows, reference prefill, clean commits and per-step cache banks have their own shape variants. No `resident` or legacy-execution switch selects an unoptimized CUDA path.

`use_compile`, `use_cuda_graph` and `warmup` can be enabled together. Compiled block functions run inside explicit CUDA Graphs; Inductor's own graph capture is disabled for this combination. Warmup covers the configured target shape and growing/full-window chunks. SF, LingbotFast, EdenLive AR and Audio AR use target-only warmup; MG4 also retains its model-specific preset shapes. A new request shape or changed external tensor storage requires a new capture. A compatible request reuses captured variants after resetting cache contents and condition identities.

For Ulysses SP with CUDA Graph on NCCL versions below 2.32.3, startup sets `NCCL_GRAPH_REGISTER=0` before process groups are created. Those releases can reuse incomplete IPC mappings for cuMem buffer views and hang or corrupt graph communication. This version-limited compatibility fix preserves compile and Graph execution; NCCL 2.32.3 and later need no override. See [torchtitan #5393](https://github.com/pytorch/torchtitan/pull/5393) and [NCCL #2362](https://github.com/NVIDIA/nccl/pull/2362).

Sequential graphs share a `CUDAGraphPool`: graph scratch, capture stream, input staging and output staging are reused by layout. Every replay refreshes its inputs. Captured computation copies its result into shared external output storage and releases temporary outputs, so additional window or step variants do not retain one output allocation each. Callers consume or stage a result before invoking another graph from the same pool. A new capture first protects its input values outside the shared pool.

`cache_commit_enabled` is the model's KV-only clean-commit policy. Compatible models skip the final block's attention output projection, cross-attention and FFN when only its updated self-attention KV is consumed. Models whose final adapter output is needed leave this flag disabled. Both full and KV-only computations use the same compile and graph lifecycle.

Model-granularity weight offload releases graphs before moving weights back to CPU. Reloaded weight tensors receive new GPU addresses, so a subsequent model reload must capture again; weight bindings include concrete TP-wrapped tensor storage to detect such changes. Block-granularity offload retains fixed GPU staging addresses and reuses graphs across compatible requests.

Implementation support and validation coverage are separate. The 2026-10-21 H200 real-model multi-GPU/offload records use FA3 for the SF/EdenLive/MG4 Flash paths; earlier SF Sage2 resident videos and cache/Graph synthetic tests do not certify every Sage2 multi-GPU/offload combination. Audio/Seko has its own backend-specific records. See [Sage backend notes](../../../scripts/self_forcing/SAGE_ATTENTION.md). RTX 40/50 adapters have no corresponding hardware acceptance. The installed FA2 runtime also produces NaNs in uncached long-sequence baselines, so four such numerical tests were excluded; the FA2 code path remains implemented.

EdenLive's earlier Sage2 resident 717-frame case replayed 275 whole-stack graphs. The current FA3 SP2 + weight block/KV offload 717-frame case replayed 9000 staged graphs with zero formal captures and exact latent/float-video SHA against same-mode main. These counts describe different backends and execution modes, not interchangeable measurements.

The non-CUDA compatibility path is retained. MG4 streaming input encoding still requires resident VAE weights, independently of DiT weight/KV offload. Seko AR `lazy_load` warmup remains unsupported because its separate adapter checkpoint lacks that loading path. Model-granularity unload/reload requires recapture; zero formal recapture only applies to compatible allocations in the tested resident/block-staging configurations.
