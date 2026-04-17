---
name: xe2-nbarrier-pipelining
description: >
  Named barrier (nbarrier) split-barrier pipelining patterns for Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU. Xe2 is the GPU architecture; LNL and BMG are product names.
  SYCL ESIMD kernels. Covers arrive/wait split barriers for latency hiding, the ESIMD nbarrier API,
  producer-consumer patterns, and proven pipelining recipes from high-performance SDP kernels.
  Use this skill whenever the user mentions named barriers, split barriers, nbarrier, barrier_arrive,
  barrier_wait, latency hiding with barriers, pipelining independent operations across a sync point,
  or overlapping memory loads with barrier synchronization on Intel GPU.
---

# Named Barrier Pipelining on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG)

## Core Concept: Split Barrier

A traditional `barrier()` is blocking — all threads stall until everyone arrives. A **split barrier** separates this into two operations:

1. **`arrive`** (non-blocking): Thread signals "I finished my pre-barrier work (e.g., SLM writes)" and **continues executing immediately**.
2. **`wait`** (blocking): Thread checks if all participants have arrived. Stalls only if someone hasn't arrived yet.

The window between `arrive` and `wait` is where you insert **independent work** — memory loads, prefetches, ALU computation — that would otherwise be wasted stall cycles.

## ESIMD Named Barrier API

```cpp
// Initialize: declare how many named barriers this kernel uses (max 32 per Xe-Core)
__esimd_nbarrier_init(count);  // Call once at kernel start, before any barrier use

// Arrive (non-blocking): signal completion, continue executing
// barrier_reg: register index (0-based), nbarrier_id: hardware barrier ID,
// producer_count: threads that will arrive, consumer_count: threads that will wait
__esimd_nbarrier_arrive(barrier_reg, nbarrier_id, producer_count, consumer_count);

// Wait (blocking): stall until all producers have arrived
__esimd_nbarrier(barrier_reg, nbarrier_id, count);
```

Typical usage with a single barrier for a 32-thread workgroup:
```cpp
__esimd_nbarrier_init(1);           // 1 named barrier
// ... later, in the loop ...
__esimd_nbarrier_arrive(0, 0, 32, 32);  // all 32 threads produce and consume
// ... independent work ...
__esimd_nbarrier(0, 0, 32);             // wait for all 32
```

### Alternative: nbarrier class API
```cpp
#include <sycl/ext/intel/esimd.hpp>
using namespace sycl::ext::intel::esimd;

nbarrier nb;
nb.init(32);    // 32 participants
nb.arrive();    // non-blocking
// ... work ...
nb.wait();      // blocking
```

Both compile to the same `send.gtwy` vISA instructions. The raw `__esimd_nbarrier*` intrinsics give more control (separate producer/consumer counts).

## Hardware Details

- Named barriers use the **Message Gateway (GTWY)** hardware unit on each Xe-Core
- Up to **32 independent named barriers** per Xe-Core
- Each barrier tracks arrive counts in hardware — near-zero overhead
- `arrive` sends a single GTWY message (~1 cycle to issue)
- `wait` checks the GTWY counter — instant if all arrivals already recorded
- On Xe2 (BMG), a workgroup can have up to 32 subgroups (threads)

## Proven Pattern: SDP Kernel Split Barrier B

From the 84 TFLOPS HD=256 Flash Attention kernel (`rev256_onednn_v2.h`), this is the most impactful split barrier pattern:

```
Timeline per outer iteration:

  QK DPAS → Softmax → S transpose → S SLM store → [arrive] → V loads (async)    → [wait] → VS DPAS
                                                              → K prefetch (next)
                                                              → sum update (ALU)
                                                              → compensation (ALU)

  |<------------- all threads must complete S stores -------->|<-- independent -->|<- need S from SLM ->|
```

### What goes between arrive and wait

The window should contain work that:
1. **Does not read SLM** that other threads are writing (that's the whole point of the barrier)
2. **Does not write SLM** that other threads will read after the barrier
3. Is **genuinely useful** — memory loads that will be needed soon, ALU that must happen anyway

Concrete example from the 84 TFLOPS kernel:

```cpp
// All threads wrote their S tiles to SLM
// Signal: "my S writes are done"
__esimd_nbarrier_arrive(0, 0, 32, 32);

// === INDEPENDENT WORK WINDOW (~100+ cycles) ===

// 1. Deferred sum update (ALU, no SLM dependency)
fp32_sum = fp32_sum * delta + local_sum;

// 2. V global memory loads (async sends, high latency)
//    These will be consumed by VS DPAS after the wait
payloadV.set_y(vRow0);
payloadV.set_x(kvGlobalX + sg_i * 32);
V_vnni0 = lsc_load_2d<fp16, 16, 16, 1, false, true, cached, cached>(payloadV);
payloadV.set_x(kvGlobalX + sg_i * 32 + 16);
V_vnni1 = lsc_load_2d<fp16, 16, 16, 1, false, true, cached, cached>(payloadV);

// 3. K prefetch for NEXT iteration (async, hides cross-iteration latency)
payloadKpf.set_y(next_kv_row);
lsc_prefetch_2d<fp16, 16, 16, 1, false, false, cached, cached>(payloadKpf);

// 4. Deferred compensation (ALU, operates on own registers only)
for (qg) for (db) for (q)
  A_tile[...] *= delta[q];

// === END INDEPENDENT WORK ===

// Now we need to read S from SLM — must ensure all threads' writes are visible
__esimd_nbarrier(0, 0, 32);

// Safe to read S tiles from SLM
S0 = slm_block_load<uint32_t, 64>(s_base);
// ... VS DPAS phase ...
```

### Why this pattern works well

- V loads have ~200 cycle latency to global memory — perfectly hidden in the barrier window
- K prefetches warm L1/L3 cache for the next iteration — zero-cost when overlapped
- Compensation ALU (~64 multiply ops) overlaps with V load latency
- The `arrive` itself is 1 cycle — negligible overhead

## When barrier() Already Subsumes nbarrier

A regular `barrier()` (or `barrier(CLK_LOCAL_MEM_FENCE)`) synchronizes **all** SLM operations across the entire workgroup. If you already have a `barrier()` somewhere that guarantees your SLM writes are visible, an nbarrier for the same purpose is redundant.

Example from the SDP kernel: **barrier_A** (for softmax max reduction) uses `barrier()` because all threads must see each other's atomic max values. This barrier also implicitly syncs any prior SLM writes. So there's no need for a separate nbarrier before the max reduction.

Rule of thumb: Use nbarrier split barriers **only when you don't already have a full barrier nearby** that serves the same synchronization purpose.

## What Doesn't Work: Double-Arrive Deadlock

Never call `arrive` twice without an intervening `wait`:

```cpp
// WRONG — potential deadlock or undefined behavior
__esimd_nbarrier_arrive(0, 0, 32, 32);
// ... some work ...
__esimd_nbarrier_arrive(0, 0, 32, 32);  // BUG: second arrive before wait!
__esimd_nbarrier(0, 0, 32);
```

The hardware counter may overflow or wrap, causing threads to never see the expected count. Always pair each `arrive` with exactly one `wait`.

## What Doesn't Work: Double-Buffered S with nbarrier

Attempted in the HD=256 SDP kernel: use 2 S SLM buffers (S_A, S_B) so iteration N+1's S writes overlap with iteration N's S reads. Result: **no performance gain** (83.8 vs 84.1 TFLOPS).

Why it failed:
1. **barrier_A already syncs S**: The softmax max reduction requires `barrier()` which also ensures all S writes are visible — the nbarrier for S is redundant
2. **Q SLM reduction**: Fitting 2 S buffers (64KB) forced Q SLM from 16 to 15 d_blks, requiring the 16th d_blk to be loaded from global memory every iteration
3. **Register pressure**: Maintaining `delta_prev`, `local_sum_prev` for cross-iteration state consumed precious GRF
4. **Code complexity**: Prologue/epilogue for pipeline startup/drain added instruction cache pressure

Lesson: Double-buffering only helps when the barrier being eliminated is a **genuine bottleneck** and the buffer doesn't steal critical resources (SLM, registers) from the main compute.

## Multiple Named Barriers (Advanced)

For complex pipelining with independent sync points:

```cpp
__esimd_nbarrier_init(2);  // 2 independent barriers

// Barrier 0: protects S SLM writes
__esimd_nbarrier_arrive(0, 0, 32, 32);
// ... V loads, compensation ...
__esimd_nbarrier(0, 0, 32);

// ... VS DPAS ...

// Barrier 1: protects max SLM writes (separate sync point)
__esimd_nbarrier_arrive(1, 1, 32, 32);
// ... different independent work ...
__esimd_nbarrier(1, 1, 32);
```

Each barrier has its own hardware counter — they don't interfere with each other. Use distinct `barrier_reg` and `nbarrier_id` values for each.

### Producer-Consumer Pattern

For asymmetric sync where only a subset of threads produce data:

```cpp
// Only threads 0-7 produce data, all 32 consume
// Producers:
if (sg_id < 8) {
  slm_store(addr, data);
  __esimd_nbarrier_arrive(0, 0, 8, 32);  // 8 producers, 32 consumers
}
// All threads wait:
__esimd_nbarrier(0, 0, 32);
// Safe to read
```

## Checklist for Adding Split Barriers

1. Identify a `barrier()` that protects SLM writes needed by subsequent SLM reads
2. Verify there's no other `barrier()` nearby that already provides the same guarantee
3. Find independent work to insert between arrive and wait:
   - Global memory loads (high latency, async sends)
   - Prefetches for future iterations
   - ALU on private registers (no SLM dependency)
4. Replace `barrier()` with `arrive` ... work ... `wait`
5. Add `__esimd_nbarrier_init(N)` at kernel start
6. Verify: nothing between arrive and wait reads SLM that other threads are writing
7. Verify: nothing between arrive and wait writes SLM that other threads will read after wait
8. Benchmark: split barriers help most when the independent work window is 50+ cycles
