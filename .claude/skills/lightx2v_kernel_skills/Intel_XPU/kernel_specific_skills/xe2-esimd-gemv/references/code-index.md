# GEMV Code Index

## Production assets

### `assets/w4a16_simd_optimized.cpp` — W4A16, 571 GB/s
**The production W4A16 kernel.** Contains two kernel structs:

| Struct | Template params | Notes |
|--------|----------------|-------|
| `W4A16_SIMD_Optimized<ROWS, VL, K_SPLIT>` | ROWS=4, VL=1024, K=2 best | Main production kernel |
| `W4A16_2D_SIMD<WG_ROWS, VL, K_SPLIT, ROW_SPLIT>` | various | 2D split variant, slightly slower |

Key patterns:
- `slm_init(ROWS * K_SPLIT * sizeof(float))` — first line
- SIMD select interleaving: `weight_f.template select<64,2>(base+0) = lo`
- 8-accumulator rotation for latency hiding
- 32 weight copies for cache-busting
- `benchmark_ksplit<>` + `benchmark_2d<>` sweep functions

Sweep results embedded in `main()` — run to reproduce full parameter sweep.
Best: `W4A16_SIMD_Optimized<4, 1024, 2>` at 571 GB/s.

---

### `assets/w8a16_nocache.cpp` — W8A16, 552 GB/s
**The production W8A16 kernel.** Single kernel struct:

| Struct | Template params | Notes |
|--------|----------------|-------|
| `W8A16_GEMV<VL>` | VL=1024 best | Simple row-parallel, no K-split |

Key patterns:
- Per-row scale (single fp16, no blocking)
- `simd<int8_t, VL>` → `simd<float, VL>` conversion + scalar multiply
- 32–128 weight copies (auto-computed from available GPU memory)
- `test_config(q, N, K)` sweeps all VL values per problem size

Tested sizes: N=4096/8192/12288, K=4096/8192.
Best: `W8A16_GEMV<1024>` at 552 GB/s.

---

## Historical kernels (in test/ directory, not in skill assets)

These were intermediate steps — kept for reference only:

| File | Notes | BW |
|------|-------|----|
| `esimd_large_shapes.cpp` | FP16 GEMV reference kernel (no quantization) | ~287 GB/s |
| `esimd_k_split_slm_v4.cpp` | FP16 GEMV with K-split SLM | ~275 GB/s |
| `w4a16_success_pattern.cpp` | First W4A16 adapted from reference, scalar loop | ~234 GB/s |
| `w4a16_simple_rowparallel.cpp` | Before SIMD fix, best pre-optimization config | 289 GB/s |
| `w4a16_push75.cpp` | K-split exploration before SIMD fix | 273 GB/s |
| `w4a16_reach390.cpp` | More K-split variants, still scalar loop | 284 GB/s |
| `w4a16_2d_split.cpp` | 2D row+K split, scalar loop | 285 GB/s |
| `w4a16_interleaved.cpp` | Earlier interleaved attempt | ~260 GB/s |
| `w8a16_gemv_v1.cpp` | W8A16 without cache-busting | ~1200 GB/s (fake) |
| `w8a16_gemv_v2_ksplit.cpp` | W8A16 with K-split (unnecessary) | ~540 GB/s |
| `w8a16_accurate.cpp` | W8A16 correctness test harness | — |

## Parameter summary

### W4A16 optimal config
```
ROWS    = 4
VL      = 1024
K_SPLIT = 2
local_size = ROWS × K_SPLIT = 8
num_groups = ceil(N / ROWS)
```

### W8A16 optimal config
```
VL      = 1024
K_SPLIT = 1  (no split)
local_size = 1
global_size = N
```

## Performance ladder (W4A16)

```
234 GB/s  w4a16_success_pattern.cpp     scalar loop, ref pattern
258 GB/s  w4a16_push75.cpp              K-split added, still scalar loop
273 GB/s  w4a16_push75.cpp              K-split tuned
285 GB/s  w4a16_2d_split.cpp            2D split, still scalar loop
289 GB/s  w4a16_simple_rowparallel.cpp  no K-split, best before fix
--- SIMD select fix applied ---
552 GB/s  w4a16_simd_optimized.cpp      K_SPLIT=1
558 GB/s  w4a16_simd_optimized.cpp      K_SPLIT=1 R=4
571 GB/s  w4a16_simd_optimized.cpp      K_SPLIT=2 R=4  ← BEST
```

## Performance ladder (W8A16)

```
~1200 GB/s  w8a16_gemv_v1.cpp          single weight copy (cache hit, fake)
~540 GB/s   w8a16_gemv_v2_ksplit.cpp   32 copies, VL=1024, K-split=2
552 GB/s    w8a16_nocache.cpp          32+ copies, VL=1024, no K-split  ← BEST
```
