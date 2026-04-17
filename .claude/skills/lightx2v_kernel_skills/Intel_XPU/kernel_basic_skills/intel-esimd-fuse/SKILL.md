---
name: intel-esimd-fuse
description: >
  Expert guidance for implementing fused multi-operation kernels on Intel GPUs using
  ESIMD. Use this skill whenever the user needs to fuse multiple operations into a
  single kernel pass to minimize memory traffic, such as softmax + top-K + normalize,
  or any pipeline that chains reduction, selection, and normalization in one kernel.
  Also trigger for ESIMD softmax implementation, vectorized exp on simd<float,N> for
  a full row, detail::sum vs reduce pitfall (reduce silently returns 0), fused
  attention block selection with probability normalization, or any kernel that
  computes softmax probabilities and immediately selects the top-K entries. The main
  example is the fused softmax+topk+normalize V2 variant achieving 43.2 GB/s (43%
  bandwidth utilization) for seq_len=32K, N=128, K=8.
---

# Intel ESIMD Kernel Fusion Patterns

This skill covers techniques for fusing multiple operations into a single ESIMD kernel
pass, eliminating intermediate global memory round-trips. The primary example is the
fused softmax + topK + normalize pipeline, but the patterns generalize to any
chain of per-row operations that can share data through registers.

**Version**: 2.0.0

## General Fusion Principles

1. **One load, multiple operations**: Load data once into registers, then chain all
   operations (reduction, selection, normalization) without writing intermediates to
   global memory.
2. **Register data sharing**: Pass data between fused stages via `simd<T, N>` registers
   rather than SLM or global memory.
3. **Single store at the end**: Write only the final result to global memory.
4. **Compile-time dimensions**: Template on data dimensions so the compiler can unroll
   and optimize the entire fused pipeline.

> For TopK only (no fusion), see **intel-esimd-topk** (V2 register heap, V3 runtime K).

---

## Example: Fused Softmax + TopK + Normalize

Fuses three operations in a single kernel pass per row:
1. Load N fp16 values, convert to float
2. Softmax: `hmax` then subtract then `exp` then sum then divide
3. Top-K streaming heap (V2 register file, compile-time K)
4. Normalize: `v_i / sum(v_i)` over the top-K values
5. Store normalized fp16 values + int32 indices

**Achievement**: 43.2 GB/s (43% of 100 GB/s peak) for seq_len=32K, N=128, K=8

---

## Softmax in ESIMD

```cpp
simd<float, N> x_f = convert<float>(x_h);

// Max subtraction — hmax on the full N-element vector directly
float max_v = hmax<float>(x_f);
x_f -= max_v;

// exp on full vector (ADL inside using namespace esimd picks esimd::exp)
simd<float, N> exp_x = exp(x_f);

// Sum — use detail::sum (NOT reduce<float> — see Pitfalls)
float sum_e = sycl::ext::intel::esimd::detail::sum<float, float, N>(exp_x);

simd<sycl::half, N> sm = convert<sycl::half>(exp_x / sum_e);
```

Two rules that matter:
- Use `hmax<float>(x_f)` and `exp(x_f)` **directly on the full vector** — do not chunk.
- Use `detail::sum`, never `reduce<float>` — `reduce` resolves to `std::reduce` and silently returns 0.

## Normalize Top-K

```cpp
simd<float, TOP_K> fheap = convert<float>(heap);
float top_sum = sycl::ext::intel::esimd::detail::sum<float, float, TOP_K>(fheap);
simd<sycl::half, TOP_K> norm_vals = convert<sycl::half>(fheap / top_sum);
```

---

## Optimization Journey (seq_len=32K, N=128, K=8)

| Version | Time | Bandwidth | Notes |
|---------|------|-----------|-------|
| Initial (chunked exp, reduce<float>) | NaN | --- | reduce<float> returns 0, div-by-zero |
| Fixed (exp on full vector, detail::sum) | **230 us** | **43.2 GB/s** | 43% of 100 GB/s peak |

---

## Complete Fused Kernel Template (Softmax + TopK + Normalize)

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

constexpr int INPUT_DIM = 128;  // must be a compile-time constant
constexpr int TOP_K     = 8;    // must be <= 32 for single pack_mask call

sycl::event softmax_topk_normalize(
    sycl::queue&      q,
    const sycl::half* input,    // [seq_len, INPUT_DIM] fp16
    sycl::half*       out_vals, // [seq_len, TOP_K] fp16
    int*              out_idx,  // [seq_len, TOP_K] int32
    int               seq_len)
{
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class SoftmaxTopKNorm>(
            sycl::range<1>(seq_len),
            [=](sycl::id<1> item_id) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;

                const int row = (int)item_id[0];

                // 1. Load
                simd<sycl::half, INPUT_DIM> x_h =
                    block_load<sycl::half, INPUT_DIM>(input + row * INPUT_DIM);

                // 2. Softmax
                simd<float, INPUT_DIM> x_f = convert<float>(x_h);
                float max_v = hmax<float>(x_f);
                x_f -= max_v;
                simd<float, INPUT_DIM> exp_x = exp(x_f);
                float sum_e = sycl::ext::intel::esimd::detail::sum<float, float, INPUT_DIM>(exp_x);
                simd<sycl::half, INPUT_DIM> sm = convert<sycl::half>(exp_x / sum_e);

                // 3. Top-K streaming heap (V2, K <= 32)
                const simd<int, TOP_K> iota(0, 1);
                simd<sycl::half, TOP_K> heap;
                simd<int,        TOP_K> hidx;
                #pragma unroll
                for (int i = 0; i < TOP_K; i++) { heap[i] = sm[i]; hidx[i] = i; }
                #pragma unroll
                for (int i = TOP_K; i < INPUT_DIM; i++) {
                    sycl::half v  = sm[i];
                    sycl::half mn = hmin<sycl::half>(heap);
                    if (v > mn) {
                        int pos = (int)fbl(pack_mask(heap == mn));
                        simd_mask<TOP_K> m = (iota == pos);
                        heap.merge(simd<sycl::half, TOP_K>(v), m);
                        hidx.merge(simd<int,        TOP_K>(i), m);
                    }
                }

                // 4. Normalize top-K
                simd<float, TOP_K> fheap = convert<float>(heap);
                float top_sum = sycl::ext::intel::esimd::detail::sum<float, float, TOP_K>(fheap);
                simd<sycl::half, TOP_K> norm_vals = convert<sycl::half>(fheap / top_sum);

                // 5. Store
                const int ob = row * TOP_K;
                block_store<sycl::half, TOP_K>(out_vals + ob, norm_vals);
                block_store<int,        TOP_K>(out_idx  + ob, hidx);
            });
    });
}
```

---

## Pitfalls

### 1. `reduce<float>(vec, std::plus<float>())` silently returns 0

**Cause**: With `using namespace sycl::ext::intel::esimd` in scope, `reduce<float>(vec,
std::plus<float>())` resolves to C++17 `std::reduce` (iterator-based), not `esimd::reduce`.
It compiles silently and returns 0, causing NaN in softmax (division by zero) and garbage
in normalization.

```cpp
// BAD — silently returns 0
float s = reduce<float>(vec, std::plus<float>());

// GOOD — explicit, correct
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
```

### 2. `exp()` producing NaN

**Cause**: Chunking the exp computation, or skipping max subtraction before exp.

**Fix**: Apply `exp(x_f)` directly on the full vector *after* max subtraction:

```cpp
float max_v = hmax<float>(x_f);  // must subtract max first
x_f -= max_v;
simd<float, INPUT_DIM> exp_x = exp(x_f);  // direct on full vector, no chunking
```

### 3. Vexing parse: `simd<T, N> var(T(val));` declared as function

When `T(val)` is a type cast, the compiler may parse `simd<T, N> var(T(val))` as a
function declaration.

```cpp
// BAD — parsed as function declaration
simd<sycl::half, K> norm(sycl::half(ts));

// GOOD
simd<sycl::half, K> norm = simd<sycl::half, K>(sycl::half(ts));
```

---

## Quick Reference

```cpp
// Softmax on full N-element vector
float max_v = hmax<float>(x_f);                            // horizontal max
x_f -= max_v;                                              // max subtraction
simd<float, N> ex = exp(x_f);                              // vectorized exp
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(ex);  // sum (NOT reduce)
simd<half, N> sm = convert<half>(ex / s);                  // normalize

// Normalize top-K
simd<float, K> fheap = convert<float>(heap);
float ts = sycl::ext::intel::esimd::detail::sum<float, float, K>(fheap);
simd<half, K> norm = convert<half>(fheap / ts);
```

**Performance** (seq_len=32K, N=128, K=8):

| Version | Time | Bandwidth |
|---------|------|-----------|
| Initial (chunked exp, reduce<float>) | NaN | --- |
| **Fixed (full-vector exp, detail::sum)** | **230 us** | **43.2 GB/s** |
