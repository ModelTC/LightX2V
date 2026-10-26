---
name: intel-esimd-fuse
description: >-
  Implement or debug fused Intel ESIMD row operations such as softmax, TopK and normalization. Use when intermediate memory traffic or shared register state motivates fusion, and verify reduction, selection and normalization semantics together.
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

> A standalone TopK skill is not bundled. For selection-only work, start from the register-heap example below and verify K bounds, tie handling and indices against a CPU reference; the sample uses compile-time K.

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

// Sum — explicit ESIMD overload with a supported operation type (<functional>).
float sum_e = sycl::ext::intel::esimd::reduce<float>(exp_x, std::plus<>{});

simd<sycl::half, N> sm = convert<sycl::half>(exp_x / sum_e);
```

Two rules that matter:
- This template uses `hmax<float>(x_f)` and `exp(x_f)` on the full row. A chunked variant must preserve the same row-wide maximum and normalization.
- Use an explicitly qualified reduction and the operation type supported by the installed SDK; see the version note below.

## Normalize Top-K

```cpp
simd<float, TOP_K> fheap = convert<float>(heap);
float top_sum = sycl::ext::intel::esimd::reduce<float>(fheap, std::plus<>{});
simd<sycl::half, TOP_K> norm_vals = convert<sycl::half>(fheap / top_sum);
```

---

## Optimization Journey (seq_len=32K, N=128, K=8)

| Version | Time | Bandwidth | Notes |
|---------|------|-----------|-------|
| Initial (chunked exp, reduce<float>) | NaN | --- | Historical zero/NaN observation; SDK and minimal reproduction not recorded |
| Fixed (exp on full vector, detail::sum) | **230 us** | **43.2 GB/s** | 43% of 100 GB/s peak |

---

## Complete Fused Kernel Template (Softmax + TopK + Normalize)

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <functional>

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
                float sum_e = sycl::ext::intel::esimd::reduce<float>(exp_x, std::plus<>{});
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
                float top_sum = sycl::ext::intel::esimd::reduce<float>(fheap, std::plus<>{});
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

### 1. Reduction operation type and SDK version

The historical failure above is not evidence that overload resolution selected `std::reduce`. Intel's [ESIMD math implementation](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/math.hpp) has its own overload and checks for `std::plus<>` / `std::multiplies<>`, rather than typed `std::plus<float>`. Check the installed SDK and compile a minimal example before using:

```cpp
#include <functional>
float sum = sycl::ext::intel::esimd::reduce<float>(vec, std::plus<>{});
```

The `detail::sum` workaround remains part of the historical measurements; it is an internal API. The source-checked examples here do not revalidate those timings or establish compatibility with every historical SDK.

### 2. `exp()` producing NaN

Check non-finite inputs, overflow, the row maximum, reduction and normalization denominator. The historical chunked variant failed, but chunking alone does not establish the cause.

The full-row template subtracts the row maximum before exponentiation:

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
float s = sycl::ext::intel::esimd::reduce<float>(ex, std::plus<>{});  // explicit ESIMD sum
simd<half, N> sm = convert<half>(ex / s);                  // normalize

// Normalize top-K
simd<float, K> fheap = convert<float>(heap);
float ts = sycl::ext::intel::esimd::reduce<float>(fheap, std::plus<>{});
simd<half, K> norm = convert<half>(fheap / ts);
```

**Performance** (seq_len=32K, N=128, K=8):

| Version | Time | Bandwidth |
|---------|------|-----------|
| Initial (chunked exp, reduce<float>) | NaN | --- |
| **Fixed (full-vector exp, detail::sum)** | **230 us** | **43.2 GB/s** |
