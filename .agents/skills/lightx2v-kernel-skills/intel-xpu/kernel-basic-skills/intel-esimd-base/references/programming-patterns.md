# Programming Patterns

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

## Kernel Design

This is the single most impactful decision. A bad design can be **10–35× slower** before any micro-optimization.

### Rule 1: Match Kernel Range Dimensionality to Data Shape

Use multi-dimensional `range<D>` so indices map directly to data dimensions. Never decode a 1D linear index into multi-dimensional coordinates at runtime.

```cpp
// BAD: 1D range + manual 4D decode → 1% bandwidth
q.parallel_for(range<1>(bsz * heads * seq * blocks),
    [=](id<1> idx) SYCL_ESIMD_KERNEL {
        int d4 =  idx[0] % blocks;
        int d3 = (idx[0] / blocks) % seq;
        int d2 = (idx[0] / (blocks * seq)) % heads;
        int d1 =  idx[0] / (blocks * seq * heads);
        output[idx[0]] = process(input[idx[0]]);  // 1 element/thread
    });

// GOOD: 3D range → ~73% bandwidth (21.7× faster before any other change)
q.parallel_for(range<3>(bsz, heads, seq),
    [=](id<3> idx) SYCL_ESIMD_KERNEL {
        int b = idx[0], h = idx[1], s = idx[2];
        // each thread processes many output elements for this (b, h, s)
    });
```

### Rule 2: Each Thread Must Do Meaningful Work

Processing 1 element per thread creates massive scheduling overhead. Target **32–128+ elements per thread** (or one complete row/column/tile).

```cpp
// GOOD: thread processes 128 elements
constexpr int ELEMS = 128;
const int base = idx[0] * ELEMS;
simd<sycl::half, ELEMS> data = block_load<sycl::half, ELEMS>(input + base);
// ... process data ...
block_store<sycl::half, ELEMS>(output + base, data);
```

### Rule 3: Use Template Parameters for All Compile-Time Constants

Runtime variables prevent loop unrolling and constant propagation. Make all array sizes and loop bounds `constexpr` or template parameters.

```cpp
// BAD: runtime N prevents unrolling
void kernel(int N) { for (int i = 0; i < N; i++) ... }

// GOOD: compile-time N enables full unroll and SIMD vectorization
template<int N>
void kernel() {
    #pragma unroll
    for (int i = 0; i < N; i++) ...
}
```

### Rule 4: Separate Load and Compute Loops

When computing multiple dot products or reductions, issue all `block_load`s before any compute. This lets the hardware pipeline memory fetches:

```cpp
// BAD: each load stalls its own compute
for (int i = 0; i < K; i++) {
    auto v = block_load<half, D>(ptr + i * D);
    results[i] = detail::sum<half, half, D>(v * w);
}

// GOOD: all loads in-flight before compute begins (+29% measured)
simd<half, D> vecs[K];
for (int i = 0; i < K; i++) vecs[i] = block_load<half, D>(ptr + i * D);
for (int i = 0; i < K; i++) results[i] = detail::sum<half, half, D>(vecs[i] * w);
```

---

## Memory Access Patterns

### Block Load / Store (Contiguous Access)

The primary memory primitive. Maps to one or a few hardware memory transactions.

```cpp
// Load N elements; check the resulting address alignment, not N*sizeof(T).
simd<T, N> data = block_load<T, N>(ptr);
simd<T, N> data = block_load<T, N>(ptr + offset);  // offset in elements

// Store (this overload assumes 16-byte alignment without properties)
block_store<T, N>(ptr + offset, data);
```

**Efficiency rules**:
- Prefer loads of **128B or 256B** (e.g., `block_load<half, 128>` = 256B, `block_load<float, 64>` = 256B)
- All threads should access consecutive memory addresses (coalescing)
- Larger individual loads are more efficient than many small ones

**Address alignment**: for the USM overloads above, `block_load` defaults to 4-byte alignment for elements of 4 bytes or less, and 8-byte alignment for 8-byte elements. The unpredicated `block_store` overload without cache hints defaults to **16-byte alignment**. Check the actual address after adding the offset; the requirement is not the load/store width.

Without cache hints, an explicit `alignment<>` property can declare a smaller, guaranteed alignment. It describes the address; it does not realign memory. Cache hints or predicated overloads impose additional alignment and block-size restrictions; check the installed SDK against Intel's [memory API definitions](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/memory.hpp).

```cpp
// Examples below: base pointers are at least 4-byte aligned; no cache hints.
// ── half / bfloat16 (2 bytes each) ───────────────────────────────────────
// Element index 3 → byte address +6 → violates block_load's default alignment.
simd<half, 8> data = block_load<half, 8>(ptr + 3);                          // WRONG
simd<half, 8> data = block_load<half, 8>(ptr + 3, properties{alignment<2>}); // OK

// The address is 2-byte aligned; declare that explicitly for this store too.
block_store<half, 8>(ptr + 3, data, properties{alignment<2>});               // OK

// ── float / int (4 bytes each) ────────────────────────────────────────────
// 4-byte elements: any element index gives a 4-byte-aligned byte address → safe.
simd<float, 8> data = block_load<float, 8>(ptr + 3);   // OK (byte addr = +12)

// ── uint8_t (1 byte each) ────────────────────────────────────────────────
// Use alignment<1> when only byte alignment is guaranteed.
simd<uint8_t, 8> data = block_load<uint8_t, 8>(ptr + 3, properties{alignment<1>});
```

**USM block_load summary (no cache hints)**:
| Type | Size | Default required alignment | When to annotate |
|------|------|---------------------------|-----------------|
| `float`, `int`, `uint32_t` | 4 B | 4 B | element offsets preserve a 4-byte-aligned base |
| `half`, `bfloat16` | 2 B | 4 B | `alignment<2>` when only 2-byte alignment is guaranteed |
| `uint8_t`, `int8_t` | 1 B | 4 B | `alignment<1>` when only byte alignment is guaranteed |

**Symptom of missing annotation**: incorrect values loaded/stored, sporadic failures depending on offset parity, no compile error.

### Gather / Scatter (Non-Contiguous Access)

Use when elements are at irregular or strided offsets — e.g., sliding windows, strided output.

```cpp
// Gather: load N elements from per-lane byte offsets
simd<T, N> data = gather<T, N>(base_ptr, simd<uint32_t, N>(byte_offsets));

// Scatter: store N elements to per-lane byte offsets
scatter<T, N>(base_ptr, simd<uint32_t, N>(byte_offsets), data);
```

**Key rules**:
- Offsets are in **bytes**, not element indices. Convert: `byte_offsets = elem_offsets * sizeof(T)`
- Offset type must be `simd<uint32_t, N>` — unsigned 32-bit
- Negative element indices cast to `uint32_t` wrap to huge values → clamp before cast
- Optimal batch size: **N=32** for FP16 (matches hardware gather width)
- Slightly slower than `block_load` for contiguous data, much faster for strided patterns

**Batch gather pattern** (process 32 outputs in parallel):
```cpp
constexpr int BS = 32;
simd<int, BS> rel = simd<int, BS>(0, 1) * STRIDE;  // [0, S, 2S, ..., 31S]

for (int out = 0; out < num_outputs; out += BS) {
    // Clamp offsets to valid range (branchless — safe for idempotent ops like max)
    simd<int, BS> offs = max(out * STRIDE + rel + OFFSET, 0);
    offs = min(offs, max_elem);

    simd<T, BS> vals = gather<T, BS>(
        input + base, simd<uint32_t, BS>(offs) * sizeof(T));

    // ... process vals ...

    int n = std::min(BS, num_outputs - out);
    if (n == BS) block_store<T, BS>(output + out_base + out, result);
    else         for (int i = 0; i < n; i++) output[out_base + out + i] = result[i];
}
```

**Performance reference**: gather achieved 72.8% bandwidth on a strided sliding-window pooling kernel (82.9× total speedup over naive scalar 1D design).

### SLM — Shared Local Memory

Use for inter-thread communication within a workgroup. Requires `nd_range` and `barrier()`.

```cpp
// Initialize SLM (compile-time size, inside kernel)
constexpr int SLM_BYTES = GROUP_SIZE * sizeof(float);
slm_init<SLM_BYTES>();

// Per-thread write (byte offset)
slm_block_store<float, 1>(local_id * sizeof(float), simd<float, 1>(partial));

// Synchronize all threads in workgroup
barrier();

// Read all GROUP_SIZE values (thread 0 only, or all threads for broadcast)
simd<float, GROUP_SIZE> parts = slm_block_load<float, GROUP_SIZE>(0);
```

**SLM limits and cost**:
- Check the target SKU's SLM capacity and per-workgroup/compiler limit; choose allocation size together with the desired residency.
- Access latency ~30 cycles vs ~1 cycle for register file
- `slm_init` must be called inside the kernel, unconditionally
- `barrier()` must be reached by **all** threads in the workgroup — never inside a branch that some threads skip

---

## Data Types

### FP16 vs FP32

**Use FP16** for: weights, activations, intermediate results in neural network ops, any case where error < 0.1 is acceptable.

**Use FP32** for: accumulators summing >1000 terms (use `simd<float, N> acc` even when inputs are FP16), numerical stability-critical paths, softmax exponents.

```cpp
// Pattern: FP16 inputs → FP32 accumulation → FP16 output
simd<sycl::half, 128> a = block_load<sycl::half, 128>(ptr_a);
simd<sycl::half, 128> b = block_load<sycl::half, 128>(ptr_b);
simd<float, 128> acc(0.f);
acc += convert<float>(a) * convert<float>(b);
float result = sycl::ext::intel::esimd::detail::sum<float, float, 128>(acc);
block_store<sycl::half, 1>(out, simd<sycl::half, 1>(sycl::half(result)));
```

---

## Vectorization Techniques

### Arithmetic

```cpp
// Element-wise ops (all broadcast scalars automatically)
simd<T, N> c = a + b;
simd<T, N> c = a * b + scalar;   // FMA when T=float
simd_mask<N> m = a > b;
simd<T, N> r = b;
r.merge(a, m);                  // r[i] = m[i] ? a[i] : b[i]
```

### Scalar × Vector (No Explicit Broadcast Needed)

Intel GPU hardware natively supports multiplying a scalar by a `simd` vector — the scalar is broadcast implicitly by the FMA unit. **No `replicate_w` or `simd<T,N>(scalar)` constructor needed.**

```cpp
// GOOD: extract scalar from simd, multiply directly — one instruction
half k_val = k_tile[tt * SUB_HD + ii];   // simd_view → half (implicit conversion)
acc[ii] += k_val * v_row;                // scalar * simd<half, N> → simd<half, N>

// BAD: explicit broadcast via replicate_w — wastes a GRF shuffle instruction
simd<half, N> k_bcast = k_tile.template replicate_w<N, 1>(tt * SUB_HD + ii);
acc[ii] += k_bcast * v_row;
```

**Why**: `replicate_w<N, 1>(i)` emits a hardware GRF shuffle (region select) to fill an N-wide register. When used only as a multiplier, the GPU's scalar-broadcast path in the EU instruction set handles this for free inside the FMA — eliminating the shuffle saves one instruction per (tt, ii) pair.

**Extraction note**: `simd<T,N>::operator[]` returns a `simd_view`, which implicitly converts to `T` in arithmetic contexts. The result of `half k_val = simd_vec[i]` is a true scalar `half` register.

### Reductions

Use an explicitly qualified operation supported by the installed ESIMD math header:

```cpp
#include <functional>
float sum = sycl::ext::intel::esimd::reduce<float>(vec, std::plus<>{});
float maximum = sycl::ext::intel::esimd::hmax<float>(vec);
float minimum = sycl::ext::intel::esimd::hmin<float>(vec);
```

The [upstream implementation](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/math.hpp) checks for `std::plus<>` / `std::multiplies<>`; typed `std::plus<float>` is not interchangeable in that implementation. Compile a minimal example with the target SDK. Historical `detail::sum` examples below use an internal API and also require a version check.

### Sub-Range Access (`select`)

```cpp
// Read a contiguous sub-range: elements [offset, offset+COUNT)
simd<T, COUNT> sub = vec.template select<COUNT, 1>(offset);

// Read with stride: elements [offset, offset+STRIDE, offset+2*STRIDE, ...]
simd<T, COUNT> sub = vec.template select<COUNT, STRIDE>(offset);

// Write a sub-range
vec.template select<COUNT, 1>(offset) = other_vec;
```

`select` returns a `simd_view` (not `simd`) — assign to a named `simd<T,N>` when you need to pass it to a function.

### Type Conversion

```cpp
simd<float, N>      fp32 = convert<float>(fp16_vec);
simd<sycl::half, N> fp16 = convert<sycl::half>(fp32_vec);
simd<float, N>      fp32 = convert<float>(uint8_vec);

// Bit-cast reinterpret (zero cost, requires lvalue)
auto u32_vec = some_expression();  // store to named variable first
auto i32_vec = u32_vec.template bit_cast_view<int32_t>();  // reinterpret in place
```

### `simd` Constructor Patterns

```cpp
// Broadcast scalar
simd<float, N> v(0.f);            // all zeros
simd<sycl::half, N> v(-65504.f); // FP16 -inf

// Arithmetic sequence: [start, start+step, start+2*step, ...]
simd<int, 32> iota(0, 1);         // [0, 1, 2, ..., 31]
simd<int, 32> strided(0, 4);      // [0, 4, 8, ..., 124]
```

### pack_mask and fbl

Used to find the first lane matching a condition, without scalar loops:

```cpp
// pack_mask: simd_mask<N> → uint32_t bitmask (N = 8, 16, or 32 only)
uint32_t m = pack_mask(heap == min_val);   // bit i = 1 if heap[i] == min_val

// fbl (find bit lowest): index of least significant set bit; -1 if all zero
uint32_t pos = fbl(m);                     // scalar
simd<uint32_t, K> pos_vec = fbl(mask_vec); // vectorized per-element

// Full 32-element chunks only; handle a partial tail separately if needed.
static_assert(N > 32 && N % 32 == 0);
constexpr int MASK_N = N / 32;
simd<uint32_t, MASK_N> masks;
#pragma unroll
for (int mi = 0; mi < MASK_N; mi++) {
    simd<sycl::half, 32> chunk = vec.template select<32, 1>(mi * 32);
    masks[mi] = pack_mask(chunk == target);
}
simd<uint32_t, MASK_N> fbl_r   = fbl(masks);
simd<int32_t,  MASK_N> bit_idx = fbl_r.template bit_cast_view<int32_t>();
simd<int32_t,  MASK_N> glob    = bit_idx + simd<int32_t, MASK_N>(0, 32);
glob.merge(simd<int32_t, MASK_N>(N), bit_idx == -1);  // sentinel above valid indices
const int first = hmin<int32_t>(glob);
const int pos = (first == N) ? -1 : first;           // first match; -1 if none
```

---

## Workgroup and SLM Patterns

### Pattern 1: Independent Threads (No SLM)

Each thread works on a private slice of data. Use `range<D>` (not `nd_range`).

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class MyKernel>(
        sycl::range<1>(num_rows),
        [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            const int row = (int)idx[0];
            simd<sycl::half, ROW_DIM> data =
                block_load<sycl::half, ROW_DIM>(input + row * ROW_DIM);
            // ... process ...
            block_store<sycl::half, ROW_DIM>(output + row * ROW_DIM, result);
        });
});
```

### Pattern 2: Workgroup Reduction via SLM

Each workgroup computes one output element; GROUP_SIZE threads split the K dimension.

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class ReduceKernel>(
        sycl::nd_range<1>(num_outputs * GROUP_SIZE, GROUP_SIZE),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;

            const int group_id = (int)item.get_group(0);
            const int local_id = (int)item.get_local_id(0);

            slm_init<GROUP_SIZE * sizeof(float)>();

            // Each thread computes partial result over its K slice
            float partial = compute_partial(group_id, local_id);

            // Store partial to SLM
            slm_block_store<float, 1>(
                local_id * sizeof(float), simd<float, 1>(partial));
            barrier();

            // Thread 0 reduces and writes output
            if (local_id == 0) {
                simd<float, GROUP_SIZE> parts =
                    slm_block_load<float, GROUP_SIZE>(0);
                float result = sycl::ext::intel::esimd::detail::sum<
                    float, float, GROUP_SIZE>(parts);
                output[group_id] = sycl::half(result);
            }
        });
});
```

**Guidelines**:
- `GROUP_SIZE` 4–8 is optimal; larger sizes increase SLM pressure and barrier overhead
- `slm_init` must be called unconditionally inside the kernel body
- `barrier()` must be reached by all GROUP_SIZE threads — never inside a conditional

---

## Register File vs SLM

When working data fits in registers (≤ ~2 KB), prefer registers over SLM.

| | Register file | SLM |
|--|--------------|-----|
| Latency | ~1 cycle | ~30 cycles |
| Requires `nd_range` | No | Yes |
| Requires `barrier()` | No | Yes |
| Capacity | 4 KiB on A770; 8/16 KiB per thread on B580 in regular/large GRF mode | SKU-specific shared capacity; B580 has 128 KB per Xe-Core |
| GROUP_SIZE overhead | None | Yes |

```cpp
// BAD: SLM for per-thread data (30× slower than needed)
slm_init<1024>();
slm_block_store<sycl::half, 32>(offset, chunk);
simd<sycl::half, 32> v = slm_block_load<sycl::half, 32>(offset);

// GOOD: register file (1 cycle, simpler kernel)
simd<sycl::half, 512> data;                         // 1 KiB; include all other live values in the budget
data.template select<32, 1>(chunk_idx * 32) = chunk;
simd<sycl::half, 32> v = data.template select<32, 1>(chunk_idx * 32);
```

**SLM is still the right choice** for shared data across threads in a workgroup (e.g., K-dimension reduction where GROUP_SIZE > 1). The rule is: SLM for **inter-thread** sharing, registers for **intra-thread** working data.

---

## Loop Optimization

### #pragma unroll

Add to every loop with a compile-time iteration count:

```cpp
#pragma unroll
for (int i = 0; i < STATIC_N; i++) { ... }  // fully unrolled

#pragma unroll 4
for (int i = 0; i < large_static_n; i++) { ... }  // unrolled 4 at a time
```

Without `#pragma unroll`, the compiler may generate a loop counter and branch, preventing vectorization and adding overhead.

### Hoist Loop-Invariant Work

```cpp
// BAD: recompute per iteration
for (int i = 0; i < N; i++) {
    float scale = 1.f / std::sqrt((float)HEAD_DIM);  // recomputed each time
    results[i] = dot(q[i], k) * scale;
}

// GOOD: hoist
const float scale = 1.f / std::sqrt((float)HEAD_DIM);
for (int i = 0; i < N; i++) results[i] = dot(q[i], k) * scale;
```

---

## Hybrid Scalar/Vector Strategy

For kernels with boundary cases (first/last element, partial tiles), use vectorized code for the hot path and simple scalar code for edges:

```cpp
for (int i = 0; i < num_outputs; i++) {
    int start = i * STRIDE - 1;
    int end   = start + WINDOW;

    if (start < 0 || end > num_elems) {
        // Boundary: correct scalar path
        start = std::max(0, start);
        end   = std::min(num_elems, end);
        float mx = -INFINITY;
        for (int j = start; j < end; j++) mx = std::max(mx, (float)input[j]);
        output[i] = sycl::half(mx);
    } else {
        // Hot path: vectorized (handles majority of iterations)
        simd<sycl::half, WINDOW> v =
            block_load<sycl::half, WINDOW>(input + start,
                                            properties{alignment<2>});
        output[i] = sycl::half(hmax<float>(convert<float>(v)));
    }
}
```

**Benefit**: avoids complex vectorized boundary logic, keeps the hot path clean, easy to debug.
**Measured**: 1.31 ms (scalar only) → 0.808 ms (hybrid) = 1.62× speedup.

For kernels where offsets can be clamped safely (e.g., max pooling where reading duplicate elements is harmless), prefer **branchless offset clamping** over hybrid:

```cpp
// Fully branchless — no hot/cold path split needed
simd<int, BS> offs = min(max(computed_offs, 0), max_elem);
simd<T, BS> vals   = gather<T, BS>(ptr, simd<uint32_t, BS>(offs) * sizeof(T));
// max(vals, running_max) — duplicate reads are harmless for max
```

---
