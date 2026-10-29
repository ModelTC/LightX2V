# Troubleshooting

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

## Common Performance Traps

### 1D Range with Manual Index Decode (10–35× slowdown)

See [Kernel Design](programming-patterns.md#kernel-design) for the original comparison; its speedup depends on the workload and compiler.

### Element-Wise Memory Access

```cpp
// BAD: 128 separate scalar loads
for (int i = 0; i < 128; i++) val[i] = ptr[offset + i];

// GOOD: one 256B block load
simd<sycl::half, 128> val = block_load<sycl::half, 128>(ptr + offset);
```

### Non-Aligned Offset

For a 4-byte-aligned `float*` base, both `7 * sizeof(float)` and `8 * sizeof(float)` satisfy the default alignment of this USM `block_load` overload. The offset need not be a multiple of the load width. See [Block Load / Store](programming-patterns.md#block-load--store-contiguous-access) for the `half` odd-offset example, explicit alignment properties, and the different store requirements.

### Excessive Synchronization

```cpp
// BAD: barrier per iteration
for (int i = 0; i < N; i++) { compute(i); barrier(); }

// GOOD: barrier once after all writes
for (int i = 0; i < N; i++) slm_block_store<float, 1>(i*4, partial[i]);
barrier();
// ... reduce ...
```

### Register Spilling

```cpp
// BAD: many large vectors live simultaneously → register spill to L1
simd<float, 256> a, b, c, d, e, f, g, h;  // 8 KiB before other live values; may spill

// GOOD: reuse temporaries
simd<float, 128> tmp;
for (...) { tmp = load(); acc += process(tmp); }
```

### Branch Divergence

An ESIMD work-item's `local_id` selects work-item behavior; a `simd_mask<N>` selects vector lanes within that work-item. Replacing `if (local_id < 2)` with a lane mask changes the algorithm. For a genuinely per-lane choice between already computed values:

```cpp
// path_a_result and path_b_result are simd<T, N> values.
simd_mask<N> m = (simd<int, N>(0, 1) < 2);
simd<T, N> result = path_b_result;
result.merge(path_a_result, m);  // first two lanes select path_a_result
```

Masking does not skip either computation or guarantee a speedup. Preserve side effects and synchronization when changing control flow.

### Uninitialized `simd<T, N>` Gives Garbage

`simd<T, N> v;` without initialization before element reads is undefined behavior. Always initialize before use — either broadcast-construct or seed with real data:

```cpp
simd<half, K> heap(-65504.f);           // broadcast init
simd<half, K> heap = block_load<...>(); // or initialize from load
```

### `alignment<2>{}` Parse Error in Template Context

```cpp
block_load<half, N>(ptr, properties{alignment<2>});   // GOOD
block_load<half, N>(ptr, properties{alignment<2>{}});  // BAD — parse error
```

### Inner Lambda Must Use `[[intel::sycl_explicit_simd]]`

Inner lambdas calling ESIMD intrinsics inside a `SYCL_ESIMD_KERNEL` must be marked `[[intel::sycl_explicit_simd]]`. Without it, the compiler errors: "SYCL kernel without 'sycl_explicit_simd' attribute can't call a function with this attribute". A `#define` macro is a valid fallback if the compiler doesn't support the attribute on lambdas.

### `sycl::min` / `sycl::max` Illegal in ESIMD Kernels

```cpp
// BAD — compile error in ESIMD
int n = sycl::min(a, b);

// GOOD
int n = std::min(a, b);
int n = (a < b) ? a : b;
```

### Reduction operation type and SDK version

An earlier experiment reported a zero reduction and subsequent NaNs with `reduce<float>(vec, std::plus<float>())`. Its compiler version and minimal reproduction are not recorded here. That observation does not establish that overload resolution chose C++17 `std::reduce` or that every ESIMD reduction returns zero.

The [Intel math header](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/math.hpp) defines ESIMD's own reduction overload and matches `std::plus<>` / `std::multiplies<>`. Use an explicit namespace and a supported operation type, then check finite outputs and a CPU reference on the target toolchain. The historical `detail::sum` workaround is an internal interface, not the only supported reduction strategy.

---
