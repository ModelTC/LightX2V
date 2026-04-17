# LSC Memory Operations: 2D Block Load/Store and Scatter/Gather

Reference for `sycl::ext::intel::experimental::esimd` (xesimd) LSC memory intrinsics on Xe2/BMG.

---

## LSC 2D Block Operations

### `lsc_load_2d` — 2D block load

```cpp
template<
    typename T,          // element type (e.g., sycl::half, float, uint32_t)
    int BlockWidth,      // block width in elements
    int BlockHeight,     // block height in elements
    int NBlocks,         // number of blocks to load (1, 2, or 4)
    bool Transposed,     // true → transpose block on load
    bool VNNI,           // true → apply VNNI (K-major) repacking (for DPAS B matrix)
    cache_hint L1H,      // L1 cache hint
    cache_hint L2H       // L2 cache hint
>
simd<T, BlockWidth * BlockHeight * NBlocks>
lsc_load_2d(config_2d_mem_access<T, BlockWidth, BlockHeight, NBlocks>& payload);

// OR inline form (without payload object):
simd<T, BlockWidth * BlockHeight * NBlocks>
lsc_load_2d(const T* ptr,
            uint32_t surfWidth_bytes_minus1,
            uint32_t surfHeight_rows_minus1,
            uint32_t surfPitch_bytes_minus1,
            int x_offset_bytes,        // column offset in bytes from base ptr
            int y_offset_rows);        // row offset in elements from base ptr
```

#### Key constraints (BMG/Xe2)
| Parameter | Allowed values |
|-----------|---------------|
| T | `uint8_t`, `uint16_t` (= `sycl::half`), `uint32_t`, `uint64_t` |
| BlockWidth × sizeof(T) | Must be 4, 8, 16, 32, or 64 bytes |
| BlockHeight | 1–32 rows |
| NBlocks | 1, 2, or 4 |
| Transposed + VNNI | Cannot both be true simultaneously |
| Alignment (ptr) | 64-byte aligned |
| x_offset | Must be naturally aligned to element size |

#### Common tile shapes for FP16 GEMM
```
A tile (no transform):   lsc_load_2d<half, 16, 8, 1, false, false, ...>
  → 16 cols × 8 rows = 128 halfs = 256 bytes

B tile (VNNI):           lsc_load_2d<half, 16, 8, 1, false, true,  ...>
  → 16 cols × 8 rows, VNNI repacked for DPAS

B tile (transposed):     lsc_load_2d<half, 8, 16, 1, true, false,  ...>
  → transpose 8×16 → 16×8 output
```

#### `Transposed` vs `VNNI`
- **`Transposed=true`**: Swaps rows and columns of the loaded block. Useful for loading B in column-major layout.
- **`VNNI=true`**: Repacks pairs of K-adjacent half elements into one 32-bit word. Required for DPAS B input on Xe2. Output element order: `[k0_col0, k1_col0, k0_col1, k1_col1, ...]`
- **Cannot use both together**: hardware limitation.

---

### `lsc_store_2d` — 2D block store

```cpp
template<typename T, int BW, int BH, int NB,
         cache_hint L1H = cache_hint::write_back,
         cache_hint L2H = cache_hint::write_back>
void lsc_store_2d(
    config_2d_mem_access<T, BW, BH, NB>& payload,
    simd<T, BW * BH * NB> data);

// Inline form:
void lsc_store_2d(T* ptr,
                  uint32_t surfWidth_bytes_minus1,
                  uint32_t surfHeight_rows_minus1,
                  uint32_t surfPitch_bytes_minus1,
                  int x_bytes, int y_rows,
                  simd<T, BW * BH * NB> data);
```

- Store does **not** support `Transposed` or `VNNI` — output is always row-major.
- Cache hints: `write_back / write_back` for L1+L2 write-allocate (default), `streaming` to bypass L1 on write.

---

### `lsc_prefetch_2d` — 2D block prefetch

```cpp
template<typename T, int BW, int BH, int NB,
         cache_hint L1H = cache_hint::streaming,
         cache_hint L2H = cache_hint::cached>
void lsc_prefetch_2d(
    const T* ptr,
    uint32_t surfWidth_bytes_minus1,
    uint32_t surfHeight_rows_minus1,
    uint32_t surfPitch_bytes_minus1,
    int x_bytes,
    int y_rows);
```

**CRITICAL**: `lsc_prefetch_2d` does **NOT** accept `config_2d_mem_access` payload object.
It does **NOT** have `Transposed` or `VNNI` template parameters.
Always use the inline 6-argument ptr form.

```cpp
// CORRECT prefetch call:
xesimd::lsc_prefetch_2d<sycl::half, 16, 8, 1,
    xesimd::cache_hint::streaming,
    xesimd::cache_hint::cached>(
    A, surfW_A, surfH_A, surfW_A,
    (int)(k_pf * sizeof(sycl::half)),
    (int)(m_start + m * 8));

// WRONG — will not compile:
xesimd::lsc_prefetch_2d<sycl::half, 16, 8, 1, false, false, ...>(payA); // ❌
```

---

### `config_2d_mem_access` Payload Object

```cpp
template<typename T, int BW, int BH, int NBlocks>
class config_2d_mem_access {
public:
    // Constructor
    config_2d_mem_access(
        const T* base_ptr,
        uint32_t surface_width_bytes_minus1,
        uint32_t surface_height_rows_minus1,
        uint32_t surface_pitch_bytes_minus1,
        uint32_t x_offset_bytes,   // column in bytes
        uint32_t y_offset_rows);   // row index

    void set_x(uint32_t x_bytes);   // update column byte offset
    void set_y(uint32_t y_rows);    // update row offset
};
```

Surface descriptor fields:
- `surfWidth` = total surface width in **bytes** − 1
- `surfHeight` = total surface height in **rows** − 1
- `surfPitch` = stride between rows in **bytes** − 1 (usually = surfWidth for row-major)
- `x` = column offset in **bytes** from base
- `y` = row index (0-based)

```cpp
// Example: A matrix [M rows × K cols], half precision
const uint32_t surfW_A = (uint32_t)(K * sizeof(sycl::half)) - 1u;
const uint32_t surfH_A = (uint32_t)M - 1u;
xesimd::config_2d_mem_access<sycl::half, 16, 8, 1>
    payA(A, surfW_A, surfH_A, surfW_A, 0u, (uint32_t)m_start);

// Update for each K-step:
payA.set_x((uint32_t)(k * sizeof(sycl::half)));  // byte offset along K
payA.set_y((uint32_t)(m_row));                    // row index along M
```

---

## LSC Scatter / Gather (1D Vector Memory)

For irregular access patterns (non-rectangular tiles), use vector gather/scatter.

### `lsc_gather` — gather from scattered addresses

```cpp
template<typename T,
         int VS = 1,                  // vector size per channel (1,2,4,8)
         lsc_data_size DS = lsc_data_size::default_size,
         cache_hint L1H = cache_hint::none,
         cache_hint L2H = cache_hint::none,
         int N>                        // SIMD lanes (deduced from offsets)
simd<T, N * VS>
lsc_gather(const T* base,
           simd<uint32_t, N> byte_offsets,  // per-lane byte offsets from base
           simd_mask<N> mask = 1);           // optional lane mask
```

```cpp
// Example: gather 16 floats at scattered offsets
sycl_ext::simd<uint32_t, 16> offsets = ...; // byte offsets
sycl_ext::simd<float, 16> data =
    xesimd::lsc_gather<float>(base_ptr, offsets);
```

### `lsc_scatter` — scatter to scattered addresses

```cpp
template<typename T, int VS = 1,
         lsc_data_size DS = lsc_data_size::default_size,
         cache_hint L1H = cache_hint::none,
         cache_hint L2H = cache_hint::none,
         int N>
void lsc_scatter(T* base,
                 simd<uint32_t, N> byte_offsets,
                 simd<T, N * VS> data,
                 simd_mask<N> mask = 1);
```

```cpp
// Example: scatter 16 floats to scattered locations
xesimd::lsc_scatter<float>(base_ptr, offsets, data);
```

### VS > 1 (vector scatter/gather)

When `VS > 1`, each SIMD lane loads/stores `VS` consecutive elements starting at its offset:

```cpp
// VS=4: each of 16 lanes gathers 4 floats → output is simd<float, 64>
simd<float, 64> data = xesimd::lsc_gather<float, 4>(base, offsets_16);
```

### `lsc_data_size` options

| Enum | Meaning |
|------|---------|
| `default_size` | Natural element size |
| `u8u32` | Load uint8, zero-extend to uint32 |
| `u16u32` | Load uint16, zero-extend to uint32 |
| `u8s32` | Load int8, sign-extend to int32 |
| `u16s32` | Load int16, sign-extend to int32 |

---

## Cache Hints Summary

```cpp
namespace xesimd = sycl::ext::intel::experimental::esimd;

// Load hints
xesimd::cache_hint::none        // no hint (L1 default behavior)
xesimd::cache_hint::uncached    // bypass L1 (L1UC) — use for streaming writes
xesimd::cache_hint::cached      // cache in L1 and L2
xesimd::cache_hint::streaming   // stream (read once, don't pollute L1)
xesimd::cache_hint::read_invalidate  // load + invalidate L1 line after

// Store hints
xesimd::cache_hint::write_back      // L1+L2 write allocate (default)
xesimd::cache_hint::write_through   // write through L1 to L2
xesimd::cache_hint::streaming       // bypass L1 on store (useful for output C)
```

### Recommended hints for GEMM

| Operation | L1 hint | L2 hint | Rationale |
|-----------|---------|---------|-----------|
| A tile load (inner loop) | `cached` | `cached` | Reused across N dimension |
| B tile load (inner loop) | `cached` | `cached` | Reused across M dimension |
| A/B prefetch | `streaming` | `cached` | Don't pollute L1 with prefetch |
| C tile store | `streaming` | `write_back` | Written once, bypass L1 |
| Cache bust (bw kernel) | `uncached` | `uncached` | Force DRAM access |

---

## Block Size Constraints (Xe2 hardware)

| Condition | Constraint |
|-----------|-----------|
| Block byte width | 4, 8, 16, 32, or 64 bytes |
| Block height | 1 to 32 rows |
| NBlocks × BlockWidth × sizeof(T) | ≤ 64 bytes per row (for NBlocks > 1) |
| Base pointer alignment | 64 bytes |
| x offset alignment | sizeof(T) aligned |
| VNNI + Transposed | Not simultaneously allowed |

Violation causes silent wrong results or a GPU fault — not a compile error.

---

## Atomic Operations (LSC)

```cpp
// Atomic add (float)
xesimd::lsc_atomic_update<xesimd::atomic_op::fadd, float>(
    ptr, byte_offsets, data, mask);

// Atomic compare-and-swap
xesimd::lsc_atomic_update<xesimd::atomic_op::cmpxchg, uint32_t>(
    ptr, byte_offsets, new_val, expected, mask);
```

Useful for reduction across WGs. Avoid in the GEMM K-loop inner body (serialization kills performance).
