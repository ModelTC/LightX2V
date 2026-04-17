# LSC Memory Operations — BMG ESIMD SDP

## Namespace

```cpp
using namespace sycl::ext::intel::esimd;                    // __ESIMD_NS
using namespace sycl::ext::intel::experimental::esimd;      // __ESIMD_ENS
```

---

## block_load / block_store (1D contiguous)

```cpp
// Load N contiguous elements from ptr
simd<T, N> data = block_load<T, N>(ptr);

// Load with offset (byte address)
simd<T, N> data = block_load<T, N>(ptr + offset_in_elements);

// Store
block_store<T, N>(ptr, data);
```

Used in decode for Q, K, V (stride-of-HD contiguous loads):
```cpp
simd<fp16, HD> kIn = block_load<fp16, HD>(kState + kvRealOffset + t * headKv * HD);
simd<fp32, HD> chunk = block_load<fp32, HD>(out + out_ck + j * HD);
block_store<fp16, HD>(out_final + q_head_base * HD + i * HD, result_fp16);
```

---

## lsc_load_2d (2D block load)

Loads a 2D tile from a strided 2D surface.

**Payload object API** (use for loads in inner loop — enables CSE):
```cpp
// Construct payload ONCE outside the loop
__ESIMD_ENS::config_2d_mem_access<fp16, BlockWidth, BlockHeight, NBlocks>
    payloadK((fp16*)kState, widthInBytes, heightInRows, pitchInBytes, coordX, coordY);

// In the loop, only update coordinates
payloadK.set_x(newX);
payloadK.set_y(newY);

// Load
simd<fp16, BlockWidth * BlockHeight * NBlocks> data =
    __ESIMD_ENS::lsc_load_2d<fp16, BlockWidth, BlockHeight, NBlocks,
        /*transpose=*/false, /*vnni=*/false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
```

**Template parameters:**
- `BlockWidth`: elements per row in output tile
- `BlockHeight`: rows in output tile
- `NBlocks`: number of tiles to load (stacked vertically)
- `transpose`: true → transpose tile (rows become columns)
- `vnni`: true → VNNI-transform for DPAS B operand

**K load (prefill, no transpose, NBlocks=1):**
```cpp
config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kCoordX, kCoordY);
// Loads 16×16 fp16 tile = 256 elements
```

**Q load (prefill, transpose=true for DPAS A):**
```cpp
config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
    (uint32_t*)qState, widthInByteQ, heightQ, widthInByteQ, qCoordX, qCoordY);
auto data = lsc_load_2d<uint32_t, 8, 16, 1, /*transpose=*/true, false, ...>(payloadQ);
```

**V load (prefill, vnni=true for DPAS B):**
```cpp
config_2d_mem_access<fp16, 16, 16, 2> payloadV(
    (fp16*)vState, widthInByteKV, heightKv, widthInByteKV, vCoordX, vCoordY);
auto data = lsc_load_2d<fp16, 16, 16, 2, false, /*vnni=*/true, ...>(payloadV);
// Loads 2 × 16×16 tiles with VNNI layout
```

---

## lsc_prefetch_2d (2D prefetch)

Does NOT accept a payload object — use the inline 6-argument form:

```cpp
__ESIMD_ENS::lsc_prefetch_2d<T, BlockWidth, BlockHeight, NBlocks,
    /*transpose=*/false, /*vnni=*/false,
    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(
    ptr, surfaceWidth, surfaceHeight, surfacePitch, coordX, coordY);
```

Example for K prefetch in prefill:
```cpp
__ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
payloadPrefK.set_x(prefCoordX + 16 * kk);
```

Note: `payloadPrefK` here is a `config_2d_mem_access` used only for coordinates; the prefetch intrinsic reads from it. Some compiler versions accept this mixed form.

---

## lsc_scatter (irregular scatter write)

Used in prefill output write — scatter to Q-head-strided positions:

```cpp
simd<uint32_t, 16> simdOffsets;  // byte offsets for each lane
simd_mask<16> mask;              // predicate per lane

// Set offsets: each lane writes to a different Q-head position
for (int i = 0; i < 16; i++) {
    simdOffsets[i] = (q_pos * headQ + headIdx + i) * HD * sizeof(fp16);
    mask[i] = valid_q;  // only write if q_pos is within activationLength
}

__ESIMD_ENS::lsc_scatter<uint32_t, 4,
    __ESIMD_ENS::lsc_data_size::u32,
    __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back,
    16, uint32_t>(
    (uint32_t*)out_ptr, simdOffsets, data_to_write, mask);
simdOffsets += 4 * sizeof(uint32_t);  // advance to next 4 uint32s (= 16 bytes = 8 fp16s)
```

---

## SLM Operations

```cpp
// Init SLM (must be called at start of kernel)
slm_init(size_in_bytes);

// Block store to SLM (byte address)
slm_block_store<T, N>(byte_offset, data);

// Block load from SLM (byte address)
simd<T, N> data = slm_block_load<T, N>(byte_offset);

// Scatter to SLM (for non-contiguous writes)
__ESIMD_ENS::lsc_slm_scatter<uint32_t, 8,
    __ESIMD_ENS::lsc_data_size::u32, 16>(
    simd_byte_offsets, data);
```

---

## Cache Hints

| Hint | Meaning | Use case |
|------|---------|----------|
| `cache_hint::cached` | L1+L2 cached | Default for reads |
| `cache_hint::uncached` | Bypass L1 | Avoid polluting cache for streaming writes |
| `cache_hint::write_back` | Write-back to L2 | Normal output writes |
| `cache_hint::streaming` | L1 bypass, L2 fill | Large sequential reads |

For SDP: use `cached` for all K/V reads (good L2 reuse within a workgroup).

---

## Surface Dimension Fields

For `config_2d_mem_access`:
- `surfaceWidth` = total width in **bytes** minus 1
- `surfaceHeight` = total height in **rows** minus 1
- `surfacePitch` = stride in **bytes** (bytes per row, usually same as surfaceWidth+1)
- `coordX` = starting column in **elements** (not bytes)
- `coordY` = starting row index

```cpp
// Example: K matrix [kv_len, headKv, HD], fp16
uint32_t widthInByteKV = headKv * HD * sizeof(fp16) - 1;   // row width - 1
uint32_t heightKv       = kv_len - 1;                       // num rows - 1
uint32_t pitchKV        = headKv * HD * sizeof(fp16);       // row stride
uint32_t kCoordX        = kvHeadIdx * HD;                   // column = head offset (elements)
uint32_t kCoordY        = 0;                                // row = 0 initially
```
