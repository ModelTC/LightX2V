// Approach 2: Eliminate S Register Transpose via lsc_slm_scatter Write + Block Read
//
// Changes from v5 baseline (rev256_onednn_v2_qk_pipeline.h):
//   - S WRITE: Use lsc_slm_scatter<uint32_t, 2, u32, 16> to write S tiles directly
//     to transposed SLM positions. 16 addresses (one per Q row across 2 qh tiles),
//     each writing 2 u32 = 4 contiguous KV values. Pack via u32 shift+OR + stride-2 interleave.
//     4 scatter calls per qp (kv groups 0..3, 4..7, 8..11, 12..15).
//     SLM layout after scatter: Q-major, KV-fast (same as baseline after transpose)
//     Same SLM addressing as baseline: S_SLM_BASE + (sg_i*16 + sg_j*Q_GRPS + qg)*256
//   - S READ:  Unchanged — slm_block_load<uint32_t, 64>
//   - DPAS:    Unchanged
//
// Expected: Remove ~270 mov register transpose instructions.
//           Add 4 lsc_slm_scatter<uint32_t, 4, u32, 16> operations (2 qp x 2 kv_halves).

ESIMD_INLINE void flashAttnBMha256Fp16Rev_s_scatter(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* normAlpha,
  uint8_t* out,
  uint32_t activationLength,
  uint32_t kvSeqLen,
  uint32_t headQ,
  uint32_t headKv,
  sycl::nd_item<2>& ndi) {

  constexpr float INV_SQRT_HD = 0.0625f;
  constexpr float LOG2E = sycl::ext::intel::esimd::detail::log2e;
  constexpr float attnScoreMul = INV_SQRT_HD * LOG2E;
  constexpr uint32_t HD = 256;
  constexpr uint32_t HD_BLKS = 16;
  constexpr uint32_t Q_ROWS = 8;
  constexpr uint32_t WG_Q_ROWS = 128;
  constexpr uint32_t KV_CHUNK = 128;
  constexpr uint32_t KV_PER_SG = 16;
  constexpr uint32_t KV_BLKS = 8;
  constexpr uint32_t Q_PAIRS = 2;
  constexpr uint32_t Q_TILES = 8;
  constexpr uint32_t Q_GRPS = 4;
  constexpr uint32_t D_BLKS_PER_SG = 2;

  constexpr uint32_t Q_SLM_BASE   = 0x00000;
  constexpr uint32_t S_SLM_BASE   = 0x10000;
  constexpr uint32_t MAX_SLM_BASE = 0x18000;
  constexpr uint32_t SUM_SLM_BASE = 0x19000;
  constexpr uint32_t TOTAL_SLM    = 0x1A000;

  __ESIMD_NS::slm_init(TOTAL_SLM);

  __esimd_nbarrier_init(1);

  int32_t tid = ndi.get_local_id(0);
  int32_t sg_i = tid & 7;
  int32_t sg_j = tid >> 3;
  int32_t headIdx = ndi.get_group(0);
  int32_t qBlock = ndi.get_group(1);

  int32_t groupSize = headQ / headKv;
  int32_t kvHeadIdx = headIdx / groupSize;

  uint32_t widthInByteQ  = headQ  * HD * sizeof(fp16) - 1;
  uint32_t widthInByteKV = headKv * HD * sizeof(fp16) - 1;
  uint32_t heightQ  = activationLength - 1;
  uint32_t heightKv = kvSeqLen - 1;

  uint32_t qGlobalY = qBlock * WG_Q_ROWS;
  uint32_t qGlobalX = headIdx * HD;
  uint32_t kvGlobalX = kvHeadIdx * HD;

  // ============================================================
  // COOPERATIVE Q LOAD TO SLM
  // ============================================================
  {
    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
      (uint32_t*)qState, widthInByteQ, heightQ, widthInByteQ, 0, 0);

    #pragma unroll
    for (int t = 0; t < 4; t++) {
      int tile_id = tid * 4 + t;
      int d_blk = tile_id >> 3;
      int q_tile = tile_id & 7;

      payloadQ.set_x((qGlobalX >> 1) + d_blk * 8);
      payloadQ.set_y(qGlobalY + q_tile * 16);

      simd<uint32_t, 128> qTile = __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);

      uint32_t slm_off = Q_SLM_BASE + tile_id * 512;
      slm_block_store<uint32_t, 64>(slm_off, qTile.select<64, 1>(0));
      slm_block_store<uint32_t, 64>(slm_off + 256, qTile.select<64, 1>(64));
    }
  }

  barrier();

  // ============================================================
  // REGISTER DECLARATIONS
  // ============================================================
  simd<float, 1024> A_tile = 0;
  simd<float, 512> ST_tile;
  simd<float, 512> ST_next;
  simd<float, 32> fp32_max = FP32_MIN;
  simd<float, 32> fp32_sum = 0;
  simd<float, 32> delta;

  int32_t kvOuterLoops = (kvSeqLen + KV_CHUNK - 1) / KV_CHUNK;

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kvGlobalX, 0);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadKpf(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kvGlobalX, 0);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadV(
    (fp16*)vState, widthInByteKV, heightKv, widthInByteKV, kvGlobalX, 0);
  __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadVpf(
    (uint32_t*)vState, widthInByteKV, heightKv, widthInByteKV, 0, 0);

  // Prefetch K[0] + K[1]
  {
    uint32_t init_kv_row = sg_i * KV_PER_SG;
    payloadKpf.set_y(init_kv_row);
    #pragma unroll
    for (int d = 0; d < HD_BLKS; d++) {
      payloadKpf.set_x(kvGlobalX + d * 16);
      __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
    }
    if (kvOuterLoops > 1) {
      uint32_t next_kv_row = KV_CHUNK + sg_i * KV_PER_SG;
      payloadKpf.set_y(next_kv_row);
      #pragma unroll
      for (int d = 0; d < HD_BLKS; d++) {
        payloadKpf.set_x(kvGlobalX + d * 16);
        __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
          __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
      }
    }
  }

  // ============================================================
  // PROLOGUE: QK[0] -> ST_tile
  // ============================================================
  {
    ST_tile = 0;

    payloadK.set_y(sg_i * KV_PER_SG);
    payloadK.set_x(kvGlobalX);
    simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

    #pragma unroll
    for (int d = 0; d < HD_BLKS; d++) {
      payloadVpf.set_y((d >> 1) * 16);
      payloadVpf.set_x(kvGlobalX / 2 + sg_i * 16 + (d & 1) * 8);
      __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadVpf);

      simd<fp16, 128> K_sb0(K_both.select<128, 1>(0).data());
      simd<fp16, 128> K_sb1(K_both.select<128, 1>(128).data());

      uint32_t q_slm_off0 = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + 0) * 512;
      uint32_t q_slm_off1 = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + 1) * 512;

      simd<fp16, 256> Q_vnni0, Q_vnni1;
      Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(0) =
        slm_block_load<uint32_t, 64>(q_slm_off0);
      Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(64) =
        slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
      Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(0) =
        slm_block_load<uint32_t, 64>(q_slm_off1);
      Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(64) =
        slm_block_load<uint32_t, 64>(q_slm_off1 + 256);

      { auto acc = ST_tile.select<128, 1>(0);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni0.data()), K_sb0); }
      { auto acc = ST_tile.select<128, 1>(128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni0.data()), K_sb1); }
      { auto acc = ST_tile.select<128, 1>(256);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni1.data()), K_sb0); }
      { auto acc = ST_tile.select<128, 1>(384);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni1.data()), K_sb1); }

      if (d < HD_BLKS - 1) {
        payloadK.set_x(kvGlobalX + (d + 1) * 16);
        K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
          __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
      }
    }
  }

  // ============================================================
  // OUTER LOOP
  // ============================================================
  for (int32_t outerIter = 0; outerIter < kvOuterLoops; outerIter++) {
    uint32_t kv_start = outerIter * KV_CHUNK;
    uint32_t kv_row = kv_start + sg_i * KV_PER_SG;

    // ========================================
    // SOFTMAX FIRST HALF
    // ========================================
    ST_tile *= attnScoreMul;

    if (outerIter == kvOuterLoops - 1) {
      uint32_t kv_base = kv_start + sg_i * KV_PER_SG;
      #pragma unroll
      for (int qp = 0; qp < Q_PAIRS; qp++) {
        #pragma unroll
        for (int kv = 0; kv < 8; kv++) {
          if (kv_base + kv >= kvSeqLen)
            ST_tile.select<16, 1>(qp * 256 + kv * 16) = FP32_MIN;
          if (kv_base + 8 + kv >= kvSeqLen)
            ST_tile.select<16, 1>(qp * 256 + 128 + kv * 16) = FP32_MIN;
        }
      }
    }

    simd<float, 32> local_max;
    #pragma unroll
    for (int qp = 0; qp < Q_PAIRS; qp++) {
      local_max.select<16, 1>(qp * 16) = ST_tile.select<16, 1>(qp * 256);
      #pragma unroll
      for (int kv = 1; kv < 8; kv++)
        local_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
          local_max.select<16, 1>(qp * 16),
          ST_tile.select<16, 1>(qp * 256 + kv * 16));
      #pragma unroll
      for (int kv = 0; kv < 8; kv++)
        local_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
          local_max.select<16, 1>(qp * 16),
          ST_tile.select<16, 1>(qp * 256 + 128 + kv * 16));
    }

    #pragma unroll
    for (int qp = 0; qp < Q_PAIRS; qp++) {
      uint32_t q_base = sg_j * 32 + qp * 16;
      slm_block_store<float, 16>(MAX_SLM_BASE + (sg_i * 128 + q_base) * 4,
        local_max.select<16, 1>(qp * 16));
    }

    // ========================================
    // BARRIER A
    // ========================================
    __esimd_nbarrier_arrive(0, 0, 32, 32);

    if (outerIter < kvOuterLoops - 1) {
      uint32_t next_kv_start = (outerIter + 1) * KV_CHUNK;
      uint32_t next_kv_row = next_kv_start + sg_i * KV_PER_SG;

      ST_next = 0;

      payloadK.set_y(next_kv_row);
      payloadK.set_x(kvGlobalX);
      simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

      #pragma unroll
      for (int d = 0; d < HD_BLKS; d++) {
        payloadVpf.set_y(next_kv_start + (d >> 1) * 16);
        payloadVpf.set_x(kvGlobalX / 2 + sg_i * 16 + (d & 1) * 8);
        __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
          __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadVpf);

        simd<fp16, 128> K_sb0(K_both.select<128, 1>(0).data());
        simd<fp16, 128> K_sb1(K_both.select<128, 1>(128).data());

        uint32_t q_slm_off0 = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + 0) * 512;
        uint32_t q_slm_off1 = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + 1) * 512;

        simd<fp16, 256> Q_vnni0, Q_vnni1;
        Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(0) =
          slm_block_load<uint32_t, 64>(q_slm_off0);
        Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(64) =
          slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
        Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(0) =
          slm_block_load<uint32_t, 64>(q_slm_off1);
        Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(64) =
          slm_block_load<uint32_t, 64>(q_slm_off1 + 256);

        { auto acc = ST_next.select<128, 1>(0);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni0.data()), K_sb0); }
        { auto acc = ST_next.select<128, 1>(128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni0.data()), K_sb1); }
        { auto acc = ST_next.select<128, 1>(256);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni1.data()), K_sb0); }
        { auto acc = ST_next.select<128, 1>(384);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(Q_vnni1.data()), K_sb1); }

        if (d < HD_BLKS - 1) {
          payloadK.set_x(kvGlobalX + (d + 1) * 16);
          K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }
      }
    }

    __esimd_nbarrier(0, 0, 32);  // wait: max sync complete

    // ========================================
    // SOFTMAX SECOND HALF
    // ========================================
    simd<float, 32> global_max = FP32_MIN;
    #pragma unroll
    for (int si = 0; si < 8; si++) {
      #pragma unroll
      for (int qp = 0; qp < Q_PAIRS; qp++) {
        uint32_t q_base = sg_j * 32 + qp * 16;
        simd<float, 16> m = slm_block_load<float, 16>(MAX_SLM_BASE + (si * 128 + q_base) * 4);
        global_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
          global_max.select<16, 1>(qp * 16), m);
      }
    }
    global_max = __ESIMD_NS::max<float, 32, float>(global_max, fp32_max);

    delta = __ESIMD_NS::exp2<float, 32, float>(fp32_max - global_max);
    fp32_max = global_max;

    simd<float, 32> local_sum = 0;
    simd<fp16, 256> ST_fp16_0;
    simd<fp16, 256> ST_fp16_1;

    {
      simd<float, 16> gm = global_max.select<16, 1>(0);
      #pragma unroll
      for (int kv = 0; kv < 8; kv++) {
        simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
          ST_tile.select<16, 1>(kv * 16) - gm);
        ST_fp16_0.select<16, 1>(kv * 16) = s;
        local_sum.select<16, 1>(0) += s;
      }
      #pragma unroll
      for (int kv = 0; kv < 8; kv++) {
        simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
          ST_tile.select<16, 1>(128 + kv * 16) - gm);
        ST_fp16_0.select<16, 1>(128 + kv * 16) = s;
        local_sum.select<16, 1>(0) += s;
      }
    }
    {
      simd<float, 16> gm = global_max.select<16, 1>(16);
      #pragma unroll
      for (int kv = 0; kv < 8; kv++) {
        simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
          ST_tile.select<16, 1>(256 + kv * 16) - gm);
        ST_fp16_1.select<16, 1>(kv * 16) = s;
        local_sum.select<16, 1>(16) += s;
      }
      #pragma unroll
      for (int kv = 0; kv < 8; kv++) {
        simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
          ST_tile.select<16, 1>(256 + 128 + kv * 16) - gm);
        ST_fp16_1.select<16, 1>(128 + kv * 16) = s;
        local_sum.select<16, 1>(16) += s;
      }
    }

    // ========================================
    // APPROACH 2: lsc_slm_scatter<uint32_t, 2, u32, 16> TO TRANSPOSED SLM POSITIONS
    //
    // 16 addresses (8 Q rows × 2 qh tiles), each writing 2 u32 = 4 fp16 KV values.
    // Four scatter calls per qp: kv groups 0..3, 4..7, 8..11, 12..15.
    //
    // Data packing: pack 2 adjacent KV rows into u32 via shift+OR, then interleave
    // the two u32 columns using stride-2 u32 dest (HStride=2, HW-safe).
    //
    // Total: 2 qp × 4 kv_groups = 8 lsc_slm_scatter<uint32_t, 2, u32, 16> operations
    // ========================================
    #pragma unroll
    for (int qp = 0; qp < Q_PAIRS; qp++) {
      auto& ST_fp16 = (qp == 0) ? ST_fp16_0 : ST_fp16_1;
      simd<uint16_t, 256> ST_fp16_u16 = ST_fp16.template bit_cast_view<uint16_t>();

      // Tile addresses for this qp's two qh groups
      uint32_t tile_addr_qh0 = S_SLM_BASE + (sg_i * 16 + sg_j * Q_GRPS + qp * 2 + 0) * 256;
      uint32_t tile_addr_qh1 = S_SLM_BASE + (sg_i * 16 + sg_j * Q_GRPS + qp * 2 + 1) * 256;

      // 16 Q-row addresses: q=0..7 in qh0 tile, q=0..7 in qh1 tile
      simd<uint32_t, 16> q_offsets;
      #pragma unroll
      for (int q = 0; q < 8; q++)
        q_offsets[q] = tile_addr_qh0 + q * 32;
      #pragma unroll
      for (int q = 0; q < 8; q++)
        q_offsets[8 + q] = tile_addr_qh1 + q * 32;

      // 4 kv groups, each writing 4 KV values (2 u32) per Q address
      #pragma unroll
      for (int kg = 0; kg < 4; kg++) {
        int kv_base = kg * 4;
        // Pack kv_base+0 and kv_base+1 into u32 (low|high<<16)
        simd<uint32_t, 16> packed0 =
          simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 0) * 16)) |
          (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 1) * 16)) << 16);
        // Pack kv_base+2 and kv_base+3
        simd<uint32_t, 16> packed1 =
          simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 2) * 16)) |
          (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 3) * 16)) << 16);
        // Element-major layout: first 16 = elem0 for all addrs, next 16 = elem1
        simd<uint32_t, 32> data;
        data.select<16, 1>(0) = packed0;
        data.select<16, 1>(16) = packed1;
        __ESIMD_ENS::lsc_slm_scatter<uint32_t, 2, __ESIMD_ENS::lsc_data_size::u32, 16>(
          q_offsets + kv_base * 2, data);
      }
    }

    // ========================================
    // BARRIER B (reuse nbarrier 0): arrive, V loads + compensation, wait
    // ========================================
    __esimd_nbarrier_arrive(0, 0, 32, 32);

    fp32_sum = fp32_sum * delta + local_sum;

    uint32_t kpf_row = kv_row + 2 * KV_CHUNK;

    uint32_t vRow0 = kv_start;
    payloadV.set_y(vRow0);
    payloadV.set_x(kvGlobalX + sg_i * 32);
    simd<fp16, 256> V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

    payloadV.set_x(kvGlobalX + sg_i * 32 + 16);
    simd<fp16, 256> V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

    payloadKpf.set_y(kpf_row);
    payloadKpf.set_x(kvGlobalX);
    __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
    payloadKpf.set_x(kvGlobalX + 16);
    __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);

    #pragma unroll
    for (int qg = 0; qg < Q_GRPS; qg++) {
      #pragma unroll
      for (int db = 0; db < D_BLKS_PER_SG; db++) {
        #pragma unroll
        for (int q = 0; q < Q_ROWS; q++) {
          float d = delta[qg * Q_ROWS + q];
          A_tile.select<16, 1>((qg * D_BLKS_PER_SG + db) * 128 + q * 16) *= d;
        }
      }
    }

    __esimd_nbarrier(0, 0, 32);  // wait: S sync complete

    // ========================================
    // VS PHASE — UNCHANGED (block_load S + DPAS)
    // SLM layout is Q-major, KV-fast (same as baseline after transpose)
    // ========================================
    {
      uint32_t s_base = S_SLM_BASE + sg_j * Q_GRPS * 256;
      simd<fp16, 128> S0, S1, S2, S3;
      S0.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base);
      S1.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 256);
      S2.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 512);
      S3.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 768);

      { auto acc = A_tile.select<128, 1>(0 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S0.data())); }
      { auto acc = A_tile.select<128, 1>(1 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S0.data())); }
      { auto acc = A_tile.select<128, 1>(2 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S1.data())); }
      { auto acc = A_tile.select<128, 1>(3 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S1.data())); }
      { auto acc = A_tile.select<128, 1>(4 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S2.data())); }
      { auto acc = A_tile.select<128, 1>(5 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S2.data())); }
      { auto acc = A_tile.select<128, 1>(6 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S3.data())); }
      { auto acc = A_tile.select<128, 1>(7 * 128);
        acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S3.data())); }
    }

    #pragma unroll
    for (int kv_blk = 1; kv_blk < KV_BLKS; kv_blk++) {
      uint32_t vRow = kv_start + kv_blk * 16;

      payloadV.set_y(vRow);
      payloadV.set_x(kvGlobalX + sg_i * 32);
      V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

      payloadV.set_x(kvGlobalX + sg_i * 32 + 16);
      V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

      payloadKpf.set_y(kpf_row);
      payloadKpf.set_x(kvGlobalX + (kv_blk * 2) * 16);
      __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
      payloadKpf.set_x(kvGlobalX + (kv_blk * 2 + 1) * 16);
      __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);

      {
        uint32_t s_base = S_SLM_BASE + (kv_blk * 16 + sg_j * Q_GRPS) * 256;
        simd<fp16, 128> S0, S1, S2, S3;
        S0.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base);
        S1.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 256);
        S2.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 512);
        S3.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 768);

        { auto acc = A_tile.select<128, 1>(0 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S0.data())); }
        { auto acc = A_tile.select<128, 1>(1 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S0.data())); }
        { auto acc = A_tile.select<128, 1>(2 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S1.data())); }
        { auto acc = A_tile.select<128, 1>(3 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S1.data())); }
        { auto acc = A_tile.select<128, 1>(4 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S2.data())); }
        { auto acc = A_tile.select<128, 1>(5 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S2.data())); }
        { auto acc = A_tile.select<128, 1>(6 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni0.data()), simd<fp16, 128>(S3.data())); }
        { auto acc = A_tile.select<128, 1>(7 * 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(acc.data()), simd<fp16, 256>(V_vnni1.data()), simd<fp16, 128>(S3.data())); }
      }
    }

    ST_tile = ST_next;
  }

  // ============================================================
  // FINAL OUTPUT (unchanged)
  // ============================================================
  #pragma unroll
  for (int qp = 0; qp < Q_PAIRS; qp++) {
    uint32_t q_base = sg_j * 32 + qp * 16;
    slm_block_store<float, 16>(SUM_SLM_BASE + (sg_i * 128 + q_base) * 4,
      fp32_sum.select<16, 1>(qp * 16));
  }

  barrier();

  simd<float, 32> total_sum = 0;
  #pragma unroll
  for (int si = 0; si < 8; si++) {
    #pragma unroll
    for (int qp = 0; qp < Q_PAIRS; qp++) {
      uint32_t q_base = sg_j * 32 + qp * 16;
      total_sum.select<16, 1>(qp * 16) += slm_block_load<float, 16>(
        SUM_SLM_BASE + (si * 128 + q_base) * 4);
    }
  }

  simd<float, 32> inv_sum;
  inv_sum.select<16, 1>(0)  = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(0));
  inv_sum.select<16, 1>(16) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(16));

  uint32_t d_start = sg_i * D_BLKS_PER_SG * 16;
  simd<float, 32> alphaV;
  alphaV.select<16, 1>(0)  = block_load<float, 16>((float*)normAlpha + headIdx * HD + d_start);
  alphaV.select<16, 1>(16) = block_load<float, 16>((float*)normAlpha + headIdx * HD + d_start + 16);

  uint32_t outW = headQ * HD * sizeof(fp16) - 1;
  uint32_t outH = activationLength - 1;
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 8, 1> payloadO(
    (fp16*)out, outW, outH, outW, 0, 0);

  #pragma unroll
  for (int qg = 0; qg < Q_GRPS; qg++) {
    #pragma unroll
    for (int db = 0; db < D_BLKS_PER_SG; db++) {
      simd<float, 128> fOut = A_tile.select<128, 1>((qg * D_BLKS_PER_SG + db) * 128);
      simd<float, 16> alpha = alphaV.select<16, 1>(db * 16);

      #pragma unroll
      for (int q = 0; q < Q_ROWS; q++) {
        float inv = inv_sum[qg * Q_ROWS + q];
        fOut.select<16, 1>(q * 16) *= (alpha * inv);
      }

      simd<fp16, 128> outTile = fOut;
      payloadO.set_x(headIdx * HD + d_start + db * 16);
      payloadO.set_y(qGlobalY + sg_j * 32 + qg * Q_ROWS);
      __ESIMD_ENS::lsc_store_2d<fp16, 16, 8, 1,
        __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back>(payloadO, outTile);
    }
  }
}
