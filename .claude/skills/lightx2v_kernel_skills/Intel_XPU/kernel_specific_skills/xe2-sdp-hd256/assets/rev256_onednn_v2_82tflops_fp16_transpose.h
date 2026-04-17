// oneDNN-inspired Flash Attention v2 kernel for HD=256
// Cross-phase prefetch: V prefetch during QK, next-iter K prefetch during VS
// V_vnni × S approach: V loaded with VNNI transform (src1), S transposed in SLM (src2)
//
// Optimizations over baseline:
//   1. Removed redundant max_slm reset (all positions overwritten each iter)
//   2. Merged S SLM stores/loads: 1×64 u32 instead of 2×32 u32
//   3. Deferred A_tile compensation to after barrier B (overlaps with V loads)
//
// Prefetch pipeline (L1 = 256KB per Xe-core, ~128KB K+V working set):
//   QK phase: load K[this iter], prefetch V[this iter] → V warm for VS
//   VS phase: load V[this iter], prefetch K[next iter] → K warm for next QK
//
// Thread mapping: sg_i = tid & 7 (KV/D dim), sg_j = tid >> 3 (Q dim)
//   QK tile: 16K × 32Q per SG;  VS tile: 32D × 32Q per SG
//
// SLM layout (104 KB):
//   Q_slm:   64 KB @ 0x00000  [16 d_blks × 8 q_tiles × 128 u32] VNNI 16×16
//   S_slm:   32 KB @ 0x10000  [8 kv_blks × 16 q_grps × 128 fp16] row-major 8×16
//   max_slm:  4 KB @ 0x18000  [8 sg_i × 128 q] float
//   sum_slm:  4 KB @ 0x19000  [8 sg_i × 128 q] float

ESIMD_INLINE void flashAttnBMha256Fp16Rev_onednn_v2(
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
  constexpr uint32_t Q_GRPS = 4;         // VS phase: 8Q per group
  constexpr uint32_t D_BLKS_PER_SG = 2;

  constexpr uint32_t Q_SLM_BASE   = 0x00000;
  constexpr uint32_t S_SLM_BASE   = 0x10000;
  constexpr uint32_t MAX_SLM_BASE = 0x18000;
  constexpr uint32_t SUM_SLM_BASE = 0x19000;
  constexpr uint32_t TOTAL_SLM    = 0x1A000;

  __ESIMD_NS::slm_init(TOTAL_SLM);

  // Named barrier for split barrier B (arrive/wait pattern)
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
  // COOPERATIVE Q LOAD TO SLM (16×16 VNNI FORMAT)
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

  // No max_slm init needed — all positions written fresh before each barrier A
  barrier();

  // ============================================================
  // REGISTER DECLARATIONS
  // ============================================================
  simd<float, 1024> A_tile = 0;       // [4qg × 2db × 8Q × 16D] float
  simd<float, 512> ST_tile;           // [2qp × 2sb × 8KV × 16Q] float
  simd<float, 32> fp32_max = FP32_MIN;
  simd<float, 32> fp32_sum = 0;
  simd<float, 32> delta;              // persist across barrier for deferred compensation

  int32_t kvOuterLoops = (kvSeqLen + KV_CHUNK - 1) / KV_CHUNK;

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kvGlobalX, 0);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadKpf(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kvGlobalX, 0);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadV(
    (fp16*)vState, widthInByteKV, heightKv, widthInByteKV, kvGlobalX, 0);
  __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadVpf(
    (uint32_t*)vState, widthInByteKV, heightKv, widthInByteKV, 0, 0);

  // Prefetch initial K tiles for first iteration (cold start warming)
  {
    uint32_t init_kv_row = sg_i * KV_PER_SG;
    payloadKpf.set_y(init_kv_row);
    #pragma unroll
    for (int d = 0; d < HD_BLKS; d++) {
      payloadKpf.set_x(kvGlobalX + d * 16);
      __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
    }
  }

  // ============================================================
  // OUTER LOOP: iterate KV in chunks of 128
  // ============================================================
  for (int32_t outerIter = 0; outerIter < kvOuterLoops; outerIter++) {
    uint32_t kv_start = outerIter * KV_CHUNK;
    uint32_t kv_row = kv_start + sg_i * KV_PER_SG;

    // ========================================
    // QK PHASE: S^T = Q_vnni × K
    // Cross-phase: prefetch V while computing QK
    // 16 d_steps × (1 K load + 1 V prefetch + 4 DPAS)
    // No "if" in loop — V prefetch unconditional
    // ========================================
    ST_tile = 0;

    #pragma unroll
    for (int d = 0; d < HD_BLKS; d++) {
      // K load for current d_step
      payloadK.set_y(kv_row);
      payloadK.set_x(kvGlobalX + d * 16);
      simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

      // V prefetch: 1 per d_step, 16 total = 8 kv_blks × 2 D-tiles
      // d=0,1 → kv_blk=0; d=2,3 → kv_blk=1; ... d=14,15 → kv_blk=7
      payloadVpf.set_y(kv_start + (d >> 1) * 16);
      payloadVpf.set_x(kvGlobalX / 2 + sg_i * 16 + (d & 1) * 8);
      __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadVpf);

      simd<fp16, 128> K_sb0(K_both.select<128, 1>(0).data());
      simd<fp16, 128> K_sb1(K_both.select<128, 1>(128).data());

      #pragma unroll
      for (int qp = 0; qp < Q_PAIRS; qp++) {
        uint32_t q_slm_off = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + qp) * 512;

        simd<fp16, 256> Q_vnni;
        Q_vnni.template bit_cast_view<uint32_t>().select<64, 1>(0) =
          slm_block_load<uint32_t, 64>(q_slm_off);
        Q_vnni.template bit_cast_view<uint32_t>().select<64, 1>(64) =
          slm_block_load<uint32_t, 64>(q_slm_off + 256);

        {
          auto acc = ST_tile.select<128, 1>(qp * 256);
          acc = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 128>(acc.data()),
            simd<fp16, 256>(Q_vnni.data()),
            K_sb0);
        }
        {
          auto acc = ST_tile.select<128, 1>(qp * 256 + 128);
          acc = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 128>(acc.data()),
            simd<fp16, 256>(Q_vnni.data()),
            K_sb1);
        }
      }
    }

    // ========================================
    // SOFTMAX on S^T[2qp × 2sb × 8KV × 16Q]
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

    barrier();  // Barrier A: all maxes written

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

    // Compute delta for deferred compensation (applied after barrier B)
    delta = __ESIMD_NS::exp2<float, 32, float>(fp32_max - global_max);
    fp32_sum *= delta;
    fp32_max = global_max;

    // exp2 + sum → write to fp16 temp (no ST_tile write-back needed)
    // Separating fp32→fp16 conversion from strided transpose gather
    // generates wider SIMD moves instead of scalar element-by-element moves
    simd<float, 32> local_sum = 0;
    simd<fp16, 256> ST_fp16_0;  // fp16 copy of qp=0: [2sb × 8KV × 16Q]
    simd<fp16, 256> ST_fp16_1;  // fp16 copy of qp=1: [2sb × 8KV × 16Q]

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
    fp32_sum += local_sum;

    // ========================================
    // TRANSPOSE S^T → S AND WRITE TO SLM
    // Strided gathers on fp16 registers (not fp32) for wider ISA moves
    // ========================================
    #pragma unroll
    for (int qp = 0; qp < Q_PAIRS; qp++) {
      auto& ST_fp16 = (qp == 0) ? ST_fp16_0 : ST_fp16_1;
      #pragma unroll
      for (int qh = 0; qh < 2; qh++) {
        int qg = qp * 2 + qh;
        int q_off = qh * 8;
        uint32_t tile_addr = S_SLM_BASE + (sg_i * 16 + sg_j * Q_GRPS + qg) * 256;

        simd<fp16, 128> S_tile;
        #pragma unroll
        for (int q = 0; q < 8; q++) {
          S_tile.select<8, 1>(q * 16)     = ST_fp16.select<8, 16>(q_off + q);
          S_tile.select<8, 1>(q * 16 + 8) = ST_fp16.select<8, 16>(128 + q_off + q);
        }

        slm_block_store<uint32_t, 64>(tile_addr,
          S_tile.template bit_cast_view<uint32_t>());
      }
    }

    // Split barrier B: arrive (non-blocking), do useful work, then wait
    __esimd_nbarrier_arrive(0, 0, 32, 32);

    // ========================================
    // VS PHASE: A += V_vnni × S
    // Between arrive and wait: V loads + K prefetch + compensation
    // Cross-phase: prefetch K[next iter] while computing VS
    // ========================================
    uint32_t next_kv_row = kv_row + KV_CHUNK;

    // Issue first V loads (async) — these will be in-flight during compensation
    uint32_t vRow0 = kv_start;
    payloadV.set_y(vRow0);
    payloadV.set_x(kvGlobalX + sg_i * 32);
    simd<fp16, 256> V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

    payloadV.set_x(kvGlobalX + sg_i * 32 + 16);
    simd<fp16, 256> V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

    // K prefetch for next iteration (first 2 d_blks)
    payloadKpf.set_y(next_kv_row);
    payloadKpf.set_x(kvGlobalX);
    __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
    payloadKpf.set_x(kvGlobalX + 16);
    __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);

    // Deferred compensation: V loads in flight, ALU overlaps with memory latency
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

    // Wait for all S writes to complete before reading S from SLM
    __esimd_nbarrier(0, 0, 32);

    // Process kv_blk=0 with already-loaded V — pre-load all S for 8 consecutive DPAS
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

    // Remaining kv_blks (1..7)
    #pragma unroll
    for (int kv_blk = 1; kv_blk < KV_BLKS; kv_blk++) {
      uint32_t vRow = kv_start + kv_blk * 16;

      // Load V with VNNI transform
      payloadV.set_y(vRow);
      payloadV.set_x(kvGlobalX + sg_i * 32);
      V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

      payloadV.set_x(kvGlobalX + sg_i * 32 + 16);
      V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

      // K prefetch for next iteration: 2 per kv_blk
      payloadKpf.set_y(next_kv_row);
      payloadKpf.set_x(kvGlobalX + (kv_blk * 2) * 16);
      __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
      payloadKpf.set_x(kvGlobalX + (kv_blk * 2 + 1) * 16);
      __ESIMD_ENS::lsc_prefetch_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);

      // Pre-load all 4 S tiles for 8 consecutive DPAS
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
  }

  // ============================================================
  // FINAL OUTPUT: O[Q×D] → normalize and write directly
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
        float inv = 1.0f / (float)total_sum[qg * Q_ROWS + q];
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
