// Optimized FP16 Flash Attention kernel for HD=256
// Based on flash.attn.b.mha128.fp16.opt.h adapted with Approach A1:
//   - 8 Q-rows per thread (halved from 16) to fit GRF budget
//   - 16 threads/WG, kv_step=64
//   - SLM: 64KB per WG (2 × 64 × 256 × 2B ping-pong V)
//   - GRF budget: ~232/256 (24 headroom)
//   - Dispatch: 128 Q-rows/WG (was 256 for HD=128) → 2x dispatch count
// Non-causal only.
//
// GRF budget (per thread, 16 threads/WG):
//   fp16QState (Q, resident):  8×256×2B  = 4096B = 64 GRFs
//   tempOutput (scores+V tmp): 16×64×4B  = 4096B = 64 GRFs  (first 512 for 8×64 scores, rest for V SLM load temp)
//   finalOutput (accum, fp16): 8×256×2B  = 4096B = 64 GRFs
//   tempBuffer (K scratch):    16×32×4B  = 2048B = 32 GRFs
//   fp16VState (V load buf):   1024×2B   = 2048B = 32 GRFs
//   ttemp + scalars + misc:              ≈  512B ≈  8 GRFs
//   Total:                                        ≈232 GRFs
//
// DPAS geometry (dpas<8, 8> with ExecSize=8 for 8 Q-rows):
//   API order: dpas(C, B, A) — B is VNNI-encoded
//   QK: B=128 fp16 (Q: 8 Q-rows × 16 HD-cols VNNI), A=128 fp16 (K: 8 KV-pos × 16 HD-cols), C=64 float
//       nn=0..15 (256 HD-cols), kk=0..7 (64 score-cols)
//   SxV: B=128 fp16 (scores VNNI), A=128 fp16 (V data), C=64 float (ExecSize=8 requires float accum)
//        nn=0..1 per block × 2 blocks, ll=0..7 (64 output-cols per l-group)
//        4 l-groups × 64 = 256 output-cols
//   Note: ExecSize=8 forbids fp16 accumulator — SxV uses float temp with fp16↔float conversion

ESIMD_INLINE void flashAttnBMha256Fp16OptPrecomputed(
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
  constexpr float matMulQuantCoeff = 0.0625f;  // 1/sqrt(256)
  constexpr float attnScoreMul = matMulQuantCoeff * sycl::ext::intel::esimd::detail::log2e;
  constexpr uint32_t HD = 256;
  constexpr uint32_t Q_ROWS = 8;    // halved from 16 for GRF budget
  constexpr uint32_t KV_STEP = 64;
  constexpr uint32_t slmSizeV = 2 * KV_STEP * HD * sizeof(fp16);  // 2×64×256×2 = 64KB
  constexpr uint32_t slmSize = slmSizeV;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  constexpr uint32_t slmOffsetBaseV = 0;

  int32_t localLinearId = ndi.get_local_id(0);
  int32_t hhq = localLinearId & 0xf;    // 16-way thread ID
  int32_t hhv = localLinearId & 0x3;    // V load: 4-way x split
  int32_t vvv = localLinearId >> 2;     // V load: 4-way y split
  int32_t hhpref = localLinearId & 0x1;
  int32_t vvpref = localLinearId >> 1;
  int32_t h = ndi.get_group(1);         // Q-row block index
  int32_t v = ndi.get_group(0);         // head index

  int32_t headIdx = v;
  int32_t groupSize = headQ / headKv;
  int32_t kvHeadIdx = headIdx / groupSize;

  // Register declarations
  simd<fp16, Q_ROWS * HD> fp16QState;              // 8×256 = 2048 fp16 = 64 GRFs
  simd<float, 16 * 32> tempBuffer;                  // K scratch = 32 GRFs (reused for packed scores)
  simd<float, 16 * 64> tempOutput;                  // 1024 floats = 64 GRFs
                                                     // [0..511]: 8×64 scores; [512..1023]: V temp during SxV
  auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
  auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  simd<fp16, Q_ROWS * HD> finalOutput = 0;          // 8×256 = 2048 fp16 = 64 GRFs
  simd<float, Q_ROWS> fp32SoftMaxTemp = 0;
  simd<float, Q_ROWS> fp32HistoricMaxTemp = FP32_MIN;
  simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);

  int32_t kvSeqOutLoopCount = (kvSeqLen + KV_STEP - 1) / KV_STEP;

  uint32_t widthInByteQ = headQ * HD * sizeof(fp16) - 1;
  uint32_t widthInByteKV = headKv * HD * sizeof(fp16) - 1;
  uint32_t heightQ = activationLength - 1;
  uint32_t heightKv = kvSeqLen - 1;

  // Each WG processes 16T × 8 rows = 128 Q-rows (was 256 for HD=128)
  uint32_t qCoordX = headIdx * HD >> 1;  // in uint32_t units (fp16→u32)
  uint32_t qCoordY = h * 128 + hhq * Q_ROWS;
  uint32_t kCoordX = kvHeadIdx * HD;
  uint32_t kCoordY = 0;
  // V cooperative load: each thread loads 64×256/16 = 1024 fp16
  // 4 threads across X (each 64 cols via 2×32-col loads), 4 threads across Y (each 16 rows)
  uint32_t vCoordX = kvHeadIdx * HD + hhv * 64;
  uint32_t vCoordY = vvv * 16;
  uint32_t prefCoordX = (kvHeadIdx * HD >> 1) + hhpref * 32;
  uint32_t prefCoordYK = vvpref * 8;

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kCoordX, kCoordY);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadV(
    (fp16*)vState, widthInByteKV, heightKv, widthInByteKV, vCoordX, vCoordY);
  __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefK(
    (uint32_t*)kState, widthInByteKV, heightKv, widthInByteKV, prefCoordX, prefCoordYK);

  // Each thread scatters 1024 fp16 = 2048 bytes to SLM
  unsigned int slmOffsetV = slmOffsetBaseV + localLinearId * 1024 * sizeof(fp16);

  // Initial prefetch
  #pragma unroll
  for (int32_t k = 0; k < 1; k++) {
    #pragma unroll
    for (int32_t kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
      payloadPrefK.set_x(prefCoordX + 16 * kk);
    }
    prefCoordYK += KV_STEP;
    payloadPrefK.set_y(prefCoordYK);
  }

  // Load first V block: 1024 fp16 per thread via 2 load_2d calls (each 512 fp16)
  simd<fp16, 1024> fp16VState;  // 32 GRFs
  fp16VState.select<512, 1>(0) =
    __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
  payloadV.set_x(vCoordX + 32);
  fp16VState.select<512, 1>(512) =
    __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
  payloadV.set_x(vCoordX);
  vCoordY += KV_STEP;
  payloadV.set_y(vCoordY);

  // Load Q: 8 rows × 256 cols (16 blocks of 8×16 uint32_t = 8 rows × 16 HD-cols each)
  {
    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 8, 1> payloadQ(
      (uint32_t*)qState, widthInByteQ, heightQ, widthInByteQ, qCoordX, qCoordY);
    #pragma unroll
    for (int32_t kk = 0; kk < 16; kk++) {
      fp16QState.template bit_cast_view<uint32_t>().select<64, 1>(64 * kk) =
        __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 8, 1, true, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);
      qCoordX += 8;
      payloadQ.set_x(qCoordX);
    }
  }

  // Store first V to SLM: 1024 fp16 = 2048 bytes per thread
  // Each scatter writes 16 addresses × 8 u32 = 512 bytes. Need 4 scatters for 2048 bytes.
  // First 2 scatters: bytes [0..1023], second 2: bytes [1024..2047]
  {
    simd<uint32_t, 32> simdSlmOffsetsV;
    simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
    simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
    simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV;
    // First half (512 fp16 from first load_2d call)
    #pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
        simdSlmOffsetsV.select<16, 1>(16 * kk),
        fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
    }
    // Second half (512 fp16 from second load_2d call) — offset by 1024 bytes
    simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) + 32 * 16 * sizeof(fp16);
    #pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
        simdSlmOffsetsV.select<16, 1>(16 * kk),
        fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * (kk + 2)));
    }
  }

  int loopIdx;

  // ===== MAIN LOOP =====
  for (loopIdx = 0; loopIdx < kvSeqOutLoopCount - 1; loopIdx++) {
    uint32_t slmPingpongLoad = loopIdx & 0x1;
    uint32_t slmPingpongStore = (loopIdx + 1) & 0x1;
    slmPingpongLoad = slmPingpongLoad * KV_STEP * HD * sizeof(fp16);
    slmPingpongStore = slmPingpongStore * KV_STEP * HD * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 32> compensationTemp;
    // Zero only the score portion (first 512 floats)
    tempOutput.select<512, 1>(0) = 0;

    // ===== Q @ K^T =====
    // dpas<8, 8> ExecSize=8: B=Q(128), A=K(128), C=scores(64 float)
    // nn=0..15 (16 blocks × 16 = 256 HD), kk=0..7 (8 tiles × 8 = 64 scores)
    {
      #pragma unroll
      for (int32_t nn = 0; nn < 16; nn++) {
        payloadK.set_x(kCoordX + 16 * nn);
        #pragma unroll
        for (int32_t l = 0; l < 4; l++) {
          payloadK.set_y(kCoordY + 16 * l);
          tempBufferAsFp16.select<256, 1>(256 * l) =
            __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }
        #pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
          auto ccTile = tempOutput.select<64, 1>(64 * kk);
          auto aaTile = fp16QState.select<128, 1>(128 * nn);
          auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
          ccTile = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 64>(ccTile.data()),
            simd<fp16, 128>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
      kCoordY += KV_STEP;
    }

    // ===== V load (next iteration): 1024 fp16 per thread =====
    fp16VState.select<512, 1>(0) =
      __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
    payloadV.set_x(vCoordX + 32);
    fp16VState.select<512, 1>(512) =
      __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
    payloadV.set_x(vCoordX);
    vCoordY += KV_STEP;
    payloadV.set_y(vCoordY);

    // ===== Softmax over 8×64 = 512 scores (tempOutput[0..511]) =====
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<8, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<8, 1>(8);
      auto fp32Exp2Temp = tempBuffer.select<8, 1>(16);
      simd<float, 4 * 8> ttemp;
      fp32CurrentMaxTemp = fp32HistoricMaxTemp;

      // Max reduction: 8 kk-tiles × 8 cols × 8 rows = 512 floats
      // Each 8-element group = 1 column for all 8 rows
      // First tile (kk=0): 4 column-pairs → 4 partial maxes
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<8, 1>(8 * kk) = __ESIMD_NS::max<float, 8, float>(
          tempOutput.select<8, 1>(16 * kk),
          tempOutput.select<8, 1>(16 * kk + 8));
      }
      // Fold remaining 7 kk-tiles
      #pragma unroll
      for (int kkk = 1; kkk < 8; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) =
            __ESIMD_NS::max<float, 8, float>(
              ttemp.select<8, 1>(8 * kk),
              tempOutput.select<8, 1>(64 * kkk + 16 * kk));
          ttemp.select<8, 1>(8 * kk) =
            __ESIMD_NS::max<float, 8, float>(
              ttemp.select<8, 1>(8 * kk),
              tempOutput.select<8, 1>(64 * kkk + 16 * kk + 8));
        }
      }
      // Reduce 4 partials → 1
      ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
      ttemp.select<8, 1>(0) = __ESIMD_NS::max<float, 8, float>(ttemp.select<8, 1>(0), ttemp.select<8, 1>(8));
      fp32CurrentMaxTemp.merge(
        ttemp.select<8, 1>(0),
        ttemp.select<8, 1>(0) > fp32CurrentMaxTemp);

      fp32Exp2Temp.select<8, 1>(0) = fp32CurrentMaxTemp.select<8, 1>(0) * attnScoreMul;

      // Exp2 on all 512 scores — two waves per k-tile (ttemp=32 floats, 4 cols per wave)
      #pragma unroll
      for (int k = 0; k < 8; k++) {
        // Wave 1: even columns (col 0, 2, 4, 6)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) = tempOutput.select<8, 1>(64 * k + 16 * kk) * attnScoreMul - fp32Exp2Temp.select<8, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          tempOutput.select<8, 1>(64 * k + 16 * kk) = __ESIMD_NS::exp2<float, 8, float>(ttemp.select<8, 1>(8 * kk));
        }
        // Wave 2: odd columns (col 1, 3, 5, 7)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) = tempOutput.select<8, 1>(64 * k + 16 * kk + 8) * attnScoreMul - fp32Exp2Temp.select<8, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          tempOutput.select<8, 1>(64 * k + 16 * kk + 8) = __ESIMD_NS::exp2<float, 8, float>(ttemp.select<8, 1>(8 * kk));
        }
      }

      // Compensation
      fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<8, 1>(0);
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 8, float>(fp32SoftMaxCompensation);
      fp32SoftMaxTemp.select<8, 1>(0) = fp32SoftMaxTemp.select<8, 1>(0) * fp32SoftMaxCompensation.select<8, 1>(0);

      // EARLY fp16 convert — 4 copies of 8-row compensation for 32-element multiply
      compensationTemp.select<8, 1>(0) = fp32SoftMaxCompensation;
      compensationTemp.select<8, 1>(8) = fp32SoftMaxCompensation;
      compensationTemp.select<8, 1>(16) = fp32SoftMaxCompensation;
      compensationTemp.select<8, 1>(24) = fp32SoftMaxCompensation;

      // Sum reduction (same structure as max)
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<8, 1>(8 * kk) = tempOutput.select<8, 1>(16 * kk) + tempOutput.select<8, 1>(16 * kk + 8);
      }
      #pragma unroll
      for (int kkk = 1; kkk < 8; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) = ttemp.select<8, 1>(8 * kk) + tempOutput.select<8, 1>(64 * kkk + 16 * kk);
          ttemp.select<8, 1>(8 * kk) = ttemp.select<8, 1>(8 * kk) + tempOutput.select<8, 1>(64 * kkk + 16 * kk + 8);
        }
      }
      ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
      ttemp.select<8, 1>(0) = ttemp.select<8, 1>(0) + ttemp.select<8, 1>(8);
      fp32SoftMaxTemp.select<8, 1>(0) = fp32SoftMaxTemp.select<8, 1>(0) + ttemp.select<8, 1>(0);
      fp32HistoricMaxTemp = fp32CurrentMaxTemp;

      // Pack softmax weights fp32 → fp16 VNNI
      // Block 1 (score cols 0-31) → tempBufferAsFp16[0..255], A for SxV block 1
      // Block 2 (score cols 32-63) → tempQkAsFp16[0..255], A for SxV block 2
      // k=0..3: 4 groups of 64 fp16 = 256 total
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<16, 2>(64 * k + 32 * kk) = tempOutput.select<16, 1>(64 * k + 32 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<16, 2>(64 * k + 32 * kk + 1) = tempOutput.select<16, 1>(64 * k + 32 * kk + 16);
        }
      }
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<16, 2>(64 * k + 32 * kk) = tempOutput.select<16, 1>(64 * k + 256 + 32 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<16, 2>(64 * k + 32 * kk + 1) = tempOutput.select<16, 1>(64 * k + 256 + 32 * kk + 16);
        }
      }
    }

    barrier();  // V SLM ready, compensation ALU can overlap with SLM loads

    // ===== Interleaved compensation + S×V: 4 l-groups of 64 output-cols =====
    // dpas<8, 8>: C=64 float, B=128 fp16 (scores), A=128 fp16 (V data)
    // ExecSize=8 requires float accumulator — fp16↔float conversion per DPAS
    // B: packed scores from tempBufferAsFp16/tempQkAsFp16 (128 per nn)
    // A: V from SLM via tempQkAsFp16[1024..2047] (128 per ll, 8 ll per load)
    {
      // --- l=0: finalOutput[0..511] (output cols 0-63) ---
      #pragma unroll
      for (int kk = 0; kk < 16; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }
      // Block 1 (V rows 0-31, score positions 0-31), l=0
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      // Block 2 (V rows 32-63, score positions 32-63), l=0
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // --- l=1: finalOutput[512..1023] (output cols 64-127) ---
      #pragma unroll
      for (int kk = 16; kk < 32; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }
      // Block 1, l=1
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(512 + 64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      // Block 2, l=1
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(512 + 64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // --- l=2: finalOutput[1024..1535] (output cols 128-191) ---
      #pragma unroll
      for (int kk = 32; kk < 48; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }
      // Block 1, l=2
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              16 * 128 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1024 + 64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      // Block 2, l=2
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              16 * 128 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1024 + 64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // --- l=3: finalOutput[1536..2047] (output cols 192-255) ---
      #pragma unroll
      for (int kk = 48; kk < 64; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }
      // Block 1, l=3
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              16 * 192 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1536 + 64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      // Block 2, l=3
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              16 * 192 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1536 + 64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // SLM scatter for next V block: 1024 fp16 = 2048 bytes
      simd<uint32_t, 32> simdSlmOffsetsV;
      simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
      simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
      simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV + slmPingpongStore;
      // First half
      #pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
          simdSlmOffsetsV.select<16, 1>(16 * kk),
          fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
      }
      // Second half (+1024 bytes)
      simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) + 32 * 16 * sizeof(fp16);
      #pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
          simdSlmOffsetsV.select<16, 1>(16 * kk),
          fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * (kk + 2)));
      }
    }

    // Prefetch
    #pragma unroll
    for (int32_t kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
      payloadPrefK.set_x(prefCoordX + 16 * kk);
    }
    prefCoordYK += KV_STEP;
    payloadPrefK.set_y(prefCoordYK);
  }

  // ===== LAST LOOP — boundary checking =====
  {
    uint32_t slmPingpongLoad = (loopIdx) & 0x1;
    slmPingpongLoad = slmPingpongLoad * KV_STEP * HD * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 32> compensationTemp;
    tempOutput.select<512, 1>(0) = 0;

    // Q @ K^T
    {
      #pragma unroll
      for (int32_t nn = 0; nn < 16; nn++) {
        payloadK.set_x(kCoordX + 16 * nn);
        #pragma unroll
        for (int32_t l = 0; l < 4; l++) {
          payloadK.set_y(kCoordY + 16 * l);
          tempBufferAsFp16.select<256, 1>(256 * l) =
            __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }
        #pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
          auto ccTile = tempOutput.select<64, 1>(64 * kk);
          auto aaTile = fp16QState.select<128, 1>(128 * nn);
          auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
          ccTile = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 64>(ccTile.data()),
            simd<fp16, 128>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
    }

    // Apply boundary mask, then softmax
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<8, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<8, 1>(8);
      auto fp32Exp2Temp = tempBuffer.select<8, 1>(16);
      auto softmaxPositions = ui32Temp.select<16, 1>(24);
      simd<float, 4 * 8> ttemp;

      // Boundary masking: 8 kk-tiles × 64 floats each
      // Score layout: each 8-element group = 1 column for 8 rows
      // Position counter tracks KV positions, advances by 1 per column
      softmaxPositions.select<8, 1>(0) = baseOffsetInc16AsVector.select<8, 1>(0) + loopIdx * KV_STEP;
      #pragma unroll
      for (int k = 0; k < 8; k++) {  // 8 kk-tiles
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {  // 8 cols per tile
          tempOutput.select<8, 1>(64 * k + 8 * kk).merge(FP32_MIN, softmaxPositions.select<8, 0>(kk) >= kvSeqLen);
        }
        softmaxPositions.select<8, 1>(0) = softmaxPositions.select<8, 1>(0) + 8;
      }

      fp32CurrentMaxTemp = fp32HistoricMaxTemp;
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<8, 1>(8 * kk) = __ESIMD_NS::max<float, 8, float>(
          tempOutput.select<8, 1>(16 * kk),
          tempOutput.select<8, 1>(16 * kk + 8));
      }
      #pragma unroll
      for (int kkk = 1; kkk < 8; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) =
            __ESIMD_NS::max<float, 8, float>(
              ttemp.select<8, 1>(8 * kk),
              tempOutput.select<8, 1>(64 * kkk + 16 * kk));
          ttemp.select<8, 1>(8 * kk) =
            __ESIMD_NS::max<float, 8, float>(
              ttemp.select<8, 1>(8 * kk),
              tempOutput.select<8, 1>(64 * kkk + 16 * kk + 8));
        }
      }
      ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
      ttemp.select<8, 1>(0) = __ESIMD_NS::max<float, 8, float>(ttemp.select<8, 1>(0), ttemp.select<8, 1>(8));
      fp32CurrentMaxTemp.merge(
        ttemp.select<8, 1>(0),
        ttemp.select<8, 1>(0) > fp32CurrentMaxTemp);

      fp32Exp2Temp.select<8, 1>(0) = fp32CurrentMaxTemp.select<8, 1>(0) * attnScoreMul;

      // Exp2 — two waves per k-tile (ttemp=32 floats, 4 cols per wave)
      #pragma unroll
      for (int k = 0; k < 8; k++) {
        // Wave 1: even columns (col 0, 2, 4, 6)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) = tempOutput.select<8, 1>(64 * k + 16 * kk) * attnScoreMul - fp32Exp2Temp.select<8, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          tempOutput.select<8, 1>(64 * k + 16 * kk) = __ESIMD_NS::exp2<float, 8, float>(ttemp.select<8, 1>(8 * kk));
        }
        // Wave 2: odd columns (col 1, 3, 5, 7)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) = tempOutput.select<8, 1>(64 * k + 16 * kk + 8) * attnScoreMul - fp32Exp2Temp.select<8, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          tempOutput.select<8, 1>(64 * k + 16 * kk + 8) = __ESIMD_NS::exp2<float, 8, float>(ttemp.select<8, 1>(8 * kk));
        }
      }

      fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<8, 1>(0);
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 8, float>(fp32SoftMaxCompensation);

      if (loopIdx != 0) {
        fp32SoftMaxTemp.select<8, 1>(0) = fp32SoftMaxTemp.select<8, 1>(0) * fp32SoftMaxCompensation.select<8, 1>(0);
      }

      // EARLY fp16 convert
      compensationTemp.select<8, 1>(0) = fp32SoftMaxCompensation;
      compensationTemp.select<8, 1>(8) = fp32SoftMaxCompensation;
      compensationTemp.select<8, 1>(16) = fp32SoftMaxCompensation;
      compensationTemp.select<8, 1>(24) = fp32SoftMaxCompensation;

      // Sum reduction
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<8, 1>(8 * kk) = tempOutput.select<8, 1>(16 * kk) + tempOutput.select<8, 1>(16 * kk + 8);
      }
      #pragma unroll
      for (int kkk = 1; kkk < 8; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<8, 1>(8 * kk) = ttemp.select<8, 1>(8 * kk) + tempOutput.select<8, 1>(64 * kkk + 16 * kk);
          ttemp.select<8, 1>(8 * kk) = ttemp.select<8, 1>(8 * kk) + tempOutput.select<8, 1>(64 * kkk + 16 * kk + 8);
        }
      }
      ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
      ttemp.select<8, 1>(0) = ttemp.select<8, 1>(0) + ttemp.select<8, 1>(8);
      fp32SoftMaxTemp.select<8, 1>(0) = fp32SoftMaxTemp.select<8, 1>(0) + ttemp.select<8, 1>(0);
      fp32HistoricMaxTemp = fp32CurrentMaxTemp;

      // Pack softmax weights (same as main loop)
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<16, 2>(64 * k + 32 * kk) = tempOutput.select<16, 1>(64 * k + 32 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<16, 2>(64 * k + 32 * kk + 1) = tempOutput.select<16, 1>(64 * k + 32 * kk + 16);
        }
      }
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<16, 2>(64 * k + 32 * kk) = tempOutput.select<16, 1>(64 * k + 256 + 32 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<16, 2>(64 * k + 32 * kk + 1) = tempOutput.select<16, 1>(64 * k + 256 + 32 * kk + 16);
        }
      }
    }

    barrier();

    // S×V with interleaved compensation (last iteration) — 4 l-groups
    {
      // l=0
      if (loopIdx != 0) {
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }
      // l=0 Block 1
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      // l=0 Block 2
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // l=1
      if (loopIdx != 0) {
        #pragma unroll
        for (int kk = 16; kk < 32; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(512 + 64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(512 + 64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // l=2
      if (loopIdx != 0) {
        #pragma unroll
        for (int kk = 32; kk < 48; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              16 * 128 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1024 + 64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              16 * 128 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1024 + 64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }

      // l=3
      if (loopIdx != 0) {
        #pragma unroll
        for (int kk = 48; kk < 64; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * nn * sizeof(fp16) +
              16 * 192 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1536 + 64 * ll);
          auto aaTile = tempBufferAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * HD * 2 * sizeof(fp16) +
              16 * HD * nn * sizeof(fp16) +
              16 * 192 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<64, 1>(1536 + 64 * ll);
          auto aaTile = tempQkAsFp16.select<128, 1>(128 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          {
            simd<float, 64> float_cc = ccTile;
            float_cc = dpas<8, 8, float, float, fp16, fp16>(
              float_cc,
              simd<fp16, 128>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
            ccTile = float_cc;
          }
        }
      }
    }
  }

#ifdef DIAG_DUMP
  // Dump 3: finalOutput after SxV (before normalization) — minimal version
  if (hhq == 0 && h == 0 && v == 0) {
    float* _dbg = (float*)out;
    // [160..223]: finalOutput[0..63] as float (ll=0: dims 0-7)
    {
      simd<float, 32> _d = finalOutput.select<32, 1>(0);
      block_store<float, 32>(_dbg + 160, _d);
      _d = finalOutput.select<32, 1>(32);
      block_store<float, 32>(_dbg + 192, _d);
    }
    // [224..287]: finalOutput[192..255] as float (ll=3: dims 24-31)
    {
      simd<float, 32> _d = finalOutput.select<32, 1>(192);
      block_store<float, 32>(_dbg + 224, _d);
      _d = finalOutput.select<32, 1>(224);
      block_store<float, 32>(_dbg + 256, _d);
    }
  }
  return;
#endif

  // ===== Output normalization =====
  // DPAS output layout for dpas<8, 4>: 64 elements = 8 columns × 8 rows (column-of-rows)
  // Each 8-element group = 1 column for all 8 rows
  // finalOutput: 2048 fp16 = 64 chunks of 32 = 64 groups of (4 cols × 8 rows)
  // For scatter: need row-major with 8 rows as scatter lanes

  simd<float, Q_ROWS> softMaxDividor;
  simd<float, HD> alphaV;
  alphaV = block_load<float, HD>((float*)normAlpha + headIdx * HD);
  softMaxDividor.select<Q_ROWS, 1>(0) = fp32SoftMaxTemp;
  softMaxDividor = 1.0f / softMaxDividor;

  // Normalize: 64 chunks of 32 fp16, each = 4 output cols × 8 rows
  // Pack into fp16QState for scatter output
  #pragma unroll
  for (int kk = 0; kk < 64; kk++) {
    simd<float, 32> alphaMul;
    simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
    int col_base = 4 * kk;  // 4 output columns per chunk
    // Each 8-element sub-group in f16Temp is one column for all 8 rows
    // alphaMul: broadcast alpha[col] × softMaxDividor[row] for each sub-group
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      #pragma unroll
      for (int r = 0; r < Q_ROWS; r++) {
        alphaMul[8 * c + r] = alphaV[col_base + c] * softMaxDividor[r];
      }
    }
    f16Temp = f16Temp * alphaMul;
    // Deinterleave to row-pair format for scatter:
    // fp16QState layout per 32-element region:
    //   u32[0..7]: 8 rows' (col_pair_0) = fp16 pairs (c0, c1) per row
    //   u32[8..15]: 8 rows' (col_pair_1) = fp16 pairs (c2, c3) per row
    fp16QState.select<8, 2>(32 * kk) = f16Temp.select<8, 1>(0);       // col 0 rows
    fp16QState.select<8, 2>(32 * kk + 1) = f16Temp.select<8, 1>(8);   // col 1 rows
    fp16QState.select<8, 2>(32 * kk + 16) = f16Temp.select<8, 1>(16); // col 2 rows
    fp16QState.select<8, 2>(32 * kk + 17) = f16Temp.select<8, 1>(24); // col 3 rows
  }

  // ===== Output scatter: 8 rows × 256 cols =====
  // scatter<u32, 4, u32, wb, wb, 8>: 8 lanes × 4 u32 per lane
  // SoA data: data[r + 8*j] = u32 for row r, column-pair j
  // 32 scatter iterations × 8 cols = 256 output columns
  simd<uint32_t, Q_ROWS> simdOffsets;
  simd_mask<Q_ROWS> mask;
  simdOffsets = baseOffsetInc16AsVector.select<Q_ROWS, 1>(0);
  simdOffsets = simdOffsets + 128 * h + Q_ROWS * hhq;
  mask = simdOffsets < activationLength;
  simdOffsets = simdOffsets * headQ * HD * sizeof(fp16) + headIdx * HD * sizeof(fp16);
  #pragma unroll
  for (int kk = 0; kk < 32; kk++) {
    __ESIMD_ENS::lsc_scatter<uint32_t, 4, __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back, Q_ROWS, uint32_t>(
      (uint32_t*)out, simdOffsets,
      fp16QState.template bit_cast_view<uint32_t>().select<Q_ROWS * 4, 1>(Q_ROWS * 4 * kk), mask);
    simdOffsets += 4 * sizeof(uint32_t);
  }
}
