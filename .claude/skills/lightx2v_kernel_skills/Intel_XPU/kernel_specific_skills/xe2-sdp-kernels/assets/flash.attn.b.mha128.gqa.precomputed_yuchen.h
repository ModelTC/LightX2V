// New approach: Pre-compute kvSeqOutLoopCount per workgroup, remove all conditionals except mask application
// Skip blocks still execute compute but mask everything to -inf
// This ensures same barrier count for all threads

template<bool IS_CAUSAL>
ESIMD_INLINE void flashAttnBMha128GQAPrecomputed(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* normAlpha,
  uint8_t* out,
  uint32_t activationLength,
  uint32_t kvSeqLen,
  uint32_t history_len,
  uint32_t headQ,
  uint32_t headKv,
  sycl::nd_item<2>& ndi) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f;
  constexpr float attnScoreMul = matMulQuantCoeff * sycl::ext::intel::esimd::detail::log2e;
  constexpr uint32_t slmSizeV = 2 * 64 * 128 * sizeof(fp16);
  constexpr uint32_t slmSize = slmSizeV;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  constexpr uint32_t slmOffsetBaseV = 0;

  int32_t localLinearId = ndi.get_local_id(1);  // Fixed: use dimension 1
  int32_t hhq = localLinearId & 0xf;
  int32_t vvq = localLinearId >> 4;
  int32_t hhv = localLinearId & 0x3;
  int32_t vvv = localLinearId >> 2;
  int32_t hhpref = localLinearId & 0x1;
  int32_t vvpref = localLinearId >> 1;
  // Fixed: swap h and v - grid is ({groupH, groupV*16}, {1, 16})
  // groupH workgroups in dim 0, groupV workgroups in dim 1
  int32_t h = ndi.get_group(0);  // Query workgroup index (0 to groupH-1)
  int32_t v = ndi.get_group(1);  // KV head index (0 to groupV-1)

  int32_t groupSize = headQ / headKv;
  int32_t headIdx = v * groupSize;
  int32_t kvHeadIdx = v;

  simd<fp16, 16 * 128> fp16QState;
  simd<float, 16 * 32> tempBuffer;
  simd<float, 16 * 64> tempOutput;
  auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
  auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  simd<fp16, 16 * 128> finalOutput = 0;
  simd<float, 16> fp32SoftMaxTemp = 0;
  simd<float, 16> fp32HistoricMaxTemp = FP32_MIN;
  simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);

  // Pre-compute causal boundaries for this thread's query position
  // Each thread handles 1 query position across 16 heads
  simd<int32_t, 16> causal_boundaries;
  int32_t this_q_pos = h * 16 + hhq;

  if constexpr (IS_CAUSAL) {
    if (this_q_pos < (int32_t)activationLength) {
      // All 16 heads at this query position have the same causal boundary
      int32_t boundary = history_len + this_q_pos;
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        causal_boundaries[i] = boundary;
      }
    } else {
      // Invalid Q position - mark all as invalid
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        causal_boundaries[i] = -1;
      }
    }
  }

  // Pre-compute kvSeqOutLoopCount for this workgroup based on causal masking
  // Each workgroup processes 16 query positions (h*16 to h*16+15)
  int32_t kvSeqOutLoopCount;

  if constexpr (IS_CAUSAL) {
    // Find max Q position in this workgroup
    int32_t max_q_in_workgroup = h * 16 + 15;
    max_q_in_workgroup = (max_q_in_workgroup < (int32_t)activationLength) ? max_q_in_workgroup : ((int32_t)activationLength - 1);

    // Max valid KV position this workgroup can attend to
    int32_t max_kv_pos_in_workgroup = history_len + max_q_in_workgroup;

    // How many 64-token blocks do we need?
    kvSeqOutLoopCount = (max_kv_pos_in_workgroup + 64) / 64;

    // Clamp to actual KV length
    int32_t total_kv_blocks = (kvSeqLen + 63) / 64;
    kvSeqOutLoopCount = (kvSeqOutLoopCount < total_kv_blocks) ? kvSeqOutLoopCount : total_kv_blocks;
  } else {
    kvSeqOutLoopCount = (kvSeqLen + 63) / 64;
  }

  uint32_t widthInByteQ = 128 * sizeof(fp16) - 1;
  uint32_t widthInByteKV = headKv * 128 * sizeof(fp16) - 1;
  uint32_t heightQ = headQ * activationLength - 1;
  uint32_t heightKv = kvSeqLen - 1;

  uint32_t qCoordX = 0;
  uint32_t qCoordY = h * 16 * headQ + hhq * headQ + headIdx;
  uint32_t kCoordX = kvHeadIdx * 128;
  uint32_t kCoordY = 0;
  uint32_t vCoordX = kvHeadIdx * 128 + hhv * 32;
  uint32_t vCoordY = vvv * 16;
  uint32_t prefCoordX = (kvHeadIdx * 128 >> 1) + hhpref * 32;
  uint32_t prefCoordYK = vvpref * 8;

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kState, widthInByteKV, heightKv, widthInByteKV, kCoordX, kCoordY);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadV(
    (fp16*)vState, widthInByteKV, heightKv, widthInByteKV, vCoordX, vCoordY);
  __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefK(
    (uint32_t*)kState, widthInByteKV, heightKv, widthInByteKV, prefCoordX, prefCoordYK);

  unsigned int slmOffsetV = slmOffsetBaseV + localLinearId * 512 * sizeof(fp16);

  // Initial prefetch
  #pragma unroll
  for (int32_t k = 0; k < 1; k++) {
    #pragma unroll
    for (int32_t kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
      payloadPrefK.set_x(prefCoordX + 16 * kk);
    }
    prefCoordYK += 64;
    payloadPrefK.set_y(prefCoordYK);
  }

  // Load first V block
  tempBufferAsFp16.select<512, 1>(0) =
    __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
  vCoordY += 64;
  payloadV.set_y(vCoordY);

  // Load Q
  {
    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
      (uint32_t*)qState, widthInByteQ, heightQ, widthInByteQ, qCoordX, qCoordY);
    #pragma unroll
    for (int32_t kk = 0; kk < 8; kk++) {
      fp16QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk) =
        __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);
      qCoordX += 8;
      payloadQ.set_x(qCoordX);
    }
  }

  // Store first V to SLM
  {
    simd<uint32_t, 32> simdSlmOffsetsV;
    simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
    simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
    simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV;
    #pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
        simdSlmOffsetsV.select<16, 1>(16 * kk),
        tempBufferAsFp16.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
    }
  }

  int loopIdx;

  // ===== MAIN LOOP - All conditionals removed except mask application =====
  for (loopIdx = 0; loopIdx < kvSeqOutLoopCount - 1; loopIdx++) {
    uint32_t slmPingpongLoad = loopIdx & 0x1;
    uint32_t slmPingpongStore = (loopIdx + 1) & 0x1;
    slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
    slmPingpongStore = slmPingpongStore * 64 * 128 * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 512> fp16VState;
    tempOutput = 0;

    int32_t kv_block_start = loopIdx * 64;

    // ===== Q @ K^T - ALWAYS EXECUTE =====
    {
      #pragma unroll
      for (int32_t nn = 0; nn < 8; nn++) {
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
          auto ccTile = tempOutput.select<128, 1>(128 * kk);
          auto aaTile = fp16QState.select<256, 1>(256 * nn);
          auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
          ccTile = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
      kCoordY += 64;
    }

    // Apply causal mask AFTER compute using vectorized SIMD operations
    if constexpr (IS_CAUSAL) {
      #pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        #pragma unroll
        for (int m = 0; m < 8; m++) {
          int32_t kv_pos = kv_block_start + kk * 8 + m;
          int base_idx = kk * 128 + m * 16;

          // Vectorized masking: broadcast kv_pos, vector compare, vector merge
          simd<int, 16> v_kv_pos(kv_pos);
          auto mask = v_kv_pos > causal_boundaries;
          tempOutput.select<16, 1>(base_idx).merge(FP32_MIN, mask);
        }
      }
    }

    // ===== V load - ALWAYS EXECUTE =====
    fp16VState =
      __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
    vCoordY += 64;
    payloadV.set_y(vCoordY);

    // ===== Softmax - ALWAYS EXECUTE =====
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
      simd<float, 8 * 16> ttemp;
      fp32CurrentMaxTemp = fp32HistoricMaxTemp;

      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<32, 1>(32 * kk) = __ESIMD_NS::max<float, 32, float>(
          tempOutput.select<32, 1>(64 * kk),
          tempOutput.select<32, 1>(64 * kk + 32));
      }
      #pragma unroll
      for (int kkk = 0; kkk < 6; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<32, 1>(32 * kk) =
            __ESIMD_NS::max<float, 32, float>(
              ttemp.select<32, 1>(32 * kk),
              tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16));
        }
      }
      ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(64));
      ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(32));
      ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
      fp32CurrentMaxTemp.merge(
        ttemp.select<16, 1>(0),
        ttemp.select<16, 1>(0) > fp32CurrentMaxTemp);

      fp32Exp2Temp.select<16, 1>(0) = fp32CurrentMaxTemp.select<16, 1>(0) * attnScoreMul;

      #pragma unroll
      for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(128 * k + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
          ttemp.select<16, 1>(16 * kk + 32) = tempOutput.select<16, 1>(128 * k + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          ttemp.select<16, 1>(16 * kk + 64) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
          ttemp.select<16, 1>(16 * kk + 64 + 32) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          tempOutput.select<16, 1>(128 * k + 16 * kk) = __ESIMD_NS::exp2<float, 16, float>(ttemp.select<16, 1>(16 * kk));
        }
      }

      fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);
      fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);

      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<32, 1>(32 * kk) = tempOutput.select<32, 1>(64 * kk) + tempOutput.select<32, 1>(64 * kk + 32);
      }
      #pragma unroll
      for (int kkk = 0; kkk < 6; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<32, 1>(32 * kk) = ttemp.select<32, 1>(32 * kk) + tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16);
        }
      }
      ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(64);
      ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(32);
      ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
      fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);
      fp32HistoricMaxTemp = fp32CurrentMaxTemp;

      simd<fp16, 32> compensationTemp;
      compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
      compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;
      #pragma unroll
      for (int kk = 0; kk < 64; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }

      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 64 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 64 * kk + 32);
        }
      }
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk + 32);
        }
      }
    }

    barrier();

    // ===== Attention @ V - ALWAYS EXECUTE =====
    {
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int l = 0; l < 2; l++) {
          #pragma unroll
          for (int ll = 0; ll < 2; ll++) {
            tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
              slm_block_load<fp16, 512>(slmOffsetBaseV +
                slmPingpongLoad +
                16 * 128 * nn * sizeof(fp16) +
                16 * 64 * l * sizeof(fp16) +
                512 * ll * sizeof(fp16)
              );
          }
          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
            auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
            auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
            ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
              simd<fp16, 128>(ccTile.data()),
              simd<fp16, 256>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
          }
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int l = 0; l < 2; l++) {
          #pragma unroll
          for (int ll = 0; ll < 2; ll++) {
            tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
              slm_block_load<fp16, 512>(slmOffsetBaseV +
                slmPingpongLoad +
                16 * 128 * 2 * sizeof(fp16) +
                16 * 128 * nn * sizeof(fp16) +
                16 * 64 * l * sizeof(fp16) +
                512 * ll * sizeof(fp16)
              );
          }
          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
            auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
            auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
            ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
              simd<fp16, 128>(ccTile.data()),
              simd<fp16, 256>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
          }
        }
      }

      // ===== SLM scatter - ALWAYS EXECUTE =====
      simd<uint32_t, 32> simdSlmOffsetsV;
      simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
      simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
      simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV + slmPingpongStore;
      #pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
          simdSlmOffsetsV.select<16, 1>(16 * kk),
          fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
      }
    }

    // Prefetch
    #pragma unroll
    for (int32_t kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
      payloadPrefK.set_x(prefCoordX + 16 * kk);
    }
    prefCoordYK += 64;
    payloadPrefK.set_y(prefCoordYK);
  }

  // ===== LAST LOOP - boundary checking for kvSeqLen =====
  {
    uint32_t slmPingpongLoad = (loopIdx) & 0x1;
    slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    tempOutput = 0;

    int32_t kv_block_start = loopIdx * 64;

    // Q @ K^T
    {
      #pragma unroll
      for (int32_t nn = 0; nn < 8; nn++) {
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
          auto ccTile = tempOutput.select<128, 1>(128 * kk);
          auto aaTile = fp16QState.select<256, 1>(256 * nn);
          auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
          ccTile = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
    }

    // Apply both causal mask and boundary mask
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
      auto softmaxPositions = ui32Temp.select<16, 1>(48);
      simd<float, 8 * 16> ttemp;

      // Apply boundary mask (kvSeqLen limit)
      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
        }
        softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
      }

      // Apply causal mask
      if constexpr (IS_CAUSAL) {
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          #pragma unroll
          for (int m = 0; m < 8; m++) {
            int32_t kv_pos = kv_block_start + kk * 8 + m;
            int base_idx = kk * 128 + m * 16;
            #pragma unroll
            for (int q_idx = 0; q_idx < 16; q_idx++) {
              tempOutput[base_idx + q_idx].merge(FP32_MIN, kv_pos > causal_boundaries[q_idx]);
            }
          }
        }
      }

      fp32CurrentMaxTemp = fp32HistoricMaxTemp;
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<32, 1>(32 * kk) = __ESIMD_NS::max<float, 32, float>(
          tempOutput.select<32, 1>(64 * kk),
          tempOutput.select<32, 1>(64 * kk + 32));
      }
      #pragma unroll
      for (int kkk = 0; kkk < 6; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<32, 1>(32 * kk) =
            __ESIMD_NS::max<float, 32, float>(
              ttemp.select<32, 1>(32 * kk),
              tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16));
        }
      }
      ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(64));
      ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(32));
      ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
      fp32CurrentMaxTemp.merge(
        ttemp.select<16, 1>(0),
        ttemp.select<16, 1>(0) > fp32CurrentMaxTemp);

      fp32Exp2Temp.select<16, 1>(0) = fp32CurrentMaxTemp.select<16, 1>(0) * attnScoreMul;

      #pragma unroll
      for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(128 * k + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
          ttemp.select<16, 1>(16 * kk + 32) = tempOutput.select<16, 1>(128 * k + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          ttemp.select<16, 1>(16 * kk + 64) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
          ttemp.select<16, 1>(16 * kk + 64 + 32) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          tempOutput.select<16, 1>(128 * k + 16 * kk) = __ESIMD_NS::exp2<float, 16, float>(ttemp.select<16, 1>(16 * kk));
        }
      }

      fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);

      if (loopIdx != 0) {
        fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
      }

      #pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        ttemp.select<32, 1>(32 * kk) = tempOutput.select<32, 1>(64 * kk) + tempOutput.select<32, 1>(64 * kk + 32);
      }
      #pragma unroll
      for (int kkk = 0; kkk < 6; ++kkk) {
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          ttemp.select<32, 1>(32 * kk) = ttemp.select<32, 1>(32 * kk) + tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16);
        }
      }
      ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(64);
      ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(32);
      ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
      fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);
      fp32HistoricMaxTemp = fp32CurrentMaxTemp;

      if (loopIdx != 0) {
        simd<fp16, 32> compensationTemp;
        compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
        compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;
        #pragma unroll
        for (int kk = 0; kk < 64; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }

      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 64 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 64 * kk + 32);
        }
      }
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          tempQkAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk + 32);
        }
      }
    }

    barrier();

    {
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int l = 0; l < 2; l++) {
          #pragma unroll
          for (int ll = 0; ll < 2; ll++) {
            tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
              slm_block_load<fp16, 512>(slmOffsetBaseV +
                slmPingpongLoad +
                16 * 128 * nn * sizeof(fp16) +
                16 * 64 * l * sizeof(fp16) +
                512 * ll * sizeof(fp16)
              );
          }
          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
            auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
            auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
            ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
              simd<fp16, 128>(ccTile.data()),
              simd<fp16, 256>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
          }
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int l = 0; l < 2; l++) {
          #pragma unroll
          for (int ll = 0; ll < 2; ll++) {
            tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
              slm_block_load<fp16, 512>(slmOffsetBaseV +
                slmPingpongLoad +
                16 * 128 * 2 * sizeof(fp16) +
                16 * 128 * nn * sizeof(fp16) +
                16 * 64 * l * sizeof(fp16) +
                512 * ll * sizeof(fp16)
              );
          }
          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
            auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
            auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
            ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
              simd<fp16, 128>(ccTile.data()),
              simd<fp16, 256>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
          }
        }
      }
    }
  }

  // Output normalization
  // Yuchen version: each thread handles 1 query × 16 heads
  // finalOutput layout: [head0[0:128], head1[0:128], ..., head15[0:128]]
  simd<float, 16> softMaxDividor;
  softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
  softMaxDividor = 1.0f / softMaxDividor;

  // Normalize each head separately
  #pragma unroll
  for (int head_offset = 0; head_offset < 16; head_offset++) {
    // Load alpha for this specific head
    simd<float, 128> alphaV;
    alphaV = block_load<float, 128>((float*)normAlpha + (headIdx + head_offset) * 128);
    float dividor_scalar = softMaxDividor[head_offset];

    // Process this head's 128 dimensions: 4 chunks of 32 elements
    #pragma unroll
    for (int chunk = 0; chunk < 4; chunk++) {
      int base_idx = head_offset * 128 + chunk * 32;
      simd<fp16, 32> vals_fp16 = finalOutput.select<32, 1>(base_idx);
      simd<float, 32> vals_f32 = vals_fp16;
      simd<float, 32> alpha_chunk = alphaV.select<32, 1>(chunk * 32);
      vals_f32 = vals_f32 * alpha_chunk * dividor_scalar;

      // Convert back to fp16 and store with stride-2 interleaving for scatter
      simd<fp16, 32> result = vals_f32;
      int kk = (head_offset * 128 + chunk * 32) / 32;  // Which 32-element block
      fp16QState.select<16, 2>(32 * kk) = result.select<16, 1>(0);
      fp16QState.select<16, 2>(32 * kk + 1) = result.select<16, 1>(16);
    }
  }

  simd<uint32_t, 16> simdOffsets;
  simd_mask<16> mask;

  // Calculate scatter offsets for 16 heads at this query position
  // Output layout: [qlen, headQ, 128]
  // offset[i] = qpos * headQ * 128 + (headIdx + i) * 128 (in fp16 elements)
  bool valid_q = (this_q_pos < (int32_t)activationLength);
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    simdOffsets[i] = (this_q_pos * headQ + headIdx + i) * 128 * sizeof(fp16);
    mask[i] = valid_q;
  }
  #pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    __ESIMD_ENS::lsc_scatter<uint32_t, 4, __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back, 16, uint32_t>(
      (uint32_t*)out, simdOffsets, fp16QState.template bit_cast_view<uint32_t>().select<64, 1>(64 * kk), mask);
    simdOffsets += 4 * sizeof(uint32_t);
  }
}

// Functor wrapper for kernel launch (enables multiple template instantiations in same file)
template<bool IS_CAUSAL>
struct FlashAttnFunctor {
    uint8_t* Q;
    uint8_t* K;
    uint8_t* V;
    uint8_t* normAlpha;
    uint8_t* O;
    int activationLength;
    int kvSeqLen;
    int history_len;
    int headQ;
    int headKV;

    void operator()(sycl::nd_item<2> ndi) const SYCL_ESIMD_KERNEL {
        flashAttnBMha128GQAPrecomputed<IS_CAUSAL>(
            Q, K, V, normAlpha, O,
            activationLength, kvSeqLen, history_len, headQ, headKV, ndi);
    }
};

// Host-side wrapper function to launch flash attention kernel
template<bool IS_CAUSAL>
inline void launch_flash_attention(
    sycl::queue& q,
    sycl::half* d_Q,
    sycl::half* d_K,
    sycl::half* d_V,
    float* d_normAlpha,
    sycl::half* d_O,
    int q_len,
    int kv_len,
    int num_heads) {

    int groupH = num_heads;
    int groupV = (q_len + 255) / 256;
    int localH = 16, localV = 1;
    sycl::nd_range<2> nd_range({(size_t)(localH * groupH), (size_t)(groupV * localV)},
                                {(size_t)localH, (size_t)localV});

    FlashAttnFunctor<IS_CAUSAL> functor{
        reinterpret_cast<uint8_t*>(d_Q),
        reinterpret_cast<uint8_t*>(d_K),
        reinterpret_cast<uint8_t*>(d_V),
        reinterpret_cast<uint8_t*>(d_normAlpha),
        reinterpret_cast<uint8_t*>(d_O),
        q_len, kv_len, num_heads, num_heads
    };

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, functor);
    }).wait();
}
