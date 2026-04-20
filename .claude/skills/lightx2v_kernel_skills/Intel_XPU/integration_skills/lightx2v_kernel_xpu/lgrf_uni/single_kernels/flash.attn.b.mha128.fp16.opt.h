// Optimized FP16 Flash Attention kernel — barrier+interleave optimization
// Based on flash.attn.b.mha128.gqa.precomputed.h with 3 optimizations from bf16 kernel:
//   1. Barrier moved before compensation — SLM loads can issue while compensation ALU runs
//   2. Early fp16 convert — compensationTemp ready sooner, sum reduction ALU overlaps
//   3. Interleaved by l-group — l=1 compensation (32 fp16 muls) overlaps with l=0 SxV DPAS
// Non-causal only.

ESIMD_INLINE void flashAttnBMha128Fp16OptPrecomputed(
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
  constexpr float matMulQuantCoeff = 0.08838834764831844f;
  constexpr float attnScoreMul = matMulQuantCoeff * sycl::ext::intel::esimd::detail::log2e;
  constexpr uint32_t slmSizeV = 2 * 64 * 128 * sizeof(fp16);
  constexpr uint32_t slmSize = slmSizeV;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  constexpr uint32_t slmOffsetBaseV = 0;

  int32_t localLinearId = ndi.get_local_id(0);
  int32_t hhq = localLinearId & 0xf;
  int32_t vvq = localLinearId >> 4;
  int32_t hhv = localLinearId & 0x3;
  int32_t vvv = localLinearId >> 2;
  int32_t hhpref = localLinearId & 0x1;
  int32_t vvpref = localLinearId >> 1;
  int32_t h = ndi.get_group(1);
  int32_t v = ndi.get_group(0);

  int32_t headIdx = v;
  int32_t groupSize = headQ / headKv;
  int32_t kvHeadIdx = headIdx / groupSize;

  simd<fp16, 16 * 128> fp16QState;
  simd<float, 16 * 32> tempBuffer;
  simd<float, 16 * 64> tempOutput;
  auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
  auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  simd<fp16, 16 * 128> finalOutput = 0;
  simd<float, 16> fp32SoftMaxTemp = 0;
  simd<float, 16> fp32HistoricMaxTemp = FP32_MIN;
  simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);

  int32_t kvSeqOutLoopCount = (kvSeqLen + 63) / 64;

  uint32_t widthInByteQ = headQ * 128 * sizeof(fp16) - 1;
  uint32_t widthInByteKV = headKv * 128 * sizeof(fp16) - 1;
  uint32_t heightQ = activationLength - 1;
  uint32_t heightKv = kvSeqLen - 1;

  uint32_t qCoordX = headIdx * 128 >> 1;
  uint32_t qCoordY = h * 256 + hhq * 16;
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

  // ===== MAIN LOOP =====
  for (loopIdx = 0; loopIdx < kvSeqOutLoopCount - 1; loopIdx++) {
    uint32_t slmPingpongLoad = loopIdx & 0x1;
    uint32_t slmPingpongStore = (loopIdx + 1) & 0x1;
    slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
    slmPingpongStore = slmPingpongStore * 64 * 128 * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 512> fp16VState;
    simd<fp16, 32> compensationTemp;  // declared at loop scope for use after barrier
    tempOutput = 0;

    // ===== Q @ K^T =====
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

    // ===== V load =====
    fp16VState =
      __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
    vCoordY += 64;
    payloadV.set_y(vCoordY);

    // ===== Softmax =====
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
      simd<float, 8 * 16> ttemp;
      fp32CurrentMaxTemp = fp32HistoricMaxTemp;

      // Max reduction
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

      // Exp2 on scores
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

      // Compensation compute
      fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);
      fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);

      // EARLY fp16 convert — before sum reduction, so sum ALU overlaps with convert latency
      compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
      compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;

      // Sum reduction
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

      // Pack softmax weights fp32 -> fp16 VNNI (before barrier, compensation moved after)
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

    barrier();  // moved before compensation — SLM ready, comp ALU can overlap with SLM loads

    // ===== Interleaved compensation + S×V by output-dim half =====
    {
      // --- l=0: finalOutput[0..1023] ---
      // Compensate l=0 half
      #pragma unroll
      for (int kk = 0; kk < 32; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }
      // Block 1 (V rows 0-31), l=0
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(128 * ll);
          auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
      // Block 2 (V rows 32-63), l=0
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * 2 * sizeof(fp16) +
              16 * 128 * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(128 * ll);
          auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }

      // --- l=1: finalOutput[1024..2047] ---
      // Compensate l=1 half (overlaps with l=0 DPAS pipeline)
      #pragma unroll
      for (int kk = 32; kk < 64; kk++) {
        finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
      }
      // Block 1 (V rows 0-31), l=1
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(1024 + 128 * ll);
          auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
      // Block 2 (V rows 32-63), l=1
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * 2 * sizeof(fp16) +
              16 * 128 * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(1024 + 128 * ll);
          auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }

      // SLM scatter for next V block
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

  // ===== LAST LOOP - boundary checking =====
  {
    uint32_t slmPingpongLoad = (loopIdx) & 0x1;
    slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 32> compensationTemp;  // declared at scope for use after barrier
    tempOutput = 0;

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

    // Apply boundary mask, then softmax
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
      auto softmaxPositions = ui32Temp.select<16, 1>(48);
      simd<float, 8 * 16> ttemp;

      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * 64;
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
        }
        softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
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

      // EARLY fp16 convert — before sum reduction
      compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
      compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;

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

      // Pack softmax weights fp32 -> fp16 VNNI
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

    barrier();  // moved before compensation

    // S×V with interleaved compensation (last iteration)
    {
      if (loopIdx != 0) {
        // --- l=0: finalOutput[0..1023] ---
        #pragma unroll
        for (int kk = 0; kk < 32; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }
      // Block 1, l=0
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(128 * ll);
          auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
      // Block 2, l=0
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * 2 * sizeof(fp16) +
              16 * 128 * nn * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(128 * ll);
          auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }

      if (loopIdx != 0) {
        // --- l=1: finalOutput[1024..2047] ---
        #pragma unroll
        for (int kk = 32; kk < 64; kk++) {
          finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
        }
      }
      // Block 1, l=1
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
          tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
            slm_block_load<fp16, 512>(slmOffsetBaseV +
              slmPingpongLoad +
              16 * 128 * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(1024 + 128 * ll);
          auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
          auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
          ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
            simd<fp16, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
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
              16 * 128 * 2 * sizeof(fp16) +
              16 * 128 * nn * sizeof(fp16) +
              16 * 64 * sizeof(fp16) +
              512 * ll * sizeof(fp16));
        }
        #pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          auto ccTile = finalOutput.select<128, 1>(1024 + 128 * ll);
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

  // Output normalization
  simd<float, 16> softMaxDividor;
  simd<float, 128> alphaV;
  alphaV = block_load<float, 128>((float*)normAlpha + headIdx * 128);
  simd<uint32_t, 16> simdOffsets;
  simd_mask<16> mask;
  softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
  softMaxDividor = 1.0f / softMaxDividor;

  #pragma unroll
  for (int kk = 0; kk < 64; kk++) {
    simd<float, 32> alphaMul;
    simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
    alphaMul.select<16, 1>(0) = alphaV[2 * kk] * softMaxDividor.select<16, 1>(0);
    alphaMul.select<16, 1>(16) = alphaV[2 * kk + 1] * softMaxDividor.select<16, 1>(0);
    f16Temp = f16Temp * alphaMul;
    fp16QState.select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
    fp16QState.select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
  }

  simdOffsets = baseOffsetInc16AsVector;
  simdOffsets = simdOffsets + 256 * h + 16 * hhq;
  mask = simdOffsets < activationLength;
  simdOffsets = simdOffsets * headQ * 128 * sizeof(fp16) + headIdx * 128 * sizeof(fp16);
  #pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    __ESIMD_ENS::lsc_scatter<uint32_t, 4, __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back, 16, uint32_t>(
      (uint32_t*)out, simdOffsets, fp16QState.template bit_cast_view<uint32_t>().select<64, 1>(64 * kk), mask);
    simdOffsets += 4 * sizeof(uint32_t);
  }
}
