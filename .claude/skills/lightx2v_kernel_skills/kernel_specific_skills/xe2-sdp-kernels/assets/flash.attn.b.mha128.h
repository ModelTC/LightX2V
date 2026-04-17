// Shape Q [activation token length, 16, 128]  FP16,
// Shape K:  [kv len, 4, 128] FP16,
// Shape V : [kv len, 4, 128] FP16,
// output: [activation token length, 16, 128]  FP16,
ESIMD_INLINE void flashAttnBMha128(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* normAlpha,
  uint8_t* out,
  uint32_t activationLength,
  uint32_t kvSeqLen,
  uint32_t headQkv,
  sycl::nd_item<2>& ndi) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  constexpr float attnScoreMul = matMulQuantCoeff * sycl::ext::intel::esimd::detail::log2e;
  constexpr uint32_t slmSizeV = 2 * 64 * 128 * sizeof(fp16);
  constexpr uint32_t slmSizeSoftMaxSum = 0 * sizeof(float);
  constexpr uint32_t slmSizeSoftMaxHistoric = 0 * sizeof(float);
  constexpr uint32_t slmSizeSoftMaxCompensation = 0 * sizeof(float);
  constexpr uint32_t slmSize =
    slmSizeV + 
    slmSizeSoftMaxSum + 
    slmSizeSoftMaxHistoric + 
    slmSizeSoftMaxCompensation;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  __ESIMD_NS::slm_init(slmSize);
  constexpr uint32_t slmOffsetBaseV = 0;
  constexpr uint32_t slmOffsetBaseSoftMaxSum = slmOffsetBaseV + slmSizeV;
  constexpr uint32_t slmOffsetBaseSoftMaxHistoric = slmOffsetBaseSoftMaxSum + slmSizeSoftMaxSum;
  constexpr uint32_t slmOffsetBaseSoftMaxCompensation = slmOffsetBaseSoftMaxHistoric + slmSizeSoftMaxHistoric;
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
  simd<fp16, 16 * 128> fp16QState;
  simd<float, 16 * 32> tempBuffer;
  simd<float, 16 * 64> tempOutput;
  auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
  auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();
  simd<fp16, 16 * 128> finalOutput = 0;
  simd<float, 16> fp32SoftMaxTemp = 0;
  simd<float, 16> fp32HistoricMaxTemp = FP32_MIN;
  simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);

  int32_t kvSeqOutLoopCount = (kvSeqLen + 0x3f) >> 6;
  kvSeqOutLoopCount = kvSeqOutLoopCount - 1;
  uint32_t widthInByte = headQkv * 128 * sizeof(fp16) - 1;
  uint32_t heightQ = activationLength - 1;
  uint32_t heightKv = kvSeqLen - 1;
  uint32_t qCoordX = headIdx * 128 >> 1;
  uint32_t qCoordY = h * 256 + hhq * 16;
  uint32_t kCoordX = headIdx * 128;
  uint32_t kCoordY = 0;
  uint32_t vCoordX = headIdx * 128 + hhv * 32;
  uint32_t vCoordY = vvv * 16;
  uint32_t prefCoordX = (headIdx * 128 >> 1) + hhpref * 32;
  uint32_t prefCoordYK = vvpref * 8;

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kState, widthInByte, heightKv, widthInByte, kCoordX, kCoordY);

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadV(
    (fp16*)vState, widthInByte, heightKv, widthInByte, vCoordX, vCoordY);

  __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefK(
    (uint32_t*)kState, widthInByte, heightKv, widthInByte, prefCoordX, prefCoordYK);
  unsigned int slmOffsetV = slmOffsetBaseV + localLinearId * 512 * sizeof(fp16);
#pragma unroll
  for (int32_t k = 0; k < 1; k++) {
#pragma unroll
    for (int32_t kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_prefetch_2d<
        uint32_t,
        16,
        8,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >(payloadPrefK);
      payloadPrefK.set_x(prefCoordX + 16 * kk);
    }
    prefCoordYK += 64;
    payloadPrefK.set_y(prefCoordYK);
  }

  tempBufferAsFp16.select<512, 1>(0) =
    __ESIMD_ENS::lsc_load_2d<
    fp16,
    16,
    16,
    2,
    false,
    true,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>(payloadV);

  vCoordY += 64;
  payloadV.set_y(vCoordY);

  {
    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
      (uint32_t*)qState, widthInByte, heightQ, widthInByte, qCoordX, qCoordY);
#pragma unroll
    for (int32_t kk = 0; kk < 8; kk++) {
      fp16QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk) =
        __ESIMD_ENS::lsc_load_2d<
        uint32_t,
        8,
        16,
        1,
        true,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>(payloadQ);

      qCoordX += 8;
      payloadQ.set_x(qCoordX);
    }
  }

  {
    simd<uint32_t, 32> simdSlmOffsetsV;
    simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
    simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
    simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV;

#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_slm_scatter<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        16>(simdSlmOffsetsV.select<16, 1>(16 * kk), tempBufferAsFp16.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
    }
  }

  int loopIdx;
  for (loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    uint32_t slmPingpongLoad = loopIdx & 0x1;
    uint32_t slmPingpongStore = (loopIdx + 1) & 0x1;
    slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
    slmPingpongStore = slmPingpongStore * 64 * 128 * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 512> fp16VState;
    tempOutput = 0;
    {
#pragma unroll
      for (int32_t nn = 0; nn < 8; nn++) {
        payloadK.set_x(kCoordX + 16 * nn);
#pragma unroll
        for (int32_t l = 0; l < 4; l++) {
          payloadK.set_y(kCoordY + 16 * l);
          tempBufferAsFp16.select<256, 1>(256 * l) =
            __ESIMD_ENS::lsc_load_2d<
            fp16,
            16,
            16,
            1,
            false,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>(payloadK);
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

    fp16VState =
      __ESIMD_ENS::lsc_load_2d<
      fp16,
      16,
      16,
      2,
      false,
      true,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>(payloadV);

    vCoordY += 64;
    payloadV.set_y(vCoordY);
    {
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
      simd<float, 8 * 16> ttemp;
      auto ttempFp16 = ttemp.template bit_cast_view<fp16>();
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

      simd<uint32_t, 32> simdSlmOffsetsV;
      simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
      simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
      simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV + slmPingpongStore;

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        __ESIMD_ENS::lsc_slm_scatter<
          uint32_t,
          8,
          __ESIMD_ENS::lsc_data_size::u32,
          16>(simdSlmOffsetsV.select<16, 1>(16 * kk), fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
      }
    }
#pragma unroll
    for (int32_t kk = 0; kk < 2; kk++) {
      __ESIMD_ENS::lsc_prefetch_2d<
        uint32_t,
        16,
        8,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
      >(payloadPrefK);
      payloadPrefK.set_x(prefCoordX + 16 * kk);
    }
    prefCoordYK += 64;
    payloadPrefK.set_y(prefCoordYK);
  }

  // last loop
  {
    uint32_t slmPingpongLoad = (loopIdx) & 0x1;
    slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
    auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    tempOutput = 0;
    {
#pragma unroll
      for (int32_t nn = 0; nn < 8; nn++) {
        payloadK.set_x(kCoordX + 16 * nn);
#pragma unroll
        for (int32_t l = 0; l < 4; l++) {
          payloadK.set_y(kCoordY + 16 * l);
          tempBufferAsFp16.select<256, 1>(256 * l) =
            __ESIMD_ENS::lsc_load_2d<
            fp16,
            16,
            16,
            1,
            false,
            false,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>(payloadK);
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

    {
      auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
      auto softmaxPositions = ui32Temp.select<16, 1>(48);
      simd<float, 8 * 16> ttemp;
      auto ttempFp16 = ttemp.template bit_cast_view<fp16>();

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
  simdOffsets = simdOffsets * headQkv * 128 * sizeof(fp16) + headIdx * 128 * sizeof(fp16);
#pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    __ESIMD_ENS::lsc_scatter<
      uint32_t,
      4,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back,
      16,
      uint32_t
    >((uint32_t*)out, simdOffsets, fp16QState.template bit_cast_view<uint32_t>().select<64, 1>(64 * kk), mask);
    simdOffsets += 4 * sizeof(uint32_t);
  }
}

ESIMD_INLINE void inplaceNormFp16(
  uint8_t* input,
  uint8_t* output,
  uint8_t* normAlpha,
  uint32_t nTok,
  uint32_t hiddenDim,
  sycl::nd_item<2>& ndi) {
  constexpr uint32_t slmSize = 8 * 32 * sizeof(fp16);
  __ESIMD_NS::slm_init(slmSize);
  int32_t h = ndi.get_group(0);
  int32_t v = ndi.get_group(1);
  int32_t localLinearId = ndi.get_local_id(0);
  uint32_t widthInByte = hiddenDim * sizeof(fp16) - 1;
  uint32_t height = nTok - 1;
  uint32_t coordX = h * 16;
  uint32_t coordY = localLinearId * 16;
  int32_t loopCount = nTok;
  loopCount = (loopCount + 127) >> 7;
  simd<fp16, 256> fp16Input;
  simd<fp16, 32> tempMax = 0;
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadIn(
    (fp16*)input, widthInByte, height, widthInByte, coordX, coordY);

  for (int nn = 0; nn < loopCount; nn++) {
    simd<fp16, 256> fp16Abs;
    fp16Input = 
      __ESIMD_ENS::lsc_load_2d<
      fp16,
      16,
      16,
      1,
      false,
      true,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>(payloadIn);

    coordY += 128;
    payloadIn.set_y(coordY);

    fp16Abs = abs(fp16Input);
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      tempMax = __ESIMD_NS::max<fp16, 32, fp16>(tempMax, fp16Abs.select<32, 1>(32 * kk));
    }
  }
  slm_block_store<fp16, 32>(localLinearId * 32 * sizeof(fp16), tempMax);
  barrier();
  if (localLinearId == 0) {
    simd<fp16, 256> fp16MaxTemp;
    simd<fp16, 16> fp16MaxFinal;

    fp16MaxTemp = slm_block_load<fp16, 256>(0);
#pragma unroll
    for (int kk = 1; kk < 8; kk++) {
      fp16MaxTemp.select<32, 1>(0) =
        __ESIMD_NS::max<fp16, 32, fp16>(
          fp16MaxTemp.select<32, 1>(0),
          fp16MaxTemp.select<32, 1>(32 * kk));
    }

    fp16MaxFinal = __ESIMD_NS::max<fp16, 16, fp16>(
      fp16MaxTemp.select<16, 2>(0),
      fp16MaxTemp.select<16, 2>(1));

    slm_block_store<fp16, 16>(0, fp16MaxFinal);
  }

  barrier();
  simd<fp16, 16> channelMax = slm_block_load<fp16, 16>(0);
  simd<float, 32> channelMaxFp32;
  channelMaxFp32.select<16, 1>(0) = channelMax;
  channelMaxFp32.select<16, 1>(16) = channelMax;
  float sqrtNorm = (float)nTok;
  sqrtNorm = __ESIMD_NS::sqrt(sqrtNorm);
  simd<float, 16> normFactorFp32 = channelMaxFp32.select<16, 1>(0) * sqrtNorm;
  sqrtNorm = 1.0f / sqrtNorm;
  channelMaxFp32 = 1.0f / channelMaxFp32;
  channelMaxFp32.select<16, 1>(0).merge(0.0f, channelMax == 0);
  channelMaxFp32.select<16, 1>(16).merge(0.0f, channelMax == 0);
  channelMaxFp32.select<32, 1>(0) = channelMaxFp32.select<32, 1>(0) * sqrtNorm;

  coordY = localLinearId * 16;

  for (int nn = 0; nn < loopCount; nn++) {
    payloadIn.set_y(coordY);

    simd<float, 256> fp32Temp;
    simd<fp16, 128> fp16Temp0;
    simd<fp16, 128> fp16Temp1;

    fp16Input =
      __ESIMD_ENS::lsc_load_2d<
      fp16,
      16,
      16,
      1,
      false,
      false,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>(payloadIn);


    fp32Temp = fp16Input;
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      fp32Temp.select<32, 1>(32 * kk) = fp32Temp.select<32, 1>(32 * kk) * channelMaxFp32.select<32, 1>(0);
    }

    fp16Temp0 = fp32Temp.select<128, 1>(0);
    fp16Temp1 = fp32Temp.select<128, 1>(128);

    __ESIMD_ENS::lsc_store_2d<
      fp16,
      16,
      8,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)output, widthInByte, height, widthInByte, coordX, coordY, fp16Temp0);

    __ESIMD_ENS::lsc_store_2d<
      fp16,
      16,
      8,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>(
        (fp16*)output, widthInByte, height, widthInByte, coordX, coordY + 8, fp16Temp1);
    coordY += 128;
  }

  uint32_t offsetAlpha = h * 16 * sizeof(float);
  block_store<uint8_t, 64>(normAlpha + offsetAlpha, normFactorFp32.template bit_cast_view<uint8_t>());
}