#pragma once
#include <stdexcept>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"

namespace up1785_gelu_row {
constexpr int N = 13824;
constexpr int K = 5120;
using Tile = cute::Shape<cute::_256, cute::_256, cute::_64>;
using Cluster = cute::Shape<cute::_2, cute::_1, cute::_1>;
namespace fusion = cutlass::epilogue::fusion;
using Bias = fusion::Sm90RowBroadcast<0, Tile, cutlass::bfloat16_t, float,
    cute::Stride<cute::_0,cute::_1,int64_t>, 8>;
using AddBias = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    fusion::Sm90AccFetch, Bias>;
using Fusion = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::epilogue::thread::GELU_taylor, cutlass::bfloat16_t, float, cutlass::FloatRoundStyle::round_to_nearest>, AddBias>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, void, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
// Five input stages match the qualified shared-builder N256/2x1 instantiation.
// Explicitly pin the measured policy rather than deriving it from an unrelated
// FP32 baseline epilogue's shared-memory carveout.
using Base = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster, cutlass::gemm::collective::StageCount<5>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Mainloop = Base;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// Logical output BF16 [M,13824], row-major pitch13824.
// No output padding allocation is required.
inline Gemm::Arguments make_arguments(cutlass::bfloat16_t const* x,
    cutlass::bfloat16_t const* w, cutlass::bfloat16_t const* bias,
    cutlass::bfloat16_t* d, int rows) {
  for (auto p : {x, w, bias, static_cast<cutlass::bfloat16_t const*>(d)})
    if (!p || reinterpret_cast<uintptr_t>(p) % 16)
      throw std::invalid_argument("up1785_gelu requires non-null 16-byte aligned pointers");
  Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {rows, N, K, 1},
    {x, cute::make_stride(int64_t(K),cute::_1{},int64_t(0)),
     w, cute::make_stride(int64_t(K),cute::_1{},int64_t(0))},
    {{}, nullptr, cute::make_stride(int64_t(N),cute::_1{},int64_t(0)),
     d, cute::make_stride(int64_t(N),cute::_1{},int64_t(0))}};
  args.epilogue.thread = {{{}, {bias, cutlass::bfloat16_t(0), {}}, {}}, {}};
  args.scheduler.max_swizzle_size = 0;
  return args;
}
} // namespace up1785_gelu_row
