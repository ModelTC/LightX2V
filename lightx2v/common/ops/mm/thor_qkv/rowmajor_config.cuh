// Derived from official upstream 0ca5f0d; row-major BF16 + fused bias.
#pragma once
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
using namespace cute;
// A matrix configuration
using         ElementA    = cutlass::bfloat16_t;                              // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                       // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;     // Memory access granularity (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::bfloat16_t;                              // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                    // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;     // Memory access granularity (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::bfloat16_t;                                           // BF16 output type; C is unused
using         LayoutC     = cutlass::layout::RowMajor;                    // Contiguous row-major D
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;     // Memory access granularity (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag — native tensor cores

// Tile shape defaults with -D compiler flag override for search
// Production QKV: M256xN256xK64, cluster2x1, fused BF16 bias/output.
#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 256
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef TILE_CLUSTERM
#define TILE_CLUSTERM 2
#endif
#ifndef TILE_CLUSTERN
#define TILE_CLUSTERN 1
#endif
#ifndef TILE_CLUSTERZ
#define TILE_CLUSTERZ 1
#endif

#define TILE_CAT_(a,b) a##b
#define TILE_CAT(a,b)  TILE_CAT_(a,b)

using MmaTileShape_MNK = Shape<TILE_CAT(_,TILE_M), TILE_CAT(_,TILE_N), TILE_CAT(_,TILE_K)>;
using ClusterShape_MNK = Shape<TILE_CAT(_,TILE_CLUSTERM), TILE_CAT(_,TILE_CLUSTERN), TILE_CAT(_,TILE_CLUSTERZ)>;

// Fused bias: FP32 accumulator + BF16 row broadcast -> BF16 row-major.
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
namespace fusion = cutlass::epilogue::fusion;
using Bias = fusion::Sm90RowBroadcast<0, MmaTileShape_MNK, cutlass::bfloat16_t, float,
    cute::Stride<cute::_0, cute::_1, int64_t>, 8>;
using Fusion = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::plus, cutlass::bfloat16_t, float,
                       cutlass::FloatRoundStyle::round_to_nearest>,
    fusion::Sm90AccFetch, Bias>;
// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    void, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion
  >::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                   // Default to ClusterLaunchControl (CLC) based tile scheduler

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
