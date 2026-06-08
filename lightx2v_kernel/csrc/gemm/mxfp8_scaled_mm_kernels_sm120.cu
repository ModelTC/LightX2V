#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/all.h>

#include <array>
#include <mutex>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#include "sm120_utils.h"

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

#define CHECK_TYPE(x, st, m) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)


using namespace cute;

namespace cutlass::epilogue::fusion {

template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementGate_ = ElementOutput_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentGate_ = 128 / cute::sizeof_bits_v<ElementGate_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasPerColGateResidual : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementGate = ElementGate_;
  using ElementBias = ElementBias_;
  static constexpr int AlignmentGate = AlignmentGate_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerColBiasSupported = true;
};

template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementGate = ElementOutput,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentGate = 128 / sizeof_bits_v<ElementGate>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColBiasPerColGateResidual =
  // Wan 1D gate fast path: apply gate/residual in the CUTLASS epilogue using
  // accumulator values, then round once to ElementOutput. This is intentionally
  // not bitwise identical to materializing BF16 GEMM output before gate.
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // gate * (alpha * acc + bias) + C
    Sm90RowBroadcast<0, CtaTileShapeMNK, ElementGate, ElementCompute, Stride<_0,_1,int64_t>, AlignmentGate>, // gate
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch, // acc
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
    >,
    Sm90SrcFetch<ElementSource> // C / residual
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementGate,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentGate,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasPerColGateResidual<
      ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource, ElementScalar, AlignmentGate, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerColBiasPerColGateResidual<
      CtaTileShapeMNK, ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource, ElementScalar, AlignmentGate, AlignmentBias, RoundStyle
    > {

  using Impl = Sm90LinCombPerColBiasPerColGateResidual<
    CtaTileShapeMNK, ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource, ElementScalar, AlignmentGate, AlignmentBias, RoundStyle
  >;
  using Operation = fusion::LinCombPerColBiasPerColGateResidual<
    ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource, ElementScalar, AlignmentGate, AlignmentBias, RoundStyle
  >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    using StrideGate = Stride<_0,_1,int64_t>;
    ElementGate const* gate_ptr = nullptr;
    StrideGate dGate = {};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {     // ternary op : gate * (alpha * acc + bias) + C
          {gate_ptr, ElementGate(0), dGate}, // leaf args : gate
          {                     // ternary op : alpha * acc + bias
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
            {}                  // ternary args : multiply_add
          },
          {}, // leaf args : C
          {}  // ternary args : multiply_add
        };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion


struct Mxfp8GemmSm120 {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
    using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = 16;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for B matrix operand
    using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static constexpr int AlignmentB  = 128;                                            // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
    using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
    using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
    using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    // Kernel functional config
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

    // Kernel Perf config
    using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
    using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

    // use per-column bias, i.e. every column has different bias
    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,                      // Epilogue schedule policy
        EVTOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,                                                   // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));
};

struct Mxfp8GemmResidualGateSm120 {
    using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using         LayoutATag  = cutlass::layout::RowMajor;
    static constexpr int AlignmentA  = 16;

    using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using         LayoutBTag  = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB  = 128;

    using         ElementD    = cutlass::bfloat16_t;
    using         ElementC    = cutlass::bfloat16_t;
    using         LayoutCTag  = cutlass::layout::RowMajor;
    using         LayoutDTag  = cutlass::layout::RowMajor;
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator  = float;
    using ArchTag             = cutlass::arch::Sm120;
    using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

    using ThreadBlockShape    = Shape<_128,_128,_128>;
    using ClusterShape        = Shape<_1,_1,_1>;

    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBiasPerColGateResidual<
        ElementD, ElementAccumulator, ElementD, ElementD, ElementC, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        EVTOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));
};


// Populates a Gemm::Arguments structure from the given commandline options
typename Mxfp8GemmSm120::Gemm::Arguments args_from_options_mxfp8(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  using Sm1xxBlkScaledConfig = typename Mxfp8GemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(Mxfp8GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Mxfp8GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Mxfp8GemmSm120::StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  if (bias){
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename Mxfp8GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Mxfp8GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Mxfp8GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue8m0_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue8m0_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Mxfp8GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Mxfp8GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    fusion_args.beta = 0.0f;
    fusion_args.beta_ptr = nullptr;
    fusion_args.bias_ptr = static_cast<Mxfp8GemmSm120::Gemm::ElementC const*>(bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    return arguments;
  } else {
    typename Mxfp8GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Mxfp8GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Mxfp8GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue8m0_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue8m0_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Mxfp8GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Mxfp8GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    fusion_args.beta = 0.0f;
    fusion_args.beta_ptr = nullptr;
    return arguments;
  }
}

typename Mxfp8GemmResidualGateSm120::Gemm::Arguments args_from_options_mxfp8_residual_gate(
    at::Tensor& residual,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    at::Tensor const& gate,
    int64_t M,
    int64_t N,
    int64_t K) {
  using Sm1xxBlkScaledConfig = typename Mxfp8GemmResidualGateSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(Mxfp8GemmResidualGateSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Mxfp8GemmResidualGateSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Mxfp8GemmResidualGateSm120::StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename Mxfp8GemmResidualGateSm120::Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {// Mainloop arguments
     static_cast<Mxfp8GemmResidualGateSm120::Gemm::ElementA const*>(A.data_ptr()),
     stride_A,
     static_cast<Mxfp8GemmResidualGateSm120::Gemm::ElementB const*>(B.data_ptr()),
     stride_B,
     static_cast<cutlass::float_ue8m0_t const*>(A_sf.data_ptr()),
     layout_SFA,
     static_cast<cutlass::float_ue8m0_t const*>(B_sf.data_ptr()),
     layout_SFB},
    {     // Epilogue arguments
     {},  // epilogue.thread
     static_cast<Mxfp8GemmResidualGateSm120::Gemm::ElementC const*>(residual.data_ptr()),
     stride_D,
     static_cast<Mxfp8GemmResidualGateSm120::Gemm::ElementD*>(residual.data_ptr()),
     stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
  fusion_args.gate_ptr = static_cast<Mxfp8GemmResidualGateSm120::ElementD const*>(gate.data_ptr());
  if (bias) {
    fusion_args.bias_ptr = static_cast<Mxfp8GemmResidualGateSm120::ElementD const*>(bias->data_ptr());
  }
  return arguments;
}


void runGemmMxfp8Sm120(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename Mxfp8GemmSm120::Gemm gemm;

  auto arguments = args_from_options_mxfp8(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  size_t workspace_size = Mxfp8GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

void runGemmMxfp8ResidualGateSm120(
    at::Tensor& residual,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    at::Tensor const& gate,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename Mxfp8GemmResidualGateSm120::Gemm gemm;

  auto arguments = args_from_options_mxfp8_residual_gate(residual, A, B, A_sf, B_sf, alpha, bias, gate, m, n, k);
  size_t workspace_size = Mxfp8GemmResidualGateSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}


constexpr auto FP6_FP8_TYPE = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e8m0fnu;

namespace lightx2v_mxfp8_fused {

constexpr int kFusedThreads = 256;
using pack_128b_t = uint4;

struct Mxfp8GemmMeta {
  int64_t m;
  int64_t n;
  int64_t k;
};

int64_t round_up(int64_t x, int64_t y) {
  return (x + y - 1) / y * y;
}

Mxfp8GemmMeta check_mxfp8_gemm_inputs(
    torch::Tensor const& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    char const* op_name) {
  CHECK_INPUT(D, at::ScalarType::BFloat16, "out");
  CHECK_INPUT(A, FP6_FP8_TYPE, "mat_a");
  CHECK_INPUT(B, FP6_FP8_TYPE, "mat_b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");
  TORCH_CHECK(
      D.device() == A.device() && D.device() == B.device() && D.device() == A_sf.device() &&
          D.device() == B_sf.device() && D.device() == alpha.device(),
      op_name,
      " expects output, mat_a, mat_b, scale_a, scale_b, and alpha on the same CUDA device");
  TORCH_CHECK(D.dim() == 2, "out must be a matrix");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "mat_a and mat_b must be matrices");
  TORCH_CHECK(alpha.numel() == 1, "alpha must contain exactly one scalar");
  TORCH_CHECK(
      A.sizes()[1] == B.sizes()[1],
      "mat_a and mat_b shapes cannot be multiplied (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      " and ",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1];

  TORCH_CHECK(D.sizes()[0] == m, "out rows must match mat_a rows");
  TORCH_CHECK(D.sizes()[1] == n, "out cols must match mat_b rows");
  constexpr int alignment_a = 16;
  constexpr int alignment_b = 128;
  TORCH_CHECK(
      k % alignment_a == 0,
      "Expected k to be divisible by ",
      alignment_a,
      ", but got mat_a shape: (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      "), k: ",
      k,
      ".");
  TORCH_CHECK(
      n % alignment_b == 0,
      "Expected n to be divisible by ",
      alignment_b,
      ", but got mat_b shape: (",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ").");

  int64_t rounded_m = round_up(m, 128);
  int64_t rounded_n = round_up(n, 128);
  int64_t rounded_k = round_up(k / 32, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(
      A_sf.sizes()[1] == B_sf.sizes()[1],
      "scale_a and scale_b shapes cannot be multiplied (",
      A_sf.sizes()[0],
      "x",
      A_sf.sizes()[1],
      " and ",
      B_sf.sizes()[0],
      "x",
      B_sf.sizes()[1],
      ")");
  TORCH_CHECK(
      A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
      "scale_a must be padded and swizzled to a shape (",
      rounded_m,
      "x",
      rounded_k,
      "), but got a shape (",
      A_sf.sizes()[0],
      "x",
      A_sf.sizes()[1],
      ")");
  TORCH_CHECK(
      B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
      "scale_b must be padded and swizzled to a shape (",
      rounded_n,
      "x",
      rounded_k,
      "), but got a shape (",
      B_sf.sizes()[0],
      "x",
      B_sf.sizes()[1],
      ")");
  if (bias) {
    auto const& bias_tensor = bias.value();
    CHECK_INPUT(bias_tensor, at::ScalarType::BFloat16, "bias");
    TORCH_CHECK(bias_tensor.device() == A.device(), "bias must be on the same CUDA device");
    TORCH_CHECK(bias_tensor.numel() == n, "bias numel must match output columns");
  }
  lightx2v_kernel::check_sm120_or_throw(A, op_name);
  return {m, n, k};
}

void check_mxfp8_residual_gate(
    torch::Tensor const& residual,
    torch::Tensor const& gate,
    char const* op_name) {
  CHECK_INPUT(residual, at::ScalarType::BFloat16, "residual");
  CHECK_INPUT(gate, at::ScalarType::BFloat16, "gate");
  TORCH_CHECK(residual.dim() == 2, "residual must be a matrix");
  TORCH_CHECK(gate.device() == residual.device(), op_name, " expects residual and gate on the same CUDA device");
  TORCH_CHECK(gate.dim() == 1 || gate.dim() == 2, "gate must be 1D or 2D");
  if (gate.dim() == 1) {
    TORCH_CHECK(gate.sizes()[0] == residual.sizes()[1], "1D gate size must match residual columns");
  } else {
    TORCH_CHECK(gate.sizes() == residual.sizes(), "2D gate shape must match residual shape");
  }
}

__global__ void mxfp8_residual_gate_2d_bf16_kernel(
    __nv_bfloat16* residual,
    __nv_bfloat16 const* ffn_out,
    __nv_bfloat16 const* gate,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    float product = __bfloat162float(ffn_out[idx]) * __bfloat162float(gate[idx]);
    float sum = __bfloat162float(residual[idx]) + product;
    residual[idx] = __float2bfloat16(sum);
  }
}

__device__ inline __nv_bfloat162 float2_to_bfloat162_rn(float2 v) {
  return __halves2bfloat162(__float2bfloat16(v.x), __float2bfloat16(v.y));
}

__global__ void mxfp8_residual_gate_2d_bf16x2_kernel(
    __nv_bfloat16* residual,
    __nv_bfloat16 const* ffn_out,
    __nv_bfloat16 const* gate,
    int64_t num_pairs) {
  int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= num_pairs) {
    return;
  }

  auto residual_vec = reinterpret_cast<__nv_bfloat162*>(residual);
  auto ffn_out_vec = reinterpret_cast<__nv_bfloat162 const*>(ffn_out);
  auto gate_vec = reinterpret_cast<__nv_bfloat162 const*>(gate);

  float2 residual_f = __bfloat1622float2(residual_vec[pair_idx]);
  float2 ffn_out_f = __bfloat1622float2(ffn_out_vec[pair_idx]);
  float2 gate_f = __bfloat1622float2(gate_vec[pair_idx]);
  float2 out;
  out.x = residual_f.x + ffn_out_f.x * gate_f.x;
  out.y = residual_f.y + ffn_out_f.y * gate_f.y;
  residual_vec[pair_idx] = float2_to_bfloat162_rn(out);
}

__global__ void mxfp8_residual_gate_2d_bf16x8_kernel(
    __nv_bfloat16* residual,
    __nv_bfloat16 const* ffn_out,
    __nv_bfloat16 const* gate,
    int64_t num_packs) {
  int64_t pack_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pack_idx >= num_packs) {
    return;
  }

  auto residual_pack = reinterpret_cast<pack_128b_t*>(residual);
  auto ffn_out_pack = reinterpret_cast<pack_128b_t const*>(ffn_out);
  auto gate_pack = reinterpret_cast<pack_128b_t const*>(gate);

  pack_128b_t residual_reg = residual_pack[pack_idx];
  pack_128b_t ffn_out_reg = ffn_out_pack[pack_idx];
  pack_128b_t gate_reg = gate_pack[pack_idx];
  pack_128b_t out_reg;

  auto residual_vec = reinterpret_cast<__nv_bfloat162*>(&residual_reg);
  auto ffn_out_vec = reinterpret_cast<__nv_bfloat162*>(&ffn_out_reg);
  auto gate_vec = reinterpret_cast<__nv_bfloat162*>(&gate_reg);
  auto out_vec = reinterpret_cast<__nv_bfloat162*>(&out_reg);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 residual_f = __bfloat1622float2(residual_vec[i]);
    float2 ffn_out_f = __bfloat1622float2(ffn_out_vec[i]);
    float2 gate_f = __bfloat1622float2(gate_vec[i]);
    float2 out;
    out.x = residual_f.x + ffn_out_f.x * gate_f.x;
    out.y = residual_f.y + ffn_out_f.y * gate_f.y;
    out_vec[i] = float2_to_bfloat162_rn(out);
  }

  residual_pack[pack_idx] = out_reg;
}

inline bool is_aligned_16(void const* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

inline bool is_aligned_4(void const* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % 4 == 0;
}

void launch_mxfp8_residual_gate_2d(torch::Tensor& residual, torch::Tensor const& ffn_out, torch::Tensor const& gate, cudaStream_t stream) {
  TORCH_CHECK(gate.dim() == 2, "2D residual gate fallback expects a per-element 2D gate");
  int64_t total = residual.numel();
  auto* residual_ptr = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>());
  auto const* ffn_out_ptr = reinterpret_cast<__nv_bfloat16 const*>(ffn_out.data_ptr<at::BFloat16>());
  auto const* gate_ptr = reinterpret_cast<__nv_bfloat16 const*>(gate.data_ptr<at::BFloat16>());

  if (is_aligned_16(residual_ptr) && is_aligned_16(ffn_out_ptr) && is_aligned_16(gate_ptr) && total >= 8) {
    int64_t num_packs = total / 8;
    if (num_packs > 0) {
      int blocks = static_cast<int>((num_packs + kFusedThreads - 1) / kFusedThreads);
      mxfp8_residual_gate_2d_bf16x8_kernel<<<blocks, kFusedThreads, 0, stream>>>(
          residual_ptr,
          ffn_out_ptr,
          gate_ptr,
          num_packs);
      int64_t processed = num_packs * 8;
      int64_t rem = total - processed;
      if (rem >= 2) {
        int64_t num_pairs = rem / 2;
        blocks = static_cast<int>((num_pairs + kFusedThreads - 1) / kFusedThreads);
        mxfp8_residual_gate_2d_bf16x2_kernel<<<blocks, kFusedThreads, 0, stream>>>(
            residual_ptr + processed,
            ffn_out_ptr + processed,
            gate_ptr + processed,
            num_pairs);
        processed += num_pairs * 2;
        rem = total - processed;
      }
      if (rem > 0) {
        mxfp8_residual_gate_2d_bf16_kernel<<<1, 1, 0, stream>>>(
            residual_ptr + processed,
            ffn_out_ptr + processed,
            gate_ptr + processed,
            rem);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }

  if (is_aligned_4(residual_ptr) && is_aligned_4(ffn_out_ptr) && is_aligned_4(gate_ptr) && total >= 2) {
    int64_t num_pairs = total / 2;
    int blocks = static_cast<int>((num_pairs + kFusedThreads - 1) / kFusedThreads);
    mxfp8_residual_gate_2d_bf16x2_kernel<<<blocks, kFusedThreads, 0, stream>>>(
        residual_ptr,
        ffn_out_ptr,
        gate_ptr,
        num_pairs);
    int64_t processed = num_pairs * 2;
    int64_t rem = total - processed;
    if (rem > 0) {
      mxfp8_residual_gate_2d_bf16_kernel<<<1, 1, 0, stream>>>(
          residual_ptr + processed,
          ffn_out_ptr + processed,
          gate_ptr + processed,
          rem);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  int blocks = static_cast<int>((total + kFusedThreads - 1) / kFusedThreads);
  mxfp8_residual_gate_2d_bf16_kernel<<<blocks, kFusedThreads, 0, stream>>>(
      residual_ptr,
      ffn_out_ptr,
      gate_ptr,
      total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace lightx2v_mxfp8_fused

void cutlass_scaled_mxfp8_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias) {

  auto const meta = lightx2v_mxfp8_fused::check_mxfp8_gemm_inputs(
      D, A, B, A_sf, B_sf, alpha, bias, "cutlass_scaled_mxfp8_mm_sm120");
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  runGemmMxfp8Sm120(D, A, B, A_sf, B_sf, alpha, bias, meta.m, meta.n, meta.k, stream);
}

void cutlass_scaled_mxfp8_mm_residual_gate_sm120(
    torch::Tensor& residual,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    torch::Tensor const& gate) {
  auto const meta = lightx2v_mxfp8_fused::check_mxfp8_gemm_inputs(
      residual, A, B, A_sf, B_sf, alpha, bias, "cutlass_scaled_mxfp8_mm_residual_gate_sm120");
  lightx2v_mxfp8_fused::check_mxfp8_residual_gate(
      residual, gate, "cutlass_scaled_mxfp8_mm_residual_gate_sm120");
  if (gate.dim() == 1) {
    at::cuda::CUDAGuard device_guard{(char)A.get_device()};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
    runGemmMxfp8ResidualGateSm120(
        residual, A, B, A_sf, B_sf, alpha, bias, gate, meta.m, meta.n, meta.k, stream);
    return;
  }
  auto ffn_out = torch::empty_like(residual);
  cutlass_scaled_mxfp8_mm_sm120(ffn_out, A, B, A_sf, B_sf, alpha, bias);
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  lightx2v_mxfp8_fused::launch_mxfp8_residual_gate_2d(residual, ffn_out, gate, stream);
}
