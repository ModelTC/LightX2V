#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

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
template <typename T>
struct PyTorchTanhGelu {
  CUTLASS_HOST_DEVICE
  T operator()(T const& x) const {
    T const beta = T(0.7978845608028654);
    T const kappa = T(0.044715);
    return T(0.5) * x * (T(1) + ::tanhf(beta * (x + kappa * x * x * x)));
  }
};

template <typename T, int N>
struct PyTorchTanhGelu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& input) const {
    Array<T, N> output;
    PyTorchTanhGelu<T> gelu;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < N; ++index) {
      output[index] = gelu(input[index]);
    }
    return output;
  }
};


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
struct LinCombPerColBiasBf16GateResidual
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
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
using Sm90LinCombPerColBiasBf16GateResidual =
  // ElementOutput at each nested compute preserves the original BF16
  // materialization boundaries: GEMM+bias, gate multiply, then residual add.
  // Parent nodes convert those BF16 fragments back to ElementCompute.
  Sm90EVT<Sm90Compute<plus, ElementOutput, ElementCompute, RoundStyle>,
    Sm90SrcFetch<ElementSource>,
    Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>,
        Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
        Sm90AccFetch,
        Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias>
      >,
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementGate, ElementCompute, Stride<_0,_1,int64_t>, AlignmentGate>
    >
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
    fusion::LinCombPerColBiasBf16GateResidual<
      ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource, ElementScalar,
      AlignmentGate, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerColBiasBf16GateResidual<
      CtaTileShapeMNK, ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource,
      ElementScalar, AlignmentGate, AlignmentBias, RoundStyle> {
  using Impl = Sm90LinCombPerColBiasBf16GateResidual<
    CtaTileShapeMNK, ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource,
    ElementScalar, AlignmentGate, AlignmentBias, RoundStyle>;
  using Operation = fusion::LinCombPerColBiasBf16GateResidual<
    ElementOutput, ElementCompute, ElementGate, ElementBias, ElementSource, ElementScalar,
    AlignmentGate, AlignmentBias, RoundStyle>;

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
      return {
        {},
        {
          {{{alpha}, {alpha_ptr}, {dAlpha}}, {}, {bias_ptr, ElementBias(0), dBias}, {}},
          {gate_ptr, ElementGate(0), dGate},
          {}
        },
        {}
      };
    }
  };

  using Impl::Impl;
};

template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasBf16Gelu
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerColBiasSupported = true;
};

template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColBiasBf16Gelu =
  // Match the unfused path's BF16 materialization before GELU.
  Sm90EVT<Sm90Compute<PyTorchTanhGelu, ElementOutput, ElementCompute, RoundStyle>,
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
      Sm90AccFetch,
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias>
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasBf16Gelu<
      ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar,
      AlignmentBias, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerColBiasBf16Gelu<
      CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource,
      ElementScalar, AlignmentBias, RoundStyle> {
  using Impl = Sm90LinCombPerColBiasBf16Gelu<
    CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource,
    ElementScalar, AlignmentBias, RoundStyle>;
  using Operation = fusion::LinCombPerColBiasBf16Gelu<
    ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar,
    AlignmentBias, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    using StrideAlpha = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return {
        {{{alpha}, {alpha_ptr}, {dAlpha}}, {}, {bias_ptr, ElementBias(0), dBias}, {}},
        {}
      };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion


template <
    class ThreadBlockShape_,
    class ClusterShape_,
    class MainloopSchedule_,
    class EpilogueSchedule_>
struct Fp4GemmSm120Config {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
    using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
    using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
    using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
    using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
    using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    // Kernel functional config
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
#if defined(LIGHTX2V_THOR_NVFP4_ONLY)
    using ArchTag             = cutlass::arch::Sm100;
#else
    using ArchTag             = cutlass::arch::Sm120;
#endif
    using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

    // Kernel Perf config
    using ThreadBlockShape    = ThreadBlockShape_;
    using ClusterShape        = ClusterShape_;

    // use per-column bias, i.e. every column has different bias
    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        EpilogueSchedule_,
        EVTOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule_
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

using Fp4GemmSm120 = Fp4GemmSm120Config<
    Shape<_128,_128,_128>, Shape<_1,_1,_1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto>;

using Fp4GemmSm120Wan22Ffn2 = Fp4GemmSm120Config<
    Shape<_256,_256,_256>, Shape<_2,_1,_1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized2SmNvf4>;

template <
    class ThreadBlockShape_,
    class ClusterShape_,
    class MainloopSchedule_,
    class EpilogueSchedule_>
struct Fp4GemmResidualGateSm120Config {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator = float;
#if defined(LIGHTX2V_THOR_NVFP4_ONLY)
    using ArchTag = cutlass::arch::Sm100;
#else
    using ArchTag = cutlass::arch::Sm120;
#endif
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using ThreadBlockShape = ThreadBlockShape_;
    using ClusterShape = ClusterShape_;

    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBiasBf16GateResidual<
        ElementD, ElementAccumulator, ElementD, ElementD, ElementC, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        EpilogueSchedule_,
        EVTOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

using Fp4GemmResidualGateSm120 = Fp4GemmResidualGateSm120Config<
    Shape<_128,_128,_128>, Shape<_1,_1,_1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto>;

using Fp4GemmResidualGateSm120Wan22Ffn2 = Fp4GemmResidualGateSm120Config<
    Shape<_256,_256,_256>, Shape<_2,_1,_1>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized2SmNvf4>;

struct Fp4GemmGeluSm120 {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator = float;
#if defined(LIGHTX2V_THOR_NVFP4_ONLY)
    using ArchTag = cutlass::arch::Sm100;
#else
    using ArchTag = cutlass::arch::Sm120;
#endif
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using ThreadBlockShape = Shape<_256,_256,_256>;
    using ClusterShape = Shape<_2,_1,_1>;

    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBiasBf16Gelu<
        ElementD, ElementAccumulator, ElementD, ElementC, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized2SmNvf4,
        EVTOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// Populates a Gemm::Arguments structure from the given commandline options
typename Fp4GemmSm120::Gemm::Arguments args_from_options_nvfp4_nvfp4(
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
  using Sm1xxBlkScaledConfig = typename Fp4GemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  if (bias){
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename Fp4GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Fp4GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Fp4GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Fp4GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Fp4GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    // static const float beta_zero = 0.0f;
    // fusion_args.beta_ptr = &beta_zero;
    fusion_args.bias_ptr = static_cast<Fp4GemmSm120::Gemm::ElementC const*>(bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    return arguments;
  } else {
    typename Fp4GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Fp4GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Fp4GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Fp4GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Fp4GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    // static const float beta_zero = 0.0f;
    // fusion_args.beta_ptr = &beta_zero;
    return arguments;
  }
}


void runGemmNvfp4Sm120(
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
  typename Fp4GemmSm120::Gemm gemm;

  auto arguments = args_from_options_nvfp4_nvfp4(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  auto beta_dev = torch::zeros({1}, torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(A.device()));
  arguments.epilogue.thread.beta_ptr =
      static_cast<float const*>(beta_dev.data_ptr());
  size_t workspace_size = Fp4GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

template <class GemmConfig>
typename GemmConfig::Gemm::Arguments
args_from_options_nvfp4_split_n_stride_residual_gate(
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
    int64_t K,
    int64_t split_n_parts) {
  using Sm1xxBlkScaledConfig =
      typename GemmConfig::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  int batch_count = static_cast<int>(split_n_parts);
  int shard_n = n / batch_count;

  auto stride_A = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideA{}, {m, k, batch_count});
  auto stride_B = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideB{}, {shard_n, k, batch_count});
  auto stride_D = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideD{}, {m, shard_n, batch_count});
  cute::get<2>(stride_A) = 0;
  cute::get<0>(stride_D) = n;
  cute::get<2>(stride_D) = shard_n;

  auto problem_shape = cute::make_shape(m, shard_n, k, batch_count);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);
  cute::get<2, 1>(cute::stride(layout_SFA)) = 0;

  typename GemmConfig::Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kBatched,
    problem_shape,
    {
      static_cast<typename GemmConfig::Gemm::ElementA const*>(A.data_ptr()),
      stride_A,
      static_cast<typename GemmConfig::Gemm::ElementB const*>(B.data_ptr()),
      stride_B,
      static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
      layout_SFA,
      static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
      layout_SFB
    },
    {
      {},
      static_cast<typename GemmConfig::ElementC const*>(residual.data_ptr()),
      stride_D,
      static_cast<typename GemmConfig::ElementD*>(residual.data_ptr()),
      stride_D
    }
  };

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
  fusion_args.gate_ptr = static_cast<typename GemmConfig::ElementD const*>(gate.data_ptr());
  using StrideGate = Stride<cutlass::_0, cutlass::_1, int64_t>;
  auto gate_stride = StrideGate{};
  cute::get<2>(gate_stride) = shard_n;
  fusion_args.dGate = gate_stride;
  if (bias) {
    fusion_args.bias_ptr =
        static_cast<typename GemmConfig::ElementD const*>(bias->data_ptr());
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;
    auto bias_stride = StrideBias{};
    cute::get<2>(bias_stride) = shard_n;
    fusion_args.dBias = bias_stride;
  }
  return arguments;
}

template <class GemmConfig>
void runGemmNvfp4SplitNStrideResidualGateSm120(
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
    int64_t split_n_parts,
    cudaStream_t stream) {
  typename GemmConfig::Gemm gemm;
  auto arguments = args_from_options_nvfp4_split_n_stride_residual_gate<GemmConfig>(
      residual, A, B, A_sf, B_sf, alpha, bias, gate, m, n, k, split_n_parts);
  size_t workspace_size = GemmConfig::Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      workspace_size, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

typename Fp4GemmGeluSm120::Gemm::Arguments
args_from_options_nvfp4_split_n_stride_gelu(
    at::Tensor& output,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t split_n_parts) {
  using Sm1xxBlkScaledConfig =
      typename Fp4GemmGeluSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  int batch_count = static_cast<int>(split_n_parts);
  int shard_n = n / batch_count;

  auto stride_A = cutlass::make_cute_packed_stride(
      Fp4GemmGeluSm120::StrideA{}, {m, k, batch_count});
  auto stride_B = cutlass::make_cute_packed_stride(
      Fp4GemmGeluSm120::StrideB{}, {shard_n, k, batch_count});
  auto stride_D = cutlass::make_cute_packed_stride(
      Fp4GemmGeluSm120::StrideD{}, {m, shard_n, batch_count});
  cute::get<2>(stride_A) = 0;
  cute::get<0>(stride_D) = n;
  cute::get<2>(stride_D) = shard_n;

  auto problem_shape = cute::make_shape(m, shard_n, k, batch_count);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);
  cute::get<2, 1>(cute::stride(layout_SFA)) = 0;

  typename Fp4GemmGeluSm120::Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kBatched,
    problem_shape,
    {
      static_cast<Fp4GemmGeluSm120::Gemm::ElementA const*>(A.data_ptr()),
      stride_A,
      static_cast<Fp4GemmGeluSm120::Gemm::ElementB const*>(B.data_ptr()),
      stride_B,
      static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
      layout_SFA,
      static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
      layout_SFB
    },
    {
      {},
      static_cast<Fp4GemmGeluSm120::ElementC const*>(output.data_ptr()),
      stride_D,
      static_cast<Fp4GemmGeluSm120::ElementD*>(output.data_ptr()),
      stride_D
    }
  };

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
  if (bias) {
    fusion_args.bias_ptr = static_cast<Fp4GemmGeluSm120::ElementD const*>(bias->data_ptr());
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;
    auto bias_stride = StrideBias{};
    cute::get<2>(bias_stride) = shard_n;
    fusion_args.dBias = bias_stride;
  }
  return arguments;
}

void runGemmNvfp4SplitNStrideGeluSm120(
    at::Tensor& output,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t split_n_parts,
    cudaStream_t stream) {
  typename Fp4GemmGeluSm120::Gemm gemm;
  auto arguments = args_from_options_nvfp4_split_n_stride_gelu(
      output, A, B, A_sf, B_sf, alpha, bias, m, n, k, split_n_parts);
  size_t workspace_size = Fp4GemmGeluSm120::Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      workspace_size, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void cutlass_scaled_nvfp4_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias) {

  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");


  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(
      A.sizes()[1] == B.sizes()[1],
      "a and b shapes cannot be multiplied (",
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
  auto const k = A.sizes()[1] * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(
      k % alignment == 0,
      "Expected k to be divisible by ",
      alignment,
      ", but got a shape: (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      "), k: ",
      k,
      ".");
  TORCH_CHECK(
      n % alignment == 0,
      "Expected n to be divisible by ",
      alignment,
      ", but got b shape: (",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

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

  auto out_dtype = D.dtype();
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  runGemmNvfp4Sm120(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
}


// Keep split-N stride argument construction and execution isolated from the
// regular NVFP4 operator so changes here cannot alter its behavior.
// prepare the calculation parameters for the gemm
template <class GemmConfig>
typename GemmConfig::Gemm::Arguments args_from_options_nvfp4_nvfp4_split_n_stride(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t split_n_parts) {
  using Sm1xxBlkScaledConfig = typename GemmConfig::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  int batch_count = static_cast<int>(split_n_parts);
  int shard_n = n / batch_count;

  auto stride_A = cutlass::make_cute_packed_stride(typename GemmConfig::StrideA{}, {m, k, batch_count});
  auto stride_B = cutlass::make_cute_packed_stride(typename GemmConfig::StrideB{}, {shard_n, k, batch_count});
  auto stride_D = cutlass::make_cute_packed_stride(typename GemmConfig::StrideD{}, {m, shard_n, batch_count});

  // Broadcast A across batches. B batches are consecutive N shards.
  cute::get<2>(stride_A) = 0;
  // D is one [M, N] tensor: batch l starts at column l * shard_n.
  cute::get<0>(stride_D) = n;
  cute::get<2>(stride_D) = shard_n;

  auto problem_shape = cute::make_shape(m, shard_n, k, batch_count);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);
  // SFB remains packed by batch. Broadcast only SFA's nested batch mode.
  cute::get<2, 1>(cute::stride(layout_SFA)) = 0;

  if (bias) {
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename GemmConfig::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      problem_shape,
      {// Mainloop arguments
       static_cast<typename GemmConfig::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<typename GemmConfig::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<typename GemmConfig::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<typename GemmConfig::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    fusion_args.bias_ptr = static_cast<typename GemmConfig::Gemm::ElementC const*>(bias->data_ptr());
    auto stride_bias = StrideBias{};
    cute::get<2>(stride_bias) = shard_n;
    fusion_args.dBias = stride_bias;
    return arguments;
  }
  else
  {
    typename GemmConfig::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      problem_shape,
      {// Mainloop arguments
       static_cast<typename GemmConfig::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<typename GemmConfig::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<typename GemmConfig::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<typename GemmConfig::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    return arguments;
  }
}

// implement the gemm for nvfp4splitnstride
template <class GemmConfig>
void runGemmNvfp4SplitNStrideSm120(
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
    int64_t split_n_parts,
    cudaStream_t stream) {
  typename GemmConfig::Gemm gemm;

  auto arguments = args_from_options_nvfp4_nvfp4_split_n_stride<GemmConfig>(
      D, A, B, A_sf, B_sf, alpha, bias, m, n, k, split_n_parts);
  auto beta_dev = torch::zeros({1}, torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(A.device()));
  arguments.epilogue.thread.beta_ptr =
      static_cast<float const*>(beta_dev.data_ptr());
  size_t workspace_size = GemmConfig::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

// check the inputs and run the NVFP4 split-N stride GEMM kernel
void check_nvfp4_split_n_stride_inputs(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t split_n_parts) {
  CHECK_INPUT(D, at::ScalarType::BFloat16, "out");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(D.dim() == 2, "out must be a matrix");
  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(
      A.sizes()[1] == B.sizes()[1],
      "a and b shapes cannot be multiplied (",
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
  auto const k = A.sizes()[1] * 2;

  TORCH_CHECK(
      D.sizes()[0] == m && D.sizes()[1] == n,
      "out must have shape (",
      m,
      "x",
      n,
      "), but got (",
      D.sizes()[0],
      "x",
      D.sizes()[1],
      ")");
  if (bias) {
    auto const& bias_tensor = bias.value();
    CHECK_INPUT(bias_tensor, at::ScalarType::BFloat16, "bias");
    TORCH_CHECK(bias_tensor.numel() == n, "bias must contain ", n, " elements, but got ", bias_tensor.numel());
  }

  constexpr int alignment = 32;
  TORCH_CHECK(
      k % alignment == 0,
      "Expected k to be divisible by ",
      alignment,
      ", but got a shape: (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      "), k: ",
      k,
      ".");
  TORCH_CHECK(
      n % alignment == 0,
      "Expected n to be divisible by ",
      alignment,
      ", but got b shape: (",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

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

  TORCH_CHECK(split_n_parts >= 2, "split_n_parts must be at least 2, but got ", split_n_parts);
  TORCH_CHECK(
      n % split_n_parts == 0,
      "Expected n to be divisible by split_n_parts, but got n=",
      n,
      " and split_n_parts=",
      split_n_parts);
  TORCH_CHECK(
      (n / split_n_parts) % 128 == 0,
      "Each NVFP4 split-N shard must be divisible by 128 for the swizzled scale layout, but got shard n=",
      n / split_n_parts);

}

void cutlass_scaled_nvfp4_mm_split_n_stride_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t split_n_parts) {
  check_nvfp4_split_n_stride_inputs(D, A, B, A_sf, B_sf, alpha, bias, split_n_parts);

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  if (m == 75348 && n == 5120 && k == 13824 && split_n_parts == 2) {
    runGemmNvfp4SplitNStrideSm120<Fp4GemmSm120Wan22Ffn2>(
        D, A, B, A_sf, B_sf, alpha, bias, m, n, k, split_n_parts, stream);
  } else {
    runGemmNvfp4SplitNStrideSm120<Fp4GemmSm120>(
        D, A, B, A_sf, B_sf, alpha, bias, m, n, k, split_n_parts, stream);
  }
}

void cutlass_scaled_nvfp4_mm_split_n_stride_gelu_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t split_n_parts) {
  check_nvfp4_split_n_stride_inputs(D, A, B, A_sf, B_sf, alpha, bias, split_n_parts);

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  runGemmNvfp4SplitNStrideGeluSm120(
      D, A, B, A_sf, B_sf, alpha, bias, m, n, k, split_n_parts, stream);
}

void cutlass_scaled_nvfp4_mm_split_n_stride_residual_gate_sm120(
    torch::Tensor& residual,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    torch::Tensor const& gate,
    int64_t split_n_parts) {
  check_nvfp4_split_n_stride_inputs(
      residual, A, B, A_sf, B_sf, alpha, bias, split_n_parts);
  CHECK_INPUT(gate, at::ScalarType::BFloat16, "gate");
  TORCH_CHECK(gate.device() == residual.device(), "gate and residual must be on the same CUDA device");
  TORCH_CHECK(gate.dim() == 1, "gate must be a 1D per-column tensor");
  TORCH_CHECK(gate.sizes()[0] == residual.sizes()[1], "gate size must match residual columns");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  if (m == 75348 && n == 5120 && k == 13824 && split_n_parts == 2) {
    runGemmNvfp4SplitNStrideResidualGateSm120<Fp4GemmResidualGateSm120Wan22Ffn2>(
        residual, A, B, A_sf, B_sf, alpha, bias, gate, m, n, k, split_n_parts, stream);
  } else {
    runGemmNvfp4SplitNStrideResidualGateSm120<Fp4GemmResidualGateSm120>(
        residual, A, B, A_sf, B_sf, alpha, bias, gate, m, n, k, split_n_parts, stream);
  }
}
