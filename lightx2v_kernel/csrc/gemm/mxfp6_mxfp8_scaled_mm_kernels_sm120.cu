#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
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


struct Mxfp6Mxfp8GemmSm120 {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
    using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = 16;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for B matrix operand
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


// Populates a Gemm::Arguments structure from the given commandline options
typename Mxfp6Mxfp8GemmSm120::Gemm::Arguments args_from_options_mxfp6_mxfp8(
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
  using Sm1xxBlkScaledConfig = typename Mxfp6Mxfp8GemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(Mxfp6Mxfp8GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Mxfp6Mxfp8GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Mxfp6Mxfp8GemmSm120::StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  if (bias){
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename Mxfp6Mxfp8GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue8m0_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue8m0_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    fusion_args.bias_ptr = static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementC const*>(bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    return arguments;
  } else {
    typename Mxfp6Mxfp8GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue8m0_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue8m0_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Mxfp6Mxfp8GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    return arguments;
  }
}


void runGemmMxfp6Mxfp8Sm120(
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
  typename Mxfp6Mxfp8GemmSm120::Gemm gemm;

  auto arguments = args_from_options_mxfp6_mxfp8(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  size_t workspace_size = Mxfp6Mxfp8GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}


constexpr auto FP6_FP8_TYPE = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e8m0fnu;

void cutlass_scaled_mxfp6_mxfp8_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias) {

  CHECK_INPUT(A, FP6_FP8_TYPE, "a");
  CHECK_INPUT(B, FP6_FP8_TYPE, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");


  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
//   TORCH_CHECK(
//       A.sizes()[1] == B.sizes()[1],
//       "a and b shapes cannot be multiplied (",
//       A.sizes()[0],
//       "x",
//       A.sizes()[1],
//       " and ",
//       B.sizes()[0],
//       "x",
//       B.sizes()[1],
//       ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1];

  constexpr int alignment_a = 16;
  constexpr int alignment_b = 128;
  TORCH_CHECK(
      k % alignment_a == 0,
      "Expected k to be divisible by ",
      alignment_a,
      ", but got a shape: (",
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
  int rounded_k = round_up(k / 32, 4);

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

  runGemmMxfp6Mxfp8Sm120(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
}
