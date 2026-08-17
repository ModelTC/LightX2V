/***************************************************************************************************
 * Torch-callable wrapper around the cute_attn_analysis fused FMHA (example-06 kernel).
 *
 * Exposes sycl_kernels cute_fmha::sdp(q, k, v) -> o with the SAME signature/layout as
 * omni_xpu_kernel.sdp: q,k,v,o are [B, L, H, D] (B==1, D==128), fp16 or bf16,
 * XPU. PTL-H and BMG builds additionally expose a dense-BHLD D120 entry point
 * used by the validated ComfyUI workflow route.
 *
 * The kernel body is the CUTLASS-SYCL flash-attention-v2 forward used by
 * 06_xe_fmha_fwd.cpp (d=128, platform-selected tile/GRF/pipeline policy).
 * The initial PTL-H policy uses the correctness-validated BMG values; only the launch is changed:
 * instead of cutlass' global compat queue we submit onto torch's current XPU
 * queue (at::xpu::getCurrentXPUStream().queue()), with torch tensor data_ptr()s
 * as the operands, so the SYCL context matches torch's allocations.
 *
 * Launch glue pattern copied from sgl-kernel-xpu src/sycl/comm/common.h.
 **************************************************************************************************/
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/device_kernel.h"
#include "cute/util/compat.hpp"

#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cute_fmha_config.h"

using namespace cute;

namespace {

// ---- launch glue: submit cutlass device kernel onto torch's XPU queue --------
// (mirror of sgl-kernel-xpu src/sycl/comm/common.h::launch)
template <typename Kernel, int GrfSize>
class CuteFmhaKernelTag {};

template <typename Kernel, int GrfSize = 256>
static void launch_on_torch_queue(typename Kernel::Params params) {
  static_assert(GrfSize == 128 || GrfSize == 256, "GRF size must be 128 or 256");

  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  int smem_size = Kernel::SharedStorageSize;

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<GrfSize>};
  compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};

  syclex::launch_config config(policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](::sycl::handler& cgh) {
    auto KernelFunctor =
        compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<Kernel>>(cgh, policy, params);
    syclex::detail::LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())>
        ConfigAccess(config);
    cgh.parallel_for<CuteFmhaKernelTag<Kernel, GrfSize>>(
        ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
  };
  auto stream = at::xpu::getCurrentXPUStream();
  auto q = stream.queue();
  q.submit(cgf);
}

static int checked_int(int64_t value, const char* label) {
  TORCH_CHECK(
      value >= 0 && value <= std::numeric_limits<int>::max(), label,
      " exceeds the CUTE int32 index range: ", value);
  return static_cast<int>(value);
}

// ---- kernel type assembly (128-wide tile, example-06 PREFILL path) ----------
// KV tile = get<1>(ShapeQK). Default 32 (stock example-06). -DCUTE_FMHA_KV64
// switches to a 64-wide KV tile (fewer K-loop iters at large seq — omni uses 64).
template <
    typename Element,
    int PipelineStagesOverride = 0,
    int QTileOverride = 0,
    int SubgroupLayoutQOverride = 0,
    int MmaKOverride = 0,
    int VTileOverride = 0,
    int HeadDimOverride = 0>
struct D128TileKernel {
  using PlatformConfig = cute_fmha_config::ActiveConfig;
  static constexpr int QTile =
      QTileOverride > 0 ? QTileOverride : PlatformConfig::Q_TILE;
  static constexpr int SubgroupLayoutQ =
      SubgroupLayoutQOverride > 0
          ? SubgroupLayoutQOverride
          : PlatformConfig::SUBGROUP_LAYOUT_Q;
  static constexpr int MmaK =
      MmaKOverride > 0 ? MmaKOverride : PlatformConfig::MMA_K;
  static constexpr int VTile =
      VTileOverride > 0 ? VTileOverride : PlatformConfig::V_TILE;
  static constexpr int HeadDim =
      HeadDimOverride > 0 ? HeadDimOverride : PlatformConfig::HEAD_DIM;
#if defined(CUTE_FMHA_KV64)
  // KV tile = get<1>(ShapeQK) = 64. Per get_tiled_mma_pv, the PV tile must be
  // <TileQ, TileV, KVtile> — so ShapePV's K-dim (3rd) MUST equal 64, not 32.
  // TileV=32 -> VTiles = 128/32 = 4. (My earlier <256,32,32> broke the QK->PV
  // K-dim match and tripped the gemm.hpp static_assert.)
  static constexpr int KvTile = 64;
#else
  static constexpr int KvTile = PlatformConfig::KV_TILE;
#endif
  using ShapeQK = Shape<
      Int<QTile>, Int<KvTile>, Int<MmaK>>;
  using ShapePV = Shape<
      Int<QTile>, Int<VTile>, Int<KvTile>>;
  using ShapeOutput = Shape<
      Int<QTile>, Int<HeadDim>>;
  using SubgroupLayoutQK = Layout<
      Shape<Int<SubgroupLayoutQ>, _1, _1>>;
#ifdef CUTE_FMHA_STAGES
  static constexpr int PipelineStages = CUTE_FMHA_STAGES;
#else
  static constexpr int PipelineStages =
      PipelineStagesOverride > 0 ? PipelineStagesOverride
                                 : PlatformConfig::PIPELINE_STAGES;
#endif
  static constexpr int GrfSize = PlatformConfig::GRF_SIZE;

  using ElementQ = Element;
  using ElementK = Element;
  using ElementV = Element;
  using ElementO = Element;   // output dtype == input dtype (fp16/bf16)

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;

  static constexpr int SGTileQ =
      get<0>(shape_div(ShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Element>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using TiledMMAQK =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
  static constexpr int VTiles = get<1>(ShapeOutput{}) / get<1>(ShapePV{});

  static auto make_dummy(Element v, StrideQ s) {
    return make_tensor(make_gmem_ptr(&v), make_layout(repeat<rank_v<StrideQ>>(1), s));
  }
  using TensorQ = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideQ>>(1), StrideQ{})));
  using TensorK = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideK>>(1), StrideK{})));
  using TensorV = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideV>>(1), StrideV{})));
  using TensorO = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideO>>(1), StrideO{})));
  using TensorK_cache = TensorK;
  using TensorV_cache = TensorV;

  using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
      MainloopDispatchPolicy, /*Causal=*/false, /*CachedKV=*/false, /*PagedKV=*/false,
      TiledMMAQK, TiledMMAPV, VTiles,
      TensorQ, TensorK, TensorV, TensorK_cache, TensorV_cache,
      void, void, void, void, void>;

  using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      CollectiveMainloop, ShapeOutput, TensorO, void>;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<false>;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
      ProblemShapeType, CollectiveMainloop, CollectiveEpilogue,
      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
};

template <
    typename Element,
    int PipelineStagesOverride = 0,
    int QTileOverride = 0,
    int SubgroupLayoutQOverride = 0,
    int MmaKOverride = 0,
    int VTileOverride = 0,
    int HeadDimOverride = 0>
void run_d128_tile(
    const void* q_ptr, const void* k_ptr, const void* v_ptr, void* o_ptr,
    int B, int H, int Lq, int Lkv, int D, float scale,
    int64_t q_stride_seq = -1, int64_t q_stride_head = -1,
    int64_t q_stride_batch = -1, int64_t k_stride_seq = -1,
    int64_t k_stride_head = -1, int64_t k_stride_batch = -1,
    int64_t v_stride_seq = -1, int64_t v_stride_head = -1,
    int64_t v_stride_batch = -1, int64_t o_stride_seq = -1,
    int64_t o_stride_head = -1, int64_t o_stride_batch = -1) {
  using KT = D128TileKernel<
      Element,
      PipelineStagesOverride,
      QTileOverride,
      SubgroupLayoutQOverride,
      MmaKOverride,
      VTileOverride,
      HeadDimOverride>;
  using K    = typename KT::Kernel;
  using PS   = typename KT::ProblemShapeType;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  PS shape;
  shape.batch = B;
  shape.num_heads_q = H;
  shape.num_heads_kv = H;
  shape.seq_len_qo = Lq;   // cross-attention: Lq may differ from Lkv
  shape.seq_len_kv = Lkv;
  shape.seq_len_kv_cache = 0;
  shape.head_size_qk = D;
  shape.head_size_vo = D;

  // Logical cute modes are Q/K/O=(seq,dim,head,batch) and
  // V=(dim,seq,head,batch). Default strides consume contiguous BLHD. The D120
  // entry point supplies the actual dense BHLD/BLHD-backed strides so it can
  // match ComfyUI's mixed input layouts without materializing copies.
  const int HD = checked_int(static_cast<int64_t>(H) * D, "H*D");
  const int LqHD = checked_int(static_cast<int64_t>(Lq) * H * D, "Lq*H*D");
  const int LkvHD = checked_int(static_cast<int64_t>(Lkv) * H * D, "Lkv*H*D");
  if (q_stride_seq < 0) {
    q_stride_seq = HD;
    q_stride_head = D;
    q_stride_batch = LqHD;
    k_stride_seq = HD;
    k_stride_head = D;
    k_stride_batch = LkvHD;
    v_stride_seq = HD;
    v_stride_head = D;
    v_stride_batch = LkvHD;
    o_stride_seq = HD;
    o_stride_head = D;
    o_stride_batch = LqHD;
  }
  typename KT::StrideQ stride_Q =
      cute::make_stride(
          checked_int(q_stride_seq, "Q sequence stride"), _1{},
          checked_int(q_stride_head, "Q head stride"),
          checked_int(q_stride_batch, "Q batch stride"));
  typename KT::StrideK stride_K =
      cute::make_stride(
          checked_int(k_stride_seq, "K sequence stride"), _1{},
          checked_int(k_stride_head, "K head stride"),
          checked_int(k_stride_batch, "K batch stride"));
  typename KT::StrideV stride_V =
      cute::make_stride(
          _1{}, checked_int(v_stride_seq, "V sequence stride"),
          checked_int(v_stride_head, "V head stride"),
          checked_int(v_stride_batch, "V batch stride"));
  typename KT::StrideO stride_O =
      cute::make_stride(
          checked_int(o_stride_seq, "output sequence stride"), _1{},
          checked_int(o_stride_head, "output head stride"),
          checked_int(o_stride_batch, "output batch stride"));

  typename K::Arguments arguments{
      {
          shape,
          static_cast<const Element*>(q_ptr), stride_Q,
          static_cast<const Element*>(k_ptr), stride_K,
          static_cast<const Element*>(v_ptr), stride_V,
          static_cast<Element*>(o_ptr),       stride_O,
          nullptr, stride_K,   // k_cache
          nullptr, stride_V,   // v_cache
      },
      {scale, nullptr, 0, nullptr},
      {},
      hw_info};

  size_t workspace_size = K::get_workspace_size(arguments);
  auto opts = at::TensorOptions().dtype(at::kByte).device(at::kXPU);
  at::Tensor workspace = at::empty({(long)workspace_size}, opts);

  TORCH_CHECK(K::can_implement(arguments),
              "sycl_kernels cute_fmha: can_implement failed (bad problem shape)");
  K::initialize_workspace(arguments, workspace.data_ptr());
  auto kernel_params = K::to_underlying_arguments(arguments, workspace.data_ptr());
  launch_on_torch_queue<K, KT::GrfSize>(kernel_params);
}

// ---- public op --------------------------------------------------------------
at::Tensor sdp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "sycl_kernels cute_fmha: expect [B,L,H,D]");
  // All three operands must be on XPU and share q's dtype — the kernel takes raw
  // data_ptr()s and reinterprets them as q's element type, so a CPU tensor or a
  // dtype mismatch would feed invalid pointers / misread data.
  TORCH_CHECK(q.device().is_xpu() && k.device().is_xpu() && v.device().is_xpu(),
              "sycl_kernels cute_fmha: q, k, v must all be XPU tensors (got ",
              q.device(), ", ", k.device(), ", ", v.device(), ")");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "sycl_kernels cute_fmha: q, k, v must share dtype (got ",
              q.scalar_type(), ", ", k.scalar_type(), ", ", v.scalar_type(), ")");
  // Public layout is [B, L, H, D] (drop-in for omni_xpu_kernel.sdp). The kernel
  // reads this contiguous layout directly via custom strides (run_d128_tile), so no
  // permute/copy is needed — output is also [B, L, H, D].
  // q,k,v are [B, L, H, D]. The current scheduler is validated only for
  // self-attention; reject cross-attention instead of returning silently
  // inaccurate results. ComfyUI routes cross-attention to the ESIMD backend.
  const int B = checked_int(q.size(0), "batch");
  const int Lq = checked_int(q.size(1), "query length");
  const int H = checked_int(q.size(2), "head count");
  const int D = checked_int(q.size(3), "head dimension");
  const int Lkv = checked_int(k.size(1), "key/value length");
  TORCH_CHECK(B == 1, "sycl_kernels cute_fmha: only B==1 supported (got ", B, ")");
  TORCH_CHECK(D == 128, "sycl_kernels cute_fmha: only head_dim==128 supported (got ", D, ")");
  TORCH_CHECK(Lq == Lkv,
              "sycl_kernels cute_fmha: only self-attention with equal q/kv lengths is supported (got ",
              Lq, " and ", Lkv, ")");
  TORCH_CHECK(k.size(0) == B && v.size(0) == B, "sycl_kernels cute_fmha: batch mismatch");
  TORCH_CHECK(k.size(2) == H && v.size(2) == H,
              "sycl_kernels cute_fmha: q,k,v must share num_heads (got ", H, ",",
              k.size(2), ",", v.size(2), ")");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D,
              "sycl_kernels cute_fmha: q,k,v must share head_dim");
  TORCH_CHECK(v.size(1) == Lkv,
              "sycl_kernels cute_fmha: k,v seq_len must match (got ", Lkv, ",",
              v.size(1), ")");

  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  at::Tensor o = at::empty_like(qc);
  const float scale = 1.0f / std::sqrt((float)D);

  if (q.scalar_type() == at::kHalf) {
    run_d128_tile<cutlass::half_t>(
        qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), B, H, Lq,
        Lkv, D, scale);
  } else if (q.scalar_type() == at::kBFloat16) {
#if defined(OMNI_XPU_ARCH_BMG)
    // Z-Image Turbo's canonical 1024x1024 workflow uses this exact self-
    // attention contract.  On BMG, MMA-K16 reduces both forward and reverse
    // publication micro time while Krea2 L4192/H48 remains fastest at the
    // platform-default MMA-K32.  Keep the override exact instead of changing
    // the platform-wide D128 policy.
    if (Lq == 4128 && H == 30) {
      run_d128_tile<cutlass::bfloat16_t, 0, 0, 0, 16>(
          qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), B, H, Lq,
          Lkv, D, scale);
      return o;
    }
#endif
    run_d128_tile<cutlass::bfloat16_t>(
        qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), B, H, Lq,
        Lkv, D, scale);
  } else {
    TORCH_CHECK(false, "sycl_kernels cute_fmha: only fp16/bf16 supported");
  }
  return o;
}

}  // namespace

TORCH_LIBRARY(sycl_kernels_cute, m) {
  m.def("sdp(Tensor q, Tensor k, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(sycl_kernels_cute, XPU, m) {
  m.impl("sdp", &sdp);
}
