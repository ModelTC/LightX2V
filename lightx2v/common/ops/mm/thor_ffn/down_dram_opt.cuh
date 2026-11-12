// Hand-written persistent 2-SM (cta_group::2) tcgen05 BF16 GEMM + bias for Thor (sm_110a), DRAM-efficiency revision.
//   D[M,N] (bf16 row-major) = bf16( X[M,K] (row-major) * W[N,K]^T + bias[N] ), FP32 accumulation, FP32 bias add.
// Per CTA (320 threads): w0 TMA producer, w1 MMA issuer (leader CTA only), w2..w9 epilogue.
// One CTA pair = one 256 x 512 output tile at a time, one 512-column TMEM accumulator, 2-stage K128 pipeline.
//
// Changes vs optimization/down_hand (all driven by profiling, see README.md):
//  * Operand loads are raw 4D TMA boxes {64 k, 8 rows, 2 k-atoms, 16 row groups}. Thor DRAM streams at ~150 GB/s
//    when every row is fetched in 128 B pieces (the inner extent of a SW128 2D box) but ~200 GB/s when 256 B of a
//    row are fetched together. The smem layout is K-atom-fastest ([8-row group][64-wide K atom][8 rows x 128 B]):
//    still canonical SW128 K-major atoms for UMMA, only the 8-row-group stride doubles.
//  * TILE_K 128 / 2 stages (the 4D box needs two K atoms per stage; smem holds two 96 KB stages).
//  * Epilogue: 8 warps drain the accumulator into packed BF16 registers and release TMEM before any global store,
//    so stores overlap the next tile's MMAs (was: TMEM held until the last store, ~22 us MMA idle per tile).
//  * A (activations) loads at normal L2 priority: evict_last on the 35 MB row group overfilled the 32 MB L2.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
using namespace cute;

using TA = cutlass::bfloat16_t;
constexpr int TILE_M = 256, TILE_N = 512, TILE_K = 128;
constexpr int KA = TILE_K / 64;                           // 64-element (128 B) SW128 atoms per stage
constexpr int STAGES = 2;
constexpr int EPI_WARPS = 8;                              // warps 2..9: 2 per TMEM lane quadrant
constexpr int NTHREADS = 64 + 32 * EPI_WARPS;
constexpr int EPI_COLS = TILE_N / (EPI_WARPS / 4);        // accumulator columns per epilogue warp
constexpr int NCLUST = 10, PER_CLUSTER = 7;               // 70 tiles (7 row x 10 column tiles), 7 per CTA pair

struct Sched {
  const int* table;   // table[i * NCLUST + c] = row | n_tile << 8 | reverse_K << 16 (wave-major)
  uint64_t hint_a, hint_b;
};

// TMA load maps. a4/b4: 4D {64, 8, K/64, rows/8} with box {64, 8, KA, 16} (128 rows). The 128-row block of A that
// crosses M uses a4t (box over its full 8-row groups) + a3 (3D {64, M, K/64}, box {64, 8, KA}: bounds-checks the row
// index and zero fills past M) for the remaining groups, so the byte count per stage is unchanged.
struct LoadMaps { CUtensorMap a4, a4t, a3, b4; };

template <class ASL, class BSL>
struct Smem {
  alignas(1024) ArrayEngine<TA, cosize_v<ASL>> A;
  alignas(1024) ArrayEngine<TA, cosize_v<BSL>> B;
  alignas(16) uint64_t full[STAGES];
  alignas(16) uint64_t empty[STAGES];
  alignas(16) uint64_t tmem_full;
  alignas(16) uint64_t tmem_empty;
  alignas(16) uint32_t tmem_ptr;
};

template <class SharedStorage, class TiledMMA, class ASmemLayout, class BSmemLayout>
__global__ __launch_bounds__(NTHREADS, 1) void
hand_gemm(TiledMMA tiled_mma, CUTE_GRID_CONSTANT LoadMaps const lm,
          TA const* __restrict__ bias, TA* __restrict__ D, int M, int N, int K, Sched sp, ASmemLayout, BSmemLayout)
{
  extern __shared__ char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cute::block_rank_in_cluster();
  const int cluster = blockIdx.x >> 1;
  const bool leader = (rank == 0);
  const int KB = K / TILE_K;

  Tensor sA = make_tensor(make_smem_ptr(ss.A.begin()), ASmemLayout{});   // (MmaA, NumM, NumK, STAGES)
  Tensor sB = make_tensor(make_smem_ptr(ss.B.begin()), BSmemLayout{});
  ThrMMA cta_mma = tiled_mma.get_slice(rank);
  Tensor tCrA = cta_mma.make_fragment_A(sA);
  Tensor tCrB = cta_mma.make_fragment_B(sB);
  Tensor gC_shape = make_tensor(make_gmem_ptr(D), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), LayoutRight{}));
  Tensor tCtAcc = cta_mma.make_fragment_C(cta_mma.partition_C(gC_shape));

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_alloc{};
  if (warp == 0) tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns, &ss.tmem_ptr);
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { cute::initialize_barrier(ss.full[s], 1); cute::initialize_barrier(ss.empty[s], 1); }
    cute::initialize_barrier(ss.tmem_full, 1);
    cute::initialize_barrier(ss.tmem_empty, 2 * EPI_WARPS);   // epilogue warps of both CTAs
  }
  cute::cluster_sync();
  const uint32_t tmem_base = ss.tmem_ptr;
  tCtAcc.data() = tmem_base;

  constexpr int TMA_BYTES = 2 * (128 * TILE_K * 2 + 256 * TILE_K * 2);   // both CTAs: A half + B half each

  if (warp == 0) {
    // ---------------- TMA producer (both CTAs; completion is counted on the leader's barrier) ----------------
    int stage = 0, phase = 1;
    for (int i = 0; i < PER_CLUSTER; ++i) {
      const int e = sp.table[i * NCLUST + cluster];
      const int row = e & 255, nt = (e >> 8) & 255;
      const bool rev = (e >> 16) & 1;
      const int arow = row * TILE_M + int(rank) * 128;
      const int full_groups = (M - arow) / 8;    // >= 16: whole block in range
      for (int kb = 0; kb < KB; ++kb) {
        const int k = rev ? (KB - 1 - kb) : kb;
        cutlass::arch::ClusterBarrier::wait(&ss.empty[stage], phase);
        if (lane == 0) {
          if (leader) cute::set_barrier_transaction_bytes(ss.full[stage], TMA_BYTES);
          TA* a_dst = ss.A.begin() + stage * (128 * TILE_K);
          TA* b_dst = ss.B.begin() + stage * (256 * TILE_K);
          if (full_groups >= 16) {
            cute::SM100_TMA_2SM_LOAD_4D::copy(&lm.a4, &ss.full[stage], sp.hint_a, a_dst, 0, 0, k * KA, arow / 8);
          } else {
            if (full_groups > 0) cute::SM100_TMA_2SM_LOAD_4D::copy(&lm.a4t, &ss.full[stage], sp.hint_a, a_dst, 0, 0, k * KA, arow / 8);
            for (int g = full_groups > 0 ? full_groups : 0; g < 16; ++g)
              cute::SM100_TMA_2SM_LOAD_3D::copy(&lm.a3, &ss.full[stage], sp.hint_a, a_dst + g * 8 * TILE_K, 0, arow + g * 8, k * KA);
          }
          // B (W) per CTA: rows [nt*512 + j*256 + rank*128, +128) for the two N256 UMMA panels j.
          for (int j = 0; j < 2; ++j)
            cute::SM100_TMA_2SM_LOAD_4D::copy(&lm.b4, &ss.full[stage], sp.hint_b, b_dst + j * 128 * TILE_K, 0, 0, k * KA,
                                              (nt * TILE_N + j * 256 + int(rank) * 128) / 8);
        }
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if (warp == 1 && leader) {
    // ---------------- MMA issuer (leader CTA of the pair) ----------------
    int stage = 0, phase = 0;
    for (int i = 0; i < PER_CLUSTER; ++i) {
      cutlass::arch::ClusterBarrier::wait(&ss.tmem_empty, ((i & 1) ^ 1));
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      for (int kb = 0; kb < KB; ++kb) {
        cutlass::arch::ClusterBarrier::wait(&ss.full[stage], phase);
        CUTE_UNROLL
        for (int kk = 0; kk < size<2>(tCrA); ++kk) {
          gemm(tiled_mma, tCrA(_, _, kk, stage), tCrB(_, _, kk, stage), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&ss.empty[stage], uint16_t(0b11));
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(&ss.tmem_full, uint16_t(0b11));
    }
  } else if (warp >= 2) {
    // ---------------- epilogue: TMEM -> regs (+bias, bf16) -> release TMEM -> global ----------------
    const int q = warp & 3, h = (warp - 2) >> 2;            // TMEM lane quadrant, column half
    const int row_in_tile = int(rank) * 128 + q * 32 + lane;
    for (int i = 0; i < PER_CLUSTER; ++i) {
      const int e = sp.table[i * NCLUST + cluster];
      const int row = e & 255, nt = (e >> 8) & 255;
      cutlass::arch::ClusterBarrier::wait(&ss.tmem_full, (i & 1));
      cutlass::arch::fence_view_async_tmem_load();
      const long grow = (long)row * TILE_M + row_in_tile;
      const int n0 = nt * TILE_N + h * EPI_COLS;
      uint32_t pk[EPI_COLS / 2];
      CUTE_UNROLL
      for (int c = 0; c < EPI_COLS / 32; ++c) {
        uint32_t r[32];
        const uint32_t addr = tmem_base + ((uint32_t(q) * 32u) << 16) + uint32_t(h * EPI_COLS + c * 32);
        SM100_TMEM_LOAD_32dp32b32x::copy(addr, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
            r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19],
            r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
        const uint4* bp = reinterpret_cast<const uint4*>(bias + n0 + c * 32);
        uint4 b4[4] = {__ldg(bp), __ldg(bp + 1), __ldg(bp + 2), __ldg(bp + 3)};
        const uint32_t* bw = reinterpret_cast<const uint32_t*>(b4);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int j = 0; j < 16; ++j) {
          float lo = __uint_as_float(r[2 * j]) + __uint_as_float(bw[j] << 16);
          float hi = __uint_as_float(r[2 * j + 1]) + __uint_as_float(bw[j] & 0xffff0000u);
          __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
          pk[c * 16 + j] = *reinterpret_cast<uint32_t*>(&v);
        }
      }
      __syncwarp();
      if (lane == 0) cutlass::arch::ClusterBarrier::arrive(&ss.tmem_empty, 0u, 1u);   // accumulator free for the next tile
      if (grow < M) {
        uint4* dp = reinterpret_cast<uint4*>(D + grow * (long)N + n0);
        CUTE_UNROLL
        for (int j = 0; j < EPI_COLS / 8; ++j) dp[j] = make_uint4(pk[4 * j], pk[4 * j + 1], pk[4 * j + 2], pk[4 * j + 3]);
      }
    }
  }

  cute::cluster_sync();   // MMAs/TMEM of the peer are done before anything is released
  if (warp == 0) tmem_alloc.free(tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
}

// ------------------------------------------------------------------ host
#include <functional>
struct Handle {
  int* dtable = nullptr;
  std::function<cudaError_t(cudaStream_t)> run;
};

static uint64_t hint_value(int h) {
  return h == 1 ? uint64_t(cute::TMA::CacheHintSm100::EVICT_FIRST)
       : h == 2 ? uint64_t(cute::TMA::CacheHintSm100::EVICT_LAST) : uint64_t(cute::TMA::CacheHintSm100::EVICT_NORMAL);
}

static bool make_plan(Handle* h, TA const* x, TA const* w, TA const* bias, TA* y, int M, int N, int K, Sched sp) {
  auto tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<TA, TA, float, 256, 256, UMMA::Major::K, UMMA::Major::K>{});
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<TILE_M>{}, Int<TILE_K>{}));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<TILE_N>{}, Int<TILE_K>{}));
  // K-atom-fastest SW128 K-major layouts: [stage][(B: N panel)][8-row group][64-wide K atom][8 rows x 128 B].
  auto sA_s = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TA>{}, append(mma_shape_A, Int<STAGES>{}), Step<_2, _1, _3>{});
  auto sB_s = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TA>{}, append(mma_shape_B, Int<STAGES>{}), Step<_2, _1, _3>{});
  static_assert(cosize_v<decltype(sA_s)> == STAGES * 128 * TILE_K, "A stage layout");
  static_assert(cosize_v<decltype(sB_s)> == STAGES * 256 * TILE_K, "B stage layout");
  using SMEM = Smem<decltype(sA_s), decltype(sB_s)>;

  LoadMaps lm{};
  auto enc = [](CUtensorMap* m, void const* p, int r, const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box) {
    cuuint32_t es[5] = {1, 1, 1, 1, 1};
    return cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, r, const_cast<void*>(p), dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
  };
  const cuuint64_t st4[3] = {cuuint64_t(K) * 2, 128, cuuint64_t(K) * 16};
  bool ok = true;
  { cuuint64_t d[4] = {64, 8, cuuint64_t(K / 64), cuuint64_t(M / 8)}; cuuint32_t b[4] = {64, 8, KA, 16}; ok &= enc(&lm.a4, x, 4, d, st4, b);
    const int tail = (M % 128) / 8;   // full 8-row groups of the 128-row block that crosses M
    cuuint32_t bt[4] = {64, 8, KA, cuuint32_t(tail > 0 ? tail : 1)}; ok &= enc(&lm.a4t, x, 4, d, st4, bt); }
  { cuuint64_t d[3] = {64, cuuint64_t(M), cuuint64_t(K / 64)}, st[2] = {cuuint64_t(K) * 2, 128};
    cuuint32_t b[3] = {64, 8, KA}; ok &= enc(&lm.a3, x, 3, d, st, b); }
  { cuuint64_t d[4] = {64, 8, cuuint64_t(K / 64), cuuint64_t(N / 8)}; cuuint32_t b[4] = {64, 8, KA, 16}; ok &= enc(&lm.b4, w, 4, d, st4, b); }
  if (!ok) return false;

  auto* kptr = &hand_gemm<SMEM, decltype(tiled_mma), decltype(sA_s), decltype(sB_s)>;
  const int smem_bytes = sizeof(SMEM);
  if (cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess) return false;
  h->run = [=](cudaStream_t stream) {
    dim3 grid(2 * NCLUST), block(NTHREADS), cluster(2, 1, 1);
    cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_bytes, stream};
    auto st = cutlass::launch_kernel_on_cluster(params, (void const*)kptr, tiled_mma, lm, bias, y, M, N, K, sp, sA_s, sB_s);
    return st == cutlass::Status::kSuccess ? cudaGetLastError() : cudaErrorLaunchFailure;
  };
  return true;
}
