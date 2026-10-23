// Copyright (c) 2026 LightX2V contributors. SPDX-License-Identifier: Apache-2.0
// MP31 SOL: block-summary routing, approximate contributions, and exact attention.
// Uses the MUTLASS/MuTe SQMMA and TME building blocks; no dense/DSA substitution.
#include <musa_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "collective/fmha_common.hpp"
#include "mute/tensor.hpp"
#include "mutlass/arch/barrier.hpp"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/numeric_types.h"

namespace sol_mp31 {
using namespace mute;
using mutlass::fmha::collective::layout_acc_mn;
using mutlass::fmha::collective::reduction_target_n;
using E = mutlass::bfloat16_t;
using S = Stride<int, _1, Stride<int, int>>;
using SV = Stride<_1, int, Stride<int, int>>;
using QK = typename mutlass::gemm::collective::CollectiveBuilder<
    mutlass::arch::Mp31, mutlass::arch::OpClassTensorOp, E, S, 16, E, S, 16, float,
    Shape<_64, _64, _128>, Shape<_1, _1, _1>, _2, mutlass::gemm::KernelTme>::CollectiveOp;
using PV = typename mutlass::gemm::collective::CollectiveBuilder<
    mutlass::arch::Mp31, mutlass::arch::OpClassTensorOp, E, S, 16, E, SV, 16, float,
    Shape<_64, _128, _64>, Shape<_1, _1, _1>, _2, mutlass::gemm::KernelTme>::CollectiveOp;
using LQ = decltype(tile_to_shape(typename QK::SmemLayoutAtomA{}, Shape<_64, _128>{}));
using LK = decltype(tile_to_shape(typename QK::SmemLayoutAtomB{}, Shape<_64, _128>{}));
using LP = decltype(tile_to_shape(typename PV::SmemLayoutAtomA{}, Shape<_64, _64>{}));
using LV = decltype(tile_to_shape(typename PV::SmemLayoutAtomB{}, Shape<_128, _64>{}));

template <class Layout, class Stride>
auto descriptor(E const* p, int n, int h, int b, Layout l, Stride s) {
  auto dims = [&] {
    if constexpr (is_same_v<Stride, SV>)
      return make_shape(128, n, make_shape(h, b));
    else
      return make_shape(n, 128, make_shape(h, b));
  }();
  return make_tme_copy(MP31_TME_LOAD{}, make_tensor(make_gmem_ptr(p), dims, s), l, shape(l));
}
using DQ = decltype(descriptor((E const*)nullptr, 0, 0, 0, LQ{}, S{}));
using DK = decltype(descriptor((E const*)nullptr, 0, 0, 0, LK{}, S{}));
using DV = decltype(descriptor((E const*)nullptr, 0, 0, 0, LV{}, SV{}));
struct Args {
  DQ q;
  DK k, kc;
  DV v, vc;
  float const* thresholds;
  E* out;
  uint8_t* routes;
  int B, T, H, N, NPAD, sink_lo, sink_hi, has_sink;
  float scale2;
};
struct Shared {
  array_aligned<E, cosize_v<LQ>, 256> q;
  array_aligned<E, cosize_v<LK>, 256> k;
  array_aligned<E, cosize_v<LV>, 256> v;
  // Routing warp partials and probability tiles have disjoint lifetimes.
  // Keep the full QK accumulator in registers; only 4 x 64 partials go to SMEM.
  union alignas(256) {
    float route_sums[4 * 64];
    E p[cosize_v<LP>];
  } temporary;
  // Immutable throughout approximate/precise evaluation of a route group.
  unsigned route_mask[2];
};

// Caller guarantees that at least one word is nonzero. __ffs accepts the
// complete 32-bit pattern, including bit 31, and returns a one-based index.
__device__ int first_exact_offset(unsigned lo, unsigned hi) {
  return lo ? __ffs(static_cast<int>(lo)) - 1 : 32 + __ffs(static_cast<int>(hi)) - 1;
}

// This reduction is specific to the MP31 64x64 accumulator lane layout.
// For thread t, score[j + 8*r] is (row=t/8 + 16*r, col=t%8 + 8*j).
// Sum four register rows, then four lanes (distance 8/16) per warp. The
// resulting warp partial covers 16 rows, and four warps cover the Q block.
template <class Acc>
__device__ void route_column_partials(Acc const& score, Shared& s, int tid) {
  static_assert(is_same_v<typename QK::TiledMma::LayoutC_TV,
                         MP31::SQMMA::CLayout<64, 64>>,
                "Update SOL routing reduction when the SQMMA C layout changes");
  int lane = tid % 32, warp = tid / 32;
  MUTE_UNROLL for (int j = 0; j < 8; ++j) {
    float sum = score(j) + score(j + 8);
    sum += score(j + 16);
    sum += score(j + 24);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    if (lane < 8) s.temporary.route_sums[warp * 64 + j * 8 + lane] = sum;
  }
}

template <class Desc, class Layout>
__device__ void issue(Desc const& d, E* dst, Layout l, int n, int H, int B, int tile, int head,
                      int batch, uint32_t barrier) {
  constexpr bool transposed = is_same_v<Layout, LV>;
  auto dims = [&] {
    if constexpr (transposed)
      return make_shape(128, n, make_shape(H, B));
    else
      return make_shape(n, 128, make_shape(H, B));
  }();
  auto g = d.get_tme_tensor(dims);
  auto tiled = local_tile(g, shape(l), make_coord(_, _, _));
  auto src = [&] {
    if constexpr (transposed)
      return tiled(_, _, 0, tile, make_coord(head, batch));
    else
      return tiled(_, _, tile, 0, make_coord(head, batch));
  }();
  auto copy_slice = d.get_slice(0);
  auto smem = make_tensor(make_smem_ptr(dst), l);
  copy(d.with(barrier), copy_slice.partition_S(src), copy_slice.partition_D(smem));
}

template <class MMA, class Tensor>
__device__ float row_max_reduce(MMA mma, Tensor& x, int i) {
  float v = -INFINITY;
  MUTE_UNROLL for (int j = 0; j < size<1>(x); ++j) v = fmaxf(v, x(i, j));
  auto r = reduction_target_n(mma);
  for_each(make_seq<decltype(rank(r))::value>{}, [&](auto axis) {
    MUTE_UNROLL for (int k = 1; k < shape<axis>(r); k *= 2) v =
        fmaxf(v, __shfl_xor_sync(0xffffffff, v, stride<axis>(r) * k));
  });
  return v;
}
template <class MMA>
__device__ float row_sum_reduce(MMA mma, float v) {
  auto r = reduction_target_n(mma);
  for_each(make_seq<decltype(rank(r))::value>{}, [&](auto axis) {
    MUTE_UNROLL for (int k = 1; k < shape<axis>(r); k *= 2) v +=
        __shfl_xor_sync(0xffffffff, v, stride<axis>(r) * k);
  });
  return v;
}

template <bool Approx, class AccS, class AccO, class Index, class Row>
__device__ void softmax(AccS& score, AccO& output, Index const& coords, Row& rowmax, Row& rowsum,
                        Shared& s, Args const& a, int base) {
  typename QK::TiledMma qk;
  typename PV::TiledMma pv;
  auto x = make_tensor(score.data(), layout_acc_mn(qk, score.layout()));
  auto c = make_tensor(coords.data(), layout_acc_mn(qk, coords.layout()));
  auto o = make_tensor(output.data(), layout_acc_mn(pv, output.layout()));
  auto p = make_tensor(make_smem_ptr(s.temporary.p), LP{});
  unsigned route_lo = 0, route_hi = 0;
  if constexpr (Approx) {
    route_lo = s.route_mask[0];
    route_hi = s.route_mask[1];
  }
  MUTE_UNROLL for (int i = 0; i < size<0>(x); ++i) {
    MUTE_UNROLL for (int j = 0; j < size<1>(x); ++j) {
      int col = get<1>(c(i, j));
      bool valid;
      if constexpr (Approx) {
        unsigned word = col < 32 ? route_lo : route_hi;
        bool exact = ((word >> (col & 31)) & 1u) != 0;
        valid = base + col < a.N && !exact;
      } else
        valid = base * 64 + col < a.T;
      x(i, j) = valid ? x(i, j) * a.scale2 : -INFINITY;
    }
    float m = fmaxf(rowmax(i), row_max_reduce(qk, x, i));
    float alpha = isfinite(m) ? exp2f(rowmax(i) - m) : 1.f;
    float sum = 0.f;
    MUTE_UNROLL for (int j = 0; j < size<1>(x); ++j) {
      float prob = isfinite(m) ? exp2f(x(i, j) - m) : 0.f;
      int col = get<1>(c(i, j));
      int len = 1;
      if constexpr (Approx) len = max(0, min(64, a.T - (base + col) * 64));
      sum += prob * len;
      p(get<0>(c(i, j)), col) = E(prob);
    }
    rowsum(i) = rowsum(i) * alpha + row_sum_reduce(qk, sum);
    rowmax(i) = m;
    MUTE_UNROLL for (int j = 0; j < size<1>(o); ++j) o(i, j) *= alpha;
  }
  __syncthreads();
}

__global__ __launch_bounds__(128) void forward(Args a) {
  extern __shared__ __align__(256) unsigned char bytes[];
  auto& s = *reinterpret_cast<Shared*>(bytes);
  int tid = threadIdx.x, qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  mutlass::arch::allocate_async_barriers(3);
  mutlass::arch::AsyncTransactionBarrier bq(0), bk(1), bv(2);
  if (tid == 0) {
    bq.init(1);
    bk.init(1);
    bv.init(1);
  }
  __syncthreads();
  int kp = 0, vp = 0;
  if (tid == 0) {
    bq.arrive_and_expect_tx(sizeof(s.q));
    issue(a.q, s.q.data(), LQ{}, a.T, a.H, a.B, qb, h, b, bq.get_barrier_id());
  }
  typename QK::TiledMma qk;
  typename PV::TiledMma pv;
  auto tq = qk.get_thread_slice(tid);
  auto tv = pv.get_thread_slice(tid);
  auto sq = make_tensor(make_smem_ptr(s.q.data()), LQ{});
  auto sk = make_tensor(make_smem_ptr(s.k.data()), LK{});
  auto sp = make_tensor(make_smem_ptr(s.temporary.p), LP{});
  auto sv = make_tensor(make_smem_ptr(s.v.data()), LV{});
  auto rq = tq.make_fragment_A(tq.partition_A(sq));
  auto rk = tq.make_fragment_B(tq.partition_B(sk));
  auto rp = tv.make_fragment_A(tv.partition_A(sp));
  auto rv = tv.make_fragment_B(tv.partition_B(sv));
  auto acc = partition_fragment_C(pv, Shape<_64, _128>{});
  clear(acc);
  auto rowshape = make_tensor(acc.data(), layout_acc_mn(pv, acc.layout()));
  auto rowmax = make_fragment_like<float>(size<0>(rowshape));
  fill(rowmax, -INFINITY);
  auto rowsum = make_fragment_like<float>(rowmax);
  clear(rowsum);
  auto identity = make_identity_tensor(Shape<_64, _64>{});
  auto coords = tq.partition_C(identity);
  bq.wait(0);
  float thresh = a.thresholds[(int64_t(b) * a.N + qb) * a.H + h];
  for (int group = 0; group < a.N; group += 64) {
    if (tid == 0) {
      bk.arrive_and_expect_tx(sizeof(s.k));
      bv.arrive_and_expect_tx(sizeof(s.v));
      issue(a.kc, s.k.data(), LK{}, a.NPAD, a.H, a.B, group / 64, h, b, bk.get_barrier_id());
      issue(a.vc, s.v.data(), LV{}, a.NPAD, a.H, a.B, group / 64, h, b, bv.get_barrier_id());
    }
    bk.wait(kp);
    kp ^= 1;
    auto score = partition_fragment_C(qk, Shape<_64, _64>{});
    clear(score);
    gemm(qk, rq, rk, score);
    warpsquad_wait();
    route_column_partials(score, s, tid);
    __syncthreads();
    bool selected = false;
    if (tid < 64) {
      float sum = s.temporary.route_sums[tid] + s.temporary.route_sums[64 + tid];
      sum += s.temporary.route_sums[128 + tid];
      sum += s.temporary.route_sums[192 + tid];
      int block = group + tid;
      selected = block < a.N &&
                 (sum * a.scale2 / min(64, a.T - qb * 64) > thresh || abs(qb - block) <= 1 ||
                  (a.has_sink && block >= a.sink_lo && block < a.sink_hi));
      if (a.routes && block < a.N)
        a.routes[((int64_t(b) * a.H + h) * a.N + qb) * a.N + block] = selected;
    }
    unsigned mask = __ballot_sync(0xffffffff, selected);
    if (tid == 0 || tid == 32) s.route_mask[tid / 32] = mask;
    // Publish the two masks once; no shared index compaction/count is needed.
    __syncthreads();
    unsigned remaining_lo = s.route_mask[0], remaining_hi = s.route_mask[1];
    int exact_count = __popc(remaining_lo) + __popc(remaining_hi);
    // Summary QK has completed. Prefetch the first exact K while the
    // approximate softmax and PV consume the summary V tile.
    if (tid == 0 && (remaining_lo != 0 || remaining_hi != 0)) {
      int first_block = group + first_exact_offset(remaining_lo, remaining_hi);
      bk.arrive_and_expect_tx(sizeof(s.k));
      issue(a.k, s.k.data(), LK{}, a.T, a.H, a.B, first_block, h, b, bk.get_barrier_id());
    }
    bv.wait(vp);
    vp ^= 1;
    if (exact_count < min(64, a.N - group)) {
      softmax<true>(score, acc, coords, rowmax, rowsum, s, a, group);
      gemm(pv, rp, rv, acc);
      warpsquad_wait();
    }
    __syncthreads();
    while (remaining_lo != 0 || remaining_hi != 0) {
      int block = group + first_exact_offset(remaining_lo, remaining_hi);
      // Consume in ascending KV-block order, then peek at the next bit for K prefetch.
      if (remaining_lo != 0)
        remaining_lo &= remaining_lo - 1u;
      else
        remaining_hi &= remaining_hi - 1u;
      if (tid == 0) {
        // V transfer overlaps this block's QK and softmax.
        bv.arrive_and_expect_tx(sizeof(s.v));
        issue(a.v, s.v.data(), LV{}, a.T, a.H, a.B, block, h, b, bv.get_barrier_id());
      }
      bk.wait(kp);
      kp ^= 1;
      clear(score);
      gemm(qk, rq, rk, score);
      warpsquad_wait();
      // K is no longer read by SQMMA. Reuse it immediately for the next
      // block, overlapping its TME transfer with the current softmax/PV.
      if (tid == 0 && (remaining_lo != 0 || remaining_hi != 0)) {
        int next_block = group + first_exact_offset(remaining_lo, remaining_hi);
        bk.arrive_and_expect_tx(sizeof(s.k));
        issue(a.k, s.k.data(), LK{}, a.T, a.H, a.B, next_block, h, b, bk.get_barrier_id());
      }
      softmax<false>(score, acc, coords, rowmax, rowsum, s, a, block);
      bv.wait(vp);
      vp ^= 1;
      gemm(pv, rp, rv, acc);
      warpsquad_wait();
      __syncthreads();
    }
  }
  auto ci = make_identity_tensor(Shape<_64, _128>{});
  auto oc = tv.partition_C(ci);
  auto out = make_tensor(acc.data(), layout_acc_mn(pv, acc.layout()));
  auto outc = make_tensor(oc.data(), layout_acc_mn(pv, oc.layout()));
  MUTE_UNROLL for (int i = 0; i < size<0>(out); ++i) {
    MUTE_UNROLL for (int j = 0; j < size<1>(out); ++j) {
      int row = qb * 64 + get<0>(outc(i, j)), col = get<1>(outc(i, j));
      if (row < a.T)
        a.out[((int64_t(b) * a.T + row) * a.H + h) * 128 + col] = E(out(i, j) / rowsum(i));
    }
  }
}

int launch(const void* q, const void* k, const void* v, const void* kc, const void* vc,
           const float* threshold, void* out, int B, int T, int H, int NPAD, float scale,
           int sink_start_block, int sink_end_block, int has_sink, void* stream, uint8_t* routes) {
  if (B <= 0 || T <= 0 || T > INT_MAX - 63 || H <= 0 || NPAD < (T + 63) / 64 ||
      int64_t(T) * H * 128 > INT_MAX || int64_t(NPAD) * H * 128 > INT_MAX)
    return musaErrorInvalidValue;
  if (!q || !k || !v || !kc || !vc || !threshold || !out ||
      (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
       reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(kc) |
       reinterpret_cast<uintptr_t>(vc)) %
          32)
    return musaErrorInvalidValue;
  S stride = make_stride(H * 128, _1{}, make_stride(128, T * H * 128));
  S stridec = make_stride(H * 128, _1{}, make_stride(128, NPAD * H * 128));
  SV stridev = make_stride(_1{}, H * 128, make_stride(128, T * H * 128));
  SV stridevc = make_stride(_1{}, H * 128, make_stride(128, NPAD * H * 128));
  Args a{descriptor((E const*)q, T, H, B, LQ{}, stride),
         descriptor((E const*)k, T, H, B, LK{}, stride),
         descriptor((E const*)kc, NPAD, H, B, LK{}, stridec),
         descriptor((E const*)v, T, H, B, LV{}, stridev),
         descriptor((E const*)vc, NPAD, H, B, LV{}, stridevc),
         threshold,
         (E*)out,
         routes,
         B,
         T,
         H,
         (T + 63) / 64,
         NPAD,
         sink_start_block,
         sink_end_block,
         has_sink,
         scale * 1.4426950408889634f};
  auto status =
      musaFuncSetAttribute(forward, musaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
  if (status != musaSuccess) return status;
  forward<<<dim3(a.N, H, B), 128, sizeof(Shared), (musaStream_t)stream>>>(a);
  return musaGetLastError();
}
}  // namespace sol_mp31

extern "C" int lightx2v_sol_forward(const void* q, const void* k, const void* v, const void* kc,
                                    const void* vc, const float* threshold, void* out, int B, int T,
                                    int H, int NPAD, float scale, int sink_lo, int sink_hi,
                                    int has_sink, void* stream) {
  return sol_mp31::launch(q, k, v, kc, vc, threshold, out, B, T, H, NPAD, scale, sink_lo, sink_hi,
                          has_sink, stream, nullptr);
}
extern "C" int lightx2v_sol_forward_debug(const void* q, const void* k, const void* v,
                                          const void* kc, const void* vc, const float* threshold,
                                          void* out, int B, int T, int H, int NPAD, float scale,
                                          int sink_lo, int sink_hi, int has_sink, void* stream,
                                          uint8_t* routes) {
  return sol_mp31::launch(q, k, v, kc, vc, threshold, out, B, T, H, NPAD, scale, sink_lo, sink_hi,
                          has_sink, stream, routes);
}
