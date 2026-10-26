// SDP ESIMD Flash Attention DLL — PTL-H target (Xe2 ISA, same as BMG)
// Compiled with: icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device ptl-h -options -doubleGRF"
// Exports: sdp_fp16 (FP16 optimized), sdp_bf16io (BF16 I/O hybrid)
// Tensor shape: [B, L, H, D] with D=128, B=1, contiguous

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "esimd_kernel_api.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS  sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef  ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))
#define FP32_MIN -3.402823466e+38f

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

#include "single_kernels/flash.attn.b.mha128.fp16.opt.h"
#include "single_kernels/flash.attn.b.mha128.bf16io.h"

// ──────────────────────────────────────────────────────────────────────────────
// sdp_fp16: FP16 optimized Flash Attention
//   Q/K/V/out: raw device pointers to [L, H, 128] fp16 (squeezed from [B,L,H,D])
//   normAlpha: [H * 128] float32, typically all 1.0
//   q_len, kv_len, headQ, headKv: sequence and head counts
//   queue: the SYCL queue to submit into
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sdp_fp16(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr)
{
    sycl::queue& q = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);

    int groupH = headQ;
    int groupV = (q_len + 255) / 256;
    sycl::nd_range<2> ndr({(size_t)(16 * groupH), (size_t)groupV}, {16, 1});

    uint8_t* pQ = reinterpret_cast<uint8_t*>(Q);
    uint8_t* pK = reinterpret_cast<uint8_t*>(K);
    uint8_t* pV = reinterpret_cast<uint8_t*>(V);
    uint8_t* pA = reinterpret_cast<uint8_t*>(normAlpha);
    uint8_t* pO = reinterpret_cast<uint8_t*>(out);
    uint32_t aLen  = (uint32_t)q_len;
    uint32_t kvLen = (uint32_t)kv_len;
    uint32_t hQ    = (uint32_t)headQ;
    uint32_t hKv   = (uint32_t)headKv;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(ndr, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Fp16OptPrecomputed(
                pQ, pK, pV, pA, pO,
                aLen, kvLen, hQ, hKv, ndi);
        });
    }).wait();
}

// ──────────────────────────────────────────────────────────────────────────────
// sdp_bf16io: BF16 I/O hybrid Flash Attention
//   Q/K/V/out: raw device pointers to [L, H, 128] bf16 (squeezed from [B,L,H,D])
//   normAlpha: [H * 128] float32
//   Internally uses bf16 DPAS for QK, fp16 DPAS + fp16 accumulator for SxV.
//   V conversion bf16→fp16 is hidden inside SxV DPAS (zero cost on Xe2).
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sdp_bf16io(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr)
{
    sycl::queue& q = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);

    int groupH = headQ;
    int groupV = (q_len + 255) / 256;
    sycl::nd_range<2> ndr({(size_t)(16 * groupH), (size_t)groupV}, {16, 1});

    uint8_t* pQ = reinterpret_cast<uint8_t*>(Q);
    uint8_t* pK = reinterpret_cast<uint8_t*>(K);
    uint8_t* pV = reinterpret_cast<uint8_t*>(V);
    uint8_t* pA = reinterpret_cast<uint8_t*>(normAlpha);
    uint8_t* pO = reinterpret_cast<uint8_t*>(out);
    uint32_t aLen  = (uint32_t)q_len;
    uint32_t kvLen = (uint32_t)kv_len;
    uint32_t hQ    = (uint32_t)headQ;
    uint32_t hKv   = (uint32_t)headKv;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(ndr, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16IoPrecomputed(
                pQ, pK, pV, pA, pO,
                aLen, kvLen, hQ, hKv, ndi);
        });
    }).wait();
}
