// PyTorch wrapper for ESIMD SDP Flash Attention kernels (PTL-H)
// Input/output layout: [B, L, H, 128]  (B=1, contiguous)
// Unified dispatch: fp16 → sdp_fp16, bf16 → sdp_bf16io

#include <torch/extension.h>

#include "sdp_kernels.h"
#include "utils.h"

using ST = torch::ScalarType;

// Validate tensor: must be XPU, contiguous, [B, L, H, 128], B=1
static void check_sdp_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::XPU,
        name, " must be on XPU");
    TORCH_CHECK(t.is_contiguous(),
        name, " must be contiguous");
    TORCH_CHECK(t.dim() == 4,
        name, " must be 4-D [B, L, H, 128]");
    TORCH_CHECK(t.size(0) == 1,
        name, " batch size must be 1");
    TORCH_CHECK(t.size(3) == 128,
        name, " head_dim must be 128");
}

// ──────────────────────────────────────────────────────────────────────────────
// sdp: unified Flash Attention SDP
//   Q: [1, L_q,  H_q,  128]  fp16 or bf16
//   K: [1, L_kv, H_kv, 128]  same dtype as Q
//   V: [1, L_kv, H_kv, 128]  same dtype as Q
//   returns: [1, L_q, H_q, 128]  same dtype as Q
//
//   fp16  → sdp_fp16   (optimised FP16 ESIMD kernel)
//   bf16  → sdp_bf16io (BF16 I/O, FP16 internal ESIMD kernel)
// ──────────────────────────────────────────────────────────────────────────────
torch::Tensor sdp_torch(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    check_sdp_tensor(Q, "Q");
    check_sdp_tensor(K, "K");
    check_sdp_tensor(V, "V");
    TORCH_CHECK(Q.scalar_type() == K.scalar_type() && Q.scalar_type() == V.scalar_type(),
        "Q, K, V must have the same dtype");

    const int q_len  = (int)Q.size(1);
    const int kv_len = (int)K.size(1);
    const int headQ  = (int)Q.size(2);
    const int headKv = (int)K.size(2);

    auto out = torch::empty_like(Q);

    // Cache normAlpha: only reallocate when headQ changes (avoids per-call
    // XPU malloc + fill-kernel on every sdp() invocation).
    static torch::Tensor s_normAlpha;
    static int           s_headQ = -1;
    if (headQ != s_headQ) {
        s_normAlpha = torch::ones({headQ * 128},
            torch::dtype(torch::kFloat).device(Q.device()));
        s_headQ = headQ;
    }
    const auto& normAlpha = s_normAlpha;

    sycl::queue& sq = utils::get_queue(Q.device());

    auto dispatch = [&](auto kernel) {
        kernel(Q.data_ptr(), K.data_ptr(), V.data_ptr(),
               normAlpha.data_ptr(), out.data_ptr(),
               q_len, kv_len, headQ, headKv, &sq);
    };

    switch (Q.scalar_type()) {
        case ST::Half:
            dispatch(sdp_fp16);   break;
        case ST::BFloat16:
            dispatch(sdp_bf16io); break;
        default:
            TORCH_CHECK(false,
                "sdp: unsupported dtype, only FP16 and BF16 are supported");
    }

    return out;
}
