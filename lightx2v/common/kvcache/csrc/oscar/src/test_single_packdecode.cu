#include "flash_api.h"
#include <cstdio>
#include <fstream>

torch::Tensor single_mha(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, int head_dim) {
    const float sm_scale = 1.f / std::sqrt(float(head_dim));
    auto scaled_q = q * sm_scale;

    auto scores = torch::einsum("bthd,bshd->bhts", {scaled_q, k});
    auto attention = torch::softmax(scores, -1).to(v.dtype());
    auto output = torch::einsum("bhts,bshd->bthd", {attention, v});
    return output;
}


template <int num_heads, int num_heads_kv, int head_dim, int num_bits>
void TestDecodingKernelCorrectness(int seqlen_kv, const std::string& quant_mode, const int group_size) {
    // Set the random seed for reproducibility
    torch::manual_seed(42);

    const int bs        = 1;
    const int seqlen_q  = 1;
    const int pack_nums = 16 / num_bits;

    torch::Tensor Q_host = torch::rand({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::randn({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::randn({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    torch::Tensor K_device = K_host.to(torch::kCUDA);
    torch::Tensor V_device = V_host.to(torch::kCUDA);

    at::Tensor k_pack, k_params, v_pack, v_params;
    if (quant_mode == "k-channel") {
        k_pack   = torch::empty({bs, seqlen_kv / pack_nums,   num_heads_kv, head_dim}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params = torch::empty({bs, seqlen_kv / group_size, num_heads_kv, head_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    } else {
        k_pack   = torch::empty({bs, seqlen_kv, num_heads_kv, head_dim / pack_nums}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    }

    v_pack   = torch::empty({bs, seqlen_kv,   num_heads_kv, head_dim / pack_nums}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
    v_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    // Convert K, V to unpadded format
    torch::Tensor K_unpad = K_device.reshape({bs * seqlen_kv, num_heads_kv, head_dim});
    torch::Tensor V_unpad = V_device.reshape({bs * seqlen_kv, num_heads_kv, head_dim});

    auto cu_seqlens_k = torch::arange(0, (bs + 1) * seqlen_kv, seqlen_kv, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    std::optional<at::Tensor> opt_block_table = std::nullopt;

    kvcache_qpack<num_bits>(
        K_unpad, k_pack, k_params,
        V_unpad, v_pack, v_params,
        opt_block_table,
        cu_seqlens_k,
        seqlen_kv,
        quant_mode,
        group_size
    );

    // mha_fwd_kvcache
    const float sm_scale = 1 / std::sqrt(float(head_dim));
    auto out = mha_fwd_kvcache<num_bits>(Q_device,
                                         k_pack, k_params,
                                         v_pack, v_params,
                                         opt_block_table,
                                         sm_scale,
                                         quant_mode,
                                         group_size);

    torch::Tensor out_cpu = out.to(torch::kCPU);

    // CPU reference
    torch::Tensor out_ref = single_mha(Q_host, K_host, V_host, head_dim);

    // Compute the difference
    torch::Tensor diff = out_cpu - out_ref;
    float mean_absolute_error = diff.abs().mean().item<float>();
    float mean_squared_error = diff.pow(2).mean().item<float>();

    printf("\nnum_bits: %d num_heads_kv: %d seqlen_kv: %d head_dim: %d Quant_mode: %s, Group_size: %d\n", num_bits, num_heads_kv, seqlen_kv, head_dim, quant_mode.c_str(), group_size);
    if (mean_absolute_error < 1e-1 && mean_squared_error < 1e-1) {
        printf("test pass ! \n");
        printf("mean_absolute_error: %f, mean_squared_error: %f\n", mean_absolute_error, mean_squared_error);
    } else {
        printf("test fail ! \n");
        printf("mean_absolute_error: %f, mean_squared_error: %f\n", mean_absolute_error, mean_squared_error);
    }

    printf("\nFirst 10 elements of out_cpu:\n");
    auto out_cpu_accessor = out_cpu.flatten().data_ptr<at::Half>();
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", static_cast<float>(out_cpu_accessor[i]));
    }

    printf("\n\nFirst 10 elements of out_ref:\n");
    auto out_ref_accessor = out_ref.flatten().data_ptr<at::Half>();
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", static_cast<float>(out_ref_accessor[i]));
    }

    printf("\n\n");
}


int main() {
    const int num_heads    = 32;
    const int num_heads_kv = 32;
    const int head_dim     = 128;

    const std::string quant_mode = "k-channel";
    const int num_bits     = 4;
    const int group_size   = 128;

    int seqlen_kv = 1024;

    TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(seqlen_kv, quant_mode, group_size);

    return 0;
}
