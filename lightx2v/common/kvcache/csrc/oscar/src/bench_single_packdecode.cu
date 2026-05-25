#include "flash_api.h"
#include <cstdio>
#include <fstream>


template <int num_heads, int num_heads_kv, int head_dim, int num_bits>
double TestDecodingKernelPerformance(int seqlen_kv, const std::string& quant_mode, const int group_size, const int repeat) {
    const int bs = 1;
    const int seqlen_q = 1;
    const int pack_nums = 16 / num_bits;

    torch::Tensor Q_host = torch::rand({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::ones({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::ones({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

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

    at::Tensor K_new_host, V_new_host, K_new_device, V_new_device, seqlens_k;

    const float sm_scale = 1 / std::sqrt(float(head_dim));
    // Warm up
    for (int i = 0; i < 5; ++i)
        mha_fwd_kvcache<num_bits>(Q_device,
                        k_pack, k_params,
                        v_pack, v_params,
                        opt_block_table,
                        sm_scale,
                        quant_mode,
                        group_size);

    // Benchmark
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        mha_fwd_kvcache<num_bits>(Q_device,
                                  k_pack, k_params,
                                  v_pack, v_params,
                                  opt_block_table,
                                  sm_scale,
                                  quant_mode,
                                  group_size);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    msec = msec / repeat;

    return msec;
}

int main() {
    const int num_heads    = 32;
    const int num_heads_kv = 32;
    const int head_dim     = 128;

    const std::string quant_mode = "k-channel";
    const int num_bits   = 4;
    const int group_size = 128;

    const int test_num = 10;
    int len_list[test_num];
    len_list[0] = 1024;
    for (int i = 1; i < test_num; i++) {
        len_list[i] = len_list[i - 1] * 2;
    }

    const int outer_repeat = 3, inner_repeat = 3;
    printf("\n######## Benchmark single decode ########\n");
    for (int j = 0; j < test_num; j++) {

        int seqlen_kv = len_list[j];
        double max_msec = 0.0;
        double min_msec = DBL_MAX;
        double total_msec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = TestDecodingKernelPerformance<num_heads, num_heads_kv, head_dim, num_bits>(seqlen_kv, quant_mode, group_size, inner_repeat);
            max_msec = max(max_msec, this_sec);
            min_msec = min(min_msec, this_sec);
            total_msec += this_sec;
        }

        double avg_msec = total_msec / outer_repeat;
        printf("seqlen_kv num_heads head_dim = %6d %6d %6d, ", seqlen_kv, num_heads, head_dim);
        printf("Time = %12.8lf %12.8lf %12.8lf ms, \n", min_msec, avg_msec, max_msec);
    }

    return 0;
}
