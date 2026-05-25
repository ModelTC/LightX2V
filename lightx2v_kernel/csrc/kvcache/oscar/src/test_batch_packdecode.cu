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

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int> _generate_block_kvcache(
    int seqlen_k,
    int paged_kv_block_size,
    int batch_size,
    int nheads_k,
    int d,
    int num_bits,
    const std::string& quant_mode,
    torch::Device device,
    torch::ScalarType dtype) {

    // Calculate number of blocks needed
    int num_blocks = std::ceil(float(seqlen_k) / paged_kv_block_size) * batch_size;

    int num_per_params = 16 / num_bits;

    // Generate random k/v blocks
    auto k_cache_paged = torch::randn(
        {num_blocks, paged_kv_block_size, nheads_k, d},
        torch::TensorOptions().device(device).dtype(dtype)
    );

    auto v_cache_paged = torch::randn(
        {num_blocks, paged_kv_block_size, nheads_k, d},
        torch::TensorOptions().device(device).dtype(dtype)
    );

    // Pack
    torch::Tensor k_cache_paged_pack, v_cache_paged_pack;
    if (quant_mode == "k-channel") {
        k_cache_paged_pack = torch::randn(
            {num_blocks, paged_kv_block_size / num_per_params, nheads_k, d},
            torch::TensorOptions().device(device)
        ).to(torch::kUInt16);

        v_cache_paged_pack = torch::randn(
            {num_blocks, paged_kv_block_size, nheads_k, d / num_per_params},
            torch::TensorOptions().device(device)
        ).to(torch::kUInt16);

    } else {
        k_cache_paged_pack = torch::randn(
            {num_blocks, paged_kv_block_size, nheads_k, d / num_per_params},
            torch::TensorOptions().device(device)
        ).to(torch::kUInt16);

        v_cache_paged_pack = torch::randn(
            {num_blocks, paged_kv_block_size, nheads_k, d / num_per_params},
            torch::TensorOptions().device(device)
        ).to(torch::kUInt16);
    }

    // Generate block_table: for each batch, create a permutation of blocks
    // First create a randperm of all blocks
    auto block_table = torch::randperm(num_blocks,
        torch::TensorOptions().device(device).dtype(torch::kInt32)
    );

    // Reshape to (batch_size, num_blocks_per_batch)
    int nblocks_per_batch = num_blocks / batch_size;
    block_table = block_table.reshape({batch_size, nblocks_per_batch});

    return std::make_tuple(k_cache_paged, v_cache_paged, k_cache_paged_pack, v_cache_paged_pack, block_table, num_blocks);
}

template <int num_heads, int num_heads_kv, int head_dim, int num_bits>
void TestDecodingKernelCorrectness(int bs, int seqlen_kv, const std::string& quant_mode, const int group_size) {
    // Set the random seed for reproducibility
    torch::manual_seed(42);

    const int seqlen_q = 1;
    const int page_block_size = 256;

    torch::Tensor Q_host = torch::rand({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor K_host = torch::randn({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::randn({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    torch::Tensor K_device = K_host.to(torch::kCUDA);
    torch::Tensor V_device = V_host.to(torch::kCUDA);

    // Page
    auto [k_cache_paged, v_cache_paged, k_cache_paged_pack, v_cache_paged_pack, block_table, num_blocks] = _generate_block_kvcache<at::Half>(
        seqlen_kv,
        page_block_size,
        bs,
        num_heads_kv,
        head_dim,
        num_bits,
        quant_mode,
        torch::kCUDA,
        torch::kHalf
    );

    at::Tensor k_params = quant_mode == "k-channel"
        ? torch::empty({bs, seqlen_kv / group_size, num_heads_kv, head_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA)
        : torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    at::Tensor v_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    auto cu_seqlens_k = torch::arange(0, (bs + 1) * seqlen_kv, seqlen_kv, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    std::optional<at::Tensor> opt_block_table        = std::make_optional(block_table);

    kvcache_qpack<num_bits>(
        k_cache_paged, k_cache_paged_pack, k_params,
        v_cache_paged, v_cache_paged_pack, v_params,
        opt_block_table,
        cu_seqlens_k,
        seqlen_kv,
        quant_mode,
        group_size
    );

    // mha_fwd_kvcache
    const float sm_scale = 1 / std::sqrt(float(head_dim));
    auto out = mha_fwd_kvcache<num_bits>(Q_device,
                                         k_cache_paged_pack, k_params,
                                         v_cache_paged_pack, v_params,
                                         opt_block_table,
                                         sm_scale,
                                         quant_mode,
                                         group_size);

    torch::Tensor out_cpu = out.to(torch::kCPU);

    // torch reference
    // Page
    int nblocks_per_batch = block_table.size(1);
    auto flat_block_table = block_table.flatten().to(torch::kInt64);
    auto k_cache = k_cache_paged.index_select(0, flat_block_table);
    auto v_cache = v_cache_paged.index_select(0, flat_block_table);

    k_cache = k_cache.reshape({bs, nblocks_per_batch * page_block_size, num_heads_kv, head_dim})
                     .slice(1, 0, seqlen_kv);
    v_cache = v_cache.reshape({bs, nblocks_per_batch * page_block_size, num_heads_kv, head_dim})
                     .slice(1, 0, seqlen_kv);

    torch::Tensor out_ref = single_mha(Q_device, k_cache, v_cache, head_dim);
    out_ref = out_ref.to(torch::kCPU);

    // Compute the difference
    torch::Tensor diff = out_cpu - out_ref;
    float mean_absolute_error = diff.abs().mean().item<float>();
    float mean_squared_error = diff.pow(2).mean().item<float>();

    printf("batch_size: %d num_heads_kv: %d seqlen_kv: %d head_dim: %d Quant_mode: %s\n", bs, num_heads_kv, seqlen_kv, head_dim, quant_mode.c_str());
    if (mean_absolute_error < 1e-2 && mean_squared_error < 1e-2) {
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
    const int batch_size   = 4;
    const int num_heads    = 32;
    const int num_heads_kv = 32;
    const int head_dim     = 128;

    const std::string quant_mode = "k-channel";
    const int num_bits     = 4;
    const int group_size   = 128;

    int seqlen_kv = 1024;

    TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

    return 0;
}
