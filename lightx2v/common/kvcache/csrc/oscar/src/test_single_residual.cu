#include "flash_api.h"
#include <cstdio>
#include <fstream>

torch::Tensor single_mha(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, int head_dim) {
    // Optional RoPE
    const float sm_scale = 1.f / std::sqrt(float(head_dim));
    auto scaled_q = q * sm_scale;

    auto scores = torch::einsum("bthd,bshd->bhts", {scaled_q, k});
    auto attention = torch::softmax(scores, -1).to(v.dtype());
    auto output = torch::einsum("bhts,bshd->bthd", {attention, v});
    return output;
}

template <int num_heads, int num_heads_kv, int head_dim, int num_bits>
void TestDecodingKernelCorrectness(const int bs, int seqlen_kv, const std::string& quant_mode, const int group_size) {
    // Set the random seed for reproducibility
    torch::manual_seed(42);

    const int seqlen_q  = 1;
    const int pack_nums = 16 / num_bits;
    const int residual_block_size = num_bits == 4 ? 128 : 256;
    int residual_len = seqlen_kv % residual_block_size == 0 ? residual_block_size : seqlen_kv % residual_block_size;
    seqlen_kv = seqlen_kv - residual_len;

    printf("\n\n################## Round 0 ##################\n\n");

    torch::Tensor Q_host = torch::rand({bs, seqlen_q, num_heads, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor K_host = torch::randn({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    torch::Tensor V_host = torch::randn({bs, seqlen_kv, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    torch::Tensor Q_device = Q_host.to(torch::kCUDA);
    torch::Tensor K_device = K_host.to(torch::kCUDA);
    torch::Tensor V_device = V_host.to(torch::kCUDA);

    at::Tensor k_pack, k_params, v_pack, v_params, k_pack_new, k_params_new, v_pack_new, v_params_new;
    if (quant_mode == "k-channel") {
        k_pack   = torch::empty({bs, seqlen_kv / pack_nums,   num_heads_kv, head_dim}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params = torch::empty({bs, seqlen_kv / group_size,  num_heads_kv, head_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
        k_pack_new   = torch::empty({bs, residual_block_size / pack_nums, num_heads_kv, k_pack.size(-1)}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params_new = torch::empty({bs, residual_block_size / group_size, num_heads_kv, k_params.size(-1)}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    } else {
        k_pack   = torch::empty({bs, seqlen_kv, num_heads_kv, head_dim / pack_nums}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
        k_pack_new   = torch::empty({bs, residual_block_size, num_heads_kv, k_pack.size(-1)}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
        k_params_new = torch::empty({bs, k_params.size(1), num_heads_kv, residual_block_size}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    }

    v_pack   = torch::empty({bs, seqlen_kv,   num_heads_kv, head_dim / pack_nums}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
    v_params = torch::empty({bs, head_dim / group_size, num_heads_kv, seqlen_kv}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    v_pack_new   = torch::empty({bs, residual_block_size, num_heads_kv, v_pack.size(-1)}, torch::dtype(torch::kUInt16)).to(torch::kCUDA);
    v_params_new = torch::empty({bs, v_params.size(1), num_heads_kv, residual_block_size}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    // Convert K, V to unpadded format
    torch::Tensor K_unpad = K_device.reshape({bs * seqlen_kv, num_heads_kv, head_dim});
    torch::Tensor V_unpad = V_device.reshape({bs * seqlen_kv, num_heads_kv, head_dim});

    auto cu_seqlens_k = torch::arange(0, (bs + 1) * seqlen_kv, seqlen_kv, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    // auto cu_seqlens_k = std::nullopt;

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

    at::Tensor K_residual_host, V_residual_host, K_new_host, V_new_host, K_new_device, V_new_device, seqlens_k;
    int new_lens = 0;

    seqlens_k       = torch::full({bs}, seqlen_kv, torch::dtype(torch::kInt32).device(torch::kCUDA));

    K_residual_host = torch::zeros({bs, residual_block_size, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    V_residual_host = torch::zeros({bs, residual_block_size, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    K_new_host   = torch::randn({bs, residual_len, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    V_new_host   = torch::randn({bs, residual_len, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    new_lens = residual_len;

    // Copy data from K_new_host to K_residual_host
    K_residual_host.slice(1, 0, residual_len).copy_(K_new_host);
    V_residual_host.slice(1, 0, residual_len).copy_(V_new_host);

    K_new_device = K_residual_host.to(torch::kCUDA);
    V_new_device = V_residual_host.to(torch::kCUDA);

    // mha_fwd_kvcache
    const float sm_scale = 1 / std::sqrt(float(head_dim));
    std::optional<const at::Tensor> opt_K_new_device = std::make_optional(K_new_device);
    std::optional<const at::Tensor> opt_V_new_device = std::make_optional(V_new_device);
    std::optional<const at::Tensor> opt_seqlens_k = std::make_optional(seqlens_k);

    auto [out, k_pack_new_1, k_params_new_1, v_pack_new_1, v_params_new_1]
        = mha_fwd_kvcache<num_bits>(Q_device,
            k_pack, k_params,
            v_pack, v_params,
            opt_K_new_device, opt_V_new_device, opt_seqlens_k,
            k_pack_new, k_params_new, v_pack_new, v_params_new,
            opt_block_table,
            sm_scale,
            quant_mode,
            group_size,
            residual_block_size,
            new_lens);

    torch::Tensor out_cpu = out.to(torch::kCPU);

    // CPU reference
    torch::Tensor K_host_cat, V_host_cat;

    K_host_cat = torch::cat({K_host, K_new_host}, 1);
    V_host_cat = torch::cat({V_host, V_new_host}, 1);

    torch::Tensor out_ref = single_mha(Q_host, K_host_cat, V_host_cat, head_dim);

    // Compute the difference
    torch::Tensor diff = out_cpu - out_ref;
    float mean_absolute_error = diff.abs().mean().item<float>();
    float mean_squared_error  = diff.pow(2).mean().item<float>();
    float max_error           = diff.abs().max().item<float>();

    printf("\nnum_bits: %d num_heads_kv: %d seqlen_kv: %d head_dim: %d Quant_mode: %s, Group_size: %d\n", num_bits, num_heads_kv, seqlen_kv + new_lens, head_dim, quant_mode.c_str(), group_size);
    if (mean_absolute_error < 1e-1 && mean_squared_error < 1e-1) {
        printf("test pass ! \n");
        printf("max_error: %f, mean_absolute_error: %f, mean_squared_error: %f\n", max_error, mean_absolute_error, mean_squared_error);
    } else {
        printf("test fail ! \n");
        printf("max_error: %f, mean_absolute_error: %f, mean_squared_error: %f\n", max_error, mean_absolute_error, mean_squared_error);
    }

    printf("\nFirst head output (out_cpu[0,0,0,:]):\n");
    auto out_cpu_accessor = out_cpu.index({0,0,1}).data_ptr<at::Half>();
    for (int i = 0; i < head_dim; i++) {
        printf("%.6f ", static_cast<float>(out_cpu_accessor[i]));
    }
    printf("\n\nFirst head output (out_ref[0,0,0,:]):\n");
    auto out_ref_accessor = out_ref.index({0,0,1}).data_ptr<at::Half>();
    for (int i = 0; i < head_dim; i++) {
        printf("%.6f ", static_cast<float>(out_ref_accessor[i]));
    }
    printf("\n\n");

    //
    // Next round
    //

    printf("\n\n################## Round 1 ##################\n\n");

    auto seqlen_new = 1;

    K_new_host = torch::randn({bs, seqlen_new, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
    V_new_host = torch::randn({bs, seqlen_new, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));

    if (new_lens == residual_block_size) {
        k_pack    = torch::cat({k_pack, k_pack_new},      1);
        k_params  = torch::cat({k_params, k_params_new},  1);
        v_pack    = torch::cat({v_pack, v_pack_new},      1);
        v_params  = torch::cat({v_params, v_params_new}, -1);
        seqlens_k = torch::full({bs}, seqlen_kv + new_lens, torch::dtype(torch::kInt32).device(torch::kCUDA));

        K_residual_host = torch::zeros({bs, residual_block_size, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
        V_residual_host = torch::zeros({bs, residual_block_size, num_heads_kv, head_dim}, torch::dtype(torch::kHalf));
        K_residual_host.slice(1, 0, 0 + seqlen_new).copy_(K_new_host);
        V_residual_host.slice(1, 0, 0 + seqlen_new).copy_(V_new_host);
    } else {
        seqlens_k = torch::full({bs}, seqlen_kv, torch::dtype(torch::kInt32).device(torch::kCUDA));

        K_residual_host.slice(1, new_lens, new_lens + seqlen_new).copy_(K_new_host);
        V_residual_host.slice(1, new_lens, new_lens + seqlen_new).copy_(V_new_host);
    }

    seqlen_kv = seqlen_kv + new_lens + seqlen_new;
    residual_len = seqlen_kv % residual_block_size == 0 ? residual_block_size : seqlen_kv % residual_block_size;
    new_lens = residual_len;

    K_new_device = K_residual_host.to(torch::kCUDA);
    V_new_device = V_residual_host.to(torch::kCUDA);

    std::optional<const at::Tensor> opt_K_new_device_1 = std::make_optional(K_new_device);
    std::optional<const at::Tensor> opt_V_new_device_1 = std::make_optional(V_new_device);
    std::optional<const at::Tensor> opt_seqlens_k_1    = std::make_optional(seqlens_k);

    auto [out_2, k_pack_new_2, k_params_new_2, v_pack_new_2, v_params_new_2]
        = mha_fwd_kvcache<num_bits>(Q_device,
            k_pack, k_params,
            v_pack, v_params,
            opt_K_new_device_1, opt_V_new_device_1, opt_seqlens_k_1,
            k_pack_new, k_params_new, v_pack_new, v_params_new,
            opt_block_table,
            sm_scale,
            quant_mode,
            group_size,
            residual_block_size,
            new_lens);

    torch::Tensor out_2_cpu = out_2.to(torch::kCPU);

    K_host_cat = torch::cat({K_host_cat, K_new_host}, 1);
    V_host_cat = torch::cat({V_host_cat, V_new_host}, 1);

    torch::Tensor out_ref_2 = single_mha(Q_host, K_host_cat, V_host_cat, head_dim);

    diff = out_2_cpu - out_ref_2;
    mean_absolute_error = diff.abs().mean().item<float>();
    mean_squared_error = diff.pow(2).mean().item<float>();
    max_error = diff.abs().max().item<float>();

    printf("\nnum_bits: %d num_heads_kv: %d seqlen_kv: %d head_dim: %d Quant_mode: %s, Group_size: %d\n", num_bits, num_heads_kv, seqlen_kv, head_dim, quant_mode.c_str(), group_size);
    if (mean_absolute_error < 1e-1 && mean_squared_error < 1e-1) {
        printf("test pass ! \n");
        printf("max_error: %f, mean_absolute_error: %f, mean_squared_error: %f\n", max_error, mean_absolute_error, mean_squared_error);
    } else {
        printf("test fail ! \n");
        printf("max_error: %f, mean_absolute_error: %f, mean_squared_error: %f\n", max_error, mean_absolute_error, mean_squared_error);
    }

    printf("\nFirst head output (out_cpu[0,0,0,:]):\n");
    out_cpu_accessor = out_2_cpu.index({0,0,0}).data_ptr<at::Half>();
    for (int i = 0; i < head_dim; i++) {
        printf("%.6f ", static_cast<float>(out_cpu_accessor[i]));
    }
    printf("\n\nFirst head output (out_ref[0,0,0,:]):\n");
    out_ref_accessor = out_ref_2.index({0,0,0}).data_ptr<at::Half>();
    for (int i = 0; i < head_dim; i++) {
        printf("%.6f ", static_cast<float>(out_ref_accessor[i]));
    }
    printf("\n\n");
}


int main() {
    const int num_heads = 32;
    const int num_heads_kv = 32;
    const int head_dim = 128;

    const int batch_size = 1;

    const std::string quant_mode = "k-channel";
    const int num_bits = 4;
    const int group_size = 128;
    const int seqlen_kvs[] = {1024};

    for (int base_seqlen : seqlen_kvs) {
        int seqlen_kv = base_seqlen - 33;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen - 32;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen - 10;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen - 1;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 31;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 32;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 - 40;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 - 23;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 - 1;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 + 10;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 + 31;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 + 32;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 + 50;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 128 + 130;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 256 - 1;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 256;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);

        seqlen_kv = base_seqlen + 256 + 1;
        TestDecodingKernelCorrectness<num_heads, num_heads_kv, head_dim, num_bits>(batch_size, seqlen_kv, quant_mode, group_size);
    }


    return 0;
}
