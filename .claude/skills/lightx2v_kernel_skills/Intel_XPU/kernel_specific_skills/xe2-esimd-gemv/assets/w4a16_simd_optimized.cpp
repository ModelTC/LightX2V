#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

constexpr int BLOCK_SIZE = 128;

// CPU Reference
void cpu_gemv_w4a16(const std::vector<sycl::half>& input,
                    const std::vector<uint8_t>& weight_packed,
                    const std::vector<sycl::half>& scale,
                    std::vector<float>& output,
                    int N, int K) {
    int scale_stride = K / BLOCK_SIZE;

    for (int n = 0; n < N; n++) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            float input_f = static_cast<float>(input[k]);

            int packed_idx = n * (K/2) + k/2;
            uint8_t packed = weight_packed[packed_idx];
            uint8_t w4 = (k % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);

            int scale_idx = n * scale_stride + k / BLOCK_SIZE;
            float scale_f = static_cast<float>(scale[scale_idx]);
            float weight_f = (static_cast<float>(w4) - 8.0f) * scale_f;

            sum += input_f * weight_f;
        }

        output[n] = sum;
    }
}

// Optimized with SIMD select for interleaving
template<int ROWS, int VL, int K_SPLIT>
struct W4A16_SIMD_Optimized {
    const sycl::half* input;
    const uint8_t* weight;
    const sycl::half* scale;
    sycl::half* output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int SLM_SIZE = ROWS * K_SPLIT * sizeof(float);
        slm_init(SLM_SIZE);

        int local_id = item.get_local_id(0);
        int group_id = item.get_group(0);

        int n_start = group_id * ROWS;
        if (n_start >= N) return;

        int k_thread_id = local_id % K_SPLIT;
        int row_thread_id = local_id / K_SPLIT;

        int K_PER_THREAD = K / K_SPLIT;
        constexpr int NUM_BLOCKS = VL / BLOCK_SIZE;
        int K_BLOCKS = K / BLOCK_SIZE;

        simd<float, 8> partial_sums = 0.0f;
        int accumulator_idx = 0;

        if (row_thread_id < ROWS) {
            int n = n_start + row_thread_id;
            if (n < N) {
                int k_start = k_thread_id * K_PER_THREAD;
                int k_end = k_start + K_PER_THREAD;

                for (int k_base = k_start; k_base < k_end; k_base += VL) {
                    simd<sycl::half, VL> input_vec = block_load<sycl::half, VL>(input + k_base);
                    simd<float, VL> input_f = input_vec;

                    simd<uint8_t, VL/2> weight_packed =
                        block_load<uint8_t, VL/2>(weight + n * (K/2) + k_base/2);

                    simd<sycl::half, NUM_BLOCKS> scale_vec =
                        block_load<sycl::half, NUM_BLOCKS>(scale + n * K_BLOCKS + k_base / BLOCK_SIZE);

                    simd<float, VL> weight_f;

                    #pragma unroll
                    for (int blk = 0; blk < NUM_BLOCKS; blk++) {
                        sycl::half sh = scale_vec[blk];
                        float sc = static_cast<float>(sh);

                        int offset = blk * 64;
                        auto p = weight_packed.template select<64, 1>(offset);

                        simd<float, 64> lo = p & 0x0F;
                        simd<float, 64> hi = (p >> 4) & 0x0F;
                        lo = (lo - 8.0f) * sc;
                        hi = (hi - 8.0f) * sc;

                        // CRITICAL OPTIMIZATION: Use SIMD select for interleaving
                        // Directly select on weight_f, not on a temporary view
                        int base = blk * BLOCK_SIZE;
                        weight_f.template select<64, 2>(base + 0) = lo;  // Even positions
                        weight_f.template select<64, 2>(base + 1) = hi;  // Odd positions
                    }

                    float dot_product = reduce<float>(input_f * weight_f, std::plus<>());
                    partial_sums[accumulator_idx] += dot_product;
                    accumulator_idx = (accumulator_idx + 1) & 0x7;
                }
            }
        }

        float partial_sum = reduce<float>(partial_sums, std::plus<>());

        if (row_thread_id < ROWS) {
            uint32_t slm_offset = (row_thread_id * K_SPLIT + k_thread_id) * sizeof(float);
            simd<float, 1> sum_vec = partial_sum;
            slm_block_store<float, 1>(slm_offset, sum_vec);
        }

        barrier();

        if (k_thread_id == 0 && row_thread_id < ROWS) {
            int n = n_start + row_thread_id;
            if (n < N) {
                uint32_t slm_base = row_thread_id * K_SPLIT * sizeof(float);

                float final_sum = 0.0f;

                if constexpr (K_SPLIT == 1) {
                    simd<float, 1> partial_results = slm_block_load<float, 1>(slm_base);
                    final_sum = partial_results[0];
                } else if constexpr (K_SPLIT == 2) {
                    simd<float, 2> partial_results = slm_block_load<float, 2>(slm_base);
                    final_sum = partial_results[0] + partial_results[1];
                } else if constexpr (K_SPLIT == 4) {
                    simd<float, 4> partial_results = slm_block_load<float, 4>(slm_base);
                    final_sum = reduce<float>(partial_results, std::plus<>());
                } else if constexpr (K_SPLIT == 8) {
                    simd<float, 8> partial_results = slm_block_load<float, 8>(slm_base);
                    final_sum = reduce<float>(partial_results, std::plus<>());
                }

                output[n] = sycl::half(final_sum);
            }
        }
    }
};

// 2D split with SIMD optimization
template<int WG_ROWS, int VL, int K_SPLIT, int ROW_SPLIT>
struct W4A16_2D_SIMD {
    const sycl::half* input;
    const uint8_t* weight;
    const sycl::half* scale;
    sycl::half* output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int SLM_SIZE = WG_ROWS * K_SPLIT * sizeof(float);
        slm_init(SLM_SIZE);

        int local_id = item.get_local_id(0);
        int group_id = item.get_group(0);

        int n_start = group_id * WG_ROWS;
        if (n_start >= N) return;

        int k_thread_id = local_id % K_SPLIT;
        int row_thread_id = local_id / K_SPLIT;

        constexpr int ROWS_PER_THREAD = WG_ROWS / ROW_SPLIT;
        int K_PER_THREAD = K / K_SPLIT;
        constexpr int NUM_BLOCKS = VL / BLOCK_SIZE;
        int K_BLOCKS = K / BLOCK_SIZE;

        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int local_row = row_thread_id * ROWS_PER_THREAD + r;
            if (local_row >= WG_ROWS) continue;

            int n = n_start + local_row;
            if (n >= N) continue;

            simd<float, 8> partial_sums = 0.0f;
            int accumulator_idx = 0;

            int k_start = k_thread_id * K_PER_THREAD;
            int k_end = k_start + K_PER_THREAD;

            for (int k_base = k_start; k_base < k_end; k_base += VL) {
                simd<sycl::half, VL> input_vec = block_load<sycl::half, VL>(input + k_base);
                simd<float, VL> input_f = input_vec;

                simd<uint8_t, VL/2> weight_packed =
                    block_load<uint8_t, VL/2>(weight + n * (K/2) + k_base/2);

                simd<sycl::half, NUM_BLOCKS> scale_vec =
                    block_load<sycl::half, NUM_BLOCKS>(scale + n * K_BLOCKS + k_base / BLOCK_SIZE);

                simd<float, VL> weight_f;

                #pragma unroll
                for (int blk = 0; blk < NUM_BLOCKS; blk++) {
                    sycl::half sh = scale_vec[blk];
                    float sc = static_cast<float>(sh);

                    int offset = blk * 64;
                    auto p = weight_packed.template select<64, 1>(offset);

                    simd<float, 64> lo = p & 0x0F;
                    simd<float, 64> hi = (p >> 4) & 0x0F;
                    lo = (lo - 8.0f) * sc;
                    hi = (hi - 8.0f) * sc;

                    // SIMD select interleaving
                    int base = blk * BLOCK_SIZE;
                    weight_f.template select<64, 2>(base + 0) = lo;
                    weight_f.template select<64, 2>(base + 1) = hi;
                }

                float dot_product = reduce<float>(input_f * weight_f, std::plus<>());
                partial_sums[accumulator_idx] += dot_product;
                accumulator_idx = (accumulator_idx + 1) & 0x7;
            }

            float partial_sum = reduce<float>(partial_sums, std::plus<>());
            uint32_t slm_offset = (local_row * K_SPLIT + k_thread_id) * sizeof(float);
            simd<float, 1> sum_vec = partial_sum;
            slm_block_store<float, 1>(slm_offset, sum_vec);
        }

        barrier();

        if (k_thread_id == 0) {
            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                int local_row = row_thread_id * ROWS_PER_THREAD + r;
                if (local_row >= WG_ROWS) continue;

                int n = n_start + local_row;
                if (n >= N) continue;

                uint32_t slm_base = local_row * K_SPLIT * sizeof(float);

                float final_sum = 0.0f;

                if constexpr (K_SPLIT == 1) {
                    simd<float, 1> partial_results = slm_block_load<float, 1>(slm_base);
                    final_sum = partial_results[0];
                } else if constexpr (K_SPLIT == 2) {
                    simd<float, 2> partial_results = slm_block_load<float, 2>(slm_base);
                    final_sum = partial_results[0] + partial_results[1];
                } else if constexpr (K_SPLIT == 4) {
                    simd<float, 4> partial_results = slm_block_load<float, 4>(slm_base);
                    final_sum = reduce<float>(partial_results, std::plus<>());
                } else if constexpr (K_SPLIT == 8) {
                    simd<float, 8> partial_results = slm_block_load<float, 8>(slm_base);
                    final_sum = reduce<float>(partial_results, std::plus<>());
                }

                output[n] = sycl::half(final_sum);
            }
        }
    }
};

void init_data(std::vector<sycl::half>& input,
               std::vector<uint8_t>& weight_packed,
               std::vector<sycl::half>& scale,
               int N, int K, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> w_dist(0, 15);

    for (int i = 0; i < K; i++) {
        input[i] = sycl::half(dist(gen));
    }

    int scale_size = N * (K / BLOCK_SIZE);
    for (int i = 0; i < scale_size; i++) {
        scale[i] = sycl::half(dist(gen) * 0.1f);
    }

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k += 2) {
            uint8_t low = static_cast<uint8_t>(w_dist(gen));
            uint8_t high = static_cast<uint8_t>(w_dist(gen));
            uint8_t packed = (high << 4) | low;
            weight_packed[n * (K/2) + k/2] = packed;
        }
    }
}

bool verify_results(const std::vector<float>& cpu_output,
                   const std::vector<sycl::half>& gpu_output,
                   int N) {
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < N; i++) {
        float cpu_val = cpu_output[i];
        float gpu_val = static_cast<float>(gpu_output[i]);
        float diff = std::abs(cpu_val - gpu_val);
        float rel_error = std::abs(cpu_val) > 1e-6 ? diff / std::abs(cpu_val) : 0.0f;

        max_diff = std::max(max_diff, diff);

        if (diff > 1.0f && rel_error > 0.02f) {
            if (errors < 3) {
                std::cout << "  [" << i << "]: CPU=" << cpu_val << ", GPU=" << gpu_val << std::endl;
            }
            errors++;
        }
    }

    std::cout << "Verification: " << (errors == 0 ? "PASSED ✓" : "FAILED ✗")
              << " (" << errors << " / " << N << "), max diff=" << max_diff << std::endl;

    return errors == 0;
}

template<int ROWS, int VL, int K_SPLIT>
double benchmark_ksplit(queue& q,
                       sycl::half* d_input,
                       std::vector<uint8_t*>& d_weights,
                       sycl::half* d_scale,
                       sycl::half* d_output,
                       int N, int K,
                       const std::string& name) {
    int num_copies = d_weights.size();
    int num_groups = (N + ROWS - 1) / ROWS;
    int local_size = K_SPLIT * ROWS;
    int global_size = num_groups * local_size;

    std::cout << "  " << name << "..." << std::flush;

    for (int i = 0; i < 5; i++) {
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, local_size),
                W4A16_SIMD_Optimized<ROWS, VL, K_SPLIT>{d_input, d_weights[i % num_copies], d_scale, d_output, N, K});
        }).wait();
    }

    std::vector<double> times;
    for (int i = 0; i < 50; i++) {
        auto event = q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, local_size),
                W4A16_SIMD_Optimized<ROWS, VL, K_SPLIT>{d_input, d_weights[i % num_copies], d_scale, d_output, N, K});
        });
        event.wait();

        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        times.push_back((end - start) / 1e6);
    }

    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];

    size_t bytes = K * 2 + (size_t)N * (K/2) + (size_t)N * (K/BLOCK_SIZE) * 2 + N * 2;
    double bandwidth_gb_s = (bytes / 1e9) / (median_ms / 1000.0);

    std::cout << " " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s ("
              << (bandwidth_gb_s/520*100) << "%)";
    if (bandwidth_gb_s >= 390.0) std::cout << " ✓✓✓ TARGET!";
    else if (bandwidth_gb_s >= 350.0) std::cout << " ✓✓";
    else if (bandwidth_gb_s >= 300.0) std::cout << " ✓";
    std::cout << std::endl;

    return bandwidth_gb_s;
}

template<int WG_ROWS, int VL, int K_SPLIT, int ROW_SPLIT>
double benchmark_2d(queue& q,
                   sycl::half* d_input,
                   std::vector<uint8_t*>& d_weights,
                   sycl::half* d_scale,
                   sycl::half* d_output,
                   int N, int K,
                   const std::string& name) {
    int num_copies = d_weights.size();
    int num_groups = (N + WG_ROWS - 1) / WG_ROWS;
    int local_size = K_SPLIT * ROW_SPLIT;
    int global_size = num_groups * local_size;

    std::cout << "  " << name << "..." << std::flush;

    for (int i = 0; i < 5; i++) {
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, local_size),
                W4A16_2D_SIMD<WG_ROWS, VL, K_SPLIT, ROW_SPLIT>{d_input, d_weights[i % num_copies], d_scale, d_output, N, K});
        }).wait();
    }

    std::vector<double> times;
    for (int i = 0; i < 50; i++) {
        auto event = q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, local_size),
                W4A16_2D_SIMD<WG_ROWS, VL, K_SPLIT, ROW_SPLIT>{d_input, d_weights[i % num_copies], d_scale, d_output, N, K});
        });
        event.wait();

        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        times.push_back((end - start) / 1e6);
    }

    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];

    size_t bytes = K * 2 + (size_t)N * (K/2) + (size_t)N * (K/BLOCK_SIZE) * 2 + N * 2;
    double bandwidth_gb_s = (bytes / 1e9) / (median_ms / 1000.0);

    std::cout << " " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s ("
              << (bandwidth_gb_s/520*100) << "%)";
    if (bandwidth_gb_s >= 390.0) std::cout << " ✓✓✓ TARGET!";
    else if (bandwidth_gb_s >= 350.0) std::cout << " ✓✓";
    else if (bandwidth_gb_s >= 300.0) std::cout << " ✓";
    std::cout << std::endl;

    return bandwidth_gb_s;
}

int main() {
    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);

    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "W4A16 with SIMD Select Optimization" << std::endl;
    std::cout << "Using weight_f.select<64,2>(0)=lo and select<64,2>(1)=hi for efficient interleaving" << std::endl;
    std::cout << "Target: 390 GB/s (75% roofline)\n" << std::endl;

    int N = 16384;
    int K = 4096;
    int num_copies = 32;

    std::vector<sycl::half> input(K);
    std::vector<sycl::half> scale(N * (K / BLOCK_SIZE));
    std::vector<float> cpu_output(N);
    std::vector<sycl::half> gpu_output(N);

    std::vector<std::vector<uint8_t>> weights(num_copies);
    for (int i = 0; i < num_copies; i++) {
        weights[i].resize(N * (K/2));
        init_data(input, weights[i], scale, N, K, 42 + i);
    }

    std::cout << "[1/3] CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemv_w4a16(input, weights[0], scale, cpu_output, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "  Time: " << std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count()
              << " ms\n" << std::endl;

    std::cout << "[2/3] GPU allocation..." << std::endl;
    sycl::half* d_input = malloc_device<sycl::half>(K, q);
    sycl::half* d_scale = malloc_device<sycl::half>(N * (K / BLOCK_SIZE), q);
    sycl::half* d_output = malloc_device<sycl::half>(N, q);

    std::vector<uint8_t*> d_weights(num_copies);
    for (int i = 0; i < num_copies; i++) {
        d_weights[i] = malloc_device<uint8_t>(N * (K/2), q);
    }

    q.memcpy(d_input, input.data(), K * sizeof(sycl::half)).wait();
    q.memcpy(d_scale, scale.data(), N * (K / BLOCK_SIZE) * sizeof(sycl::half)).wait();
    for (int i = 0; i < num_copies; i++) {
        q.memcpy(d_weights[i], weights[i].data(), N * (K/2)).wait();
    }
    std::cout << "Done.\n" << std::endl;

    std::cout << "[3/3] Benchmarking with SIMD select optimization...\n" << std::endl;

    double best_bw = 0.0;
    std::string best_config;

    std::cout << "=== K-Split Tests ===" << std::endl;
    double bw1 = benchmark_ksplit<4, 1024, 1>(q, d_input, d_weights, d_scale, d_output, N, K, "R=4 VL=1024 K=1");
    if (bw1 > best_bw) { best_bw = bw1; best_config = "K-split: R=4 VL=1024 K=1"; }

    double bw2 = benchmark_ksplit<8, 1024, 1>(q, d_input, d_weights, d_scale, d_output, N, K, "R=8 VL=1024 K=1");
    if (bw2 > best_bw) { best_bw = bw2; best_config = "K-split: R=8 VL=1024 K=1"; }

    double bw3 = benchmark_ksplit<4, 1024, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "R=4 VL=1024 K=2");
    if (bw3 > best_bw) { best_bw = bw3; best_config = "K-split: R=4 VL=1024 K=2"; }

    double bw4 = benchmark_ksplit<8, 1024, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "R=8 VL=1024 K=2");
    if (bw4 > best_bw) { best_bw = bw4; best_config = "K-split: R=8 VL=1024 K=2"; }

    double bw5 = benchmark_ksplit<12, 1024, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "R=12 VL=1024 K=2");
    if (bw5 > best_bw) { best_bw = bw5; best_config = "K-split: R=12 VL=1024 K=2"; }

    double bw6 = benchmark_ksplit<16, 1024, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "R=16 VL=1024 K=2");
    if (bw6 > best_bw) { best_bw = bw6; best_config = "K-split: R=16 VL=1024 K=2"; }

    double bw7 = benchmark_ksplit<8, 1024, 4>(q, d_input, d_weights, d_scale, d_output, N, K, "R=8 VL=1024 K=4");
    if (bw7 > best_bw) { best_bw = bw7; best_config = "K-split: R=8 VL=1024 K=4"; }

    double bw8 = benchmark_ksplit<8, 512, 4>(q, d_input, d_weights, d_scale, d_output, N, K, "R=8 VL=512 K=4");
    if (bw8 > best_bw) { best_bw = bw8; best_config = "K-split: R=8 VL=512 K=4"; }

    double bw9 = benchmark_ksplit<8, 512, 8>(q, d_input, d_weights, d_scale, d_output, N, K, "R=8 VL=512 K=8");
    if (bw9 > best_bw) { best_bw = bw9; best_config = "K-split: R=8 VL=512 K=8"; }

    std::cout << "\n=== 2D Split Tests ===" << std::endl;
    double bw10 = benchmark_2d<8, 1024, 2, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=8 VL=1024 K=2 R=2");
    if (bw10 > best_bw) { best_bw = bw10; best_config = "2D: WG=8 VL=1024 K=2 R=2"; }

    double bw11 = benchmark_2d<8, 1024, 4, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=8 VL=1024 K=4 R=2");
    if (bw11 > best_bw) { best_bw = bw11; best_config = "2D: WG=8 VL=1024 K=4 R=2"; }

    double bw12 = benchmark_2d<12, 1024, 2, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=12 VL=1024 K=2 R=2");
    if (bw12 > best_bw) { best_bw = bw12; best_config = "2D: WG=12 VL=1024 K=2 R=2"; }

    double bw13 = benchmark_2d<16, 1024, 2, 2>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=16 VL=1024 K=2 R=2");
    if (bw13 > best_bw) { best_bw = bw13; best_config = "2D: WG=16 VL=1024 K=2 R=2"; }

    double bw14 = benchmark_2d<8, 512, 4, 4>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=8 VL=512 K=4 R=4");
    if (bw14 > best_bw) { best_bw = bw14; best_config = "2D: WG=8 VL=512 K=4 R=4"; }

    double bw15 = benchmark_2d<16, 512, 4, 4>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=16 VL=512 K=4 R=4");
    if (bw15 > best_bw) { best_bw = bw15; best_config = "2D: WG=16 VL=512 K=4 R=4"; }

    double bw16 = benchmark_2d<8, 1024, 2, 4>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=8 VL=1024 K=2 R=4");
    if (bw16 > best_bw) { best_bw = bw16; best_config = "2D: WG=8 VL=1024 K=2 R=4"; }

    double bw17 = benchmark_2d<16, 1024, 2, 4>(q, d_input, d_weights, d_scale, d_output, N, K, "WG=16 VL=1024 K=2 R=4");
    if (bw17 > best_bw) { best_bw = bw17; best_config = "2D: WG=16 VL=1024 K=2 R=4"; }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BEST RESULT WITH SIMD SELECT OPTIMIZATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Config: " << best_config << std::endl;
    std::cout << "Bandwidth: " << std::fixed << std::setprecision(2) << best_bw << " GB/s" << std::endl;
    std::cout << "Utilization: " << (best_bw/520*100) << "%" << std::endl;
    std::cout << "Progress to 390 GB/s target: " << (best_bw/390*100) << "%" << std::endl;
    std::cout << "Gap to target: " << (390 - best_bw) << " GB/s" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Verify
    std::cout << "\n[Verification]" << std::endl;
    constexpr int VERIFY_ROWS = 8;
    constexpr int VERIFY_K_SPLIT = 2;
    int num_groups = (N + VERIFY_ROWS - 1) / VERIFY_ROWS;
    int local_size = VERIFY_ROWS * VERIFY_K_SPLIT;  // Should be 16
    int global_size = num_groups * local_size;
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(global_size, local_size),
            W4A16_SIMD_Optimized<VERIFY_ROWS, 1024, VERIFY_K_SPLIT>{d_input, d_weights[0], d_scale, d_output, N, K});
    }).wait();
    q.memcpy(gpu_output.data(), d_output, N * sizeof(sycl::half)).wait();
    verify_results(cpu_output, gpu_output, N);

    free(d_input, q);
    free(d_scale, q);
    free(d_output, q);
    for (auto d_w : d_weights) free(d_w, q);

    return 0;
}
