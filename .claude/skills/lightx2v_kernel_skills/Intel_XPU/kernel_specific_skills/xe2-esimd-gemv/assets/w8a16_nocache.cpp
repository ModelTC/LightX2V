#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// CPU Reference
void cpu_gemv_w8a16(const std::vector<sycl::half>& input,
                    const std::vector<int8_t>& weight,
                    const std::vector<sycl::half>& scale,
                    std::vector<float>& output,
                    int N, int K) {
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        float scale_f = static_cast<float>(scale[n]);
        for (int k = 0; k < K; k++) {
            float input_f = static_cast<float>(input[k]);
            float weight_f = static_cast<float>(weight[n * K + k]);
            sum += input_f * (weight_f * scale_f);
        }
        output[n] = sum;
    }
}

// GPU Kernel
template<int VL>
struct W8A16_GEMV {
    const sycl::half* input;
    const int8_t* weight;
    const sycl::half* scale;
    sycl::half* output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_global_id(0);
        if (n >= N) return;

        float sum = 0.0f;
        sycl::half scale_val = scale[n];
        float scale_f = static_cast<float>(scale_val);

        for (int k = 0; k < K; k += VL) {
            simd<sycl::half, VL> input_vec = block_load<sycl::half, VL>(input + k);
            simd<float, VL> input_f = input_vec;

            simd<int8_t, VL> weight_vec = block_load<int8_t, VL>(weight + n * K + k);
            simd<float, VL> weight_f = weight_vec;
            weight_f = weight_f * scale_f;

            sum += reduce<float>(input_f * weight_f, std::plus<>());
        }

        simd<sycl::half, 1> result = sycl::half(sum);
        block_store<sycl::half, 1>(output + n, result);
    }
};

void init_data(std::vector<sycl::half>& input, std::vector<int8_t>& weight,
               std::vector<sycl::half>& scale, int N, int K, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> w_dist(-127, 127);

    for (int i = 0; i < K; i++) input[i] = sycl::half(dist(gen));
    for (int n = 0; n < N; n++) {
        scale[n] = sycl::half(dist(gen) * 0.1f);
        for (int k = 0; k < K; k++)
            weight[n * K + k] = static_cast<int8_t>(w_dist(gen));
    }
}

bool verify_results(const std::vector<float>& cpu_output,
                   const std::vector<sycl::half>& gpu_output,
                   int N) {
    int errors = 0;
    float max_diff = 0.0f;
    float max_rel_error = 0.0f;
    float tolerance_abs = 0.5f;
    float tolerance_rel = 0.01f;

    for (int i = 0; i < N; i++) {
        float cpu_val = cpu_output[i];
        float gpu_val = static_cast<float>(gpu_output[i]);
        float diff = std::abs(cpu_val - gpu_val);
        float rel_error = std::abs(cpu_val) > 1e-6 ? diff / std::abs(cpu_val) : 0.0f;

        max_diff = std::max(max_diff, diff);
        max_rel_error = std::max(max_rel_error, rel_error);

        if (diff > tolerance_abs && rel_error > tolerance_rel) {
            if (errors < 3) {
                std::cout << "  Mismatch [" << i << "]: CPU=" << cpu_val
                         << ", GPU=" << gpu_val << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    std::cout << "Verification: " << (errors == 0 ? "PASSED ✓" : "FAILED ✗")
              << " (" << errors << " / " << N << " errors)" << std::endl;
    std::cout << "  Max abs diff: " << max_diff
              << ", max rel error: " << (max_rel_error*100) << "%" << std::endl;

    return errors == 0;
}

template<int VL>
double benchmark_no_cache(queue& q,
                         sycl::half* d_input,
                         std::vector<int8_t*>& d_weights,  // Multiple weight copies
                         sycl::half* d_scale,
                         sycl::half* d_output,
                         int N, int K) {
    int num_weight_copies = d_weights.size();

    std::cout << "  Using " << num_weight_copies << " weight copies to avoid cache" << std::endl;

    // Warmup with all weight copies
    for (int i = 0; i < 5; i++) {
        int weight_idx = i % num_weight_copies;
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(N, 1),
                W8A16_GEMV<VL>{d_input, d_weights[weight_idx], d_scale, d_output, N, K});
        }).wait();
    }

    // Benchmark - rotate through all weight copies
    constexpr int NUM_ITERS = 100;
    std::vector<double> times;
    times.reserve(NUM_ITERS);

    for (int i = 0; i < NUM_ITERS; i++) {
        int weight_idx = i % num_weight_copies;

        auto event = q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(N, 1),
                W8A16_GEMV<VL>{d_input, d_weights[weight_idx], d_scale, d_output, N, K});
        });
        event.wait();

        auto start_time = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end_time = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        double time_ns = end_time - start_time;
        times.push_back(time_ns / 1e6);
    }

    // Use median for robustness
    std::sort(times.begin(), times.end());
    double median_ms = times[NUM_ITERS / 2];
    double min_ms = times[0];
    double max_ms = times[NUM_ITERS - 1];

    // Calculate bandwidth based on actual memory traffic
    size_t bytes_read =
        K * sizeof(sycl::half) +           // Input
        (size_t)N * K * sizeof(int8_t) +   // Weight (one copy)
        N * sizeof(sycl::half);             // Scale

    size_t bytes_write = N * sizeof(sycl::half);  // Output
    size_t total_bytes = bytes_read + bytes_write;

    double bandwidth_gb_s = (total_bytes / 1e9) / (median_ms / 1000.0);

    std::cout << "  Timing statistics (ms):" << std::endl;
    std::cout << "    Min: " << std::fixed << std::setprecision(4) << min_ms << std::endl;
    std::cout << "    Median: " << median_ms << std::endl;
    std::cout << "    Max: " << max_ms << std::endl;
    std::cout << "  Memory traffic per kernel: " << (total_bytes / 1e6) << " MB" << std::endl;

    return bandwidth_gb_s;
}

void test_config(queue& q, int N, int K) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing N=" << N << ", K=" << K << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Calculate memory requirements
    size_t weight_size_mb = (size_t)N * K / (1024 * 1024);
    size_t input_size_mb = K * 2 / (1024 * 1024);
    size_t scale_output_size_mb = N * 4 / (1024 * 1024);

    std::cout << "Memory per weight copy: " << weight_size_mb << " MB" << std::endl;

    // Determine how many weight copies we can fit in 32GB
    size_t available_memory_mb = 30 * 1024;  // Use 30GB to be safe
    int max_copies = (available_memory_mb - input_size_mb - scale_output_size_mb) / weight_size_mb;
    int num_copies = std::min(128, std::max(32, max_copies));  // Between 32-128 copies

    std::cout << "Allocating " << num_copies << " weight copies ("
              << (num_copies * weight_size_mb) << " MB)" << std::endl;

    // Allocate host data
    std::vector<sycl::half> input(K);
    std::vector<sycl::half> scale(N);
    std::vector<float> cpu_output(N);
    std::vector<sycl::half> gpu_output(N);

    // Allocate multiple weight copies with different data
    std::vector<std::vector<int8_t>> weights(num_copies);
    for (int i = 0; i < num_copies; i++) {
        weights[i].resize(N * K);
        init_data(input, weights[i], scale, N, K, 42 + i);  // Different seed per copy
    }

    // CPU reference using first weight copy
    std::cout << "\n[1/3] Computing CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemv_w8a16(input, weights[0], scale, cpu_output, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "  CPU time: " << cpu_ms << " ms" << std::endl;

    // Allocate GPU memory
    std::cout << "\n[2/3] Allocating GPU memory..." << std::endl;
    sycl::half* d_input = malloc_device<sycl::half>(K, q);
    sycl::half* d_scale = malloc_device<sycl::half>(N, q);
    sycl::half* d_output = malloc_device<sycl::half>(N, q);

    // Allocate multiple weight copies on GPU
    std::vector<int8_t*> d_weights(num_copies);
    for (int i = 0; i < num_copies; i++) {
        d_weights[i] = malloc_device<int8_t>(N * K, q);
    }

    // Copy data to GPU
    q.memcpy(d_input, input.data(), K * sizeof(sycl::half)).wait();
    q.memcpy(d_scale, scale.data(), N * sizeof(sycl::half)).wait();

    for (int i = 0; i < num_copies; i++) {
        q.memcpy(d_weights[i], weights[i].data(), N * K * sizeof(int8_t)).wait();
    }

    std::cout << "Memory allocation complete." << std::endl;

    // Benchmark different VL values
    std::cout << "\n[3/3] Benchmarking with cache-busting..." << std::endl;
    std::cout << "\nVL   | BW (GB/s) | Util %  | 75%?" << std::endl;
    std::cout << "-----+-----------+---------+------" << std::endl;

    double best_bw = 0.0;
    int best_vl = 0;

    std::vector<int> vl_options = {128, 256, 512, 1024};

    for (int vl : vl_options) {
        double bw = 0.0;

        try {
            switch(vl) {
                case 128:  bw = benchmark_no_cache<128>(q, d_input, d_weights, d_scale, d_output, N, K); break;
                case 256:  bw = benchmark_no_cache<256>(q, d_input, d_weights, d_scale, d_output, N, K); break;
                case 512:  bw = benchmark_no_cache<512>(q, d_input, d_weights, d_scale, d_output, N, K); break;
                case 1024: bw = benchmark_no_cache<1024>(q, d_input, d_weights, d_scale, d_output, N, K); break;
            }
        } catch (std::exception& e) {
            std::cout << std::setw(5) << vl << "| ERROR: " << e.what() << std::endl;
            continue;
        }

        double util = bw / 520.0 * 100.0;
        bool passes = bw >= 390.0;

        std::cout << std::setw(5) << vl << "| "
                  << std::setw(9) << std::fixed << std::setprecision(2) << bw << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << util << " | "
                  << (passes ? "YES" : "no") << std::endl;

        if (bw > best_bw) {
            best_bw = bw;
            best_vl = vl;
        }
    }

    // Verify correctness using first weight copy
    std::cout << "\n[Verification] Checking correctness..." << std::endl;
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, 1),
            W8A16_GEMV<1024>{d_input, d_weights[0], d_scale, d_output, N, K});
    }).wait();

    q.memcpy(gpu_output.data(), d_output, N * sizeof(sycl::half)).wait();
    bool correct = verify_results(cpu_output, gpu_output, N);

    // Summary
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "SUMMARY for N=" << N << ", K=" << K << ":" << std::endl;
    std::cout << "  Weight copies used: " << num_copies << std::endl;
    std::cout << "  Best VL: " << best_vl << std::endl;
    std::cout << "  Best Bandwidth: " << std::fixed << std::setprecision(2)
              << best_bw << " GB/s" << std::endl;
    std::cout << "  Utilization: " << std::fixed << std::setprecision(2)
              << (best_bw/520*100) << "%" << std::endl;
    std::cout << "  Correctness: " << (correct ? "PASSED ✓" : "FAILED ✗") << std::endl;
    std::cout << "  75% Target (390 GB/s): "
              << (best_bw >= 390 ? "ACHIEVED ✓" : "Not reached ✗") << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Cleanup
    free(d_input, q);
    free(d_scale, q);
    free(d_output, q);
    for (auto d_w : d_weights) {
        free(d_w, q);
    }
}

int main() {
    try {
        property_list props{property::queue::enable_profiling()};
        queue q(gpu_selector_v, props);

        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Global memory size: "
                  << (q.get_device().get_info<sycl::info::device::global_mem_size>() / 1024 / 1024 / 1024)
                  << " GB" << std::endl;
        std::cout << "\nUsing multiple weight copies to prevent cache pollution" << std::endl;

        // Test different sizes
        test_config(q, 4096, 4096);
        test_config(q, 8192, 8192);
        test_config(q, 12288, 12288);

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
