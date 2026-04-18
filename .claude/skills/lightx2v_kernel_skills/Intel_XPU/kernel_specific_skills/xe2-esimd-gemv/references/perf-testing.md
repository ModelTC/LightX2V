# GEMV Performance Testing

## Why cache-busting is mandatory

GEMV weight matrices are large but fixed across iterations.
Example: N=8192, K=4096, W8A16 → weight = 32 MB. BMG L3 ≈ 16 MB per tile.
Without rotation, the GPU L2/L3 cache serves most requests → measured ~1200 GB/s (fake).
With 32 weight copies (1 GB total), each iteration accesses a cold cache line → true DRAM BW.

**Rule**: always use ≥ 32 weight copies. Use 128 if memory allows.

## Cache-bust boilerplate

```cpp
// Host: allocate N_COPIES weight buffers with different seeds
constexpr int N_COPIES = 32;
std::vector<T*> d_weights(N_COPIES);
for (int i = 0; i < N_COPIES; i++) {
    d_weights[i] = sycl::malloc_device<T>(weight_elems, q);
    // init with seed (42 + i) so each copy has different data
    std::vector<T> h_w(weight_elems);
    init_weights(h_w, 42 + i);
    q.memcpy(d_weights[i], h_w.data(), weight_elems * sizeof(T)).wait();
}

// Benchmark loop
for (int i = 0; i < NUM_ITERS; i++) {
    int w_idx = i % N_COPIES;
    auto ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(nd_range, Kernel{d_input, d_weights[w_idx], d_scale, d_output, N, K});
    });
    ev.wait();
    // collect timing ...
}
```

## Timing harness

```cpp
property_list props{sycl::property::queue::enable_profiling()};
sycl::queue q(sycl::gpu_selector_v, props);

// Warmup (5 iters minimum)
for (int i = 0; i < 5; i++) {
    q.submit([&](sycl::handler& h) { h.parallel_for(...); }).wait();
}

// Timed iterations
constexpr int NUM_ITERS = 50;
std::vector<double> times_ms;
times_ms.reserve(NUM_ITERS);

for (int i = 0; i < NUM_ITERS; i++) {
    int w_idx = i % N_COPIES;
    auto ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(nd_range, Kernel{d_input, d_weights[w_idx], d_scale, d_output, N, K});
    });
    ev.wait();
    auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    times_ms.push_back((t1 - t0) / 1e6);
}

// Use median for robustness
std::sort(times_ms.begin(), times_ms.end());
double median_ms = times_ms[NUM_ITERS / 2];
```

## Bandwidth calculation

```cpp
// W4A16
size_t bytes = (size_t)K * sizeof(sycl::half)          // input
             + (size_t)N * (K/2) * sizeof(uint8_t)      // packed weight
             + (size_t)N * (K/128) * sizeof(sycl::half) // scale
             + (size_t)N * sizeof(sycl::half);           // output

// W8A16
size_t bytes = (size_t)K * sizeof(sycl::half)           // input
             + (size_t)N * K * sizeof(int8_t)            // weight
             + (size_t)N * sizeof(sycl::half)            // scale
             + (size_t)N * sizeof(sycl::half);           // output

double bw_gbs = (bytes / 1e9) / (median_ms / 1000.0);
double roofline_pct = bw_gbs / 520.0 * 100.0;   // 520 GB/s = BMG peak
double target_pct   = bw_gbs / 390.0 * 100.0;   // 390 GB/s = 75% target
```

## Random init (mandatory — no all-zeros)

```cpp
// W4A16 weight init
std::mt19937 gen(seed);
std::uniform_int_distribution<int> w4_dist(0, 15);
for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
        uint8_t lo = w4_dist(gen);
        uint8_t hi = w4_dist(gen);
        weight_packed[n * (K/2) + k/2] = (hi << 4) | lo;
    }
}

// W8A16 weight init
std::uniform_int_distribution<int> w8_dist(-127, 127);
for (int i = 0; i < N * K; i++)
    weight[i] = static_cast<int8_t>(w8_dist(gen));

// Input init
std::uniform_real_distribution<float> in_dist(-1.0f, 1.0f);
for (int k = 0; k < K; k++)
    input[k] = sycl::half(in_dist(gen));

// Scale init (keep small to avoid fp16 overflow)
for (int i = 0; i < scale_size; i++)
    scale[i] = sycl::half(in_dist(gen) * 0.1f);
```

## Reporting

```cpp
std::cout << std::fixed << std::setprecision(2);
std::cout << bw_gbs << " GB/s (" << roofline_pct << "% roofline)";
if (bw_gbs >= 390.0) std::cout << " TARGET HIT";
```

## Expected results on BMG

| Kernel | N | K | BW |
|--------|---|---|----|
| W4A16 R=4 VL=1024 K=2 | 8192 | 4096 | ~571 GB/s |
| W4A16 R=4 VL=1024 K=2 | 16384 | 4096 | ~571 GB/s |
| W8A16 VL=1024 | 8192 | 8192 | ~552 GB/s |
| W8A16 VL=1024 | 4096 | 4096 | ~548 GB/s |
