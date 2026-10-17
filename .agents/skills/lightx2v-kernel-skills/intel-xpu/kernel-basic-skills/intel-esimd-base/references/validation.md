# Validation

Detailed guidance from the [skill entry](../SKILL.md). Read the sections relevant to the task.

## Testing New Kernels

Every new kernel needs an **accuracy test**. Add a **performance test** when delivering or claiming performance. Follow these rules:

### Rules

| Rule | Detail |
|------|--------|
| **Correctness before performance** | Validate each new kernel; add a benchmark when delivering or claiming performance. Documentation review does not require running either workload. |
| **Representative and boundary inputs** | Use seeded random data plus zero, near-zero, tail and non-finite cases for the verifier. All-zero inputs alone do not validate a kernel. |
| **Accuracy: small input** | Use a small input size to keep the CPU reference fast |
| **Accuracy: run once** | Run kernel once, run reference once, compare all outputs |
| **Accuracy: finite, per-element bounds** | Reject non-finite outputs/references. For each element require `abs_err <= max(atol, rtol * abs(ref))`; report both absolute and relative errors. Choose tolerances for the algorithm and dtype. |
| **Accuracy: print worst elements** | For the element with `max_abs_err` and the element with `max_rel_err`, print the index, `out` value, and `ref` value. This immediately shows whether errors are isolated outliers or systematic, and reveals near-zero reference values that inflate relative error. |
| **Perf: large input** | Use a larger input size to saturate memory/compute |
| **Perf: five-run recipe** | The example reports the minimum of five kernel-event timings. Record all runs, warmup/JIT and cache state; use a separate warmup for a steady-state claim. |
| **Perf: report FLOPS + BW** | Compute and print both TFLOPS and GB/s where applicable |

### Accuracy Test Pattern

```cpp
// Small input — fast CPU reference
constexpr int ACC_LEN = 1024;  // smaller than perf test

half* in_host  = sycl::malloc_host<half>(ACC_LEN, q);
half* out_host = sycl::malloc_host<half>(ACC_LEN, q);
half* in_dev   = sycl::malloc_device<half>(ACC_LEN, q);
half* out_dev  = sycl::malloc_device<half>(ACC_LEN, q);

// Random input
srand(42);
for (int i = 0; i < ACC_LEN; i++)
    in_host[i] = half(((float)rand() / RAND_MAX) * 2.f - 1.f);

q.memcpy(in_dev, in_host, ACC_LEN * sizeof(half)).wait();

// Run kernel once
my_kernel(q, in_dev, out_dev, ACC_LEN).wait();
q.memcpy(out_host, out_dev, ACC_LEN * sizeof(half)).wait();

// CPU reference once — check both absolute and relative error, record worst elements
bool pass = true;
int nonfinite = 0;
double max_abs_err = 0.0, max_rel_err = 0.0;
int    max_abs_idx = 0,   max_rel_idx = 0;
double max_abs_ref = 0.0, max_rel_ref = 0.0;
double max_abs_out = 0.0, max_rel_out = 0.0;
for (int i = 0; i < ACC_LEN; i++) {
    float ref = cpu_reference(in_host, i);
    if (!std::isfinite(ref) || !std::isfinite((double)out_host[i])) {
        nonfinite++;
        pass = false;
        continue;
    }
    double abs_err = std::abs((double)out_host[i] - (double)ref);
    double rel_err = abs_err / (std::abs((double)ref) + 1e-6);
    if (abs_err > 1e-2 && abs_err > 1e-3 * std::abs((double)ref)) pass = false;
    if (abs_err > max_abs_err) {
        max_abs_err = abs_err; max_abs_idx = i;
        max_abs_ref = ref;     max_abs_out = (double)out_host[i];
    }
    if (rel_err > max_rel_err) {
        max_rel_err = rel_err; max_rel_idx = i;
        max_rel_ref = ref;     max_rel_out = (double)out_host[i];
    }
}
// Acceptance is per element; maxima are diagnostics, not the acceptance predicate.
printf("  non-finite values: %d\n", nonfinite);
printf("  accuracy: max_abs_err=%.6f (idx=%d out=%.6f ref=%.6f)"
       "  max_rel_err=%.6f (idx=%d out=%.6f ref=%.6f)  %s\n",
       max_abs_err, max_abs_idx, max_abs_out, max_abs_ref,
       max_rel_err, max_rel_idx, max_rel_out, max_rel_ref,
       pass ? "PASS" : "FAIL");
sycl::free(in_host, q); sycl::free(out_host, q);
sycl::free(in_dev, q); sycl::free(out_dev, q);
// The enclosing executable must return nonzero if pass is false.
```

### Performance Test Pattern

```cpp
// Large input — saturate memory/compute
constexpr int PERF_LEN = 1 << 20;  // larger than accuracy test
constexpr int RUNS = 5;
// This is a separate example: allocate for PERF_LEN, not ACC_LEN.
half* in_host = sycl::malloc_host<half>(PERF_LEN, q);
half* in_dev = sycl::malloc_device<half>(PERF_LEN, q);
half* out_dev = sycl::malloc_device<half>(PERF_LEN, q);

// Random input (same srand(42) for reproducibility)
srand(42);
for (int i = 0; i < PERF_LEN; i++)
    in_host[i] = half(((float)rand() / RAND_MAX) * 2.f - 1.f);
q.memcpy(in_dev, in_host, PERF_LEN * sizeof(half)).wait();

// Run 5 times, keep best (use profiling events for kernel-only time)
double best_ns = std::numeric_limits<double>::max();
for (int r = 0; r < RUNS; r++) {
    sycl::event ev = my_kernel(q, in_dev, out_dev, PERF_LEN);
    ev.wait();
    double ns = (double)(
        ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
        ev.get_profiling_info<sycl::info::event_profiling::command_start>());
    if (ns < best_ns) best_ns = ns;
}

double ms    = best_ns * 1e-6;
double bytes = (double)(PERF_LEN) * sizeof(half) * 2;  // read + write
double bw_gb = (bytes / (best_ns * 1e-9)) / 1e9;
double flops = 2.0 * PERF_LEN;  // adjust per kernel
double tflops= (flops / (best_ns * 1e-9)) / 1e12;

printf("  perf: %.3f ms   %.2f TFLOPS   %.1f GB/s\n", ms, tflops, bw_gb);
sycl::free(in_host, q); sycl::free(in_dev, q); sycl::free(out_dev, q);
```

### Notes

- Use `sycl::property::queue::enable_profiling{}` on the queue for event-based timing.
- Event profiling measures **kernel-only** time, excluding host↔device memcpy; also report wall-clock time when transfer or launch overhead is part of the task.
- For kernels with no FLOP count (e.g., pure memory ops like topk), report only GB/s.
- A best-of-five result is an observed lower bound; it does not establish steady-state behavior or eliminate the need to disclose warmup and cache state.
- Apply the finite, per-element acceptance rule above with tolerances justified by the operation and precision.

---
