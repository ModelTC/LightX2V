# Kernel Basic Skills

Foundational skills for writing and optimizing SYCL-ESIMD kernels on Intel Xe2/Xe3 GPUs.

## Sub-Skills

### Hardware & Architecture

| Skill | Description |
|-------|-------------|
| [intel-gpu-hw-info](./intel-gpu-hw-info/SKILL.md) | Intel GPU hardware specs across Xe2 and Xe3: XE cores, memory bandwidth, GRF sizes, SLM limits, thread counts, L3 cache, TDP |
| [xe2-dpas-patterns](./xe2-dpas-patterns/SKILL.md) | DPAS operand load/store patterns on Xe2: all four operand load patterns and hardware contracts |
| [xe2-nbarrier-pipelining](./xe2-nbarrier-pipelining/SKILL.md) | Named barrier split-barrier pipelining patterns for latency hiding and overlapping operations on Xe2 |

### ESIMD Programming Model

| Skill | Description |
|-------|-------------|
| [intel-esimd-base](./intel-esimd-base/skill.md) | Foundational ESIMD kernel development: design patterns, memory access, vectorization, and debugging |
| [esimd-lsc-2d-gather-scatter](./esimd-lsc-2d-gather-scatter/SKILL.md) | LSC 2D block load/store, 1D block load/store, and gather/scatter memory access patterns |
| [esimd-lsc-slm](./esimd-lsc-slm/SKILL.md) | Shared Local Memory operations: initialization, block load/store, barriers, layout design, and advanced patterns |

### Optimization Techniques

| Skill | Description |
|-------|-------------|
| [intel-gpu-kernel-opt](./intel-gpu-kernel-opt/SKILL.md) | Systematic methodology for profiling, diagnosing, and optimizing GPU kernels via roofline analysis and bottleneck identification |
| [intel-esimd-fuse](./intel-esimd-fuse/SKILL.md) | Fused multi-operation kernels using ESIMD to minimize memory traffic (e.g., softmax + top-K + normalize) |
