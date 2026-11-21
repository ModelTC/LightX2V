---
name: benchmark-core-ops
description: Compare production backends and execution configurations for core LightX2V operators, including GEMM, MoE, convolution, dense and sparse attention, and standalone SP attention. Use for fixed-input operator tuning and candidate coverage checks. Use benchmark-module-parallelism to compare TP/SP configurations for a complete module.
---

# Benchmark Core Operators

Measure one complete production operator call. Standalone SP attention tuning includes the attention computation, layout transformations, and communication. Reuse this SP workflow as needed when comparing a full module's TP/SP configurations.

Read the tool navigation and Benchmark Agent Guide first. Use the README for commands, input contracts, and metrics. Use the existing benchmark tooling to determine the backend list. [Tool navigation](../../../tools/AGENTS.md) · [Benchmark Agent Guide](../../../tools/benchmarks/AGENTS.md) · [README](../../../tools/benchmarks/README.md)

## Choose an Entry Point

| Goal | Tool |
|---|---|
| Single-GPU GEMM, MoE, Conv2d/3d, and dense or sparse attention | `tools.benchmarks.operator_bench` |
| Backends and execution configurations for standalone SP attention | `tools.benchmarks.sp_bench` with a separate suite and `torchrun` |

When shapes are known and you only need offline inspection or operation counts, use `inspect` and the shared operation-count interface. For performance comparisons, probe and time candidates on the target device. [Shared operation-count interface](../../../tools/benchmarks/README.md#计算量统计与真实moe回放)

The current SP CLI supports real QKV replay only on sparse paths and generates inputs for dense SP by default. For dense SP with fixed real QKV, wire the inputs into the task code and verify that the call actually uses them before reporting replay.

## Run the Benchmark

1. Trace the production call to establish the inputs, precision, and timing scope. Choose representative fixed inputs and generate a suite. For sparse attention, use real QKV captured before the backend, after the model's existing transformations such as RoPE. For SP, also verify the full global input at the pre-SP boundary and how the production path handles replicated auxiliary tokens. Prepare real MoE inputs and model-specific layouts in the task code following the guide. Use actual routing data for replay and treat histograms as summaries.
2. After `inspect`, probe the current production registry, dependencies, and input support in the target environment. Choose candidates within the requested scope and budget. Record included, rejected, failed, and untested candidates, along with entries awaiting integration. If a new production entry point needs an adapter, implement one that follows the real call or document the specific integration gap. Use measured results for performance exclusions.
3. Hold inputs and other conditions fixed. Warm up in the same process and complete any required library autotuning before timing with the tool's event-list mechanism. Get operation counts from the shared core or backend data and disable statistics collection for formal timing. Use the production call boundary to determine whether routing, quantization, and layout transformations belong in the measurement.
4. Compare valid, stable measurements first and run the smallest diagnostics needed to answer remaining questions. Rank SP configurations by the full-path L1 measurement. Capture a production timeline to explain head pipelining. The component simulation covers bulk execution only. Run numerical validation independently of fixed-workload timing and finite-output checks.

Follow the resource guide and Benchmark Agent Guide for execution and recovery. Check authorization and available resources, warm up in the same process, set deadlines, monitor resources during the run, invalidate timings affected by interference, and bound retries. After an interruption, check existing processes and artifacts first. `--append` skips complete records, including errors. Rerun failed cases in a separate recovery directory and retain the original errors. [Resource guide](../../../tools/runtime/AGENTS.md)

## Deliverables

Write the report in Chinese using the operator report template. Include the English operator and backend names, inputs and operation counts, the fastest complete configuration, a coverage table, and a link to the raw JSON. Group SP results by degree and rank dense, sparse, and SP results independently. [Operator report template](../../../tools/benchmarks/REPORT_TEMPLATE.md)

Report the available numerical evidence and its scope. Leave quality approval to the developer. Keep commands, the suite, input and source identities, logs, exit status, and resource checks in the current task record. Reuse evidence that still applies and rerun the benchmark matrix only when the measurements need to be revalidated.
