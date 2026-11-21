---
name: benchmark-module-parallelism
description: Compare TP/SP strategies for the same real LightX2V model module with fixed global inputs and resources. Measure post-warmup completion latency, GPU memory, and numerical results. Use SP operator screening and module profiling as needed to explain differences. For standalone SP tuning when only QKV inputs are available, use benchmark-core-ops.
---

# Compare Module Parallelism Strategies

Read the tool navigation and the full-layer TP/SP section of the Benchmark Agent Guide first. Refer to the README for API and input contracts. Compare the same complete production module across configurations. Base full-layer rankings on full-module measurements, even when an attention-only benchmark has already identified a leading option. [Tool navigation](../../../tools/AGENTS.md) · [Benchmark Agent Guide](../../../tools/benchmarks/AGENTS.md) · [README](../../../tools/benchmarks/README.md#完整模型层tpsp-比较)

## Prepare Comparable Modules

1. Fix the global inputs, weights and source code, precision, attention implementation, and authorized resources. Define the layer boundary and state whether the inputs are real or synthetic. If the user has not specified an attention backend, use FA2 as the task baseline. Record unavailable dependencies or execution paths as gaps and make any backend change explicit.
2. Build a `PreparedModule` using the production loading path and parallel mesh. Provide `call` to run the production module, `reset` to restore inputs and state, `synchronize` to wait for all target streams and communication, and `check` to reconstruct the output and compare it with the reference. Record the common identity and the actual process, TP, and SP groups. Use the production call boundary to decide whether loading and final output reconstruction count toward per-layer cost.
3. List the TP/SP layouts and complete SP configurations to compare under the same resource constraints. Prefer the existing `sp_bench` probe and candidate enumeration. Determine local heads and inputs from actual TP sharding and generate candidates for the actual SP degree. Keep the SP-disabled candidate for SP=1. Record rejected, failed, and untested candidates. Describe the search coverage based on what was actually tested. [Configuration search guidelines](../../../tools/benchmarks/AGENTS.md#完整层tpsp与sp配置遍历)

## Choose the Tools

| Goal | Tool |
|---|---|
| Compare full-module completion latency, GPU memory, and numerical results | Run `tools.benchmarks.module_bench.benchmark` and compare the reports with `compare` |
| Narrow down SP configurations before benchmarking the module | Follow the SP workflow in [benchmark-core-ops](../benchmark-core-ops/SKILL.md) using the actual local inputs after TP sharding |
| Explain computation, communication, or gaps in a selected configuration | Use [profile-model-module](../profile-model-module/SKILL.md) to profile the same real module in a separate run |

When full-module candidates are already defined, compare them directly. Use preliminary screening and profiling as needed to answer the current question. For each configuration, preserve the attention implementation actually used, the TP/SP groups, the complete SP candidate configuration, and the input identity. Check that the production call uses the intended configuration.

For the main result, warm up in the same process and benchmark the complete module with the profiler disabled. For each iteration, take the maximum host-side completion latency across ranks, including completion of the target's asynchronous work. Use SP operator event-list timings, profiler timelines, and full-layer completion latency only within their respective measurement scopes. Follow the resource guide for monitoring, deadlines, cleanup of processes you started, invalidation of timings affected by interference, and bounded reruns. [Resource guide](../../../tools/runtime/AGENTS.md)

## Deliverables

Save the task factory and configuration, JSON reports and per-iteration raw data, commands, identities, and exit status. Write the report in Chinese using the full-layer report template. If screening was used, document the search dimensions, rejection reasons, and how many candidates were measured in each stage. [Full-layer report template](../../../tools/benchmarks/MODULE_REPORT_TEMPLATE.md) · [Search scope template](../../../tools/benchmarks/SEARCH_SCOPE_TEMPLATE.md)

Rank configurations within groups that share the GPU count, global target, and inputs. Report post-warmup latency, GPU memory deltas, and numerical ranges. Leave quality decisions about nonzero numerical differences to the developer. Use final image or video review when making product quality decisions. Limit conclusions to the tested module, inputs, and resources. Draw conclusions about the full model or serving only from measurements at those scopes.
