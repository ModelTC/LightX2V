---
name: profile-model-module
description: Profile real functions or modules in LightX2V models and explain their performance. Capture inputs, set up replay, correlate operation counts, and analyze multi-rank communication as needed. Use for transformer layers, attention, encoders, and VAEs. Use benchmark tools to compare backend speed or rank parallel configurations for a full module.
---

# Profile a Model Module

Read the tool navigation and Profiler Agent Guide first. Use the README to choose the CLI or API. Choose the smallest scope that answers the question. This can be one synchronous function or module, or the same production call across ranks. [Tool navigation](../../../tools/AGENTS.md) · [Profiler Agent Guide](../../../tools/profile/AGENTS.md) · [README](../../../tools/profile/README.md)

For host-only observation or standalone operation counts, use `--observe-only` or `--stats-only`, respectively, as described in the README. Both modes skip profiler startup. Use a clean or joint capture when you need a timeline.

## Set Up the Production Call

1. Trace the model script to the loaded weights, configuration, inputs, and target call. Check existing request costs or normal measurements first. Verify that measurements attributed to the model came from its real execution and identify any standard synthetic layers as such. Stay within the scope of a local analysis if that is what the user requested.
2. For replay, save the actual tensors, weights or quantized representation, necessary state, and capture point. Capture values before buffers are overwritten, then reconstruct the production call and its parallel groups. Provide loading and reset logic for the task and compare outputs from the original call and replay first. Use the captured tensors as replay inputs and treat shapes, routing summaries, and seeds as supporting metadata.
3. Verify that the entry point can make enough real calls and restore state within the same process before choosing the `Capture` CLI or API. The profiler scheduler's warmup applies within that process. Check the README for CLI wrapping limits and import safety. Make any repeat execution of a stateful call explicit.

## Capture and Interpret

| What you need | Capture mode |
|---|---|
| Normal timeline, gaps, and possible synchronization issues | Clean `Capture` with operation counting and `record_shapes` disabled |
| Shapes, operation counts, actual sparse regions, and kernel attribution from the same call | Run a separate joint capture, then use `trace_report` |
| The same module across ranks and its communication | Use a separate directory for each rank, record the shared run or batch identity and actual TP/SP groups, then use `parallel_report` |

Use `trace_report` for ordinary single-GPU traces as well. When loading and reset are outside the target, analyze the actual scope of the active calls. Include asynchronous GPU activity that continues after the host call returns. For multi-rank analysis, get TP/SP groups from actual group metadata and use `WORLD_SIZE` only as a rank count. Combine captures only from the same configuration and batch.

Get operation counts from facts recorded by the owning core or backend during the same call. Reserve `record_shapes` for shape debugging. Record coverage gaps for compiled graphs, uninstrumented entry points, or missing final regions. A separate eager statistics run can add context. Claim per-call attribution only for facts collected with the original trace. For large gaps, check same-process warmup and capture overhead first, then trace the actual dependencies behind CPU waits and GPU activity.

Use benchmarks with the profiler disabled for formal performance comparisons. Preserve the provenance of both the clean timeline and the joint capture.

Follow the Profiler Agent Guide and resource guide for execution, process cleanup, and recovery after an interruption. Verify existing processes, artifacts, and run identity before retrying. Keep timings affected by interference and mark them invalid. [Resource guide](../../../tools/runtime/AGENTS.md)

## Deliverables

When a trace is available, follow the shared module report structure and core compute table conventions for both single-GPU and multi-GPU reports. Add communication analysis for multi-GPU runs alongside computation, data movement, gaps, and synchronization. Provide one verified, accessible TensorBoard entry point for the raw capture. Keep detailed evidence in the facts artifacts. [Shared module report structure](../../../tools/profile/MODULE_PROFILE_REPORT_TEMPLATE.md) · [Core compute table conventions](../../../tools/profile/CORE_COMPUTE_TABLE.md) · [Communication analysis](../../../tools/profile/PARALLEL_REPORT_TEMPLATE.md)

For host-only observation or standalone operation counts, deliver the session and call records or statistics artifacts that were produced, and state their scope. Include GPU timeline analysis and a TensorBoard entry only when supported by a trace. Collect additional data only when the task calls for it.

Attribute measurements to the tools and interpretations to the agent. Report any task-specific wiring and what you actually verified. Supply any capture or reconstruction scripts the new model needs as part of the task. Deliver stable, reusable functionality with the code it belongs to. Base replay support claims on the wiring actually provided.
