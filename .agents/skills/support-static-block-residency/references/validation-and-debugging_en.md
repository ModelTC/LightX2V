# Validation and Debugging

English | [简体中文](validation-and-debugging.md)

## Select validation by change

First inspect current tests and the target environment. Run checks that cover the change and unresolved issues. Documentation changes need link/content checks; launcher changes add syntax, path, and argument checks without automatically rerunning every model experiment.

| Change | Main checks |
|---|---|
| Count calculation or placement | Zero for `d<=c`; intermediate and near-full residency; minimum integer budget; index count, bounds, uniqueness, and uniformity |
| Model estimation case | Production weights and dtype, input shapes/text lengths, RoPE, finite outputs, input reset excluded from compute timing |
| Config output | Other source settings retained, count consistent with report, failed writes do not leave partial JSON |
| Inference callback or slot scheduling | Block/state ordering, actual transferred layers, ready/free dependencies, numerical equivalence for both layouts |
| Lifetime | Repeated execution, rebuilding after release, cancellation/exceptions, warmup and pre-VAE release, CPU sources retained |
| Launchers or configs | Check shell syntax; launch from outside the repository and verify config/input/output paths and the unchanged working directory; run consumes estimator output, baseline runs independently, inputs/environment match, profiling comes from `base.sh` |

## Current validation entry points

Use the existing launchers for the target model:

- [Wan](../../../../scripts/wan/inference/offload_residency/README.md)
- [Qwen Image](../../../../scripts/qwen_image/inference/offload_residency/README.md)
- [Qwen-Image-2.1-Turbo](../../../../scripts/qwen_image_21/inference/offload_residency/README.md)
- [MiniMax-H3](../../../../scripts/minimax_h3/inference/offload_residency/README.md)

Each directory provides `estimate.sh`, `run_baseline.sh`, `run.sh`, and NPU counterparts. When full call-chain validation is needed, generate the residency config first, then run baseline and resident inference and check config loading, outputs, and logs.

`test_cases/` now matches upstream, and the former `*_static_residency.py` check files have been removed. Select checks for the formula, uniform placement, event execution, and cleanup based on the current implementation and the change. Do not invoke pytest or `python -m test_cases.*` commands for files that no longer exist.

Record results and uncovered cases. Missing hardware or skipped checks do not count as passes. Small-shape block checks do not establish default-resolution, full-pipeline, or target peak-memory coverage. Check `per_tensor` and `contiguous` separately when both layouts are affected.

## Verify the mechanism

Check generated config and report values: `offload_resident_blocks == resident_blocks`, `resident_blocks + streamed_blocks == num_blocks`, and actual indices in runtime logs. Preparation must implement uniform placement rather than only changing a field or log message.

When needed, instrument `copy_block_to_buffer()` during validation. One complete block traversal should transfer only nonresident layers, in logical order, exactly `N-r` times. Count initial resident-weight copies separately. Current Qwen CFG steps with two traversals perform `2*(N-r)` such copies; use the actual traversal count for other models or conditions.

Use representative counts such as `r=0`, partial residency, and `r=N` to check that all layers still compute and resident layers skip repeated transfer. Execute repeatedly on the same instance and again after release, checking that CPU source addresses/state are not replaced by lost temporary copies. After a controlled exception, verify cleanup and successful re-execution.

These mechanism checks do not run inside `estimate.sh` or search for an optimal `r`. Legacy regression comparisons must remove `offload_resident_blocks`. Production baseline launchers retain the event path with zero residency to isolate the effect of residency itself.

## Correctness and actual launchers

On each platform, match checkpoint, task, inputs, seed, shape, steps, dtype, compute operators, weight layout, and other component offload settings between baseline and residency. Change only the resident count and necessary estimator-generated settings. A smaller input or different operator must not be presented as a successful target-config run.

Prefer comparisons of final latents, single-block/intermediate states, and image pixels or decoded video frames. Use bitwise comparisons when computation is repeatable and the path is unchanged. For nondeterministic operators, use a justified tolerance and report errors. Visual similarity alone does not prove correctness, and video-container byte differences need not indicate model-output differences.

When results diverge, check actual weights, dtype/scale handling, a single block, the first denoising step, and then final output, so multi-step error amplification does not hide the origin. Include Qwen text-state comparisons rather than checking only the image branch.

Complete integration also executes the target platform's estimation, residency, and baseline entry points, verifying that inference reads the generated config and produces output. The baseline runs independently. Do not repeat estimation merely as a formality when a valid measurement already exists and conditions are unchanged. Without NPU hardware, finish feasible checks and explicitly report missing device coverage; CUDA success is not a substitute.

## Timing and memory

Run representative performance comparisons when requested. Use the same device and conditions, run the paths sequentially, and record cold/warm state and repeated measurements. Do not generalize from the single fastest run. Launchers inherit `PROFILING_DEBUG_LEVEL` from `base.sh`, currently `2`, which enables the relevant timing output. Adjust it in `base.sh` when needed and use the same level for both paths.

| Model | Log label | Current timing scope |
|---|---|---|
| Wan | `Run Dit every step` | Scheduler step_pre, model inference, and step_post; see [default_runner.py](../../../../lightx2v/models/runners/default_runner.py) |
| Qwen Image | `infer_main` | Model inference, including positive/negative CFG passes when enabled and weight transfers; excludes scheduler pre/post work, text encoding, and VAE; see [qwen_image_runner.py](../../../../lightx2v/models/runners/qwen_image/qwen_image_runner.py) |

Report estimator time, loading/encoding, first step, subsequent steady steps, VAE, and end-to-end time separately. Do not directly compare labels with different scopes or substitute single-block `c`. Initial residency preparation and kernel initialization may enter the first step; report it separately. Custom asynchronous timing must wait for device completion; Python enqueue time is not actual device execution time.

Report actual `N`, `r`, transfer counts, resident indices, and denoising allocated/reserved peaks. Distinguish additional resident weights, two slots, other weights, activations, and workspace. `r*block_bytes` is not total device memory. Reserved memory may stay high after release due to allocator caching; inspect allocated memory and live references first.

Residency consumes additional device memory and reduces repeated transfers; speed gains depend on the input and hardware. Use a device timeline when quantifying hidden transfer time. Neither the formula nor an overall speedup proves that every copy is fully hidden.

## Diagnose problems

| Symptom | First checks |
|---|---|
| Estimation finishes quickly | It measures one real block with 3 warmups and 10 samples per measurement; inspect the report rather than expecting full denoising |
| `r=0` | Check for `d<=c`, a valid outcome, then confirm input size and dtype match |
| Residency remains slow | Generated config actually used, real indices/transfer counts, first-step cost, and non-DiT phases; aggregate budgets do not guarantee complete local overlap |
| Out of device memory | `r`, slots, activations, and other allocations; there is no automatic clipping. Explicitly adjust the generated count when reducing memory instead of silently changing task inputs |
| Config edit has no effect | Run reads a generated config; check whether base-config changes were re-estimated, and align CLI, environment, and independent checkpoint paths |
| Second execution fails or outputs vary unexpectedly | Slot ready/free events, caller input/output lifetime, synchronization before release, and CPU source restoration |
| Contiguous-layout results drift | Production storage binding, dtype/scale conversions, and matching estimation/inference layouts |
| NPU launcher fails | `PLATFORM=ascend_npu`, effective `AI_DEVICE`, platform dependencies, NPU operators, and model paths; do not silently switch to CUDA and claim acceptance |

The final report records current code version, actual config, platform/assets, test results, and uncovered cases. Historical `r` values, speedups, and test counts are background, not guarantees for a new machine.
