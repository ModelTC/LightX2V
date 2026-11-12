# LightX2V Compile Casebook

English | [简体中文](casebook.md)

For a matching symptom, confirm it, identify the cause, and apply the corresponding fix. Verify the evidence on the target model.

## Contents

- [1. Staging block recompiles](#1-staging-block-recompiles)
- [2. Layer-index state in the shared dispatcher](#2-layer-index-state-in-the-shared-dispatcher)
- [3. LTX2.3 guider reaches the recompile limit](#3-ltx23-guider-reaches-the-recompile-limit)
- [4. Qwen uses different RMSNorm kernels with the same config](#4-qwen-uses-different-rmsnorm-kernels-with-the-same-config)
- [5. Kernel launch count with a custom op](#5-kernel-launch-count-with-a-custom-op)
- [6. Unstable Qwen one-pass RMSNorm compilation](#6-unstable-qwen-one-pass-rmsnorm-compilation)
- [7. Step 1 remains slow after warmup](#7-step-1-remains-slow-after-warmup)
- [8. Cache cleanup slows Step 1 again](#8-cache-cleanup-slows-step-1-again)
- [9. Offload costs outside the graph hide gains](#9-offload-costs-outside-the-graph-hide-gains)
- [10. Incorrect scale rank in scaled_mm](#10-incorrect-scale-rank-in-scaled_mm)
- [11. Checking CUDA Graph usage with torch.compile](#11-checking-cuda-graph-usage-with-torchcompile)
- [12. OOM or interference from other processes](#12-oom-or-interference-from-other-processes)
- [13. Faster Step 1 with higher total cold-start latency](#13-faster-step-1-with-higher-total-cold-start-latency)
- [14. Similar visuals with different numerical results](#14-similar-visuals-with-different-numerical-results)
- [15. First request recompiles after shared config grows](#15-first-request-recompiles-after-shared-config-grows)
- [16. Multimodal timestep layout recompiles on later steps](#16-multimodal-timestep-layout-recompiles-on-later-steps)
- [17. Inner Triton compilation behind a stable custom op](#17-inner-triton-compilation-behind-a-stable-custom-op)

## 1. Staging block recompiles

- **Symptom.** Qwen-Image or LTX2 block offload uses a few CUDA staging blocks, while `compiled_blocks` approaches the number of logical layers.
- **Confirm.** Compare object identities in `cuda_buffers` with cache keys. Check whether the graph reads `self.block_idx`.
- **Cause.** Staging buffers independent of layer indices still use the default `block_idx` key.
- **Fix.** Override `get_compile_block_key()` in the offload infer class to return `id(block)`. Wan reads the layer index and requires a layer-index key.

## 2. Layer-index state in the shared dispatcher

- **Symptom.** A call to the shared `run_block()` adds `self.block_idx` even with `use_compile=false`.
- **Confirm.** Check which downstream operations read the layer index and whether only Wan or AR subclasses need it.
- **Cause.** State required by a specialized path was moved into the shared dispatcher.
- **Fix.** Keep the shared dispatcher responsible for eager/compile dispatch. Set the index in Wan's loop and set `_ar_block_idx` in LTX2 AR. Test that eager objects without a layer dependency retain their original fields.

## 3. LTX2.3 guider reaches the recompile limit

- **Symptom.** One `self.compiled_block` repeatedly recompiles across guider branches and hits Dynamo's cache limit.
- **Confirm.** Run `TORCH_LOGS="recompiles,graph_breaks" <command>` and check guards for `_mm_skip_*`, block identity, and guider branches.
- **Cause.** All layers share one entry point that reads changing Python sets and `self` state.
- **Fix.** Use the shared per-key cache. Compute guider state in the loop and pass `skip_*` booleans into the block. Address these causes before changing the cache limit.

## 4. Qwen uses different RMSNorm kernels with the same config

- **Symptom.** Eager and compile both use `rms_norm_type: "sgl-kernel"`, yet the actual kernels differ.
- **Confirm.** Search the repository and dependencies for `is_compiling|is_dynamo_compiling`, then trace kernel names and counts. The `sglang-kernel 0.4.4` version in this case dispatched differently under Dynamo.
- **Cause.** The third-party wrapper selects a runtime branch independently of the backend label.
- **Fix.** Decide whether the comparison is between complete production stacks or requires the same kernel. For complete stacks, report the implementations used. For a same-kernel comparison, call the target SGL kernel through a custom op and confirm agreement with a trace.

## 5. Kernel launch count with a custom op

- **Symptom.** Making RMSNorm or RoPE a leaf allows compile to run, while dispatch remains frequent and steady-state gains are limited.
- **Confirm.** Profile the custom ops, CPU dispatch, and CUDA kernel count.
- **Cause.** The custom op keeps Dynamo out of its internals. It preserves the calls and prevents fusion across that boundary.
- **Fix.** Use small leaves for compatibility. Use a larger fused operator when the goal is to reduce launch count.

## 6. Unstable Qwen one-pass RMSNorm compilation

- **Symptom.** One-pass RMSNorm runs faster in eager. In a compiled block it can cause compatibility errors, first-use Triton compilation, or unstable dispatch.
- **Confirm.** Minimize the RMSNorm test, compare numerical results and kernels, and inspect dynamic graph objects and warmup specialization.
- **Cause.** The current one-pass implementation fails Qwen block compile stability requirements.
- **Fix.** Qwen weights reject the `use_compile` and one-pass combination explicitly. Reproduce the issue on a new model before adopting the same restriction.

## 7. Step 1 remains slow after warmup

- **Symptom.** Formal Step 1 is slower than later steps after warmup. Compile can show a larger relative gap than eager.
- **Confirm.**
  1. Calculate `Step1 - steady` and `Step1 / steady - 1`. Keep individual results for at least three trials.
  2. Inspect guards with `TORCH_LOGS="recompiles,graph_breaks"`. Use these logs to verify recompile behavior even when the ordinary logs are silent.
  3. Compare warmup and formal shapes, tokens and frames, CFG and guider, dtype, layout, stage, object identity, and leaf kernels.
  4. Verify timing boundaries, then inspect allocator reserved memory, attention and grouped-MM workspaces, request caches, and data-dependent work.
- **Cause.** Possibilities include an uncovered graph signature, first allocation or initialization at the production shape, different MoE routing, and timing noise. A faster steady state also makes fixed overhead larger as a percentage.
- **Fix.**
  - If logs show a recompile or new kernel, address warmup coverage, guards, or leaf dispatch.
  - If Step 2 onward is stable and compilation evidence is absent, run one controlled warmup at the production signature to test first-use resource cost. Use that evidence to decide whether to extend production warmup.
  - Preserve scheduler and request-cache cleanup and the built-in warmup lifecycle in [support-model-warmup](../../support-model-warmup/SKILL_EN.md).

With `dynamic=None`, an accepted new shape can still allocate resources or initialize a workspace for the first time. Text length changes the sequence passed into the transformer even when the measurement excludes the text encoder.

**LingBot-Video example.** In this run, warmup used two fixed resolutions, a short prompt, and an empty negative prompt. Formal T2I and T2V used resolutions outside exact warmup coverage and long CFG text. Compiled T2I Step 1 overhead was 13.6% compared with 9.7% in eager. The absolute overhead was smaller at 23.7 ms compared with 32.6 ms. One I2V run showed no first-step penalty. These observations are consistent with incomplete signature coverage and first use of resources.

## 8. Cache cleanup slows Step 1 again

- **Symptom.** Warmup covers the formal graph, yet Step 1 is slow. RoPE or position-cache cleanup, `empty_cache()`, or GC occurs between the two phases.
- **Confirm.** Search for `rope|position.*cache|empty_cache|maybe_empty_cache|gc.collect`. Compare live allocations, allocator reserved memory, and request-specific caches.
- **Cause.** Unconditional `empty_cache()` releases the workspace. Correct cleanup of request-specific caches can also be mistaken for a performance problem.
- **Fix.** Use the existing pressure-aware allocator path. Clear caches that depend on request shape or content. Reuse the verified shape-independent portions.

## 9. Offload costs outside the graph hide gains

- **Symptom.** Model offload has slow first and last steps. Block offload sees little or negative compile gain.
- **Confirm.** Profile compute, H2D and D2H, prefetch and swap, and stream synchronization separately.
- **Cause.** Transfers and synchronization occur outside the graph. Step 1 can include full-model H2D, and the final step can include D2H.
- **Fix.** Keep the full `infer_main` metric and explain first- and last-step costs. Compare pure compute separately. Address transfer overlap as an offload optimization.

## 10. Incorrect scale rank in scaled_mm

- **Symptom.** Quantized matrix multiplication raises a `scale_a` or `scale_b` rank or shape error only under Inductor.
- **Confirm.** Reproduce with a minimal weight `apply()`, record input and weight scale shapes, and rule out checkpoint loading.
- **Cause.** Scalar, `(1,)`, `(1,1)`, and `(1,N)` scale shapes are mixed.
- **Fix.** Normalize ranks for the confirmed case. Use `(1,)` on both sides for per-tensor scales. When the weight scale is `(1,N)`, use `(1,1)` for the input scale. Validate affected quantization paths if changing shared `mm_weight.py`.

## 11. Checking CUDA Graph usage with torch.compile

- **Symptom.** An integration assumes compile eliminates launches or adds `mode="reduce-overhead"` directly.
- **Confirm.** The default call is `torch.compile(callable, dynamic=None)`. Check capture separately before claiming CUDA Graph use.
- **Cause.** Inductor, operator fusion, custom-op boundaries, and CUDA Graph capture were treated as interchangeable.
- **Fix.** Preserve the default baseline. Validate CUDA Graph separately for dynamic shapes, mutations, staging buffers, offload streams, memory, and repeatable gains.

## 12. OOM or interference from other processes

- **Symptom.** The production shape runs out of memory during eager or compile loading, warmup, or the request. Another PID may appear on the target GPU during retesting.
- **Confirm.** Test the production shape with no-warmup eager. Keep the traceback and allocated, reserved, and peak memory. Separate compile peaks, graph pools, and live tensors. Check GPU PIDs with `nvidia-smi`.
- **Cause.** The base path may exceed single-GPU capacity, compile may add tracing, Inductor, or graph-pool memory, or concurrent load may interfere.
- **Fix.** If eager also runs out of memory, mark the base path as unavailable for comparison. If only compile runs out of memory, report its memory cost. Label smaller shapes and replacement configs as smoke tests. For interference from an identified external process, stop your trial and rerun from warmup on an idle GPU.

## 13. Faster Step 1 with higher total cold-start latency

- **Symptom.** Formal Step 1 after compile and warmup approaches steady state, and the one-shot task takes longer overall.
- **Confirm.** Record compile/warmup preparation, the formal pipeline, and their sum alongside `infer_main cost`.
- **Cause.** Compilation moved before service readiness. Steady-state savings may be too small to amortize it in a short task.
- **Fix.** Report first request, steady-state throughput, and total cold-start latency separately. Choose eager or compile for the deployment lifecycle.

## 14. Similar visuals with different numerical results

- **Symptom.** Eager and compile with the same seed produce similar composition and semantics, while pixels, latents, or audio differ numerically.
- **Confirm.** Establish eager reproducibility. Compare one block, the first denoising step, and the final output in order. Check dtype, kernel dispatch, and tolerance.
- **Cause.** Small differences in BF16 or Inductor operation order can accumulate across denoising steps. Branch, kernel, and state differences can also cause divergence.
- **Fix.** Locate the first point outside tolerance. Investigate the implementation if one step already differs substantially. For long-term accumulation alone, apply the project's numerical contract and report the result.

## 15. First request recompiles after shared config grows

This case concerns a legacy service path. Current `BaseRunner.prepare_request()` writes request fields to `InputInfo`.

- **Confirm.** When logs show guard failures for `len(self.config)` or the key set, trace actual mutations by custom workers or `LightX2VPipeline.modify_config()`.
- **Fix.** Cache the graph's stable settings as initialization attributes. Pass request data through `InputInfo` and preserve the rest of the runtime config.
- **Validate.** Exercise the real `warmup → request` lifecycle and check guards, numerical agreement, and first-step time. Verify mutations on the actual call path.

## 16. Multimodal timestep layout recompiles on later steps

- **Symptom.** Warmup covers formal spatial and temporal dimensions, and Step 1 hits an existing graph. Step 2 or the first step after a branch switch still recompiles.
- **Confirm.** Compare the complete block signature at representative steps, especially timestep embeddings, AdaLN indices, and modality-specific sigma or flow shift. Use `TORCH_LOGS="recompiles,dynamic"` to find the dimension guard.
- **Cause.** Modalities can share a timestep on the first step and have multiple unique timesteps later. In MiniMax-H3, `temb.shape[0]` is 1 at Step 0 and 2 on later steps. Both layouts require coverage.
- **Fix.** Execute the fewest representative steps that cover the stable timestep layouts on the same scheduler state. Preserve `step_pre → infer → step_post` and include layout in the signature alongside `(H,W,T)`.
- **Validate.** Check that recompiles finish before service readiness. Send a real HTTP request and verify each formal step. Report extra warmup time and formal end-to-end latency separately.

## 17. Inner Triton compilation behind a stable custom op

- **Symptom.** Formal Step 1 is still substantially slower than steady state after warmup. The custom op removes the target Dynamo graph break or recompile and only part of the first-step cost.
- **Confirm.** Compare local q/k lengths and sparse block counts on each rank after packing, padding, and SP. Record Triton or third-party JIT logs and cache artifacts before and after readiness. Use a dense or alternate-backend control to verify the runner's overall warmup.
- **Cause.** The custom op isolates outer Dynamo. The inner leaf kernel can still specialize on sequence length or alignment. Different prompts can produce new local lengths after packing and SP at the same video dimensions.
- **Fix.** Include the smallest semantic unit that produces and consumes data-dependent intermediates within the custom op. Use a Triton runtime scalar only when length leaves static shape, constexpr control flow, and layout unchanged. Use `do_not_specialize` to prevent specialization on runtime values and alignment. Use finite buckets or padding for static parameters.
- **Validate.** Run the full `warmup → ready → formal request` lifecycle from a cold cache. Confirm that inner compilation completes before readiness. Check numerics at multiple lengths, including a nondivisible tail, and verify that formal Step 1 approaches steady-state timing with no steady-state regression.
