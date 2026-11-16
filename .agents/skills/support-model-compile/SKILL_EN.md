---
name: support-model-compile
description: Implement, review, debug, and benchmark `use_compile` and `torch.compile` for LightX2V models and inference paths. Covers normal inference and existing CPU model, block, phase, and lazy offload modes. Audit mode support, select compilation boundaries and cache keys, handle layer indices, staging buffers, dynamic guiders, third-party operators, and Dynamo recompiles, and use warmup to measure compilation cost, request latency, and steady-state gains separately.
---

# Support LightX2V Compile

English | [简体中文](SKILL.md)

## Goal

Use the shared LightX2V block compile mechanism with minimal changes. Add compile support for modes that already run in eager mode and preserve the behavior of `use_compile=false`.

Match the work to the task. A review provides code evidence and identifies verification gaps. Implementation requires functional validation. Run steady-state benchmarks when the task calls for performance conclusions.

`use_compile`, `compile_backend`, and `warmup` are startup JSON settings. Select them through `--config_json`.

This skill covers `torch.compile` through `use_compile`. Evaluate `use_magi_compile` and CUDA Graph separately if a comparison is requested.

When the task also requires warmup, or a performance acceptance test depends on warmup that the target lacks, first use [support-model-warmup](../support-model-warmup/SKILL_EN.md) to implement eager warmup. Warmup and compile share the production operators and keep their lifecycle and dispatch logic separate.

## 1. Audit the inference path

Start with the user’s script and trace the config, model, transformer inference, weights, and offload manager.

```bash
rg -n "use_compile|cpu_offload|offload_granularity|lazy_load" \
  scripts/<family> configs/<family> lightx2v/models/networks/<family>
rg -n "transformer_infer_class|infer_func|infer_with_.*offload|run_block|infer_block|infer_phase" \
  lightx2v/models/networks/<family>
rg -n "class .*TransformerInfer|def (infer_block|run_block)|\\.infer_block\\(" \
  lightx2v/models/networks/<family>
rg -n "cuda_buffers|prefetch|swap_|compute_stream|synchronize" \
  lightx2v/models/networks/<family> lightx2v/common/offload
```

Resolve the effective config and confirm the checkpoint, LoRA or adapter, and input media before using a GPU. Report any missing asset that blocks the original config. A valid repository config for the same task can serve as a functional smoke test. Preserve the original LoRA and model semantics for target-path acceptance.

Trace the target’s own production call chain. Run a no-warmup eager baseline with the original user script to confirm that the production shape works. A failure in this run identifies an issue in the original path. Complete the support matrix.

| Mode | Eager | Compile | Compute entry point | Block or phase object lifecycle |
|---|---|---|---|---|
| Normal | Supported / unsupported / unverified | To implement / not applicable | | Resident / rebuilt |
| CPU model | | | | |
| CPU block | | | | Staging count and reuse |
| CPU phase | | | | Phase families and staging count |
| Lazy block | | | | Rebuild between requests |
| Lazy phase | | | | Rebuild between requests |

Limit compile integration to the requested modes with eager support. Verify loading, inference, buffer management, and cleanup for each mode.

Record the design before editing code.

```text
Target infer class:
Compute entry point per mode:
Block or phase families and their signatures:
Compilation boundary for each family:
Reads the logical layer index:
Number of resident blocks and staging buffers:
Block or phase cache key and expected entry count:
Runtime branches and request-specific state:
Third-party operators that may dispatch differently:
Production graph signature (original shapes, tokens, and frames, per-rank local tensor shapes after packing, padding, and SP, dtype, layout, branch, and stage):
Whether warmup covers the same signature and actual leaf kernels:
```

Ground each item in the code and eager runs. Resolve gaps before implementing compile support.

## 2. Choose boundaries and cache keys

By default, compile one transformer block per family of blocks with the same signature and semantics. Normal inference, model offload, and block offload can share the boundary for a family. Keep weight transfers outside the graph.

For a model with one block signature, prefer `BaseTransformerInfer`.

1. Inherit from it in the target infer class.
2. Call `self.init_compile(config)` once during initialization.
3. Keep `infer_block(block, *args)` as the compute entry point.
4. Call `self.run_block(block_idx, block, *args)` in the block loop.
5. Override `get_compile_block_key()` only when the cache semantics differ.

For multiple block families, give each stable signature its own dispatcher and cache, or include the family in the cache key. Keep the compute signatures focused. Promote a shared abstraction when a second model needs it.

Enumerate overrides and direct callers before changing a signature. Adapt the infer class selected by the target script and scope support claims to verified paths, including feature caching, AR, and specialized adapters.

Choose the key from the identities the graph actually depends on.

| Weight object and graph dependencies | Cache key |
|---|---|
| Resident block for each layer | `block_idx` (default) |
| Staging block reused across layers, graph independent of layer index | `id(block)` |
| Staging block reused across layers, graph reads layer index or layer-level cache | `block_idx` |
| Fixed phase staging buffer, graph independent of layer index | `phase_idx` |
| Phase reads the layer index | `(block_idx, phase_idx)` |
| Multiple block or phase families | Separate caches or `(family, key)` using the rules above |

A cache entry also stores the identity of the block or phase object. Recreate the callable when the object for a key changes.

Keep layer-index state in the model’s own loop or specialized subclass when needed. The shared `run_block()` stays independent of `self.block_idx`.

See [implementation patterns](references/implementation-patterns_en.md) for the shared structure and Wan, Qwen-Image, and LTX2/LTX2.3 examples.

## 3. Handle runtime state

Inspect Python state inside the compilation boundary.

```bash
rg -n "self\\.block_idx|self\\.[A-Za-z0-9_]+\\s*=|is_compiling|is_dynamo_compiling" \
  lightx2v/models/networks/<family>/infer lightx2v/common/ops
```

- If attention, KV cache, or an adapter uses the layer index, retain the model’s own field and cache by logical layer.
- Compute small guider or branch flags in the loop and pass them explicitly to `infer_block` as booleans or small scalars.
- Create or refresh request-specific tensors before the block loop when the first block currently writes them into `self`.
- Resolve stable Python values such as dtype, dimensions, and config flags during initialization.
- Standard requests use `prepare_request()` and store request fields in `InputInfo`. Check custom embedded callers and `LightX2VPipeline.modify_config()` for mutations to the shared config. Address this path when code or guard evidence shows a mutation. Cache only the initialization constants the graph needs.
- Override `run_block()` in the specific subclass if it alone needs layer-index state.

Make state explicit where it enters the graph, affects correctness, or triggers recompiles.

## 4. Integrate existing offload modes

- **Model offload**. The model layer usually handles full-model H2D and D2H. Verify that transformer compute reaches the shared block loop.
- **Block offload**. Replace the direct `infer_block` call with `run_block` on the existing compute stream. Preserve prefetch, swap, and synchronization order.
- **Phase offload**. Where eager already supports it, integrate `infer_phase → get_compiled_phase → run_phase`. Key the cache according to layer-index dependencies and leave unsupported phases in eager.
- **Lazy load**. Reuse the CUDA staging buffers and caches of regular offload. Compile cache lifetime follows the infer instance. Release compiled closures and old staging objects held by the scheduler or runner at teardown.

CPU-to-GPU copies, buffer swaps, and stream synchronization remain outside the graph, along with the offload manager.

## 5. Review operators

Inspect attention, RoPE, RMSNorm, quantized matrix multiplication, and Triton or third-party operators.

- Check for a kernel switch through `is_compiling()` or `is_dynamo_compiling()`.
- Identify graph objects created dynamically, mutations, data-dependent Python branches, and missing fake or meta implementations.
- Verify numerical agreement between eager and compile.
- Check that performance comparisons use the same actual kernels.

To isolate third-party code, define a module-level `torch.library.custom_op` and register its fake implementation. Dynamo treats the custom op as a graph boundary. Its kernel launches still run, and fusion stops at that boundary. Triton, CUDA extensions, and third-party JIT code inside the op may still compile or specialize on runtime signatures. See [compilation inside custom ops and dynamic Triton scalars](references/implementation-patterns_en.md#compilation-inside-custom-ops-and-dynamic-triton-scalars).

Reject confirmed incompatible combinations at initialization when a reliable fallback is unavailable. Change `lightx2v/common/ops` only after isolating the shared operator and validating all affected paths.

Keep the shared default.

```python
torch.compile(callable, dynamic=None)
```

`compile_backend="default"` uses this call. Instantiate explicit platform backends once through `COMPILE_BACKEND_REGISTER` and pass them as `backend=...`. The repository has an Ascend `mindie` backend. Unknown names raise an error. Wan and Qwen use the default `torch.compile` call for their separate phase compilation paths. Verify MindIE phase support independently of the shared block backend.

Keep `mode="reduce-overhead"` and CUDA Graph outside the default integration.

For unexpected cache counts, recompiles, kernel dispatch changes, a slow Step 1, offload regressions, or OOM, follow the evidence in the [casebook](references/casebook_en.md).

## 6. Validate correctness

Targeted unit tests cover at least the following.

- Eager runs bypass `torch.compile` and avoid unnecessary layer-index state.
- The same key and object compile once. Changing the object refreshes the callable.
- Normal and model offload use the shared dispatcher.
- Block, phase, and lazy modes preserve cache counts and prefetch/swap order.
- Representative guider branches agree.
- Unsupported operator combinations fail early.

```bash
python -m pytest -q <targeted-compile-tests>
ruff check <changed-files>
python -m py_compile <changed-python-files>
git diff --check
```

Run the supported modes with the original user scripts. Keep the model, checkpoint, task, shape, seed, prompt, and offload settings consistent between eager and compile.

If compatibility requires different kernels, report the result as a comparison of complete optimization stacks. Attribute a gain to compile alone only when the actual kernels match.

If final diffusion outputs differ, establish eager reproducibility and then compare one block or the first denoising step before errors accumulate. Validate numerical agreement even when the images look similar. See [similar visuals with different numerical results](references/casebook_en.md#14-similar-visuals-with-different-numerical-results).

Use two levels of evidence.

- **Functional smoke test**. A replacement config, smaller shape, or minimal operator input verifies dispatch, Dynamo, or operator behavior.
- **Target-path acceptance**. Eager and compile complete with the original script, assets, production shape, and mode. Compare output tensors or hashes with the same seed, using a reasonable tolerance for nondeterministic paths.

Check eager at the production shape before attributing an OOM. If eager fits and compile runs out of memory, record the extra peak from tracing, Inductor, or graph pools. A smaller shape validates only that shape.

## 7. Benchmark steady-state gains

Enable warmup for both paths. Use the production call chain to cover the same graph signature, offload object lifecycle, and leaf-operator dispatch. Include original shapes, tokens, frames, dtype, layout, branch, and stage. For multimodal or sequence-parallel paths, also compare local tensors after packing, padding, and SP, especially the actual query and key lengths on each rank. Fix unreliable warmup using [support-model-warmup](../support-model-warmup/SKILL_EN.md) before benchmarking compile.

- Complete a warmup and formal request for each mode as a functional check.
- Run at least three trials per group.
- Use the same GPU or equivalent idle GPUs.
- Stop a trial affected by another process and rerun from warmup when the device is idle.
- If the target graph or a Triton kernel compiles during formal Step 1, address coverage or dynamic guards before benchmarking steady state.
- Report stages separately for multistage models.
- Report compilation during warmup separately from steady-state throughput.

Record the following for each trial.

- Compile and warmup preparation time.
- Formal pipeline latency.
- One-time cold-start latency, including preparation and the formal pipeline.
- `infer_main cost` per stage for Step 1, Step 2, mean from Step 3 onward, and mean from Step 6 onward.
- Device memory before and after warmup, and the peak during the formal request.

```text
speedup = eager / compile
latency reduction = (eager - compile) / eager
steady = mean(Step 3 ... last)
absolute Step 1 overhead = Step 1 - steady
relative Step 1 overhead = Step 1 / steady - 1
```

Keep individual measurements and report their mean and spread. A faster steady state can make relative Step 1 overhead larger even when the absolute overhead falls. Check the denominator and the recompile logs before attributing the difference.

For a slow Step 1 after warmup, use the [casebook](references/casebook_en.md#7-step-1-remains-slow-after-warmup) to distinguish graph or kernel compilation, first use of resources at the production shape, data-dependent work, and timing noise.

Report first-request latency, steady-state throughput, and total cold-start cost separately. Measure total latency when compilation moves into warmup. Eager may still be suitable for one-shot use.

## Completion criteria

1. Support matrix and reasons for unsupported modes.
2. Compilation boundaries, cache keys, and expected cache counts.
3. Minimal code changes and operator compatibility details.
4. Eager and compile correctness evidence.
5. Evidence covered by the functional smoke test and by target-path acceptance.
6. Post-warmup timing and memory by mode and stage.
7. Separate results for initial compilation, formal requests, total cold start, offload outside the graph, and steady-state compute gains.
