---
name: support-model-warmup
description: Implement, maintain, review, and validate built-in warmup for LightX2V models and tasks. Define warmup shapes and per-round prompts in the runner, audit normal inference, CPU offload, and lazy-load support, and verify encoder, DiT, scheduler, and VAE decode coverage, recompilation, and state isolation.
---

# Support LightX2V Warmup

English | [简体中文](SKILL.md)

## Purpose and scope

A review checks the call chain, state, and available evidence. Implementation includes functional validation for the affected path. Follow the requested number and scope of benchmark runs. Compare warmup and no-warmup runs when evaluating the benefit of warmup.

Reuse the production operator path. Start the first formal request after warmup completes. Reuse eligible kernel, allocator, and compile state for covered signatures. Isolate warmup seeds, latents, solver history, branch state, and temporary models. Preserve existing behavior for eager, lazy-load, and other tasks.

Scope changes to the runner and scheduler needed for the task. Keep production pipeline refactors and checkpoint or offload format changes in shared weight infrastructure such as `mm_weight.py` separate.

If the task also requires compile, first establish eager warmup and state isolation. Then integrate compile with [support-model-compile](../support-model-compile/SKILL_EN.md), which owns the compilation and dispatch logic.

Set `"warmup": true` in the startup JSON selected by `--config_json`. The default is `false`. The Python API reads the setting before runner initialization in `create_generator(config_json=...)`. Run the original inference entry point so built-in warmup runs once at startup.

## 1. Audit execution modes

Check the exact `model_cls`, task, runner, model, transformer inference, and weights.

```bash
rg -n "cpu_offload|offload_granularity|lazy_load" \
  scripts/<family> configs/<family> \
  lightx2v/models/{runners,networks}/<family>
rg -n "infer_with_.*offload|init_lazy_load|init_(cpu|cuda)_buffer|prefetch|swap_" \
  lightx2v/models/networks/<family> lightx2v/common/offload
```

Complete the matrix before implementation. Assess production inference and warmup independently.

| Mode | Production inference | Warmup | Code evidence and lifecycle |
|---|---|---|---|
| Normal | Supported / unsupported / unverified | Supported / unsupported / to implement | Runner → model → infer |
| CPU model | | | Owner of full-model transfers to and from the device |
| CPU block | | | CUDA staging, prefetch, and swap |
| CPU phase | | | Phase buffers and phase inference |
| Lazy block | | | CPU staging, disk prefetch, and cleanup |
| Lazy phase | | | CPU phase staging, disk prefetch, and cleanup |

- Verify the loading, inference, weight-buffer, and unload paths for each setting.
- Check specialized Wan runners, Qwen layered paths, and LTX2 versions and tasks individually.
- Implement or explicitly reject warmup for modes with an existing production path. Keep unsupported production modes out of the integration.
- When `lazy_load` requires block or phase offload, treat it as part of that mode.

See the [execution mode snapshot](references/implementation-patterns_en.md#execution-mode-snapshot) for Wan, Qwen-Image, and LTX2 evidence. Recheck the code after changes.

## 2. Derive warmup from the production request

Start with the user’s entry script. Resolve the effective config, runner registration, and `model_cls`, then trace the shared entry point and production request path.

```bash
rg -n "config_json|model_cls|task|runner" scripts/<family> configs/<family>
rg -n "def (warmup|run_warmup|init_modules|run_pipeline|end_run)" \
  lightx2v/models/runners
rg -n "def (prepare|reset|step_pre|step_post|clear)" \
  lightx2v/models/schedulers/<family>
rg -n "run_input_encoder|run_(text|image|vae)_encoder|run_vae_decoder|model\.infer" \
  lightx2v/models/runners/<family>
```

Before starting a GPU job, confirm the resolved config, model or checkpoint, LoRA or adapter, and input media. Report missing assets that block the original config. A valid config for the same task can serve as a functional smoke test. Target-path acceptance requires the original assets and model semantics.

With `warmup=true`, the startup hook runs once before the first formal `run_pipeline()`. The hook can run one or more rounds for the required coverage. Each round has its own inputs and can include multiple DiT steps, video chunks, or model stages. Model-specific warmup relies on this lifecycle.

`build_runner()` initializes the runner. `prepare_request()` validates request fields and creates `InputInfo`. `run_request()` calls `run_pipeline()`. Warmup input preparation follows the task’s `InputInfo` and shape contract, including `size=[height, width]`.

`BaseRunner` calls `warmup()` after the outermost `init_modules()` completes. Eager models are loaded, the config is locked, and any compile initialization is complete. `DefaultRunner.warmup()` checks `warmup=true` and rejects unsupported modes such as disaggregation, `unload_modules`, and feature caching. For a `DefaultRunner` model without warmup, add `run_warmup()` to the closest concrete runner and reuse the shared startup entry point.

When `warmup=true` is explicit, raise `NotImplementedError` with the reason for an unsupported model, version, task, or mode. For runners shared by multiple versions, check the version-specific features.

Trace `run_pipeline()` stage by stage.

| Stage | Production method | Shape source | State retained |
|---|---|---|---|
| Input | Text, image, or VAE encoder | Input or config | Conditioning and masks |
| Denoising | prepare → step_pre → infer → step_post | Latent shape | Generator, latents, and solver state |
| Stage transition | Upsampling, unpatchify, or equivalent | Previous stage output | New scheduler state |
| Output | VAE decode | Final latents | Iterator and temporary VAE |
| Cleanup | end_run or clear | Request boundary | Caches and runner fields |

Reuse the existing stage methods. Keep request saving, full request profiling, and request-level pipeline cleanup out of the warmup call.

## 3. Determine coverage

Within supported modes, determine the complete shape signature. Include resolution, frame count, text and image tokens, dtype, strides and layout, CFG and MoE branches, stage count, parallelism, and loading lifecycle.

- For a typical multi-resolution path, declare two representative shapes in `_WARMUP_SHAPES` near the start of the runner class. Prefer a smaller or square shape and a runnable production or non-square shape. Exercise the real DiT and decoder for each.
- Choose the number of rounds from the production path. A fixed camera size or a specialized path that warms the configured request can use one round. Paths that cover the configured size alongside presets can append `config["size"]`, deduplicate, and index the final execution order.
- When adding representative shapes with unknown feasibility, check that the production shape runs on the target device. If it runs out of memory, smaller shapes can cover height, width, and aspect ratio for functional or eager-kernel warmup. Record the remaining production-shape and compiled-graph gaps.
- For OOM, distinguish live tensor allocations from allocator reserved memory. `empty_cache()` cannot free live tensors.
- If presets leave mutually exclusive graphs uncovered, identify the gap and explain any extra representative inputs.
- Run image and VAE encoders at each I2V or I2I resolution as required by the task.
- For FLF2V, send the first and last frames together to the image and VAE encoders.
- Reuse outputs across resolutions only when they are shape-independent.
- Match production dtype, layout, branches, and leaf-op dispatch for compile warmup. Check the full graph and kernel signatures.
- For MoE or high- and low-noise models, cover every transformer branch.
- Run real stage transitions for multistage models. In parallel modes, keep the collective order the same on all ranks.
- Require explicit opt-in for specialized subclasses. Reject unsupported paths such as VACE, audio, animate, or self-forcing where they would otherwise inherit generic warmup.

Ask about version, task, resolution, or lazy support when the required information is unclear and the code cannot resolve it.

## 4. Implement the smallest path

`run_warmup()` is the public hook. Split out `_run_warmup()` and `clear_warmup_state()` where eager or lazy loading, the main loop, or error cleanup needs its own boundary.

### Runner input conventions

- Declare `_WARMUP_SHAPES` and then `_WARMUP_PROMPTS` before the warmup methods. Implement `_prepare_warmup_inputs()` for multi-round, single-round, and frame-only paths.
- Read `_WARMUP_PROMPTS[warmup_index]` in actual round order. A shared loop can select the prompt and pass it to the subclass helper. Supply a prompt for any round added by appending the configured size.
- Prompts for different rounds must differ clearly in length. Write each prompt as one string literal on one physical source line. Use one prompt for a single round and an empty tuple `()` for frame-only inputs.
- Within the same runner, T2I and I2I and single- and multi-GPU modes index the same prompt tuple by round. Keep the length variation in compile mode and use logs to check recompiles. For a specialized single-round path that warms a fully configured request, record its configured or segment-specific text source. Select the default prompt from the tuple.
- Run the text encoder separately for distinct prompts. Reuse the output only when it is shape-independent and the encoding conditions match. Key the cache by the actual text and other inputs that affect encoding. Keep the corresponding token lengths or masks with the result.

Run the stages required by the target task in each round. A diffusion image or video path follows this order.

```text
Required text, image, and VAE encoders
  → scheduler.prepare/reset
  → step_pre
  → model.infer
  → step_post
  → VAE decode
  → synchronize
```

Set `scheduler.generator` to `None` before any image or VAE encoder that might consume the request RNG. If I2V first samples conditioning with that generator and then uses it in `scheduler.prepare()` to generate noise, preserve the order and keep the generator through both operations.

Use the target model's `InputInfo`, shape calculation, scheduler, and decoder. Dummy inputs replace user content while exercising the production stages.

Choose representative steps.

- Include `step_index=0` to cover formal Step 1 in the logs.
- A single graph usually needs Step 0.
- A multimodal scheduler can vary the number of unique timesteps, embedding shapes, or AdaLN index layouts across steps. Inspect `step_pre()` and pre-infer and choose the fewest steps that cover each stable layout.
- In branching models, use the first effective step of each branch.
- For gaps between steps, reset or prepare again only when solver history, branch state, or other internal state cannot carry across the gap.
- Run the final step if unpatchify or finalization is needed for VAE decode or the next stage.
- Call `step_post()` after each inference.

Fully consume any generator or iterator returned by the decoder. Feed the actual Stage 1 output into the existing upsampler and Stage 2 preparation.

CPU model offload often returns the model to CPU on the final formal step. If warmup runs only the first step, call the existing `model.to_cpu()` in each shape's `finally` block. Restore all models that were loaded in a multibranch path.

See [implementation patterns](references/implementation-patterns_en.md) for the skeleton and family-specific details.

## 5. Isolate state and memory

Save only initialization state that warmup changes and the formal request needs, such as Stage 1 `infer_steps` or MoE guidance. Name saved values by purpose, such as `stage1_infer_steps`.

Warmup runs before the first formal request. Restore the startup baseline with `input_info = None` and remove `inputs`.

After each shape or exception, clear the following.

- Latents, predictions, masks, timesteps, sigmas, and solver history.
- CFG or MoE branch state and request-specific RoPE or position caches.
- Temporary inputs, conditioning, and transient modules.

Request-level `scheduler.clear()` releases the generator. Leave it as `None` so the next request creates one from its own seed. Keep it across Stage 1 → Stage 2 within the same request.

Preserve compile graphs, kernels, resident eager weights, and CUDA allocator cache. Look for unconditional `empty_cache()` calls between warmup and the first formal DiT step.

```bash
rg -n "empty_cache|maybe_empty_cache|gc\.collect" \
  lightx2v/models/runners/<family>
```

Unconditional `empty_cache()` releases reusable allocator blocks or workspaces. The first formal step may need to allocate them again. In eager mode, remove the unconditional call or use the existing pressure-aware cleanup. For lazy cleanup, retain the necessary cleanup after releasing temporary references.

Lazy-load lifecycle

```text
load transformer → attach scheduler → warmup → synchronize
→ remove offload manager/transient modules → drop references
→ maybe_empty_cache(collect_garbage=True)
```

Confirm CPU and CUDA buffers, prefetch and swap, and per-inference shape-cache reset before using lazy-load. In eager mode, call `gc.freeze()` only after `_run_warmup()` returns and temporary references are gone. Skip freeze in lazy mode. The shared gate rejects `unload_modules` unless support is explicitly in scope.

For lazy compile cleanup, break references from `scheduler.transformer_infer`, compiled closures, and offload managers to staging buffers and infer objects. Release local model references before collecting them. Use a `weakref` test to confirm release.

Reuse only immutable caches safely keyed by the complete shape signature across requests. Clear request-specific RoPE and position caches. For block and phase offload, verify that staging buffers, prefetch indices, and stream state are ready for the next inference.

## 6. Validate

```bash
python -m pytest -q <targeted-warmup-tests>
ruff check <changed-files>
python -m py_compile <changed-python-files>
git diff --check
```

Find existing tests with `rg --files tests test_cases` and reuse warmup cases. For input changes, check one-line prompts, actual indices, distinct lengths, separate encodes, and indices after appending or deduplicating shapes. Verify token counts when a local tokenizer is available. Cover mode guards, stage ordering, iterator consumption, full-model offload, generator state, exception cleanup, and the `clear()` request boundary as relevant to the changes. For documentation-only work, validate the text and examples.

Test supported production modes. For functional acceptance, complete built-in warmup and one formal request in each target mode. Keep the original script, checkpoint, inputs, resolution, and save behavior. Run each case once when requested. Use multiple samples or controls when evaluating stability or the benefit of warmup. Label alternative configs or sizes as smoke tests and preserve evidence of the original failure.

Record the following.

- Actual shape, prompt length, and encoder, DiT branch, stage, and decode coverage for each round.
- Dynamo recompiles with `TORCH_LOGS=recompiles`, split by each rank's `Warmup cost`.
- Formal text encoder, VAE encoder where applicable, VAE decoder, individual DiT steps, total DiT, and image save time. LightX2V reports `Run Dit every step` and `Run DiT`.
- `Warmup cost` separately. Use `RUN pipeline cost` for formal end-to-end latency, including saving when it runs.
- Device memory before and after warmup and cache cleanup before formal Step 1.
- Whether any traceback occurs during loading, warmup, or a formal request.

Acceptance checks

1. Warmup covers the target shape signature and removes avoidable compilation, kernel, and allocator cold starts.
2. Formal seeds, inputs, and subsequent requests stay isolated. Compare hashes or tensors with the same seed on deterministic paths and use reasonable tolerances for nondeterministic paths.
3. When lazy load is in scope, a formal request completes after warmup.
4. Eager, no-warmup, other tasks, and formal saving keep their existing behavior.

A single-run benchmark reports that run's timing. Use guard logs to locate recompiles and [cold-start boundaries and common pitfalls](references/implementation-patterns_en.md#cold-start-boundaries-and-common-pitfalls) to distinguish warmup gaps, CPU model transfers, per-step block or phase transfers and synchronization, lazy rebuilds, request-cache recomputation, and profiling bias.

Deliver run instructions, configs, source changes, warmup rounds, representative steps, test results, and reproduction logs as appropriate. For performance comparisons, report one-time warmup, the first formal request, and measured steady-state throughput separately.
