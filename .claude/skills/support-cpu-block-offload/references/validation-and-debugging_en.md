# Validation and Debugging

English | [简体中文](validation-and-debugging.md)

Choose validation according to the change. Distinguish CPU contract tests, CUDA/multiprocess mechanism checks, and target-model acceptance. Documentation-only changes need reference and instruction consistency checks; real model integration requires more than static checks.

## Launcher and configuration acceptance

Check the unified shared launcher and both READMEs under `scripts/<model>/offload/`. Current launchers explicitly pass `--shared_cpu_weight_scope host`. For NUMA validation, change the value to `numa` or run the corresponding full Python command directly. `SHARED_CPU_WEIGHT_SCOPE=numa bash script.sh` does not override these scripts. Trace `--config_json` and CLI merging to confirm the effective scope. Sharing flags must reach the actual loader, not merely parse successfully.

Run `bash -n` and JSON parsing on delivered files. Compare effective inference settings between modes, allowing only scope to differ, and check effective dtype, model paths, inputs, seed, output paths, GPU list, and process count. CLI scope must override the same field in legacy JSON and stay out of request inputs. Omitting the CLI option must preserve JSON/internal defaults. Use temporary copies when validation needs different paths or parameters; do not add temporary configuration-generation logic back to production launchers. Use `git diff --check` for formatting, and inspect new untracked files separately because git diff does not automatically include them.

Before launch, confirm:

- Checkpoints, required AdaLN caches/quantized files, and input media exist. Match H3 caches to model variant, weights, steps, and flow shifts. Hunyuan additionally needs upstream code and compatible FlashInfer.
- The `torchrun` world size matches `tensor_p_size * cfg_p_size * seq_p_size` in the effective configuration, and visible-device/local-rank mappings are valid.
- Linux SysV/CUDA registration requirements and actual GPU NUMA topology support the target mode. Do not conceal strict NUMA failures by silently changing scope or disabling strict binding.
- Launch from the repository root as required by the current scripts. Check all relative asset paths in JSON, not only `model_path`. Implement and test arbitrary-working-directory support only when the task asks for it.

Run the real host and NUMA entry points with identical inputs. Record commands, effective configurations, code version, asset fingerprints, hardware topology, logs, and output locations. If the run covers only one NUMA domain, state that multiple NUMA replicas were not exercised.

## Select focused tests

Discover tests that actually exist in the current repository rather than assuming temporary test names from historical commits are available:

```bash
rg --files test_cases | rg '(shared|offload|hunyuan|qwen|wan|h3)'
```

The repository currently lacks the previously documented sharing-specific tests such as `test_shared_pinned_arena.py` and `test_shared_weight_map.py`. Existing shell cases do not automatically cover sharing mechanisms either. When new or temporary validation is needed, select coverage by the following dimensions rather than fixed filenames:

| Changed area | Focused validation |
|---|---|
| Arena/manifest | View layouts, physical SysV sharing across processes, registration regions, rollback, and close-failure state |
| Replica planner | Host/NUMA/auto grouping, noncontiguous ranks, device ordering, and topology boundaries |
| Coordinator | Store status exchange, stage mismatch, delays/timeouts, failure propagation, and cleanup; CPU waits must not enqueue NCCL collectives |
| Shared weight map | Multiple shared consumers, owner retention, operator adoption without copies, and private-branch behavior |
| Wan adapter | FP8 dtypes, scale rounding, schemas/content signatures, and configuration constraints |
| Qwen adapter | Manifests spanning shards, scope propagation, BF16 resolution, and transpose requirements |
| Hunyuan adapter/slots | Storage-TP slices, SP/CFG reuse, MoE pack layouts, logical KV layer indices, and heterogeneous block families |
| H3 shared components | Component selection, source/runtime dtypes, schemas and signatures, and AdaLN variant matching |
| H3 native module offload | Parameter aliases, nonpersistent buffers, restoration of original storage, slot copying, tiles, and release/rebuild |

For common arena/coordinator changes, add appropriate common and affected-model checks. Run Python compilation and existing repository lint checks according to the actual change. Do not write tests that merely duplicate implementation for reversible documentation or script-naming edits.

Meaningful new-adapter tests cover header/index and content changes in small synthetic checkpoints, target dtype/layout, baseline-equivalent conversions, private/shared boundaries, scope propagation, and storage identity after binding. Failure checks should represent real semantic constraints, not exhaustive type checking of internal parameters.

Distinguish CPU fake-runtime tests from CUDA tests, which may skip. Report passed, failed, and skipped counts with reasons. Mocks cannot prove real `cudaHostRegister`, device events, or DMA correctness. `--help` validates entry-point imports and argument parsing only; it does not prove cache generation or model inference. A default-dtype change also requires runtime evidence at the corresponding precision.

## Prove that weights are actually shared

Mechanism evidence must cover at least:

1. Processes within a replica attach to the same SysV segment, and allocations across host/IPC/NUMA groups match the plan. Interpret segment IDs together with host and IPC namespace; comparing integer `shmid` values across machines is insufficient.
2. Only each group's leader populates the shared payload. Followers do not construct another full private copy of the shared weights. Account for digest I/O separately from payload materialization.
3. Each process's final operator CPU-view addresses, dtypes, shapes, and strides match its local manifest, with successful registration/pinned checks. Checking only loader return values is insufficient.
4. Private non-block weights and dynamic state remain independent, and shared sources stay unchanged after inference. Use synthetic test segments to check cross-process write visibility; do not mutate real shared model weights.

Expected physical capacity for the shared payload is approximately:

```text
sum(manifest.nbytes for each actual replica)
```

Calculate this per component. Account separately for private weights, metadata, registration overhead, checkpoint mappings, and temporary loading memory. NUMA mode commonly has multiple full replicas; do not evaluate it against host mode's single-copy size.

Do not sum rank RSS values as physical memory use: the same shared pages may appear in every process's RSS. Combine PSS/Private fields from `/proc/<pid>/smaps_rollup`, unique segments and arena sizes, and changes in system physical memory. When permitted, use `/proc/<pid>/numa_maps` or `numastat -p` to check page placement. PSS is not an exact measure of one weight component either; control other processes, file mappings, and sampling times.

Record both initialization peaks and steady memory use. Lower final usage must not conceal an initialization peak where every rank first loads a complete copy.

## Correctness, consecutive requests, and failure paths

Prefer comparisons among private block offload, shared host, and shared NUMA at the same parallel scale. Keep checkpoint, task, inputs, shape, steps, seed, precision, and operators identical. Comparisons between single-GPU and SP execution also reflect computation-order differences and need separate explanation.

Use tensor/hash comparison when the baseline is reproducible and the computation path matches. For nondeterministic operators, use justified tolerances and record metrics such as maximum and mean error. Video-container byte hashes may reflect encoder or metadata differences; prefer tensors before encoding or decoded frames/audio. Visual similarity alone does not establish correct weight conversion.

When results differ, first compare actual CPU weights, quantization scales, one block, and the first denoising step, then trace final outputs. This avoids hiding the root cause behind error amplification across diffusion steps.

Choose lifetime checks according to affected paths:

- Run at least two consecutive requests on the same model instance, checking slot state, first-block preparation, source addresses, and valid outputs.
- If GPU buffers are released between phases, validate release → rebuild → infer while CPU sources remain in the original arena.
- Cover VAE tiles, image-conditioned encoding, and text/vision branches actually used by the task.
- Simulate local preflight, populate, register, binding, or computation failures in controlled tests. Check propagation and cleanup; closing an arena still in use does not count as leak-free behavior.

Peak GPU use is not simply the size of two blocks. Include non-block weights, activations, attention workspaces, other components, and allocator-reserved memory. If allocated memory drops after slot release but reserved memory does not, distinguish retained objects from caching-allocator behavior.

## Performance evidence

Run representative repeated measurements only when performance conclusions are requested. Keep the same idle devices, assets, operators, and parallel scale, and retain each run's results. Identify cold/warm cache or JIT states; do not base conclusions on the best single run.

Record separately:

- Checkpoint inspection, digest, and loading time.
- SysV creation, NUMA binding, CUDA registration, and populate/attach time.
- First request and first denoising step.
- Steady denoising steps, text encoding, VAE, and end-to-end time.
- CPU initialization peak/steady physical use and GPU allocated/reserved peaks.

Host scope reduces replica count. NUMA may improve local memory access and H2D paths, depending on topology and bandwidth. Sharing does not automatically reduce H2D bytes or guarantee acceleration. Use device timelines to attribute copy/compute overlap; Python enqueue timing alone misses asynchronous device work.

When warmup/compile is in scope, use the corresponding skill to check actual operators and shape coverage, reporting preparation and steady costs separately. Do not introduce a compilation system solely for one offload acceptance run.

## Diagnose from evidence

| Symptom | Evidence to inspect first | Likely direction |
|---|---|---|
| Sharing is enabled but CPU physical use still resembles multiple copies | Final operator pointers, Private/PSS, loading peak | Look for clone/cast/pin operations in consumers, full private preloads, and unreleased temporary dictionaries |
| Pinned checks or registration fail | Tensor-aware registration regions, CUDA error, local mapping, system limits | Fix the actual registration failure; falling back to private pinned copies is not successful sharing |
| NUMA mode fails | GPU PCI/NUMA discovery, container-visible topology, mbind errors | Distinguish unknown topology from binding permission/policy failures; do not silently select host |
| A rank stalls during initialization | Last Store stage, component order, rank preflight states, missing ranks, and wait deadline | Align status-exchange order and inspect CPU-loading progress; use the shared coordination timeout rather than substituting a larger NCCL timeout for diagnosis |
| Outputs drift | Effective dtype, scale rounding, transpose strides, kernels, seed | Align loading semantics, then trace the first numerical divergence |
| Intermittent errors or second-request failure | Ready/free/completion records, slot overwrite timing, owner validity | Repair stream/request dependencies and release order |
| Host and NUMA behave identically | Effective scope, participating GPU NUMA nodes, replica logs | Confirm scope reached the runtime configuration; one NUMA domain may legitimately produce the same replica count |
| Sharing does not improve speed | Loading/registration costs, H2D bytes, steady steps, topology | Report memory and speed separately; sharing does not imply acceleration |

After each fix, rerun checks covering the root cause and affected paths. Do not repeatedly broaden checks that already passed when no new risk has appeared.

## Final evidence record

Fill delivery notes with actual values and explain items not run:

| Mode | Script/effective configuration | Task/components/parallel scale | Replicas and sharing validation | Correctness/consecutive requests | CPU/GPU memory | Validation status |
|---|---|---|---|---|---|---|
| Private block baseline | Actual paths | Actual conditions | Sharing checks do not apply | Measured results | Measured results | Validated/unvalidated |
| Host | Actual paths | Matched to baseline | Measured results | Measured results | Measured results | Validated/unvalidated |
| NUMA | Actual paths | Matched to baseline | Measured results | Measured results | Measured results | Validated/unvalidated |

Distinguish completed code/script delivery, functional smoke tests, and target-path acceptance. Small shapes, substitute assets, single-rank runs, and mocks cover only their own conditions. Historical reports may provide context but cannot replace evidence for the current task.
