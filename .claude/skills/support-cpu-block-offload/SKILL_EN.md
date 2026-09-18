---
name: support-cpu-block-offload
description: Integrate, review, and debug CPU block offload and multiprocess CPU weight sharing (shared_cpu_weights) for LightX2V models. Use for weight adapters, shared loading, GPU buffer scheduling, asynchronous copy lifetimes, and host/NUMA launchers and validation. Follow the user's scope for a focused review or a complete integration.
---

# Integrating CPU Block Offload and Weight Sharing in LightX2V

English | [简体中文](SKILL.md)

## Purpose and responsibilities

Use the model's production inference path to implement CPU-resident weights, reusable GPU block buffers, and CPU weight sharing across processes. A complete integration must deliver runnable scripts and configurations under `scripts/<model>/offload/` for both host and NUMA modes. Adding only a flag or a loading interface is insufficient.

Block offload reduces GPU weight residency; CPU weight sharing reduces duplicate CPU weights within a sharing domain. Each rank still owns its GPU staging buffers and performs H2D copies. Do not describe this as activation offload, GPU weight sharing, or elimination of weight transfers.

The current common sharing backend requires Linux SysV shared memory and CUDA host registration. Ordinary block offload supports a broader set of devices; that does not establish XPU or other device support for the sharing backend.

Read references according to the task:

- Loading, operator binding, or inference scheduling: read the relevant sections of [implementation-patterns_en.md](references/implementation-patterns_en.md).
- Reusing Wan, Qwen Image, MiniMax H3, or Hunyuan Image 3.0: read the model's section in [model-adapters_en.md](references/model-adapters_en.md), distinguishing existing restrictions from common constraints.
- Selecting tests, validating memory savings, or investigating failures: read [validation-and-debugging_en.md](references/validation-and-debugging_en.md).

If the model lacks a native inference implementation, also consult [support_new_model](../support_new_model/SKILL.md). If the task includes compile or warmup, consult [support_model_compile](../support_model_compile/SKILL.md) or [support_model_warmup](../support_model_warmup/SKILL.md), respectively. Do not add those capabilities merely because offload is being introduced. Keep explanations, reviews, investigations, and tasks explicitly limited to private block offload within the user's scope; do not automatically expand them into a full sharing integration.

## 1. Trace the complete call chain from the launcher

Inspect the script, effective configuration, runner, model, weights, inference code, and offload manager in order. Configuration includes JSON, model configuration, CLI overrides, and environment settings such as `DTYPE` and `SENSITIVE_LAYER_DTYPE`.

```bash
model=wan
rg -n 'cpu_offload|offload_granularity|shared_cpu|lazy_load|release_block' \
  "scripts/${model}" configs "lightx2v/models/networks/${model}"
rg -n '_load_shared_cpu_weights|_init_weights|_init_offload_manager|infer_with_blocks_offload' \
  "lightx2v/models/networks/${model}" lightx2v/models/networks/base_model.py
rg -n 'prefetch|swap_blocks|wait_ready|record_free|close_shared_cpu_weights' \
  lightx2v/common/offload "lightx2v/models/networks/${model}"
```

Inspect the text encoder, VAE, and runner as needed. Do not assume that `shared_cpu_weights=true` shares the entire pipeline. Record each component separately:

| Component | Weight format and inference dtype | CPU-resident and private weights | Block types and GPU buffers | Shared loading and closing entry points |
|---|---|---|---|---|
| DiT | Determine from source | Determine from source | Count, schema, stride | Callers and owner |
| Text encoder, VAE, and other in-scope components | Record separately | Record separately | Whole module, blocks, or no offload | Check each component independently |

Build a support matrix covering ordinary execution, private CPU blocks, shared host mode, shared NUMA mode, and the SP, TP, quantization, LoRA, compile, or other combinations relevant to the task. Mark each item as validated, explicitly unsupported, or unvalidated, with code or runtime evidence. A configuration that parses does not prove a complete implementation.

First reproduce a baseline with the same task, checkpoint, inputs, and precision. If full GPU residency is not feasible, use a validated private CPU block path as the baseline and state the comparison's limits. Do not remove requested quantization, LoRA, or parallel settings to manufacture a successful result.

## 2. Connect block offload for private weights

Identify the block execution entry point, CPU weight source, and GPU slot schema. Plan compatible buffers separately for different block types; do not assume every layer has the same structure.

- For LightX2V `WeightModule` paths, reuse `state_dict()`, `load_state_dict()`, `offload_block_cuda_buffers`, and `_init_offload_manager()`.
- For native `nn.Module` paths, prefer `ModuleCPUWeights`, `NativeModuleBlockSource`, and `NativeModuleBlockSlot`, preserving parameter aliases and runtime buffer handling.
- Prefer the model's existing `WeightAsyncStreamManager` scheduling. Use `EventSlotWeightAsyncStreamManager` when fixed slots need a ready/free event protocol; do not rewrite a correct model loop just for consistency of style.

Check first-block loading, last-block handling, the next denoising step, dependencies between the caller and compute streams, and synchronization when returning outputs to the caller. Slot reuse must preserve shape, dtype, stride, and operator-specific layouts.

## 3. Integrate shared CPU weights

The model adapter owns checkpoint schemas, tensor selection, precision conversions, and weight signatures. The common coordinator owns topology grouping, creation, attachment, registration, and error propagation.

1. Build the manifest from headers, indexes, and meta tensors, separating shared tensors from rank-private tensors. Do not first load a full shared payload privately on every rank.
2. Include final runtime dtypes, layouts, and numerically significant conversions in the adapter contract. Quantization scale conversion order must match the baseline.
3. Have every rank call `materialize_shared_weight_arena()` in the same component order. Only each replica's leader populates the shared payload.
4. Return a `SharedWeightViewMap` so consumers adopt shared views directly through `consume_weight()`. Allow multiple consumers to reference the same shared weight.
5. Retain the owner on the model or component. After binding, check consumption completeness, CPU addresses, dtype, shape, stride, and pinned status.

Once a shared view enters the consumer path, do not create private copies through unconditional `clone()`, `contiguous()`, dtype conversions, or repinning. Perform materializing conversions during leader population. View operations such as transpose are allowed when they preserve storage identity.

Processes may map the same storage at different virtual addresses. Validate against each process's local arena base address and manifest layout; do not compare `data_ptr()` values across processes for equality.

`host` groups replicas by host, IPC namespace, and weight signature. `numa` additionally groups by the NUMA node associated with each participating rank. Actual topology determines leaders and replica counts; do not hard-code rank 0, eight GPUs, or two NUMA nodes.

See the model case studies for current TP, LoRA, quantization, AdaLN cache, and other restrictions. Add explicit checks only for confirmed incompatible combinations. Do not turn one adapter's restrictions into bans for every model.

## 4. Define lifetimes explicitly

- Keep shared CPU sources immutable. Request state, KV caches, and mutable runtime buffers remain local to each rank.
- Wait for consumers before overwriting a GPU slot, and wait for H2D completion before computation. Resetting event bookkeeping does not replace waiting for the previous request or tile to finish.
- Keep the owner alive after temporary weight maps are destroyed. Before closing an arena, stop all accesses and complete all relevant DMA.
- Separate phase-specific GPU buffer release from final CPU arena closure. A later request must be able to rebuild GPU buffers from the original CPU sources.
- Keep component and status-exchange order identical across ranks during shared initialization. CPU loading status currently uses the distributed Store; waiting for CPU loading does not enqueue NCCL collectives. Propagate local preflight and binding failures through the existing coordination mechanism and release local resources already created.

When changing streams, release paths, or reconstruction, validate consecutive requests and failure paths. Do not rely on incidental Python GC timing for device-operation safety.

## 5. Deliver a consistent launch entry point

Keep one directly editable shared-offload launcher, an English `README.md`, and a Chinese `README_CN.md` under each model's `scripts/<model>/offload/`, referencing a version-controlled JSON configuration. Use task-neutral script and configuration names while preserving existing model naming. Select supported tasks with `--task` in the command. Default examples use H3 `t2av`, Qwen/Hunyuan `t2i`, and Wan `i2v`:

```text
scripts/<model>/offload/
├── run_<model>_block_shared_offload.sh
├── README.md
└── README_CN.md
```

Prefer `configs/<model>/offload/` for configurations; Wan retains `configs/offload/block/`. The four current examples default to host sharing and eight GPUs, explicitly setting `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and `--nproc_per_node=8`. Hunyuan defaults to TP2 × SP2 × CFG2; the other three use SP8.

Select scope with `--shared_cpu_weight_scope host|numa` in the inference command. This startup argument overrides the same field in a legacy JSON file and stays out of request inputs. The four current JSON files do not repeat the scope setting. If the CLI option is omitted, the JSON value applies; if both are absent, the internal default remains `auto`. Distinguish the launcher's host default from the internal auto default.

The current scripts do not use `TASK`, `CONFIG_JSON`, `SHARED_CPU_WEIGHT_SCOPE`, or similar environment variables to override their commands. They do not forward arguments appended to `bash script.sh` or infer GPU counts. Edit the full command to change paths, GPUs, task, scope, and inputs; update the JSON when changing parallel settings. To avoid editing files, run the full Python command directly. Do not reintroduce environment-variable wrappers, temporary JSON generators, or duplicate launchers for scope, GPU count, or task.

Default shared configurations include `cpu_offload=true`, `offload_granularity="block"`, `shared_cpu_weights=true`, `shared_cpu_weight_backend="sysv"`, `shared_cpu_weight_strict_numa=true`, `shared_cpu_weight_register_chunk_mb=128`, and `lazy_load=false`. Enable additional component-sharing flags only when their paths are implemented. Private block baselines may be used for validation; that alone does not justify new permanent scripts or dedicated configurations in the shared entry directory. Existing entry points elsewhere are not an automatic cleanup target.

Scripts and configurations must also follow these constraints:

- When switching the same task between host and NUMA, keep checkpoint, task, inputs, shape, steps, seed, dtype, operators, and parallel scale identical. Change only the sharing scope.
- Set `lightx2v_path` and `model_path` at the top of the script, follow the variable contract of `scripts/base/base.sh`, and launch with the current environment's `python -m torch.distributed.run ... -m lightx2v.infer ...`. Run from the repository root; relative checkpoint paths in JSON also depend on the working directory.
- Manually keep the GPU list, `--nproc_per_node`, and JSON topology consistent. The process count must equal `tensor_p_size * cfg_p_size * seq_p_size` in the effective configuration, treating missing dimensions as 1. Do not silently replace the user's parallel settings.
- Read defaults before retaining environment assignments. The four current launchers inherit `DTYPE=BF16` and `SENSITIVE_LAYER_DTYPE=None` from `base.sh`; `None` follows the main dtype. Existing environment values are preserved, so record effective dtypes when comparing results. Old FP16 validation does not establish BF16 correctness after Wan's default changes.
- `torchrun` defaults to `OMP_NUM_THREADS=1` only for multiple processes when the environment has not set it. The Hunyuan runner adds the upstream import path using `HUNYUAN_IMAGE3_REPO_PATH`; the launcher need not append that directory to `PYTHONPATH` again. Do not add redundant exports or `set -e` just to standardize a template. Explain settings required for actual semantics separately.
- Explicit NUMA mode must not silently fall back to host. Unknown topology and strict NUMA binding failures must raise errors. GPU counts remain constrained by attention heads and the parallel implementation.
- Both READMEs must explain default tasks, supported `--task` values and inputs, NUMA selection, other GPU counts, configuration locations, defaults and precedence, environment and asset requirements, and component-sharing scope. When changing tasks, update inputs, model variant, caches, and outputs together; changing the task name alone is insufficient. Keep experimental reports out of usage guides.

## 6. Validate and deliver

Choose checks relevant to the change using [validation-and-debugging_en.md](references/validation-and-debugging_en.md). Validate schema, storage identity, and synchronization contracts first, then run the real host and NUMA entry points. Sharing and registration across ranks require corresponding runtime evidence; CPU mocks do not replace device validation.

A final delivery includes:

1. Component scope, support matrix, key call chains, and remaining limitations.
2. Weight-adapter, scheduling, and lifetime changes, with focused validation results.
3. One host/NUMA launcher under `scripts/<model>/offload/`, one shared configuration, and English and Chinese READMEs.
4. Private/shared comparisons under matched configurations, consecutive-request checks, and evidence of shared storage and memory use.
5. If performance is in scope, separate time and memory measurements for loading/registration, the first request, and steady inference.

An interface without a working route for either scope is not a complete integration. If assets or hardware are missing, retain completed work and identify unvalidated items and missing prerequisites. Do not describe the existence of a script, successful mocks, or a small-shape smoke test as acceptance of the target path.
