# Common Implementation and Integration Contracts

English | [简体中文](implementation-patterns.md)

Paths are relative to the repository root. This is a guide to the existing implementation; locate current source by function name rather than fixed line numbers. Read the loading, scheduling, or lifetime sections relevant to the task.

## From launch to weight binding

```text
scripts/<model>/offload/*.sh
  → lightx2v/infer.py
  → utils/set_config.py: build_cli_inputs → build_startup_config
  → init_parallel
  → models/runners/runner_factory.py: build_runner
  → runner.init_modules / load_model / load_transformer
  → model constructor
  → BaseTransformerModel._init_weights → _init_weights_impl
      ├─ private path: _load_ckpt / _load_quant_ckpt
      └─ shared path: model._load_shared_cpu_weights
          → model adapter preflight
          → materialize_shared_weight_arena
          → SharedWeightViewMap(private, shared, owner=allocation)
      → pre / transformer / post weight construction and _apply_weights
      → _validate_shared_cpu_weights
  → model._init_infer → _init_offload_manager
```

See `lightx2v/utils/set_config.py` and `lightx2v/models/runners/default_runner.py` for configuration and device initialization. `set_init_device()` selects CPU according to offload settings. The runner handles orchestration; model and component loaders own weights. Preserve checkpoint formats already supported by the family; adding sharing should not require converting every model to a different format.

`build_startup_config()` reads `shared_cpu_weight_scope` from JSON, overriding the internal `auto` default. The four current shared configurations explicitly set `host`; change this field to `numa` in JSON for NUMA sharing. There is no CLI option for this field. The startup configuration returned by `build_cli_inputs()` contains the value; per-request inputs do not.

Key contracts in [base_model.py](../../../../lightx2v/models/networks/base_model.py):

| Entry point | Integration responsibility |
|---|---|
| `_load_shared_cpu_weights(unified_dtype, sensitive_layer)` | Model override returning a shared mapping with an owner; private non-block weights retain the correct loading and dtype policy |
| `_init_weights_impl()` | Retain `_shared_cpu_weight_owner` before `_apply_weights()` consumes the mapping, construct weight objects, and validate shared bindings |
| `_init_weights()` | Coordinate model-initialization errors across ranks and close retained owners on failure |
| `_validate_shared_cpu_weights()` | Validate transformer blocks by default; models or components add stricter layout checks and checks for additional shared components |
| `_init_offload_manager()` | Supply GPU block/phase buffers to the inference manager; extra CPU staging belongs to the lazy-loading branch and is distinct from a complete shared CPU source |
| `close_shared_cpu_weights()` | Close the owner after device transfers and accesses have stopped; old views must not be used afterward |

If an integration passes `weight_dict` directly, check whether it bypasses the default shared branch and owner retention. A configuration flag does not automatically manage lifetimes for every custom loading route.

## Manifests, replicas, and shared arenas

Core files:

- [checkpoint_metadata.py](../../../../lightx2v/common/offload/checkpoint_metadata.py): `read_safetensors_header()`, `checkpoint_content_digest()`.
- [shared_pinned_arena.py](../../../../lightx2v/common/offload/shared_pinned_arena.py): `SharedWeightManifest`, `ReplicaPlanner`, `SharedPinnedArena`, `CudaHostRegistration`.
- [shared_weight_coordinator.py](../../../../lightx2v/common/offload/shared_weight_coordinator.py): `validate_shared_weight_config()`, `coordinate_rank_local_error()`, `materialize_shared_weight_arena()`.

The model adapter first checks shard/index completeness, tensor names, layer indices, shapes, source dtypes, and target runtime dtypes. Describe final CPU storage with meta tensors and build a deterministic manifest. Shared signatures should cover checkpoint contents, component selection, and versions of conversions that affect interpretation or numerical values. Filenames or file sizes alone cannot identify weights.

`checkpoint_content_digest()` reuses a cached digest when still valid; otherwise, it scans the file. Followers may therefore perform file I/O. Only leaders populating shared tensor payloads does not mean only leaders ever read checkpoint bytes.

Coordination proceeds as follows:

1. All ranks discover host, IPC namespace, and current GPU PCI/NUMA information, then exchange manifest and policy data.
2. Check layouts and policies across ranks and plan replicas; select the lowest rank in each group as leader.
3. Each leader creates a SysV segment, applies NUMA binding policy, registers its local mapping, and invokes the adapter's `populate(views)`.
4. Exchange shared-segment descriptors. Followers in each group attach to the same segment and CUDA-register their own mappings.
5. Coordinate attachment/registration results and return the process-local `SharedArenaAllocation` owner.

`_CPUStatusExchange` exchanges status through the job's default Store and a separate `PrefixStore`; it does not enqueue CPU-loading wait collectives on the default NCCL group. `LIGHTX2V_SHARED_WEIGHT_TIMEOUT_SECONDS` limits each exchange's wait, defaulting to 3600 seconds. It does not change NCCL timeouts. Stage sequence numbers and names must match. The first failure is retained for waiting ranks, late arrivals, and enclosing initialization error handlers.

Preserve staged error propagation. A rank must not return alone while its peers have entered a status exchange. A rank busy loading CPU weights observes a failure when it next participates in coordination. Hard exits are detected through timeout; a disconnected Store may permit only local cleanup, with no guarantee that every peer receives the original error.

| Scope | Grouping rule |
|---|---|
| `host` | One complete shared-weight replica per host, IPC namespace, and weight signature |
| `numa` | Within those boundaries, one complete shared-weight replica per participating GPU NUMA node |
| `auto` | Decide independently for each host/IPC/signature cohort: use NUMA when every node is known, otherwise host |

NUMA replicates complete shared weights across memory domains; it does not shard weights by rank. Explicit `numa` fails on unknown topology. `shared_cpu_weight_strict_numa` controls memory-binding failure policy, not fallback from unknown topology to host. Host scope cannot share one SysV segment across IPC namespaces either.

`SharedWeightManifest.from_tensors()` defines aligned layouts; existing adapters typically use 4096-byte alignment. `tensor_views()` creates tensor views over the segment. `shared_cpu_weight_register_chunk_mb` is a target registration-region size, not arena capacity. Tensor-aware partitioning avoids splitting individual tensors, so a large tensor's registration region may exceed that value.

Set NUMA memory policy before first touch or registration. Ordinary `pin_memory()` or `Tensor.share_memory_()` does not replace the contract of a shared segment, per-process CUDA registration, and topology policy.

## From shared views to operators

[shared_weight_map.py](../../../../lightx2v/common/offload/shared_weight_map.py) manages private and shared weights separately:

- `SharedWeightViewMap.take()` removes private values when consumed. For shared values, it records consumption but retains the value for other consumers.
- `consume_weight()` returns `(tensor, is_shared)` while supporting consumption from ordinary dictionaries.
- `validate_shared_operator_views()` checks that the entire manifest was consumed and that final operator tensors retain shared storage.

In the shared branch, `lightx2v/common/ops/utils.py:create_default_tensors()` retains tensors or `.t()` views directly. The private branch uses the original pinned-tensor construction logic. Operator `state_dict()` methods should expose CPU pinned sources; `load_state_dict()` copies them into existing GPU buffers. Audit every operator used by the target model, including embeddings, ordinary tensors, norms, quantization scales, and model-specific weight paths.

The current validator checks this pointer relationship:

```text
expected_ptr = arena.address + spec.offset + spec.storage_offset * spec.itemsize
```

It also checks CPU device, dtype, shape/stride, and `is_pinned()`. Ordinary two-dimensional weights may use their original or transposed layout. If an operator requires a particular orientation, add semantic validation as Qwen does, so equal shapes in square matrices do not conceal a transpose error.

Sharing does not make GPU use free: CPU binding avoids an extra payload copy, but each rank still performs its own device copies. Shared CPU storage follows an immutability contract; the current mechanism does not provide OS-enforced read-only mappings. In-place dtype, quantization, or LoRA updates need a separate design and must not mutate the shared source.

## Two GPU buffer scheduling patterns

### WeightModule and stream swapping

`WeightAsyncStreamManager` in [manager.py](../../../../lightx2v/common/offload/manager.py):

- `init_cuda_buffer()` registers existing staging objects; `init_first_buffer()` prepares the first block and updates initialization state.
- `prefetch_weights()` copies the next block into the spare buffer on the loading stream.
- The current block executes on the compute stream; `swap_blocks()` waits for relevant work before exchanging the two buffers' roles.

CPU storage holds every block's weights. A family of structurally compatible blocks typically needs only two GPU slots. Immutable weights do not need to be copied back to CPU after each step. Account separately for non-block weights, activations, workspaces, KV caches, and other components.

First- and last-block policies belong to the caller protocol. Some models prefetch block 0 across step boundaries; others reinitialize each loop. Trace `need_init_first_buffer` and the actual loop instead of imposing one policy on every model. The compute stream must also wait for inputs produced by the caller, and outputs need the dependencies required for subsequent caller use.

### Fixed slots with ready/free events

`EventSlotWeightAsyncStreamManager` in [event_manager.py](../../../../lightx2v/common/offload/event_manager.py):

```text
prefetch_to_slot(slot, block)
  → load stream waits for the slot's previous free event
  → H2D → record ready
wait_ready(slot, caller_compute_stream)
  → compute stream waits for ready → execute block
record_free(slot, caller_compute_stream)
  → record completion dependency, allowing later overwrite
```

Record `record_free()` after the last computation that consumes the slot. Clearing Python-side pending state does not mean the GPU has finished. Callers must use the stream that actually performs computation, not an unrelated default stream.

`reset_slots()` only clears bookkeeping. Before consecutive requests, VAE tiles, or phase reentry, complete dependencies on previous work using a completion event or synchronization, then reset. Event scheduling may reduce unnecessary waits, but lower total latency must be measured.

## Native nn.Module adapters and lifetimes

[module_adapter.py](../../../../lightx2v/common/offload/module_adapter.py) provides:

| Utility | Purpose and limitations |
|---|---|
| `module_tensors()` | Enumerate parameters and buffers, retaining tied names and nonpersistent buffers so aliases referencing old devices are not missed |
| `assign_module_tensor()` | Bind parameters/buffers explicitly; does not own synchronization or owner lifetime |
| `ModuleCPUWeights` | Retain CPU sources; `activate()` moves selected submodules to GPU and `restore()` restores original sources |
| `NativeModuleBlockSource` | Expose CPU weights through detached views without cloning storage |
| `NativeModuleBlockSlot` | Copy a meta skeleton, allocate GPU tensors, and copy weights through `load_state_dict()` without deep-copying a full CPU block |

The model validates which block schemas a slot can serve and preserves operator-required strides. A dynamic buffer does not automatically belong in the shared manifest just because `module_tensors()` enumerates it; shareability depends on immutability and the checkpoint contract.

Lifetime order:

```text
create owner → bind CPU views → initialize/rebuild GPU slots → asynchronous copies and computation
  → wait for phase completion → optionally release GPU slots (retain CPU views and owner)
  → rebuild slots for the next request
  → stop all accesses and complete DMA → unregister → detach
```

Deleting a temporary mapping does not close the owner; the actual holder must retain it. Releasing GPU slots should not reread checkpoints or destroy shared CPU sources. Native modules restore original CPU views at phase boundaries rather than creating private copies through `.cpu()`.

`CudaHostRegistration.close()` retains the state needed to retry when synchronization or unregister fails. Do not force-detach pages still in device use. The common implementation owns SysV automatic-removal markers and final detachment. Do not use global `ipcrm` cleanup on segments belonging to other jobs.

If compile is in scope, keep copying and scheduling outside the compute graph and manage compilation caches according to staging-object lifetimes. Do not bypass the existing compile/warmup lifecycle with a new offload manager.
