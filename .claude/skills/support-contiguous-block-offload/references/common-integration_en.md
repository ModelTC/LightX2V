# Common Integration

English | [简体中文](common-integration.md)

## Source index and call chain

| File | Main responsibilities |
|---|---|
| [weight_module.py](../../../../lightx2v/common/modules/weight_module.py) | `register_offload_group`, `offload_group`, `named_weight_leaves`, and dispatch to the load plan |
| [base_model.py](../../../../lightx2v/models/networks/base_model.py) | Original checkpoint metadata, `_apply_weights`, and `_init_offload_manager` |
| [block_loader.py](../../../../lightx2v/common/offload/block_loader.py) | `OffloadGroup`, `BlockLoadPlan`, `prepare_contiguous_group`, and configuration/checkpoint validation |
| [block_layout.py](../../../../lightx2v/common/offload/block_layout.py) | `BlockLayout`, `BlockBuffer`, `BlockLoadContext`, and `ContiguousBlockTransfer` |
| [manager.py](../../../../lightx2v/common/offload/manager.py) | Single-group initialization, first-block loading, prefetching, double-buffer swapping, and cleanup |
| [weight_storage.py](../../../../lightx2v_platform/ops/offload/weight_storage.py) | `TensorMetadata`, `WeightStorage`, and `StorageDescription` |
| [offload/__init__.py](../../../../lightx2v_platform/ops/offload/__init__.py) | `get_block_offload_backend` and the memory-operation interface |
| [offload/template.py](../../../../lightx2v_platform/ops/offload/template.py) | `TorchBlockOffload` allocation and asynchronous copies |
| [offload/ascend_npu.py](../../../../lightx2v_platform/ops/offload/ascend_npu.py) | ND storage preparation and checkpoint validation requirements in `NpuBlockOffload` |

Preserve the existing initialization flow in `ops/__init__.py` and `set_ai_device.py`. Device classes declare `block_offload_backend` as a fully qualified module path string, avoiding imports of `ops` during device registration. After device initialization and operator registration, `get_block_offload_backend()` resolves the selected backend; direct backend class declarations remain supported. Do not import `ops.offload` at device module scope, which would trigger the parent package's operator registration too early.

```text
Model weights register a group
  → BaseTransformerModel._apply_weights
  → prepare_contiguous_group
  → BlockLoadPlan reads describe_storage from each leaf
  → WeightModule.load → BlockLoadPlan.load
  → BlockBuffer.allocate → BlockLoadContext → load / bind_storage

BaseTransformerModel._init_offload_manager
  → WeightAsyncStreamManager.init_contiguous_group
  → init_first_buffer → prefetch_weights → ContiguousBlockTransfer.copy
  → Existing block computation and swap_blocks
```

Registering a group does not replace the model's inference loop. The model must consume the selected slot at the correct computation boundary and preserve its existing synchronization order.

## Model declarations

Inspect how the model constructs block weights and offload buffers. The current `OffloadGroup` requires nonempty CPU blocks, exactly two device slots, and one corresponding checkpoint prefix ending in `.` for each block.

```python
self.register_offload_group(
    self.blocks,
    self.offload_block_cuda_buffers,
    (f"transformer_blocks.{i}." for i in range(self.blocks_num)),
)
```

This example uses Qwen prefixes; adapt them to the target checkpoint. Register only in the branch that actually creates block offload slots. Do not treat a phase buffer as an entire block.

- Every block and both slots in a group must share the same relative operator paths, shapes, dtypes, transpositions, and offsets. Equal byte counts do not establish compatibility.
- `TensorSpec.name` is excluded from layout equality, allowing checkpoint names with different layer indices to share a layout. All other structure must still match.
- Each weight container registers one group, accessed directly through `self.offload_group`; duplicate registration is rejected. Components with different layouts or lifetimes need independent weight containers and managers and cannot share one group.
- The manager uses the group bound during initialization, without group selection or switching. Registration and inference must use the same blocks container; passing another container is rejected.
- Checkpoint tensors under the declared prefixes must be described completely and uniquely. Do not assume shared/tied weight ownership, multiple consumers, or native `nn.Module` representations fit this interface. Check what it can represent and propose any required extension.

## Operator storage contracts

`TensorMetadata` distinguishes original checkpoint dtype from the dtype after loading. Describe final storage with `WeightStorage(name, attr, shape, dtype, transpose)` and device state copied alongside each block with `StorageDescription.auxiliary`.

1. Describe the loading view with the actual checkpoint shape and express the operator's compute view through `transpose`. Do not create a separate transposed weight copy first.
2. Preserve metadata before checkpoint tensors are cast together. Checking dtype only after casting cannot detect an FP8 checkpoint mistakenly used in a BF16 path.
3. Describe weights, biases, scales, and other runtime state, not just matrices. The owning operator defines quantization formats and scale layouts/conversions; do not infer them from model names.
4. Prefer having the existing `load()` consume `BlockLoadContext` through common helpers. Operators needing special binding implement `bind_storage()` and keep buffer attributes consistent with compute attributes.
5. Stateless operators may describe empty state, but their initialization through `load()` must still run. Reject unknown stateful operators explicitly.

Platform RMSNorm/LayerNorm implementations with ordinary floating weights inherit storage descriptions and binding from the [norm templates](../../../../lightx2v_platform/ops/norm/norm_template.py); new chip subclasses only need to implement computation. The common loader prefers `bind_storage()`, so subclasses that add transformations or auxiliary-state initialization in `load()` must also adapt continuous binding explicitly; those steps do not run automatically.

Standard per-channel prequantized MM uses [MMWeightPerChannelQuantTemplate](../../../../lightx2v_platform/ops/mm/template.py). Subclasses declare `checkpoint_dtype`, `weight_need_transpose`, and compute kernels, inheriting checkpoint validation, FP32 scales, optional bias, and contiguous binding. NPU INT8, MLU INT8, and MUSA FP8 share this template. Packed formats or formats requiring extra transformations retain their own storage contracts.

Platforms exposing standard PyTorch pinned allocation, typed views, asynchronous copies, and stream/event APIs can declare `block_offload_backend = "lightx2v_platform.ops.offload.template.TorchBlockOffload"`, as MLU/MUSA do, without duplicating the backend. Device classes supply special D2H/H2D behavior through `copy_to_cpu` / `copy_transposed_weight_to_device`; common helpers select these through the platform registry and otherwise retain native copies. Declaring support still requires target-hardware validation. Existing model kernels and the runtime must also work; configuration cannot supply missing kernels.

Implementation references: [MM](../../../../lightx2v/common/ops/mm/mm_weight.py), [RMSNorm](../../../../lightx2v/common/ops/norm/rms_norm_weight.py), [LayerNorm](../../../../lightx2v/common/ops/norm/layer_norm_weight.py), and [DefaultTensor](../../../../lightx2v/common/ops/tensor/tensor.py). Avoid identical forwarding functions for every operator.

## Loading and memory

`prepare_contiguous_group` builds all plans and completes consistency checks before any block consumes checkpoint tensors. Without original metadata, it can describe only the current tensors; that cannot establish their precision before conversion.

`BlockBuffer` holds mixed dtypes in one contiguous one-dimensional `uint8` storage. Tensor starts are aligned to the current `ALIGNMENT_BYTES`, and padding is initialized once. `view()` creates typed views. `BlockLoadContext.take()` copies CPU sources into their final views and consumes source entries; device loading only binds views.

After loading, compare planned and actual addresses, devices, dtypes, shapes, strides, pinned status, and state entries. Unconditional `clone()`, `contiguous()`, or pinning after binding can detach a view from its planned storage. Loading contexts reference checkpoint tensors and should not be retained after use.

This path may still hold temporary checkpoint tensors during loading. Do not describe it as reading directly from disk into final storage or eliminating peak loading memory. Removing inference-time packing and reducing peak loading memory are separate goals.

## Transfers and lifetimes

`ContiguousBlockTransfer.copy()` copies the main block storage as a whole and copies auxiliary device state separately. Keep source weights read-only and retain CPU storage and device views until all users have finished.

- Computation must wait for first-block initialization. A prefetch target slot must no longer be in use by the previous block.
- `init_first_buffer` and `swap_blocks` preserve the existing scheduling synchronization. The transfer's initialization ready event and `record_stream` do not replace dependencies for each slot reuse.
- The next step, the next CFG branch, and subsequent requests must start from the correct first block. A single successful traversal does not establish correctness.
- Cleanup waits for computation and then closes the current transfer, waiting for copies to finish. Partially initialized resources must also be releasable.
- Names such as `cuda_buffers` and `create_cuda_buffer` are existing interfaces; the platform determines the actual device. Do not rewrite all callers merely to standardize terminology.

Use `validate_contiguous_config()` as the source of truth for current boundaries: a single rank, block granularity, static private weights, eager execution, NoCaching, and matching sensitive-layer/main precision. Sharing, lazy loading, parallelism, LoRA/adapters, compile/graph modes, dynamic quantization, and related combinations currently have explicit restrictions. Do not silently disable user settings to pass validation or present current restrictions as permanent impossibilities.
