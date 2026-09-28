# Wan CPU Block Offload Layout

English | [简体中文](README_CN.md)

The baseline allocates CPU weights separately and copies them to the device tensor by tensor. The contiguous variant loads weights directly into a contiguous pinned buffer for each block, then transfers the entire block during inference. Both variants use the same computation and double-buffer scheduling.

## CUDA inference

Run from the repository root. Each script generates one video:

```bash
bash scripts/wan/layout/run_baseline.sh
bash scripts/wan/layout/run_contiguous.sh
```

Set the model path and GPU index at the top of each script. The configurations are `configs/wan/layout/baseline.json` and `continuous.json`; the latter only adds `"cpu_offload_layout": "contiguous"`.

The defaults are Wan2.1 I2V 14B LightX2V with FP8 DiT/T5/CLIP weights, BF16 computation, and FlashAttention 3: one GPU, seed 42, 40 steps, 81 frames, and CFG enabled. `size: [480, 832]` specifies an area budget; the actual width and height depend on the input image. Outputs are `save_results/wan_layout/baseline.mp4` and `solution2.mp4`. Running a script again overwrites its output video.

## Ascend 910B inference

On a machine with the driver, CANN, PyTorch, torch_npu, and project dependencies installed, run from the repository root:

```bash
MODEL_PATH=/workspace/code/models/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_baseline_npu.sh

MODEL_PATH=/workspace/code/models/Wan2.1-I2V-14B-720P \
bash scripts/wan/layout/run_contiguous_npu.sh
```

The defaults are one device (`ASCEND_RT_VISIBLE_DEVICES=0`), BF16, and the original Wan2.1 I2V weights, using `baseline_npu.json` and `continuous_npu.json`. The model directory must contain DiT, T5, CLIP, VAE, and tokenizer files. An FP8 checkpoint cannot be used as an original BF16 checkpoint. T5, CLIP, and VAE CPU offload is enabled in both configurations.

`t5_rms_norm_type` selects the two RMSNorm operators in each T5 offload block, independently of the DiT `rms_norm_type`. Both NPU configurations explicitly set it to `"torch"`; custom NPU configurations with T5 offload should also set this field. When omitted, it defaults to `"sgl-kernel"`, preserving the CUDA default. T5 no longer chooses these operators based on the device type. This option does not change the non-offloaded T5 path or the encoder's final normalization.

Both scripts call `python -m lightx2v.infer` directly and perform one complete inference run, producing `baseline_npu.mp4` and `solution2_npu.mp4`. Set `ASCEND_RT_VISIBLE_DEVICES` to select a device or `CONFIG_PATH` to select a configuration.

Wan NPU block offload disables private formats during model initialization, before creating weight buffers, and uses ND storage. Floating-point and INT8 checkpoints are checked for matrix precision before dtype conversion. NPU UniPC solves its small coefficient systems on the CPU while preserving the original sampling schedule.

When writing NPU weights back to a non-contiguous CPU pinned view, `copy_to_cpu` in `lightx2v_platform/base/ascend_npu.py` first completes a synchronous transfer into a contiguous CPU tensor, then copies into the destination according to its strides. This prevents transposed weights from being reordered incorrectly. The path preserves the original pinned storage and adds one CPU copy; CUDA keeps its existing direct-copy path.

The layout scripts inherit the profiling settings from `scripts/base/base.sh`, currently `PROFILING_DEBUG_LEVEL=2`, which synchronizes the device at profiling boundaries. To disable debug profiling, add `export PROFILING_DEBUG_LEVEL=0` after the script's `source` line.

## Contiguous layout

Each CPU block's layout is planned using the final dtype, shape, and transpose requirements. One pinned `uint8` allocation holds the block. Each tensor is a view into this storage, with its starting offset aligned to 256 bytes. Checkpoint data is copied directly into the final views, including any required dtype conversion. Contiguous refers to virtual addresses; physical pages do not need to be adjacent.

`BlockLayout` describes the byte ranges; `BlockBuffer.view()` creates the typed views. `BlockBuffer.allocate(layout, device)` allocates independent storage and initializes CPU alignment padding. `BlockBuffer(layout, storage)` instead binds existing contiguous, one-dimensional `uint8` storage with a 256-byte-aligned starting address and sufficient capacity, without copying or changing its contents. The caller initializes externally supplied padding and provides pinned storage for CPU offload. `buffer.storage` covers exactly `layout.nbytes`, even when the supplied region is larger, and views keep the underlying allocation alive. External regions must not be reused until their transfers and computations finish. Current launchers still allocate one buffer per CPU block; they do not enable a model-wide pool or per-step allocation/free.

Both device block buffers use the same layout. Prefetching transfers the entire CPU block into the spare device buffer with one `storage.copy_` call. CPU block weights remain read-only during inference, with no packing, staging buffer, or reallocation. CUDA RMSNorm auxiliary placeholder scalars retain separate D2D copies.

```text
Model weight container: register_offload_group(blocks, device_slots, prefixes)
  → BaseTransformerModel records original checkpoint metadata before casting
  → prepare_contiguous_groups
    → operator.describe_storage → BlockLayout.build
  → WeightModule.load → BlockLoadPlan.load
    → BlockBuffer.allocate → operator.load / operator.bind_storage
    → Validate views, checkpoint coverage, and pinned storage

BaseTransformerModel._init_offload_manager
  → init_contiguous_groups
  → init_first_buffer(blocks): select the group and fill its first slot
  → prefetch_weights → ContiguousBlockTransfer.copy
  → run_block → swap_blocks
```

The loader is model-independent: `lightx2v/common/offload/block_loader.py` plans and binds weights, while `block_layout.py` owns storage and transfer logic. Wan and Qwen Image register their block boundaries and existing device slots through `WeightModule.register_offload_group`. There is no Wan-specific layout loader or model/task whitelist. Each group requires matching tensor layouts and two device slots; different block structures use separate groups. Call `init_first_buffer` at a group boundary before prefetching. The existing model inference code remains responsible for execution order.

Operators describe checkpoint names, accepted source dtypes, final shapes/dtypes, transposed views, and auxiliary device state through the storage contract in `lightx2v_platform/ops/weight_storage.py`. Original checkpoint dtypes are retained before the normal loader casts tensors, so incorrect quantized weights cannot pass validation merely because they were cast to BF16. The common loader does not import concrete operator classes or choose operators by device type. Unsupported operators and undeclared state fail during loading; storage that escapes the planned views is rejected.

Operators that accept `BlockLoadContext` reuse their existing `load` method. Only operators that need different binding behavior implement `bind_storage`. Auxiliary device tensors are recorded once per block or slot and paired when copied; the manager closes every registered transfer group during cleanup.

`lightx2v_platform/base/offload.py` selects the memory backend through the existing platform device registry. NVIDIA uses standard PyTorch allocation and asynchronous copies. Ascend reuses those operations and prepares ND storage in its own backend. A new platform must explicitly declare and validate its block-offload backend; a common device type such as `cuda` does not automatically enable an untested platform. CUDA imports do not load Ascend operators. The Ascend strided D2H fix remains in `base/ascend_npu.py`.

The current contracts cover ordinary floating MM/Norm/Tensor storage, FP8-vLLM, and INT8-NPU operators with floating or quantized matrix transposes. They do not imply support for arbitrary packed quantization formats, tied storage, dynamic layouts, or every operator in every model. Shared weights, lazy loading, parallel execution, LoRA, and CUDA Graph combinations remain unsupported for contiguous layout. These launchers retain single-device eager inference and NoCaching.

Storage and integration checks can be run on a configured CUDA machine:

```bash
PLATFORM=cuda DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
python -m pytest -q test_cases/test_block_buffer.py test_cases/test_block_offload_groups.py
```

On Ascend, use `PLATFORM=ascend_npu` with the same command. CUDA-only cases are skipped. The Ascend operator storage tests also run on CUDA for coverage of loading rules; that does not replace Ascend device validation.

The baseline allocates tensors independently, so adjacent addresses within a block are not guaranteed. In the CUDA FP8 path, subsequent dtype conversion may leave some scale tensors unpinned; the actual pinned state depends on the loading result.
