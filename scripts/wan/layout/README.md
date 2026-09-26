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

Both scripts call `python -m lightx2v.infer` directly and perform one complete inference run, producing `baseline_npu.mp4` and `solution2_npu.mp4`. Set `ASCEND_RT_VISIBLE_DEVICES` to select a device or `CONFIG_PATH` to select a configuration.

Wan NPU block offload disables private formats during model initialization, before creating weight buffers, and uses ND storage. Floating-point and INT8 checkpoints are checked for matrix precision before dtype conversion. NPU UniPC solves its small coefficient systems on the CPU while preserving the original sampling schedule.

When writing NPU weights back to a non-contiguous CPU pinned view, `copy_to_cpu` in `lightx2v_platform/base/ascend_npu.py` first completes a synchronous transfer into a contiguous CPU tensor, then copies into the destination according to its strides. This prevents transposed weights from being reordered incorrectly. The path preserves the original pinned storage and adds one CPU copy; CUDA keeps its existing direct-copy path.

The layout scripts inherit the profiling settings from `scripts/base/base.sh`, currently `PROFILING_DEBUG_LEVEL=2`, which synchronizes the device at profiling boundaries. To disable debug profiling, add `export PROFILING_DEBUG_LEVEL=0` after the script's `source` line.

## Contiguous layout

Each CPU block's layout is planned using the final dtype, shape, and transpose requirements. One pinned `uint8` allocation holds the block. Each tensor is a view into this storage, with its starting offset aligned to 256 bytes. Checkpoint data is copied directly into the final views, including any required dtype conversion. Contiguous refers to virtual addresses; physical pages do not need to be adjacent.

Both device block buffers use the same layout. Prefetching transfers the entire CPU block into the spare device buffer with one `storage.copy_` call. CPU block weights remain read-only during inference, with no packing, staging buffer, or reallocation. CUDA RMSNorm auxiliary placeholder scalars retain separate D2D copies.

```text
WanTransformerAttentionBlock.load
  → load_contiguous_block
    → BlockLayout / BlockBuffer
    → WeightModule.load → BlockLoadContext.take / bind
    → Validate final views and pinned state

WanModel._init_offload_manager
  → init_contiguous_blocks
  → init_first_buffer / prefetch_weights
    → ContiguousBlockTransfer.copy
  → run_block
  → swap_blocks: synchronize loading and computation, then swap device buffers
```

Common storage and transfer logic lives in `lightx2v/common/offload/block_layout.py`; Wan loading constraints live in `lightx2v/models/networks/wan/weights/block_layout.py`. The contiguous variant supports single-device Wan2.1 I2V with standard CPU block offload and NoCaching. CUDA supports native floating-point and FP8-vLLM weights; Ascend supports native floating-point and INT8-NPU weights. Shared weights, lazy loading, parallel execution, LoRA, and CUDA Graph combinations are outside the current supported scope.

The CUDA Wan loading path does not import Ascend operators. NPU operators are integrated only when NPU is selected. Contiguous view binding and dtype rules live in the existing concrete operators under `lightx2v_platform/ops/mm/ascend_npu/` and `ops/norm/ascend_npu/`; shared platform templates retain their standard loading behavior. Wan NPU initialization, T5 default operator selection, and scheduler coefficient device selection remain in explicit platform branches in the existing model files.

The baseline allocates tensors independently, so adjacent addresses within a block are not guaranteed. In the CUDA FP8 path, subsequent dtype conversion may leave some scale tensors unpinned; the actual pinned state depends on the loading result.
