# Qwen Image CPU block offload layouts

English | [简体中文](README_CN.md)

Compare two DiT weight layouts for Qwen-Image-2512 BF16 on one NVIDIA GPU or Ascend NPU. The baseline allocates pinned storage for each tensor and copies tensors individually. The contiguous variant allocates one pinned buffer per block during loading and copies the whole block during inference. Both variants use the same compute operators and double-buffer scheduling on each platform.

## Run

For NVIDIA GPUs, run from the repository root:

```bash
bash scripts/qwen_image/layout/run_baseline.sh
bash scripts/qwen_image/layout/run_contiguous.sh
```

Set the repository path, model path, and GPU index at the top of each script. The model directory must contain the original BF16 `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`, and related files. An FP8 checkpoint cannot replace the BF16 weights. Install the project's CUDA dependencies, including FlashAttention 3, FlashInfer, and Triton, and use a GPU supported by those operators.

Defaults are T2I, 50 steps, 16:9, CFG 4.0, and seed 42, with the existing Qwen prompt and compute operators. Both scripts explicitly select BF16 with matching sensitive-layer precision and disable debug timing after sourcing `base.sh`, avoiding timing synchronizations in the offload schedule.

The configs are `configs/qwen_image/layout/baseline.json` and `continuous.json`. Their only difference is `"cpu_offload_layout": "contiguous"` in the latter. Both enable the existing whole-model CPU offload for the Qwen2.5-VL text encoder and VAE; contiguous block storage applies only to the DiT.

Outputs are `save_results/qwen_layout/baseline.png` and `continuous.png`. Running a script again overwrites its image. Keep inputs, steps, and operators identical when comparing the two layouts.

### Ascend 910B

Run in an environment with a working `torch_npu` installation:

```bash
MODEL_PATH=/workspace/code/models/Qwen-Image-2512 \
  bash /workspace/code/scripts/qwen_image/layout/run_baseline_npu.sh
MODEL_PATH=/workspace/code/models/Qwen-Image-2512 \
  bash /workspace/code/scripts/qwen_image/layout/run_contiguous_npu.sh
```

The NPU scripts locate the repository automatically, set `PLATFORM=ascend_npu`, and use device 0 by default. Select another single device with `ASCEND_RT_VISIBLE_DEVICES`. `MODEL_PATH` defaults to `models/Qwen-Image-2512` inside the repository and requires the same original BF16 weights, text encoder, and VAE files. Install PyTorch and torch_npu versions compatible with the driver and CANN, along with the project's Transformers, Diffusers, and other dependencies.

The configs are `baseline_npu.json` and `continuous_npu.json`; only the latter adds `"cpu_offload_layout": "contiguous"`. Both explicitly select `npu_flash_attn` and reuse the project's Qwen NPU choices of `torch_real_rope`, PyTorch RMSNorm, LayerNorm, and modulation. These PyTorch operations execute on NPU tensors. BF16, 50 steps, 16:9, CFG 4.0, seed 42, and component offload settings match the GPU launchers.

Outputs are `save_results/qwen_layout/baseline_npu.png` and `continuous_npu.png` inside the repository, regardless of the working directory at launch. The development machine has no NPU; launcher checks and local regression tests do not replace full generation and output comparisons on a 910B.

## Integration

`QwenImageTransformerWeights` registers the blocks under `transformer_blocks.{i}.` and its two existing device slots. A block includes all four phases: image attention, text attention, joint attention, and FFN.

```text
register_offload_group
  → BaseTransformerModel._apply_weights
  → prepare_contiguous_groups → operator storage descriptions → BlockLoadPlan.load
  → tensor views in pinned CPU blocks and device slots

QwenImageOffloadTransformerInfer
  → init_first_buffer → prefetch_weights
  → ContiguousBlockTransfer.copy → run_block → swap_blocks
```

This reuses `lightx2v/common/offload/` and `lightx2v_platform/` without a Qwen-specific layout loader. CPU weights remain immutable during inference, with no per-step packing. Separate blocks do not need adjacent addresses. RMSNorm auxiliary device scalars are still copied separately.

These launchers cover single-device BF16 block offload with eager execution and NoCaching. Shared weights, lazy loading, parallelism, LoRA, and compilation are excluded. Existing `fp8-sgl` distillation configs are outside this scope. Contiguous storage reduces allocation and transfer submission counts, not the main weight payload; inference speedup must be measured.
