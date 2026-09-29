# MiniMax-H3 CPU Block Offload Layout

English | [简体中文](README_CN.md)

These launchers compare ordinary per-tensor CPU block offload with persistent contiguous pinned storage for the MiniMax-H3 DiT. Each platform's configuration pair differs only in `cpu_offload_layout`. The contiguous path fills the final CPU block storage during loading and copies its byte buffer to one of two device slots during inference. It does not pack weights at each step. Operator auxiliary device state is copied separately.

## Scope

- One NVIDIA GPU or one Ascend NPU, original BF16 checkpoint, eager inference, and no CFG.
- Launchers select `--model-variant fl2av --task t2av`, 29 steps, 124 frames, 768 × 1344, and seed 42.
- Only DiT transformer blocks use the contiguous layout. Qwen3-VL retains its existing per-layer offload; video/audio VAEs retain model offload. Pre/post weights keep their original loading and precision, including FP32 media projections.
- A matching persistent AdaLN cache is required. Cached AdaLN projection weights are excluded from both variants' block storage.
- Contiguous H3 currently rejects fused QKV projection, quantized DiT checkpoints, and `dit_release_block_offload_buffers`. The common loader also rejects shared CPU weights, disk lazy loading, multi-rank parallelism, LoRA/adapters, feature caching, and compile/graph execution.

## Checkpoint and cache

Use the original Diffusers component layout under `MODEL_PATH`: `transformer/`, `text_encoder/`, `tokenizer/`, `processor/`, `vae/`, and `audio_vae/`. An original-format `FL2VA/` directory alone is insufficient. See the [model instructions](../README.md).

Run from the repository root. Set the model path once:

```bash
export MODEL_PATH=/workspace/code/models/MiniMax-H3
```

Generate the cache on the chosen platform before running inference. The builder refuses to overwrite an existing directory; if a matching cache already exists for these weights and settings, reuse it. Both variants on a platform use the same cache. Keep `infer_steps`, flow shifts, model variant, and weights unchanged when reusing it. The configured cache root is `~/.cache/lightx2v/adaln`.

NVIDIA:

```bash
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline.json \
  --model-variant fl2av
```

Ascend:

```bash
PLATFORM=ascend_npu ASCEND_RT_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline_npu.json \
  --model-variant fl2av
```

## Run

| Platform | Baseline | Contiguous |
|---|---|---|
| NVIDIA | `bash scripts/minimax_h3/layout/run_baseline.sh` | `bash scripts/minimax_h3/layout/run_contiguous.sh` |
| Ascend | `bash scripts/minimax_h3/layout/run_baseline_npu.sh` | `bash scripts/minimax_h3/layout/run_contiguous_npu.sh` |

Scripts resolve the repository automatically, accept `MODEL_PATH`, and select device 0 unless the corresponding visible-device environment variable is set. Precision defaults come from `scripts/base/base.sh`: `DTYPE=BF16` and `SENSITIVE_LAYER_DTYPE=None`, unless set in the environment. Output files are `save_results/minimax_h3_layout/{baseline,contiguous,baseline_npu,contiguous_npu}.mp4` under that repository. Additional CLI arguments are forwarded, for example `--save_result_path /workspace/code/results/h3.mp4`; the runner creates the parent directory automatically.

CUDA uses SageAttention2, SGL RMSNorm, and H3 Triton RoPE. Ascend uses `npu_flash_attn`, `npu_rms_norm`, and `minimax_h3_npu_rope`; the latter uses MindIE-SD when available and otherwise the existing torch real-RoPE fallback. Both NPU variants keep `qwen3vl_attn_type=torch_sdpa` to preserve causal/GQA semantics, and `vae_attn_type=torch_sdpa`. Setting DiT attention does not change the text encoder's attention. The current NPU flash-attention wrapper must not be substituted for causal text attention without implementing and validating masking.

## Integration and validation

`MiniMaxH3TransformerWeights` registers `transformer_blocks.{i}.` and the existing two slots. The model's checkpoint reader records original metadata while preserving its native mixed precision. `BaseTransformerModel`, `BlockLoadPlan`, `BlockBuffer`, and `ContiguousBlockTransfer` perform the common loading and transfer work. `MiniMaxH3OffloadTransformerInfer` keeps the original block loop and synchronization. Platform memory operations stay in `lightx2v_platform`; NVIDIA's device class is unchanged.

Development-only storage and scheduling test scripts are not included in this repository. For validation, run the baseline and contiguous launchers above on each target platform with identical inputs and settings. Compare video and audio latents, then inspect decoded outputs. CUDA validation does not establish Ascend 910B correctness; the NPU launchers require validation on that machine.

Contiguity refers to virtual addresses within each block. It reduces separate allocations and H2D submissions, not the amount of model data or attention computation. Performance must be measured separately after correctness is established.
