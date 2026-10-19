# MiniMax-H3 Resident Blocks

English | [简体中文](README_CN.md)

Run MiniMax-H3 text-to-audio/video on one device to compare a zero-residency baseline with resident blocks. Baseline transfers weights from the CPU as needed; residency keeps the estimated number of blocks evenly distributed across the model to reduce repeated transfers.

Prepare the full BF16 Diffusers components described in the [model instructions](../../README.md). Set `lightx2v_path` and `model_path` in the scripts, keeping prompts and inference seeds consistent. Generate the matching [AdaLN cache](../../README.md#adaln-cache) first; estimation, baseline, and resident inference share it for the same weights and schedule.

Run the commands below from the repository root: estimate and generate the residency config first, then compare inference runs. Baseline also runs independently.

## NVIDIA GPU

Set `CUDA_VISIBLE_DEVICES` in all three CUDA scripts. In the cache-generation script, select `PLATFORM=cuda`, `--model-variant fl2av`, and `configs/minimax_h3/residency/baseline.json`.

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline.sh
bash scripts/minimax_h3/inference/offload_residency/run.sh
```

## Ascend NPU

Set `ASCEND_RT_VISIBLE_DEVICES` in all three NPU scripts. In the cache-generation script, select `PLATFORM=ascend_npu`, `--model-variant fl2av`, and `configs/minimax_h3/residency/baseline_npu.json`.

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_npu.sh
```

Base configs are in `configs/minimax_h3/residency/`: `baseline.json` for CUDA and `baseline_npu.json` for NPU. Outputs go to `save_results/minimax_h3_residency/`: `resident.json` is the generated config, `resident.profile.json` is the measurement report, and `baseline.mp4` and `resident.mp4` are inference outputs. NPU filenames use the `_npu` suffix. Rerunning overwrites files with the same names.

In the generated config, `offload_resident_blocks` is the resident count; `0` means no resident blocks. Re-estimate after changing the device, inputs, or base config. Profiling follows [base.sh](../../../base/base.sh).

Regenerate the matching AdaLN cache after changing weights, step count, or flow shifts.

`cpu_offload_block_contiguous` controls contiguous storage and defaults to `false`. Re-estimate after setting it to `true`, using the same value for baseline and residency.
