# Qwen Image Resident Blocks

English | [简体中文](README_CN.md)

Run Qwen Image text-to-image on one device to compare a zero-residency baseline with resident blocks. Baseline transfers weights from the CPU as needed; residency keeps the estimated number of blocks evenly distributed across the model to reduce repeated transfers.

Prepare the complete `Qwen-Image-2512` BF16 model and set `lightx2v_path` and `model_path` at the top of the scripts. Keep positive and negative prompts identical across all three scripts, and use the same seed for inference. Defaults are 16:9 and 50 steps.

Run the commands below from the repository root: estimate and generate the residency config first, then compare inference runs. Baseline also runs independently.

## NVIDIA GPU

Set `CUDA_VISIBLE_DEVICES` in all three CUDA scripts.

```bash
bash scripts/qwen_image/inference/offload_residency/estimate.sh
bash scripts/qwen_image/inference/offload_residency/run_baseline.sh
bash scripts/qwen_image/inference/offload_residency/run.sh
```

## Ascend NPU

Set `ASCEND_RT_VISIBLE_DEVICES` in all three NPU scripts.

```bash
bash scripts/qwen_image/inference/offload_residency/estimate_npu.sh
bash scripts/qwen_image/inference/offload_residency/run_baseline_npu.sh
bash scripts/qwen_image/inference/offload_residency/run_npu.sh
```

Base configs are in `configs/qwen_image/residency/`: `baseline.json` for CUDA and `baseline_npu.json` for NPU. Outputs go to `save_results/qwen_residency/`: `resident.json` is the generated config, `resident.profile.json` is the measurement report, and `baseline.png` and `resident.png` are inference outputs. NPU filenames use the `_npu` suffix. Rerunning overwrites files with the same names.

In the generated config, `offload_resident_blocks` is the resident count; `0` means no resident blocks. Re-estimate after changing the device, inputs, or base config. Profiling follows [base.sh](../../../base/base.sh).
