# Qwen-Image-2.1-Turbo Resident Blocks

English | [简体中文](README_CN.md)

Run Qwen-Image-2.1-Turbo on one device to compare a zero-residency baseline with resident blocks. Baseline transfers weights from the CPU as needed; residency keeps the estimated number of blocks evenly distributed across the model to reduce repeated transfers.

Prepare the complete BF16 model with `transformer`, `text_encoder`, `processor`, and `vae`. Set `lightx2v_path` and `model_path` in the scripts, keeping prompts and inference seeds consistent. Defaults are text-to-image at 2048 × 2048, 8 steps, with CFG disabled.

Run the commands below from the repository root: estimate and generate the residency config first, then compare inference runs. Baseline also runs independently.

## NVIDIA GPU

Set `CUDA_VISIBLE_DEVICES` in all three CUDA scripts.

```bash
bash scripts/qwen_image_21/inference/offload_residency/estimate.sh
bash scripts/qwen_image_21/inference/offload_residency/run_baseline.sh
bash scripts/qwen_image_21/inference/offload_residency/run.sh
```

## Ascend NPU

Set `ASCEND_RT_VISIBLE_DEVICES` in all three NPU scripts.

```bash
bash scripts/qwen_image_21/inference/offload_residency/estimate_npu.sh
bash scripts/qwen_image_21/inference/offload_residency/run_baseline_npu.sh
bash scripts/qwen_image_21/inference/offload_residency/run_npu.sh
```

## Image editing

Append `--task i2i --prompt 'Edit instruction'` to the commands above. Also append `--image-path /path/to/input.png` to estimation and `--image_path /path/to/input.png` to both inference commands. Separate multiple image paths with commas and use the same inputs for all three commands.

After switching to image editing, estimate again before running both inference scripts. The default config and image files are overwritten.

Base configs are in `configs/qwen_image_21/residency/`: `baseline.json` for CUDA and `baseline_npu.json` for NPU. Outputs go to `save_results/qwen_image_21_residency/`: `resident.json` is the generated config, `resident.profile.json` is the measurement report, and `baseline.png` and `resident.png` are inference outputs. NPU filenames use the `_npu` suffix. Rerunning overwrites files with the same names.

In the generated config, `offload_resident_blocks` is the resident count; `0` means no resident blocks. Re-estimate after changing the device, inputs, or base config. Profiling follows [base.sh](../../../base/base.sh).
