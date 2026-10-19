# Wan Resident Blocks

English | [简体中文](README_CN.md)

Run Wan2.1 image-to-video on one device to compare a zero-residency baseline with resident blocks. Baseline transfers weights from the CPU as needed; residency keeps the estimated number of blocks evenly distributed across the model to reduce repeated transfers.

Set `lightx2v_path` and `model_path` at the top of the scripts. Use the same image for estimator `--image-path` and inference `--image_path`, and keep prompts and seed identical between inference scripts.

Run the commands below from the repository root: estimate and generate the residency config first, then compare inference runs. Baseline also runs independently.

## NVIDIA GPU

Use `Wan2.1-I2V-14B-720P-Lightx2v` FP8 weights and set `CUDA_VISIBLE_DEVICES` in all three scripts. Set `dit_quantized_ckpt`, `t5_quantized_ckpt`, and `clip_quantized_ckpt` in `configs/wan/residency/baseline.json`, preferably to absolute paths; changing `model_path` does not update them.

```bash
bash scripts/wan/inference/offload_residency/estimate.sh
bash scripts/wan/inference/offload_residency/run_baseline.sh
bash scripts/wan/inference/offload_residency/run.sh
```

## Ascend NPU

Use the complete `Wan2.1-I2V-14B-720P` BF16 weights, including the text encoder, CLIP, and VAE. Set `ASCEND_RT_VISIBLE_DEVICES` in all three NPU scripts.

```bash
bash scripts/wan/inference/offload_residency/estimate_npu.sh
bash scripts/wan/inference/offload_residency/run_baseline_npu.sh
bash scripts/wan/inference/offload_residency/run_npu.sh
```

Base configs are in `configs/wan/residency/`: `baseline.json` for CUDA and `baseline_npu.json` for NPU. Outputs go to `save_results/wan_residency/`: `resident.json` is the generated config, `resident.profile.json` is the measurement report, and `baseline.mp4` and `resident.mp4` are inference outputs. NPU filenames use the `_npu` suffix. Rerunning overwrites files with the same names.

In the generated config, `offload_resident_blocks` is the resident count; `0` means no resident blocks. Re-estimate after changing the device, inputs, or base config. Profiling follows [base.sh](../../../base/base.sh).
