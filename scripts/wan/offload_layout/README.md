# Wan CPU Offload

English | [简体中文](README_CN.md)

Run Wan image-to-video on one device to compare CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Install the project dependencies for your platform, then run from the repository root.

## NVIDIA GPU

Set `lightx2v_path`, `model_path`, and the GPU index at the top of both scripts. The defaults use `Wan2.1-I2V-14B-720P-Lightx2v` FP8 weights.

```bash
bash scripts/wan/offload_layout/run_baseline.sh
bash scripts/wan/offload_layout/run_contiguous.sh
```

## Ascend NPU

Use the original `Wan2.1-I2V-14B-720P` BF16 weights, including the text encoders and VAE. The scripts use device 0 by default.

```bash
MODEL_PATH="$PWD/models/Wan2.1-I2V-14B-720P" \
  bash scripts/wan/offload_layout/run_baseline_npu.sh
MODEL_PATH="$PWD/models/Wan2.1-I2V-14B-720P" \
  bash scripts/wan/offload_layout/run_contiguous_npu.sh
```

Configs are in `configs/wan/layout/`. Set the input image through `--image_path` in each script. Videos are saved to `save_results/wan_layout/`; rerunning a script overwrites its output.
