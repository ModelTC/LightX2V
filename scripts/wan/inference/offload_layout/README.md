# Wan CPU Offload

English | [简体中文](README_CN.md)

Run Wan image-to-video on one device to compare CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Install the project dependencies for your platform, then run from the repository root.

## NVIDIA GPU

Use `Wan2.1-I2V-14B-720P-Lightx2v` FP8 weights. Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` at the top of both scripts.

In `configs/wan/layout/baseline.json` and `contiguous.json`, set `dit_quantized_ckpt` to the absolute path of the FP8 DiT directory and `clip_quantized_ckpt` to the actual quantized CLIP file. Replace the example filename `clip-fp8.pth` if your file has a different name. These paths are independent of `model_path`.

```bash
bash scripts/wan/inference/offload_layout/run_baseline.sh
bash scripts/wan/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

Use the original `Wan2.1-I2V-14B-720P` BF16 weights, including the text encoders and VAE. Set `lightx2v_path`, `model_path`, and `ASCEND_RT_VISIBLE_DEVICES` at the top of both NPU scripts; the default device is 0.

```bash
bash scripts/wan/inference/offload_layout/run_baseline_npu.sh
bash scripts/wan/inference/offload_layout/run_contiguous_npu.sh
```

Configs are in `configs/wan/layout/`. Set the input image through `--image_path` in each script. Videos are saved to `save_results/wan_layout/`; rerunning a script overwrites its output.

`cpu_offload_block_contiguous` is `false` in baseline configs and `true` in contiguous configs.
