# Qwen Image CPU Offload

English | [简体中文](README_CN.md)

Run Qwen-Image-2512 text-to-image on one device to compare CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Install the project dependencies for your platform and prepare the full BF16 weights, including the text encoder and VAE. Run from the repository root.

## NVIDIA GPU

Set `lightx2v_path`, `model_path`, and the GPU index at the top of both scripts.

```bash
bash scripts/qwen_image/offload_layout/run_baseline.sh
bash scripts/qwen_image/offload_layout/run_contiguous.sh
```

## Ascend NPU

The scripts use device 0 by default. Set the weight directory with `MODEL_PATH`:

```bash
MODEL_PATH="$PWD/models/Qwen-Image-2512" \
  bash scripts/qwen_image/offload_layout/run_baseline_npu.sh
MODEL_PATH="$PWD/models/Qwen-Image-2512" \
  bash scripts/qwen_image/offload_layout/run_contiguous_npu.sh
```

Configs are in `configs/qwen_image/layout/`. Set the prompt through `--prompt` in each script. Images are saved to `save_results/qwen_layout/`; rerunning a script overwrites its output.
