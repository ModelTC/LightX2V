# Qwen Image CPU Offload

English | [简体中文](README_CN.md)

Run Qwen-Image-2512 text-to-image on one device to compare CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Install the project dependencies for your platform and prepare the full BF16 weights, including the text encoder and VAE. Commands below use repository-relative paths; absolute script paths also work.

## NVIDIA GPU

Set `lightx2v_path`, `model_path`, and the GPU index at the top of both scripts.

```bash
bash scripts/qwen_image/inference/offload_layout/run_baseline.sh
bash scripts/qwen_image/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

Set `lightx2v_path`, `model_path`, and `ASCEND_RT_VISIBLE_DEVICES` in each script:

```bash
bash scripts/qwen_image/inference/offload_layout/run_baseline_npu.sh
bash scripts/qwen_image/inference/offload_layout/run_contiguous_npu.sh
```

Configs are in `configs/qwen_image/layout/`. Set the prompt through `--prompt` in each script. Images are saved to `save_results/qwen_layout/`; rerunning a script overwrites its output.

`cpu_offload_block_contiguous` is `false` in baseline configs and `true` in contiguous configs.
