# MiniMax-H3 CPU Offload

English | [简体中文](README_CN.md)

Run MiniMax-H3 text-to-audio/video on one device to compare DiT CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Prepare the full BF16 Diffusers components described in the [model instructions](../../README.md). Set `lightx2v_path` and `model_path` in both scripts. Generate the matching [AdaLN cache](../../README.md#adaln-cache) first; baseline and contiguous share a cache for the same weights and schedule.

## NVIDIA GPU

Set `CUDA_VISIBLE_DEVICES` in both scripts. In the cache-generation script, select `PLATFORM=cuda`, `--model-variant fl2av`, and `configs/minimax_h3/layout/baseline.json`.

```bash
bash scripts/minimax_h3/inference/offload_layout/run_baseline.sh
bash scripts/minimax_h3/inference/offload_layout/run_contiguous.sh
```

## Ascend NPU

Set `ASCEND_RT_VISIBLE_DEVICES` in both NPU scripts. In the cache-generation script, select `PLATFORM=ascend_npu`, `--model-variant fl2av`, and `configs/minimax_h3/layout/baseline_npu.json`.

```bash
bash scripts/minimax_h3/inference/offload_layout/run_baseline_npu.sh
bash scripts/minimax_h3/inference/offload_layout/run_contiguous_npu.sh
```

Configs are in `configs/minimax_h3/layout/`. Set the prompt through `--prompt` in each script. Outputs are saved to `save_results/minimax_h3_layout/`; rerunning a script overwrites its output.

`cpu_offload_block_contiguous` is `false` in baseline configs and `true` in contiguous configs.
