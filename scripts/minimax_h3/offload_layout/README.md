# MiniMax-H3 CPU Offload

English | [简体中文](README_CN.md)

Run MiniMax-H3 text-to-audio/video on one device to compare DiT CPU block offload layouts: `baseline` copies weights tensor by tensor; `contiguous` stores each block in contiguous pinned memory and copies it as a whole.

Install the project dependencies for your platform and prepare the full BF16 Diffusers weights, including the text encoder and video/audio VAEs; see the [model instructions](../README.md). Set the weight path from the repository root:

```bash
export MODEL_PATH="$PWD/models/MiniMax-H3"
```

Generate a matching AdaLN cache before the first run. Skip cache generation if a cache already matches your weights and config. Baseline and contiguous use the same cache.

## NVIDIA GPU

```bash
# First run: generate the cache
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline.json \
  --model-variant fl2av

# Generate audio/video
bash scripts/minimax_h3/offload_layout/run_baseline.sh
bash scripts/minimax_h3/offload_layout/run_contiguous.sh
```

## Ascend NPU

```bash
# First run: generate the cache
PLATFORM=ascend_npu ASCEND_RT_VISIBLE_DEVICES=0 DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$MODEL_PATH" \
  --config_json configs/minimax_h3/layout/baseline_npu.json \
  --model-variant fl2av

# Generate audio/video
bash scripts/minimax_h3/offload_layout/run_baseline_npu.sh
bash scripts/minimax_h3/offload_layout/run_contiguous_npu.sh
```

The scripts use device 0 by default. Configs are in `configs/minimax_h3/layout/`; set the prompt through `--prompt` in each script. Outputs are saved to `save_results/minimax_h3_layout/`; rerunning a script overwrites its output.
