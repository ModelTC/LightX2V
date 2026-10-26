#!/bin/bash
set -eo pipefail

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0

# Set environment variables.
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Build the cache on one GPU using dit_original_ckpt from the inference config.
python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_cls minimax_h3_causal \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_causal/5090/realtime_768p_fp8_sp8.json"
