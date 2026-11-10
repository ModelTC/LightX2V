#!/bin/bash
set -eo pipefail

lightx2v_path=/data/LightX2V
model_path=/MiniMaxAI/MiniMax-H3-Image

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

source "$lightx2v_path/scripts/base/base.sh"

# Enable use_adaln_cache and set adaln_cache_dir in the inference config first.
python "$lightx2v_path/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_cls minimax_h3_image \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image_native.json"
