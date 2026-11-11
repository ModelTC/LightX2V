#!/bin/bash

lightx2v_path=/data/nvme1/yongyang/dan/x2v_infer/h3-image/LightX2V
model_path=/data/nvme1/models/MiniMaxAI/MiniMax-H3-Image

export CUDA_VISIBLE_DEVICES=0

source "$lightx2v_path/scripts/base/base.sh"

python "$lightx2v_path/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_cls minimax_h3_image \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image.json"
