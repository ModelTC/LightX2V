#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Reference generation loads transformer_ref instead of the base transformer.
python -m lightx2v.server \
  --model_cls minimax_h3 \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/minimax_h3.json" \
  --host 0.0.0.0 \
  --port 8000
