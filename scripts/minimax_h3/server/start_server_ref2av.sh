#!/bin/bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --task in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --task fl2av for t2av/i2av/l2av/fl2av, or --task ref2av for ref2av.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
# Cache generation and inference must use the same JSON config and adaln_cache_dir.

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
