#!/bin/bash
set -eo pipefail

# Set repository/model paths and dit_original_ckpt in the selected JSON config.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Warmup compiles the configured size/num_frames before serving requests.
torchrun --standalone --nproc_per_node=8 -m lightx2v.server \
  --model_cls minimax_h3_causal \
  --task refa2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_causal/minimax_h3_causal_compile.json" \
  --host 0.0.0.0 \
  --port 8000
