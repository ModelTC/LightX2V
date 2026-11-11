#!/bin/bash
set -eo pipefail

# Set repository/model paths and dit_original_ckpt in the selected JSON config.
lightx2v_path=/data/zhangbilang/LightX2V
model_path=/data/zhangbilang/models/MiniMaxAI/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /data/zhangbilang/cuda130/use_cuda13_compat.sh
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=0

# Eight-GPU BF16 inference with the existing post_refa2av.py request.
python -m torch.distributed.run --standalone --nproc_per_node=8 -m lightx2v.server \
  --model_cls minimax_h3_causal \
  --task refa2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_causal/h100/refa2av_realtime_768p_sp8.json" \
  --host 0.0.0.0 \
  --port 8000
