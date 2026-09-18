#!/bin/bash
set -e

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

python -m torch.distributed.run --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
  --shared_cpu_weight_scope host \
  --prompt "A cinematic fox walks through a snowy pine forest at dawn, with soft wind, crunching snow, and distant birds." \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_t2av_block_shared_offload.mp4" \
  --seed 42
