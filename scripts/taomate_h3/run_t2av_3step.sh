#!/bin/bash
set -eo pipefail

lightx2v_path=/data/nvme6/gushiqiao/codes/report/LightX2V
model_path=/data/nvme6/gushiqiao/models/MiniMax-H3

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

python -m lightx2v.infer \
  --model_cls taomate_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/taomate_h3/t2av_3step.json" \
  --prompt "A red panda plays a small guitar on a quiet forest stage at sunrise." \
  --num_frames 120 \
  --size 480 864 \
  --seed 8301 \
  --save_result_path "${lightx2v_path}/save_results/taomate_h3_t2av_3step.mp4"
