#!/bin/bash
set -eo pipefail

# Set repository/model paths and dit_original_ckpt in the selected JSON config.
lightx2v_path=/data/zhangbilang/LightX2V
model_path=/data/zhangbilang/models/MiniMaxAI/MiniMax-H3

global_file=/data/zhangbilang/models/MiniMaxAI/minimax_h3_causal/assets/refa2v/global.txt
motion_file=/data/zhangbilang/models/MiniMaxAI/minimax_h3_causal/assets/refa2v/motions.yaml
image_path=/data/zhangbilang/models/MiniMaxAI/minimax_h3_causal/assets/refa2v/image.png
audio_path=/data/zhangbilang/models/MiniMaxAI/minimax_h3_causal/assets/refa2v/audio.wav

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=0

# Load the same prompt and frame-based motions as post_refa2v.py.
prompt=$(< "${global_file}")
motions=$(python -c 'import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding="utf-8")), ensure_ascii=False))' "${motion_file}")

# Eight-GPU BF16 inference. --size is HEIGHT WIDTH.
python -m torch.distributed.run --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3_causal \
  --task refa2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_causal/h100/refa2v_realtime_768p_sp8.json" \
  --prompt "${prompt}" \
  --image_path "${image_path}" \
  --audio_path "${audio_path}" \
  --motions "${motions}" \
  --seed 42 \
  --num_frames 974 \
  --size 1344 768 \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_causal_refa2v.mp4"
