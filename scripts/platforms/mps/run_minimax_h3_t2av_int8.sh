#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3
export LIGHTX2V_METALLIB_DIR=/path/to/h3-metallib

export PLATFORM=mps
export SENSITIVE_LAYER_DTYPE=FP32
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls minimax_h3 \
    --model-variant fl2av \
    --task t2av \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_int8.json" \
    --prompt "A cinematic fox walking through a snowy forest" \
    --save_result_path "${lightx2v_path}/save_results/minimax_h3_t2av_int8.mp4" \
    --seed 42
