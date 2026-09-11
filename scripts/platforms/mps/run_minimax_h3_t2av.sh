#!/bin/bash

# set paths firstly
lightx2v_path=/Users/yongyang/Documents/x2v/LightX2V
model_path=/Users/yongyang/Documents/x2v/models/MiniMaxAI/diffusers/MiniMax-H3

# set environment variables
export PLATFORM=mps
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${lightx2v_path}:$PYTHONPATH"

prompt='A cinematic fox walking through a snowy forest'

mkdir -p "${lightx2v_path}/save_results"

/opt/miniconda3/envs/torch/bin/python -m lightx2v.infer \
    --model_cls minimax_h3 \
    --model-variant fl2av \
    --task t2av \
    --model_path "$model_path" \
    --config_json "${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_512_22.json" \
    --prompt "$prompt" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av.mp4" \
    --seed 42
