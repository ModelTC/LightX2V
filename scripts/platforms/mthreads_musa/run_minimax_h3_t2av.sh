#!/bin/bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --task in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --task fl2av for t2av/i2av/l2av/fl2av, or --task ref2av for ref2av.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
# Cache generation and inference must use the same JSON config and adaln_cache_dir.

# System management interface: mthreads-gmi

# set path firstly
lightx2v_path=/data/yongyang/LightX2V
model_path=/data/MiniMax-H3

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES=7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

prompt='In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.'

nohup python -m lightx2v.infer \
    --model_cls minimax_h3 \
    --task t2av \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/mthreads_musa/minimax_h3_t2av.json \
    --prompt "$prompt" \
    --save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av.mp4 \
    --seed 0 \
     > ${lightx2v_path}/save_results/minimax_h3_t2av_768p_362_1gpu_tp1-a.log 2>&1 &
