#!/bin/bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --model-variant in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --model-variant fl2av with the same JSON config and adaln_cache_dir.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh

set -euo pipefail

# System management interface: mthreads-gmi

# paths and runtime settings
lightx2v_path=/data/LightX2V
model_path=/data/MiniMax-H3

# set environment variables
export PLATFORM=musa
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=${MOORCAT_PYTHONPATH:-/data/wushuo/LightX2V-0905/third_party}:${PYTHONPATH:-}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls minimax_h3 \
    --model-variant fl2av \
    --task ref2av \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mthreads_musa/minimax_h3_ref2av_lora8_tp8_moorcat_h3_sparse.json" \
    --prompt "${prompt}" \
    --image_path "01_character.png,02_ui_style.png" \
    --size 736 1280 \
    --num_frames 362 \
    --save_result_path "${lightx2v_path}/save_results/minimax_h3_ref2av_lora8_tp8_moorcat_h3_sparse.mp4" \
    --seed 42
