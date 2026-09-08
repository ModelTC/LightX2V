#!/bin/bash

lightx2v_path=/data/wushuo/LightX2V
model_path=/data/MiniMax-H3

export PLATFORM=musa
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

config_path=${lightx2v_path}/configs/platforms/mthreads_musa/minimax_h3_t2av_tp8_sparse.json
output_path=${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp8_sparse.mp4
log_path=${lightx2v_path}/save_results/minimax_h3_t2av_544p_124_8gpu_tp8_sparse.log

prompt='In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.'

nohup torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls minimax_h3 \
    --task t2av \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --prompt "${prompt}" \
    --save_result_path "${output_path}" \
    --seed 0 \
    > "${log_path}" 2>&1 &
