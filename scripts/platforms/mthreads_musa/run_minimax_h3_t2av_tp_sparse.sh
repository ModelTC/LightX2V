#!/bin/bash

lightx2v_path=/data/wushuo/LightX2V
model_path=/work/data-backup-afs/MiniMax-H3
mate_path=/data/wushuo/mate_h3_ratio1_026/python

export PLATFORM=musa
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${mate_path}:${PYTHONPATH}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

config_path=${lightx2v_path}/configs/platforms/mthreads_musa/minimax_h3_t2av_bf16_tp8_sparse.json
output_path=${lightx2v_path}/save_results/minimax_h3_t2av_tp_sparse.mp4
log_path=${lightx2v_path}/save_results/minimax_h3_t2av_tp_sparse.log

prompt='In a snowy blue-purple forest, a small white spirit walks carefully past a sleeping giant while soft wind and distant birds create an immersive soundscape.'

nohup torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls minimax_h3 \
    --task t2av \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --prompt "${prompt}" \
    --save_result_path "${output_path}" \
    --seed 0 \
    --warmup > "${log_path}" 2>&1 &
