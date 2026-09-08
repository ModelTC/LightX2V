#!/bin/bash

lightx2v_path=/data/nvme1/yongyang/dan/LightX2V
wan_dancer_github_path=/data/nvme1/yongyang/dan/Wan-Dancer
model_path=/data/nvme1/yongyang/dan/Wan-Dancer/models/Wan-AI/Wan-Dancer-14B

export CUDA_VISIBLE_DEVICES=0,1,2,3

source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=4 -m lightx2v.infer \
    --model_cls wan_dancer \
    --task s2v \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/wan_dancer/global_lora_4step.json \
    --seed 0 \
    --image_path ${wan_dancer_github_path}/gen_video/ref_image/3001.jpg \
    --audio_path ${wan_dancer_github_path}/gen_video/music/KPopDance.WAV \
    --prompt "$(<${wan_dancer_github_path}/gen_video/prompt/kpop_global.txt)" \
    --save_result_path ${lightx2v_path}/save_results/wan_dancer_global_lora_4step.mp4
