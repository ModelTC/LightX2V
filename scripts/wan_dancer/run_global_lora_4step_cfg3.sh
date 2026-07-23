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
    --config_json ${lightx2v_path}/configs/wan_dancer/global_lora_4step_cfg3.json \
    --seed 0 \
    --image_path ${wan_dancer_github_path}/gen_video/ref_image/3001.jpg \
    --audio_path ${wan_dancer_github_path}/gen_video/music/KPopDance.WAV \
    --prompt "$(<${wan_dancer_github_path}/gen_video/prompt/kpop_global.txt)" \
    --negative_prompt '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走' \
    --save_result_path ${lightx2v_path}/save_results/wan_dancer_global_lora_4step_cfg3.mp4
