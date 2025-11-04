#!/bin/bash

# set path and first
lightx2v_path=
model_path=
video_path=
refer_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# process
python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
    --ckpt_path ${model_path}/process_checkpoint \
    --video_path $video_path  \
    --refer_path $refer_path \
    --save_path ${lightx2v_path}/save_results/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag \

python -m lightx2v.infer \
--model_cls wan2.2_animate \
--task animate \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22/wan_animate_lora.json \
--prompt "视频中的人在做动作" \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_animate_lora.mp4
