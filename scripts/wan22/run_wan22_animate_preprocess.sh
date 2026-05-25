#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/dok/bugs/v1/LightX2V
model_path=/data/nvme1/wushuo/hf_models/Wan2.2-Animate-14B
video_path=/data/nvme1/yongyang/dok/bugs/examples/qqqq/input1.mp4
refer_path=/data/nvme1/yongyang/dok/bugs/examples/qqqq/src_ref.png

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# animate preprocess without drop_tail_invalid_frames
python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
    --ckpt_path ${model_path}/process_checkpoint \
    --video_path $video_path  \
    --refer_path $refer_path \
    --save_path ${lightx2v_path}/save_results/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag

# animate preprocess with drop_tail_invalid_frames
# python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
#     --ckpt_path ${model_path}/process_checkpoint \
#     --video_path $video_path  \
#     --refer_path $refer_path \
#     --save_path ${lightx2v_path}/save_results/animate/process_results \
#     --resolution_area 1280 720 \
#     --retarget_flag \
#     --drop_tail_invalid_frames

# replace preprocess without drop_tail_invalid_frames
# python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
#     --ckpt_path ${model_path}/process_checkpoint \
#     --video_path $video_path  \
#     --refer_path $refer_path \
#     --save_path ${lightx2v_path}/save_results/animate/process_results \
#     --resolution_area 1280 720 \
#     --iterations 3 \
#     --k 7 \
#     --w_len 1 \
#     --h_len 1 \
#     --replace_flag

# replace preprocess with drop_tail_invalid_frames
# python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
#     --ckpt_path ${model_path}/process_checkpoint \
#     --video_path $video_path  \
#     --refer_path $refer_path \
#     --save_path ${lightx2v_path}/save_results/animate/process_results \
#     --resolution_area 1280 720 \
#     --iterations 3 \
#     --k 7 \
#     --w_len 1 \
#     --h_len 1 \
#     --replace_flag \
#     --drop_tail_invalid_frames
