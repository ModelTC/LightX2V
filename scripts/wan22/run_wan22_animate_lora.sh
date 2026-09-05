#!/bin/bash

# set path firstly
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
--src_pose_path ${lightx2v_path}/save_results/animate/process_results/src_pose.mp4 \
--src_face_path ${lightx2v_path}/save_results/animate/process_results/src_face.mp4 \
--src_ref_images ${lightx2v_path}/save_results/animate/process_results/src_ref.png \
--prompt "视频中的人在做动作" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_animate_lora.mp4
