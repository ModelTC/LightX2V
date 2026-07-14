#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-VACE-Fun-A14B
# model_path=/path/to/Wan2.2-VACE-Fun-A14B-INT8

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.2_moe_vace \
--task vace \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22_vace/a800/bf16/wan22_moe_vace.json \
--prompt "图片的女人，穿着白色连衣裙，模仿视频的动作，翩翩起舞." \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--src_video /path/to/post+depth.mp4 \
--src_ref_images /path/to/image.png \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_vace.mp4\
