#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1 \
--task flf2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan/wan_flf2v.json \
--prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective." \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path ${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png \
--last_frame_path ${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_flf2v.mp4
