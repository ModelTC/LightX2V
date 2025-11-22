#!/bin/bash

lightx2v_path=/path/to/Lightx2v
model_path=/path/to/SekoTalk-Distill-fp8

export CUDA_VISIBLE_DEVICES=0,1,2,3

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


export SENSITIVE_LAYER_DTYPE=None

torchrun --nproc-per-node 4 -m lightx2v.infer \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/seko_talk_10_fp8_dist_fixed_min_area.json \
--prompt  "The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk.mp4
