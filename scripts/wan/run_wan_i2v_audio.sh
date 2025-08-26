#!/bin/bash

# set path and first
lightx2v_path=/mtc/gushiqiao/llmc_workspace/LightX2V
model_path=/data/nvme0/gushiqiao/models/Lightx2v_models/Wan2.1-R2V721-Audio-14B-720P


export CUDA_VISIBLE_DEVICES=2

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ENABLE_GRAPH_MODE=false
export SENSITIVE_LAYER_DTYPE=None

python -m lightx2v.infer \
--model_cls wan2.1_audio \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/audio_driven/wan_i2v_audio.json \
--prompt  "The video features a old lady is saying something and knitting a sweater." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path ${lightx2v_path}/assets/inputs/audio/15.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/15.wav \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v_audio.mp4
