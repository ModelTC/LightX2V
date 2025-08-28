#!/bin/bash

# set path and first

lightx2v_path=
model_path=

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ENABLE_GRAPH_MODE=false
export SENSITIVE_LAYER_DTYPE=None

#for debugging
#export TORCH_NCCL_BLOCKING_WAIT=1 #启用 NCCL 阻塞等待模式（否则 watchdog 会杀死卡顿的进程）
#export NCCL_BLOCKING_WAIT_TIMEOUT=1800 #设置 watchdog 的等待超时

torchrun --nproc-per-node 4 -m lightx2v.infer \
--model_cls wan2.1_audio \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/audio_driven/wan_i2v_audio_dist.json \
--prompt  "The video features a old lady is saying something and knitting a sweater." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path ${lightx2v_path}/assets/inputs/audio/15.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/15.wav \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v_audio.mp4
