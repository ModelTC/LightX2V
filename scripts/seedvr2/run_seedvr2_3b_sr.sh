#!/bin/bash

# set path and first
lightx2v_path=/path/to/LightX2V
model_path=/path/to/ByteDance-Seed/SeedVR2-3B

video_path=/path/to/input.mp4


export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls seedvr2 \
--task sr \
--sr_ratio 2.0 \
--video_path $video_path \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seedvr/seedvr2_3b.json \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seedvr2_sr.mp4
