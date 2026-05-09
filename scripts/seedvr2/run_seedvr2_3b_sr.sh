#!/bin/bash

# set path and first
lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme0/gushiqiao/models/seedvr2-3b/

video_path=/data/nvme4/gushiqiao/videos/output_lightx2v_wan_t2v_sf.mp4


export CUDA_VISIBLE_DEVICES=1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls seedvr2 \
--task sr \
--sr_ratio 2.0 \
--video_path $video_path \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seedvr/4090/seedvr2_3b.json \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seedvr2_sr.mp4
