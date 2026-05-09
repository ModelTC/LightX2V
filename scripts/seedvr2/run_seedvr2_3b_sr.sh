#!/bin/bash

# set path and first
lightx2v_path=/mnt/devsft_afs_2/gushiqiao/new/LightX2V
model_path=/models/seedvr2-3b/

video_path=/mnt/devsft_afs_2/gushiqiao/output_lightx2v_lingbot_fast_i2v_quant.mp4

export CUDA_VISIBLE_DEVICES=0

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
