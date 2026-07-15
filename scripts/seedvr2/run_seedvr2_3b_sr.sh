#!/bin/bash

# set path and first
lightx2v_path=
model_path=path/to/seedvr2-3b/

video_path=path/to/test.mp4

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
