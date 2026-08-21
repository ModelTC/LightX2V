#!/bin/bash

# set path firstly
lightx2v_path=path/to/LightX2V
model_path=path/to/SwiftVR_lightx2v
video_path=path/to/test.mp4

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Choose one output-size option: --target_shape uses HEIGHT WIDTH; --sr_ratio scales both input dimensions.
# The command below uses --sr_ratio. Replace it with --target_shape 1440 2520 to set an exact output size.
python -m lightx2v.infer \
--model_cls swiftvr \
--task sr \
--video_path ${video_path} \
--sr_ratio 2 \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/swiftvr/swiftvr.json \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_swiftvr_sr.mp4
