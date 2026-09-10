#!/bin/bash

lightx2v_path=/path/to/Lightx2v
model_path=/path/to/SekoTalk-Distill-fp8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


torchrun --nproc-per-node 8 -m lightx2v.infer \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/seko_talk_13_fp8_dist_bucket_shape_8gpus_5s_realtime.json \
--prompt  "The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze." \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk.mp4
