#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=2

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--seed 123 \
--model_cls hunyuan_video_1.5 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hunyuan_video_15/hunyuan_video_t2v_480p.json \
--prompt "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style." \
--negative_prompt "" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_hunyuan_video_15_t2v.mp4
