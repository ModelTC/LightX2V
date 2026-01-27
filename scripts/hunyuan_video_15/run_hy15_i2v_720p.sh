#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--seed 123 \
--model_cls hunyuan_video_1.5 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hunyuan_video_15/hunyuan_video_i2v_720p.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--negative_prompt "" \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_hunyuan_video_15_i2v.mp4
