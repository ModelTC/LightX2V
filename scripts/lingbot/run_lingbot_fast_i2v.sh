#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/lingbot-world-base-cam

export CUDA_VISIBLE_DEVICES=7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls lingbot_world_fast \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/lingbot_fast/lingbot_fast_i2v.json \
--prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere." \
--image_path /path/to/lingbot-world/examples/03/image.jpg \
--action_path /path/to/lingbot-world/examples/03 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_fast_i2v.mp4
