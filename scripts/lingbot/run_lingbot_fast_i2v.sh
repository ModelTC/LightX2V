#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme4/gushiqiao/LightX2V
model_path=/data/nvme4/models/lingbot-world-base-cam

export CUDA_VISIBLE_DEVICES=1,2,3,4

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=4 -m lightx2v.infer \
--model_cls lingbot_world_fast \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/lingbot_fast/lingbot_fast_i2v.json \
--prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere." \
--negative_prompt "" \
--image_path /data/nvme4/gushiqiao/lingbot-world/examples/03/image.jpg \
--action_path /data/nvme4/gushiqiao/lingbot-world/examples/03/ \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_fast_i2v.mp4
