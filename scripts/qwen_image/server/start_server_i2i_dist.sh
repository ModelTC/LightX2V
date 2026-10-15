#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Qwen-Image-Edit-2511

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.server \
  --model_cls qwen_image \
  --task i2i \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i_2511_dist.json \
  --host 0.0.0.0 \
  --port 8000
