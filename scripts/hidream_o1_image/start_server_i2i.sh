#!/bin/bash

lightx2v_path=/data/nvme1/yongyang/nb/LightX2V
model_path=/data/nvme1/yongyang/nb/models/HiDream-ai/HiDream-O1-Image

host=0.0.0.0
port=8000

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
--model_cls hidream_o1_image \
--task i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_i2i.json \
--host "${host}" \
--port "${port}"
