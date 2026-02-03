#!/bin/bash

# set path firstly
lightx2v_path=
model_path=
lora_dir=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start API server with LoRA support
python -m lightx2v.server \
--model_cls qwen_image \
--task i2i \
--model_path $model_path \
--lora_dir $lora_dir \
--config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i_2511.json \
--port 8000

echo "Service stopped"
