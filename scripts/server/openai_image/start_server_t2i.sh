#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/nb/LightX2V
model_path=/data/nvme1/models/Qwen/Qwen-Image-2512

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls qwen_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512_distill.json \
--port 8000

echo "Service stopped"


# {
#   "prompt": "a beautiful sunset over the ocean",
#   "aspect_ratio": "16:9",
#   "infer_steps": 50
# }
