#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls qwen_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/qwen_image/qwen_image_t2i.json \
--port 8000

echo "Service stopped"


# {
#   "prompt": "a beautiful sunset over the ocean",
#   "aspect_ratio": "16:9",
#   "infer_steps": 50
# }
