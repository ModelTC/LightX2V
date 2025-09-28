#!/bin/bash

# set path and first
lightx2v_path=/path/to/Lightx2v
model_path=/path/to/Wan2.1-R2V0909-Audio-14B-720P-fp8

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/seko_talk_05_offload_fp8_4090.json \
--port 8000

echo "Service stopped"
