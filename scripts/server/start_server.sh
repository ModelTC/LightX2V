#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/ddc/yong/LightX2V
model_path=/data/nvme0/models/Wan-AI/Wan2.1-T2V-1.3B

export CUDA_VISIBLE_DEVICES=7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan/wan_t2v.json \
--port 8000

echo "Service stopped"
