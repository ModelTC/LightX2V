#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
torchrun -m lightx2v.server \
--model_cls seko_talk \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/seko_talk_05_offload_fp8_4090.json \
--port 8000 \
--nproc_per_node 1

echo "Service stopped"
