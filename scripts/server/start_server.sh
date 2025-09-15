#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.api_server \
--model_cls wan2.1_distill \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/distill/wan_i2v_distill_4step_cfg.json \
--port 8000 \
--nproc_per_node 1

echo "Service stopped"
