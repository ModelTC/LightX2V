#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan/wan_t2v.json \
--host 0.0.0.0 \
--port 8000

echo "Service stopped"
