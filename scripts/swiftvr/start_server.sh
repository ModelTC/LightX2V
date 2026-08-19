#!/bin/bash

# set path firstly
lightx2v_path=path/to/LightX2V
model_path=path/to/SwiftVR_lightx2v

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start API server
python -m lightx2v.server \
--model_cls swiftvr \
--task sr \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/swiftvr/swiftvr.json \
--host 0.0.0.0 \
--port 8000

echo "Service stopped"
