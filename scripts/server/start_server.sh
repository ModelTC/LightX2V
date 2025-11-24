#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls hunyuan_video_1.5_distill \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hunyuan_video_15/hunyuan_video_t2v_480p_distill.json \
--port 8000

echo "Service stopped"
