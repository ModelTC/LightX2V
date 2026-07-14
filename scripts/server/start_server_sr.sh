#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/ByteDance-Seed/SeedVR2-3B

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls seedvr2 \
--task sr \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seedvr/seedvr2_3b.json \
--port 8000

echo "Service stopped"

# {
#     "video_path": "input.mp4",
#     "seed": 42,
#     "save_result_path": "./output_lightx2v_seedvr2_sr.mp4"
# }
