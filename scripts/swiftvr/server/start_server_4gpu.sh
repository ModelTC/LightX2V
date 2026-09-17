#!/bin/bash

# set path firstly
lightx2v_path=path/to/LightX2V
model_path=path/to/SwiftVR_lightx2v

config_path=${lightx2v_path}/configs/swiftvr/h100/swiftvr_parallel.json

export CUDA_VISIBLE_DEVICES=0,1,2,3

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

# Reuse post_video.py; the request format is the same for single- and multi-GPU services.
torchrun --nproc_per_node 4 -m lightx2v.server \
  --model_cls swiftvr \
  --task sr \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --host 0.0.0.0 \
  --port 8000

echo "Service stopped"
