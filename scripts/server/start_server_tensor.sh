#!/bin/bash

lightx2v_path=/data/nvme4/gushiqiao/LightX2V
model_path=/data/nvme4/gushiqiao/realtime-video/wan_models/Wan2.1-T2V-1.3B

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

cd "${lightx2v_path}"

python3 -m lightx2v.server \
  --model_cls wan2.1 \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/wan/wan_t2v.json" \
  --host 0.0.0.0 \
  --port 8000

echo "Service stopped"
