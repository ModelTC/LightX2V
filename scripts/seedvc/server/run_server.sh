#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_variant=${1:-svc}
model_path=/path/to/models/Seed-VC

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
  --model_cls seedvc \
  --task a2a \
  --model_path "${model_path}/${model_variant%_lora}" \
  --model-variant "$model_variant" \
  --config_json ${lightx2v_path}/configs/seedvc/seedvc.json \
  --port 8018 \
  --metric_port 8019
