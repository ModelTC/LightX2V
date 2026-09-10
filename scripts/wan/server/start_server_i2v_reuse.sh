#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.1-I2V-14B-480P

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.server \
  --model_cls wan2.1 \
  --task i2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/wan/wan_i2v_reuse.json \
  --host 0.0.0.0 \
  --port 8000
