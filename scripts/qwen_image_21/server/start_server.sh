#!/bin/bash

lightx2v_path=/Path/To/LightX2V
model_path=/Path/To/Model

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
    --model_cls qwen_image_21 \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/qwen_image_21/qwen_image_21.json" \
    --host 0.0.0.0 \
    --port 8000
