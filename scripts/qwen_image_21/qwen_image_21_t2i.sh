#!/bin/bash

lightx2v_path=/Path/To/LightX2V
model_path=/Path/To/Model

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/qwen_image_21/qwen_image_21.json" \
    --prompt "A capybara wearing a wizard hat, oil painting" \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i.png"
