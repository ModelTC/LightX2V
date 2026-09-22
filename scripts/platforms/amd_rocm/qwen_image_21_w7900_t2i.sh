#!/bin/bash

# System management interface: amd-smi

# set path firstly
lightx2v_path=
model_path=

export PLATFORM=amd_rocm
export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/amd_rocm/qwen_image_21_w7900_int8_compile_sage.json" \
    --prompt "A capybara wearing a wizard hat, oil painting" \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_w7900_t2i.png"
