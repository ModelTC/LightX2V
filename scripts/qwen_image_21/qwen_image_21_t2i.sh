#!/bin/bash

lightx2v_path=/data/nvme1/yongyang/dan/LightX2V
model_path=/data/nvme1/yongyang/dan/models/Qwen/Qwen-Image-2.1

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/qwen_image_21/qwen_image_21.json" \
    --prompt "一只戴着巫师帽的水豚坐在书桌前，在烛光下阅读一本书，油画风格，细腻笔触，暖色调。" \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i.png"
