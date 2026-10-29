#!/bin/bash
set -e

lightx2v_path=/data/Lightx2v
model_path=/data/wushuo/models/Qwen-Image-2.1

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mthreads_musa/qwen_image_21_dmd_4step.json" \
    --prompt 'A coffee shop entrance with a chalkboard reading "Qwen Coffee, $2 per cup" and a neon sign displaying "通义千问".' \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i_dmd_4step.png" \
    "$@"
