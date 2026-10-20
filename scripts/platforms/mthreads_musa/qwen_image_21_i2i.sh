#!/bin/bash
set -e

lightx2v_path=/data/Lightx2v
model_path=/data/wushuo/models/Qwen-Image-2.1
image_path="${lightx2v_path}/assets/inputs/imgs/img_0.jpg"

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mthreads_musa/qwen_image_21_dmd_4step.json" \
    --prompt "Transform the cat into a cream-colored dog. Keep the sunglasses, pose, kayak, background and lighting unchanged." \
    --image_path "${image_path}" \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_i2i_dmd_4step.png" \
    "$@"
