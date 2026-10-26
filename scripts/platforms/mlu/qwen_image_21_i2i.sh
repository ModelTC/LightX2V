#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Qwen-Image-2.1

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mlu/qwen_image_21_dmd_4step.json" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --prompt "Transform the cat in the reference image into a dog." \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_i2i_dmd_4step.png" \
    "$@"
