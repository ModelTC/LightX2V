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
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mlu/qwen_image_21_dmd_4step.json" \
    --prompt 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition.' \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i_dmd_4step.png" \
    "$@"
