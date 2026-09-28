#!/bin/bash
set -eo pipefail

# Set the project/model paths here and the checkpoint paths in the JSON config.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/Qwen-Image-2.1
config_json=${CONFIG_JSON:-"${lightx2v_path}/configs/qwen_image_21/qwen_image_21_5090_2k.json"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/girl.png" \
    --prompt "Transform the person in the reference image into a cinematic fantasy portrait, preserving identity and facial features, with soft golden light and detailed clothing." \
    --size "${HEIGHT:-2048}" "${WIDTH:-2048}" \
    --seed 42 \
    --save_result_path "${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/qwen_image_21_i2i_5090_2k.png}"
